import json

from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    SecretValue,
    BundlingOptions,
    aws_iam as iam,
    aws_lambda as _lambda,
    aws_apigateway as apigateway,
    aws_bedrock as bedrock,
    aws_opensearchserverless as ops,
    aws_s3 as s3,
    aws_logs as logs,
    aws_secretsmanager as secretsmanager,
    aws_ssm as ssm,
    custom_resources as cr,
)
from cdk_nag import NagSuppressions
from constructs import Construct

from .utils.solution_bundling import SolutionBundling


RAG_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
SLACK_SLASH_COMMAND = "/askblue"
COLLECTION_NAME = "slack-bedrock-vector-db"
VECTOR_INDEX_NAME = "slack-bedrock-os-index"
BEDROCK_KB_NAME = "slack-bedrock-kb"
BEDROCK_KB_DATA_SOURCE = "slack-bedrock-kb-ds"
LAMBDA_MEMORY_SIZE = 265


class BedrockSlackChatbotStack(Stack):
    def __init__(self, scope: Construct, id: str, env_name="dev", **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Get secrets from context or fail if not provided
        slack_bot_token = self.node.try_get_context("slackBotToken")
        slack_signing_secret = self.node.try_get_context("slackSigningSecret")

        if not slack_bot_token or not slack_signing_secret:
            raise Exception(
                "Missing required context variables. Please provide slackBotToken and slackSigningSecret"
            )

        # Create Secrets in Secrets Manager with environment name in secret names
        slack_bot_token_secret = secretsmanager.Secret(
            self,
            f"slack-bot-token-secret-{env_name}",
            secret_name=f"/{env_name}/slack/bot-token",
            description="Slack Bot User OAuth Token",
            secret_string_value=SecretValue.unsafe_plain_text(
                json.dumps({"token": slack_bot_token})
            ),
        )

        slack_bot_signing_secret = secretsmanager.Secret(
            self,
            f"slack-bot-signing-secret-{env_name}",
            secret_name=f"/{env_name}/slack/signing-secret",
            description="Slack Signing Secret",
            secret_string_value=SecretValue.unsafe_plain_text(
                json.dumps({"secret": slack_signing_secret})
            ),
        )

        # Create SSM parameters that reference the secrets, including the env_name
        bot_token_parameter = ssm.StringParameter(
            self,
            f"slack-bot-token-parameter-{env_name}",
            parameter_name=f"/{env_name}/slack/bot-token/parameter",
            string_value=f"{{{{resolve:secretsmanager:{slack_bot_token_secret.secret_name}}}}}",
            description="Reference to Slack Bot Token in Secrets Manager",
            tier=ssm.ParameterTier.STANDARD,
        )

        signing_secret_parameter = ssm.StringParameter(
            self,
            f"slack-bot-signing-secret-parameter-{env_name}",
            parameter_name=f"/{env_name}/slack/signing-secret/parameter",
            string_value=f"{{{{resolve:secretsmanager:{slack_bot_signing_secret.secret_name}}}}}",
            description="Reference to Slack Signing Secret in Secrets Manager",
            tier=ssm.ParameterTier.STANDARD,
        )

        # Define an S3 bucket (bucket names must be lowercase and can include hyphens)
        kb_bucket = s3.Bucket(
            self,
            f"kb-bucket-{env_name}",
            bucket_name=f"kb-bucket-{env_name}-{self.account}-{self.region}",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            enforce_ssl=True,
        )

        NagSuppressions.add_resource_suppressions(
            kb_bucket,
            [
                {
                    "id": "AwsSolutions-S1",
                    "reason": "S3 access logging not required for sample code",
                }
            ],
        )

        aws_account = self.account

        # Create IAM policies for S3 access
        s3_access_list_policy = iam.PolicyStatement(
            actions=["s3:ListBucket"],
            resources=[kb_bucket.bucket_arn],
        )
        s3_access_list_policy.add_condition(
            "StringEquals", {"aws:ResourceAccount": aws_account}
        )

        s3_access_get_policy = iam.PolicyStatement(
            actions=["s3:GetObject", "s3:Delete*"],
            resources=[f"{kb_bucket.bucket_arn}/*"],
        )
        s3_access_get_policy.add_condition(
            "StringEquals", {"aws:ResourceAccount": aws_account}
        )

        # IAM policy to invoke Bedrock models and access the embedding model
        bedrock_execution_role_policy = iam.PolicyStatement(
            actions=["bedrock:InvokeModel"],
            resources=[
                f"arn:aws:bedrock:{self.region}::foundation-model/{EMBEDDING_MODEL}"
            ],
        )

        # IAM policy to delete Bedrock knowledge base
        bedrock_kb_delete_role_policy = iam.PolicyStatement(
            actions=["bedrock:Delete*"],
            resources=[
                f"arn:aws:bedrock:{self.region}:{self.account}:knowledge-base/*"
            ],
        )

        # IAM policy for OpensearchServerless access
        bedrock_oss_policy_for_kb = iam.PolicyStatement(
            actions=[
                "aoss:APIAccessAll",
                "aoss:DeleteAccessPolicy",
                "aoss:DeleteCollection",
                "aoss:DeleteLifecyclePolicy",
                "aoss:DeleteSecurityConfig",
                "aoss:DeleteSecurityPolicy",
            ],
            resources=[f"arn:aws:aoss:{self.region}:{self.account}:collection/*"],
        )

        # Define IAM Role for Bedrock execution
        bedrock_execution_role = iam.Role(
            self,
            f"bedrock-execution-role-{env_name}",
            assumed_by=iam.ServicePrincipal("bedrock.amazonaws.com"),
        )
        bedrock_execution_role.add_to_policy(bedrock_execution_role_policy)
        bedrock_execution_role.add_to_policy(bedrock_oss_policy_for_kb)
        bedrock_execution_role.add_to_policy(s3_access_list_policy)
        bedrock_execution_role.add_to_policy(s3_access_get_policy)
        bedrock_execution_role.add_to_policy(bedrock_kb_delete_role_policy)

        # Create bedrock Guardrails for the slack bot with env_name appended to the name
        guardrail = bedrock.CfnGuardrail(
            self,
            f"slack-bot-guardrail-{env_name}",
            blocked_input_messaging="Sorry, slack bot cannot provide a response for this question",
            blocked_outputs_messaging="Sorry, slack bot cannot provide a response for this question",
            name=f"slack-bedrock-guardrail-{env_name}",
            description="Bedrock Guardrails for Slack bedrock bot",
            content_policy_config={
                "filtersConfig": [
                    {
                        "type": "SEXUAL",
                        "inputStrength": "HIGH",
                        "outputStrength": "HIGH",
                    },
                    {
                        "type": "VIOLENCE",
                        "inputStrength": "HIGH",
                        "outputStrength": "HIGH",
                    },
                    {"type": "HATE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
                    {
                        "type": "INSULTS",
                        "inputStrength": "HIGH",
                        "outputStrength": "HIGH",
                    },
                    {
                        "type": "MISCONDUCT",
                        "inputStrength": "HIGH",
                        "outputStrength": "HIGH",
                    },
                    {
                        "type": "PROMPT_ATTACK",
                        "inputStrength": "HIGH",
                        "outputStrength": "NONE",
                    },
                ]
            },
            sensitive_information_policy_config={
                "piiEntitiesConfig": [
                    {"type": "EMAIL", "action": "ANONYMIZE"},
                    {"type": "PHONE", "action": "ANONYMIZE"},
                    {"type": "NAME", "action": "ANONYMIZE"},
                    {"type": "CREDIT_DEBIT_CARD_NUMBER", "action": "BLOCK"},
                ]
            },
            word_policy_config={"managedWordListsConfig": [{"type": "PROFANITY"}]},
        )

        guardrail_version = bedrock.CfnGuardrailVersion(
            self,
            f"slack-bot-guardrail-version-{env_name}",
            guardrail_identifier=guardrail.attr_guardrail_id,
            description="v1.0",
        )

        # Variables for Guardrail ID and version
        GUARD_RAIL_ID = guardrail.attr_guardrail_id
        GUARD_RAIL_VERSION = guardrail_version.attr_version

        # Define OpenSearchServerless Collection with env_name appended to the name
        os_collection = ops.CfnCollection(
            self,
            f"os-collection-{env_name}",
            name=f"{COLLECTION_NAME}-{env_name}",
            description="Slack bedrock vector db",
            type="VECTORSEARCH",
        )

        # Define AOSS vector DB encryption policy with AWSOwnedKey true, including env_name in the policy name
        aoss_encryption_policy = ops.CfnSecurityPolicy(
            self,
            f"aoss-encryption-policy-{env_name}",
            name=f"bedrock-kb-encryption-policy-{env_name}",
            type="encryption",
            policy=json.dumps(
                {
                    "Rules": [
                        {
                            "ResourceType": "collection",
                            "Resource": [f"collection/{COLLECTION_NAME}-{env_name}"],
                        }
                    ],
                    "AWSOwnedKey": True,
                }
            ),
        )
        os_collection.add_dependency(aoss_encryption_policy)

        # Define Vector DB network policy with AllowFromPublic true and env_name in the policy name
        aoss_network_policy = ops.CfnSecurityPolicy(
            self,
            f"aoss-network-policy-{env_name}",
            name=f"bedrock-kb-network-policy-{env_name}",
            type="network",
            policy=json.dumps(
                [
                    {
                        "Rules": [
                            {
                                "ResourceType": "collection",
                                "Resource": [
                                    f"collection/{COLLECTION_NAME}-{env_name}"
                                ],
                            },
                            {
                                "ResourceType": "dashboard",
                                "Resource": [
                                    f"collection/{COLLECTION_NAME}-{env_name}"
                                ],
                            },
                        ],
                        "AllowFromPublic": True,
                    }
                ]
            ),
        )
        os_collection.add_dependency(aoss_network_policy)

        # Define create-index-function execution role and policy
        create_index_function_role = iam.Role(
            self,
            f"create-index-function-role-{env_name}",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
        )
        create_index_function_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "service-role/AWSLambdaBasicExecutionRole"
            )
        )
        create_index_function_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "aoss:APIAccessAll",
                    "aoss:DescribeIndex",
                    "aoss:ReadDocument",
                    "aoss:CreateIndex",
                    "aoss:DeleteIndex",
                    "aoss:UpdateIndex",
                    "aoss:WriteDocument",
                    "aoss:CreateCollectionItems",
                    "aoss:DeleteCollectionItems",
                    "aoss:UpdateCollectionItems",
                    "aoss:DescribeCollectionItems",
                ],
                resources=[
                    f"arn:aws:aoss:{self.region}:{self.account}:collection/*",
                    f"arn:aws:aoss:{self.region}:{self.account}:index/*",
                ],
                effect=iam.Effect.ALLOW,
            )
        )

        # Define Lambda function to create an opensearch serverless index
        create_index_function = _lambda.Function(
            self,
            f"create-index-function-{env_name}",
            runtime=_lambda.Runtime.PYTHON_3_12,
            code=_lambda.Code.from_asset("deploy/custom_resources/opensearch"),
            environment={
                "INDEX_NAME": os_collection.attr_id,
                "ENVIRONMENT_NAME": env_name,
            },
            handler="index_manager.handler",
            timeout=Duration.minutes(1),
            role=create_index_function_role,
        )

        # Define OpenSearchServerless access policy to access the index and collection
        aoss_access_policy = ops.CfnAccessPolicy(
            self,
            f"aoss-access-policy-{env_name}",
            name=f"bedrock-kb-access-policy-{env_name}",
            type="data",
            policy=json.dumps(
                [
                    {
                        "Rules": [
                            {
                                "ResourceType": "collection",
                                "Resource": ["collection/*"],
                                "Permission": ["aoss:*"],
                            },
                            {
                                "ResourceType": "index",
                                "Resource": ["index/*/*"],
                                "Permission": ["aoss:*"],
                            },
                        ],
                        "Principal": [
                            bedrock_execution_role.role_arn,
                            (
                                create_index_function.role.role_arn
                                if create_index_function.role
                                else ""
                            ),
                            f"arn:aws:iam::{self.account}:root",
                        ],
                    }
                ]
            ),
        )
        os_collection.add_dependency(aoss_access_policy)

        # Compute endpoint for vector index creation
        endpoint = f"{os_collection.attr_id}.{self.region}.aoss.amazonaws.com"

        # Define a custom resource to create the vector index using the Lambda function.
        vector_index = cr.AwsCustomResource(
            self,
            f"vector-index-{env_name}",
            install_latest_aws_sdk=True,
            on_create={
                "service": "Lambda",
                "action": "invoke",
                "parameters": {
                    "FunctionName": create_index_function.function_name,
                    "InvocationType": "RequestResponse",
                    "Payload": json.dumps(
                        {
                            "RequestType": "Create",
                            "CollectionName": os_collection.name,
                            "IndexName": f"{VECTOR_INDEX_NAME}-{env_name}",
                            "Endpoint": endpoint,
                        }
                    ),
                },
                "physical_resource_id": cr.PhysicalResourceId.of(
                    f"vector-index-{env_name}"
                ),
            },
            on_delete={
                "service": "Lambda",
                "action": "invoke",
                "parameters": {
                    "FunctionName": create_index_function.function_name,
                    "InvocationType": "RequestResponse",
                    "Payload": json.dumps(
                        {
                            "RequestType": "Delete",
                            "CollectionName": os_collection.name,
                            "IndexName": f"{VECTOR_INDEX_NAME}-{env_name}",
                            "Endpoint": endpoint,
                        }
                    ),
                },
            },
            policy=cr.AwsCustomResourcePolicy.from_statements(
                [
                    iam.PolicyStatement(
                        actions=["lambda:InvokeFunction"],
                        resources=[create_index_function.function_arn],
                    )
                ]
            ),
            timeout=Duration.seconds(60),
        )
        vector_index.node.add_dependency(os_collection)

        # Define a Bedrock knowledge base using OpenSearchServerless and the embedding model
        bedrock_kb = bedrock.CfnKnowledgeBase(
            self,
            f"bedrock-kb-{env_name}",
            name=f"{BEDROCK_KB_NAME}-{env_name}",
            description="bedrock knowledge base for aws",
            role_arn=bedrock_execution_role.role_arn,
            knowledge_base_configuration={
                "type": "VECTOR",
                "vectorKnowledgeBaseConfiguration": {
                    "embeddingModelArn": f"arn:aws:bedrock:{self.region}::foundation-model/{EMBEDDING_MODEL}"
                },
            },
            storage_configuration={
                "type": "OPENSEARCH_SERVERLESS",
                "opensearchServerlessConfiguration": {
                    "collectionArn": os_collection.attr_arn,
                    "fieldMapping": {
                        "vectorField": "bedrock-knowledge-base-default-vector",
                        "textField": "AMAZON_BEDROCK_TEXT_CHUNK",
                        "metadataField": "AMAZON_BEDROCK_METADATA",
                    },
                    "vectorIndexName": f"{VECTOR_INDEX_NAME}-{env_name}",
                },
            },
        )
        bedrock_kb.node.add_dependency(vector_index)
        bedrock_kb.node.add_dependency(create_index_function)
        bedrock_kb.node.add_dependency(os_collection)
        bedrock_kb.node.add_dependency(bedrock_execution_role)

        # Define a Bedrock knowledge base data source using the S3 bucket
        bedrock_kb_data_source = bedrock.CfnDataSource(
            self,
            f"bedrock-kb-datasource-{env_name}",
            name=f"{BEDROCK_KB_DATA_SOURCE}-{env_name}",
            knowledge_base_id=bedrock_kb.attr_knowledge_base_id,
            data_source_configuration={
                "type": "S3",
                "s3Configuration": {
                    "bucketArn": kb_bucket.bucket_arn,
                },
            },
        )

        # Create IAM policies for the Slack integration Lambda
        lambda_bedrock_model_policy = iam.PolicyStatement(
            actions=["bedrock:InvokeModel"],
            resources=[
                f"arn:aws:bedrock:{self.region}::foundation-model/{RAG_MODEL_ID}"
            ],
        )

        lambda_bedrock_kb_policy = iam.PolicyStatement(
            actions=["bedrock:Retrieve", "bedrock:RetrieveAndGenerate"],
            resources=[
                f"arn:aws:bedrock:{self.region}:{self.account}:knowledge-base/{bedrock_kb.attr_knowledge_base_id}"
            ],
        )

        lambda_ssm_policy = iam.PolicyStatement(
            actions=["ssm:GetParameter"],
            resources=[
                f"arn:aws:ssm:{self.region}:{self.account}:parameter{bot_token_parameter.parameter_name}",
                f"arn:aws:ssm:{self.region}:{self.account}:parameter{signing_secret_parameter.parameter_name}",
            ],
        )

        lambda_reinvoke_policy = iam.PolicyStatement(
            actions=["lambda:InvokeFunction"],
            resources=[
                f"arn:aws:lambda:{self.region}:{self.account}:function:AmazonBedrock*"
            ],
        )

        lambda_gr_invoke_policy = iam.PolicyStatement(
            actions=["bedrock:ApplyGuardrail"],
            resources=[f"arn:aws:bedrock:{self.region}:{self.account}:guardrail/*"],
        )

        # Create the Slack bot Lambda function for handling slash commands.
        # Note: The environment variable "ENVIRONMENT_NAME" is also added.
        code_asset_path = "api"
        bedrock_kb_slackbot_function = _lambda.Function(
            self,
            f"bedrock-kb-slackbot-function-{env_name}",
            runtime=_lambda.Runtime.PYTHON_3_12,
            memory_size=LAMBDA_MEMORY_SIZE,
            environment={
                "RAG_MODEL_ID": RAG_MODEL_ID,
                "SLACK_SLASH_COMMAND": SLACK_SLASH_COMMAND,
                "KNOWLEDGEBASE_ID": bedrock_kb.attr_knowledge_base_id,
                "SLACK_BOT_TOKEN_PARAMETER": bot_token_parameter.parameter_name,
                "SLACK_SIGNING_SECRET_PARAMETER": signing_secret_parameter.parameter_name,
                "GUARD_RAIL_ID": GUARD_RAIL_ID,
                "GUARD_RAIL_VERSION": GUARD_RAIL_VERSION,
                "ENVIRONMENT_NAME": env_name,
            },
            handler="index.handler",
            code=_lambda.Code.from_asset(
                path=code_asset_path,
                bundling=BundlingOptions(
                    image=_lambda.Runtime.PYTHON_3_12.bundling_image,
                    local=SolutionBundling(source_path=code_asset_path),
                ),
            ),
            timeout=Duration.minutes(5),
        )

        # Grant the Lambda function permission to read the secrets
        slack_bot_token_secret.grant_read(bedrock_kb_slackbot_function)
        slack_bot_signing_secret.grant_read(bedrock_kb_slackbot_function)

        # Attach IAM policies to the Lambda function's execution role
        bedrock_kb_slackbot_function.add_to_role_policy(lambda_bedrock_model_policy)
        bedrock_kb_slackbot_function.add_to_role_policy(lambda_bedrock_kb_policy)
        bedrock_kb_slackbot_function.add_to_role_policy(lambda_reinvoke_policy)
        bedrock_kb_slackbot_function.add_to_role_policy(lambda_gr_invoke_policy)
        bedrock_kb_slackbot_function.add_to_role_policy(lambda_ssm_policy)
        bedrock_kb_slackbot_function.add_to_role_policy(
            iam.PolicyStatement(
                actions=["lambda:InvokeFunction"],
                resources=[bedrock_kb_slackbot_function.function_arn],
            )
        )

        # Define the API Gateway resource to trigger the Slack bot Lambda.
        # The log group name includes the environment name.
        bedrock_kb_slackbot_api = apigateway.LambdaRestApi(
            self,
            f"bedrock-kb-slackbot-api-{env_name}",
            cloud_watch_role=True,
            handler=bedrock_kb_slackbot_function,
            deploy_options=apigateway.StageOptions(
                access_log_destination=apigateway.LogGroupLogDestination(
                    logs.LogGroup(self, f"bedrock-kb-slackbot-api-loggroup-{env_name}")
                ),
                access_log_format=apigateway.AccessLogFormat.json_with_standard_fields(
                    caller=False,
                    http_method=True,
                    ip=True,
                    protocol=True,
                    request_time=True,
                    resource_path=True,
                    response_length=True,
                    status=True,
                    user=True,
                ),
            ),
            proxy=False,
        )

        slack_resource = bedrock_kb_slackbot_api.root.add_resource(
            "slack"
        ).add_resource("askblue")
        slack_resource.add_method("POST")

        # CDK NAG Suppression Rules - IAM
        NagSuppressions.add_resource_suppressions_by_path(
            self,
            [
                f"/bedrockslackchatbotstack-{env_name}/bedrock-execution-role-{env_name}/DefaultPolicy/Resource",
                f"/bedrockslackchatbotstack-{env_name}/create-index-function-role-{env_name}/DefaultPolicy/Resource",
                f"/bedrockslackchatbotstack-{env_name}/bedrock-kb-slackbot-function-{env_name}/ServiceRole/DefaultPolicy/Resource",
                f"/bedrockslackchatbotstack-{env_name}/create-index-function-role-{env_name}/Resource",
                f"/bedrockslackchatbotstack-{env_name}/bedrock-kb-slackbot-function-{env_name}/ServiceRole/Resource",
                f"/bedrockslackchatbotstack-{env_name}/bedrock-kb-slackbot-api-{env_name}/CloudWatchRole/Resource",
            ],
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "IAM policy ARN limits actions to the AWS Account and AWS Service with conditions",
                },
                {
                    "id": "AwsSolutions-IAM4",
                    "reason": "IAM managed policies used for sample/demo code",
                },
            ],
        )

        # CDK NAG Suppression Rules - Secrets Manager
        NagSuppressions.add_resource_suppressions_by_path(
            self,
            [
                f"/bedrockslackchatbotstack-{env_name}/slack-bot-token-secret-{env_name}/Resource",
                f"/bedrockslackchatbotstack-{env_name}/slack-bot-signing-secret-{env_name}/Resource",
            ],
            [
                {
                    "id": "AwsSolutions-SMG4",
                    "reason": "Secret rotation is not possible in this case",
                },
            ],
        )

        # CDK NAG Suppression Rules - API Gateway
        NagSuppressions.add_resource_suppressions_by_path(
            self,
            [
                f"/bedrockslackchatbotstack-{env_name}/bedrock-kb-slackbot-api-{env_name}/Resource",
                f"/bedrockslackchatbotstack-{env_name}/bedrock-kb-slackbot-api-{env_name}/DeploymentStage.prod/Resource",
                f"/bedrockslackchatbotstack-{env_name}/bedrock-kb-slackbot-api-{env_name}/Default/slack/askblue/POST/Resource",
            ],
            [
                {
                    "id": "AwsSolutions-APIG2",
                    "reason": "API validation is not required for demo/sample code",
                },
                {
                    "id": "AwsSolutions-APIG3",
                    "reason": "AWS WAF is not required for sample/demo code",
                },
                {
                    "id": "AwsSolutions-APIG6",
                    "reason": "Logging is enabled for the API",
                },
                {
                    "id": "AwsSolutions-APIG4",
                    "reason": "API Auth is not provided in demo/sample code",
                },
                {
                    "id": "AwsSolutions-COG4",
                    "reason": "Cognito is not being used in the sample code",
                },
            ],
        )
