import aws_cdk as core
import aws_cdk.assertions as assertions

from bedrock_slack_chatbot.bedrock_slack_chatbot_stack import BedrockSlackChatbotStack

# example tests. To run these tests, uncomment this file along with the example
# resource in bedrock_slack_chatbot/bedrock_slack_chatbot_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = BedrockSlackChatbotStack(app, "bedrock-slack-chatbot")
    template = assertions.Template.from_stack(stack)

#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })
