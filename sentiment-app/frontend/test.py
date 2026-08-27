import os
import boto3
from dotenv import load_dotenv

load_dotenv()

COGNITO_REGION = os.getenv("COGNITO_REGION", "us-east-1")
CLIENT_ID = os.getenv("COGNITO_CLIENT_ID", "your-client-id")
TEST_USER = os.getenv("COGNITO_TEST_USER", "test-user")
TEST_PASSWORD = os.getenv("COGNITO_TEST_PASSWORD", "TestPassword123!")

client = boto3.client("cognito-idp", region_name=COGNITO_REGION)

try:
    response = client.initiate_auth(
        AuthFlow="USER_PASSWORD_AUTH",
        AuthParameters={
            "USERNAME": TEST_USER,
            "PASSWORD": TEST_PASSWORD,
        },
        ClientId=CLIENT_ID,
    )
    print("Login success")
    print(response.get("AuthenticationResult", {}))
except client.exceptions.NotAuthorizedException:
    print("Not authorized: Check your credentials.")
except Exception as e:
    print(f"Auth request failed: {e}")
