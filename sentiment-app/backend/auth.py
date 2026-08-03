import os
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv

try:
    import boto3
except ImportError:
    boto3 = None

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

COGNITO_REGION = os.getenv("COGNITO_REGION", "us-east-1")
CLIENT_ID = os.getenv("COGNITO_CLIENT_ID")
MOCK_AUTH = os.getenv("MOCK_AUTH", "false").lower() in ("true", "1", "yes")

security = HTTPBearer(auto_error=False)

def get_cognito_client():
    if boto3 is None:
        raise HTTPException(
            status_code=500,
            detail="Boto3 is not installed. Install boto3 or set MOCK_AUTH=true."
        )
    if not CLIENT_ID and not MOCK_AUTH:
        raise HTTPException(
            status_code=500,
            detail="Cognito configuration error: COGNITO_CLIENT_ID is not set in environment."
        )
    return boto3.client("cognito-idp", region_name=COGNITO_REGION)

def authenticate_user(username: str, password: str) -> dict:
    if MOCK_AUTH:
        if username == "testuser" and password == "TestPassword123!":
            return {
                "access_token": "mock-access-token-xyz",
                "id_token": "mock-id-token-xyz",
                "refresh_token": "mock-refresh-token-xyz"
            }
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    client = get_cognito_client()
    try:
        response = client.initiate_auth(
            AuthFlow="USER_PASSWORD_AUTH",
            AuthParameters={"USERNAME": username, "PASSWORD": password},
            ClientId=CLIENT_ID
        )
        return {
            "access_token": response["AuthenticationResult"]["AccessToken"],
            "id_token": response["AuthenticationResult"]["IdToken"],
            "refresh_token": response["AuthenticationResult"]["RefreshToken"]
        }
    except client.exceptions.NotAuthorizedException:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    except client.exceptions.UserNotFoundException:
        raise HTTPException(status_code=404, detail="User not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header with Bearer token."
        )

    token = credentials.credentials
    if MOCK_AUTH and token.startswith("mock-"):
        return "mock_user"

    if not token or len(token) < 10:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")

    return "authenticated_user"
