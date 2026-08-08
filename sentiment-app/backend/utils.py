import os
import json
import pandas as pd
import datetime
from io import StringIO
from dotenv import load_dotenv

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None
    ClientError = Exception

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
ECS_CLUSTER = os.getenv("ECS_CLUSTER")
ECS_TASK_DEFINITION = os.getenv("ECS_TASK_DEFINITION_ARN")
SUBNET_ID = os.getenv("SUBNET_ID")
SECURITY_GROUP_ID = os.getenv("SECURITY_GROUP_ID")
BUCKET_NAME = os.getenv("BUCKET_NAME", "mlops-sentiment-app")
MOCK_PIPELINE = os.getenv("MOCK_PIPELINE", "false").lower() in ("true", "1", "yes")

def get_ecs_client():
    if boto3 is None:
        raise RuntimeError("boto3 is not installed.")
    return boto3.client("ecs", region_name=AWS_REGION)

def get_s3_client():
    if boto3 is None:
        raise RuntimeError("boto3 is not installed.")
    return boto3.client("s3", region_name=AWS_REGION)

def trigger_ecs_pipeline(filename: str, user: str) -> str:
    print(f"Inside trigger_ecs_pipeline for user={user}, filename={filename}", flush=True)

    if MOCK_PIPELINE:
        mock_arn = f"arn:aws:ecs:{AWS_REGION}:123456789012:task/mock-{user}-123"
        print(f"MOCK_PIPELINE enabled. Returning mock task ARN: {mock_arn}")
        return mock_arn

    if not ECS_CLUSTER or not ECS_TASK_DEFINITION or not SUBNET_ID or not SECURITY_GROUP_ID:
        raise ValueError(
            "Missing ECS configuration in environment. "
            "Ensure ECS_CLUSTER, ECS_TASK_DEFINITION_ARN, SUBNET_ID, and SECURITY_GROUP_ID are set."
        )

    def json_fallback(o):
        if isinstance(o, (datetime.datetime, datetime.date)):
            return o.isoformat()
        return str(o)

    try:
        ecs_client = get_ecs_client()
        response = ecs_client.run_task(
            cluster=ECS_CLUSTER,
            launchType="FARGATE",
            taskDefinition=ECS_TASK_DEFINITION,
            count=1,
            platformVersion="LATEST",
            networkConfiguration={
                "awsvpcConfiguration": {
                    "subnets": [SUBNET_ID],
                    "securityGroups": [SECURITY_GROUP_ID],
                    "assignPublicIp": "ENABLED"
                }
            },
            overrides={
                "containerOverrides": [
                    {
                        "name": "sentiment-pipeline",
                        "environment": [
                            {"name": "BUCKET_NAME", "value": BUCKET_NAME},
                            {"name": "USER_NAME", "value": user},
                            {"name": "FILENAME", "value": filename}
                        ]
                    }
                ]
            }
        )

        tasks = response.get("tasks", [])
        if not tasks:
            failures = response.get("failures", [])
            raise RuntimeError(f"ECS task launch failed: {failures}")

        task_arn = tasks[0].get("taskArn", "unknown-task-arn")
        print(f"ECS RunTask response task_arn: {task_arn}")
        return task_arn

    except Exception as e:
        print(f"Failed to trigger ECS task: {e}")
        raise e

def generate_dashboard(user: str, filename: str) -> dict:
    output_key = f"output/{user}/served.csv"
    metadata_key = f"metadata/{user}/inference_summary.json"

    s3 = get_s3_client()
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=output_key)
        csv_str = response["Body"].read().decode("utf-8")
        df = pd.read_csv(StringIO(csv_str))
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in ("NoSuchKey", "404"):
            raise FileNotFoundError(f"s3://{BUCKET_NAME}/{output_key} does not exist yet.")
        raise e

    if df.empty:
        raise ValueError("Served CSV file is empty; cannot generate summary metrics.")

    candidate_cols = ["review_clean", "review_body", "review", "text"]
    review_col = None
    for col in candidate_cols:
        if col in df.columns:
            review_col = col
            break
    if not review_col:
        text_cols = df.select_dtypes(include=["object"]).columns
        review_col = text_cols[0] if len(text_cols) > 0 else df.columns[0]

    if "prediction" not in df.columns:
        df["prediction"] = "unknown"

    pos_mask = df["prediction"].astype(str).str.lower() == "positive"
    neg_mask = df["prediction"].astype(str).str.lower() == "negative"
    neu_mask = df["prediction"].astype(str).str.lower() == "neutral"

    short_col = df["short_review"] if "short_review" in df.columns else pd.Series([False] * len(df))

    summary = {
        "total_reviews": len(df),
        "positive_reviews": int(pos_mask.sum()),
        "negative_reviews": int(neg_mask.sum()),
        "neutral_reviews": int(neu_mask.sum()),
        "average_review_length": round(float(df[review_col].astype(str).str.len().mean()), 2),
        "short_reviews_flagged": int(short_col.sum()),
        "top_positive_reviews": df[pos_mask][review_col].head(5).tolist(),
        "top_negative_reviews": df[neg_mask][review_col].head(5).tolist()
    }

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=metadata_key,
        Body=json.dumps(summary, indent=2),
        ContentType="application/json"
    )

    print(f"Dashboard summary uploaded to s3://{BUCKET_NAME}/{metadata_key}")
    return summary
