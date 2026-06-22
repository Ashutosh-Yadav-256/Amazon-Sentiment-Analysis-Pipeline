import argparse
import os
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel

def parse_args():
    parser = argparse.ArgumentParser(description="Deploy DistilBERT model to SageMaker Endpoint")
    parser.add_argument("--model-data", type=str, default=None,
                        help="S3 URI to model.tar.gz (defaults to latest model in bucket)")
    parser.add_argument("--bucket", type=str, default="mlops-sentiment-analysis-data",
                        help="S3 bucket containing models")
    parser.add_argument("--instance-type", type=str, default="ml.m5.large",
                        help="SageMaker inference instance type")
    parser.add_argument("--instance-count", type=int, default=1,
                        help="Number of inference instances")
    parser.add_argument("--endpoint-name", type=str, default="sentiment-analysis-distilbert",
                        help="Name of the SageMaker endpoint")
    return parser.parse_args()

def get_latest_model_s3_uri(bucket_name: str) -> str:
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix="models/")
    candidates = [
        obj["Key"] for obj in response.get("Contents", [])
        if obj["Key"].endswith("model.tar.gz")
    ]
    if not candidates:
        raise FileNotFoundError(f"No model.tar.gz artifacts found under s3://{bucket_name}/models/")
    latest_key = sorted(candidates)[-1]
    return f"s3://{bucket_name}/{latest_key}"

def deploy_model(model_data_uri: str, instance_type: str, instance_count: int, endpoint_name: str):
    role = sagemaker.get_execution_role()

    hub_config = {
        'HF_TASK': 'text-classification'
    }

    huggingface_model = HuggingFaceModel(
        model_data=model_data_uri,
        role=role,
        transformers_version="4.17.0",
        pytorch_version="1.10.2",
        py_version="py38",
        env=hub_config
    )

    print(f"Deploying model from {model_data_uri} to endpoint '{endpoint_name}'...")
    predictor = huggingface_model.deploy(
        initial_instance_count=instance_count,
        instance_type=instance_type,
        endpoint_name=endpoint_name
    )
    print(f"Endpoint successfully deployed: {predictor.endpoint_name}")
    return predictor

if __name__ == "__main__":
    args = parse_args()
    model_uri = args.model_data or get_latest_model_s3_uri(args.bucket)
    deploy_model(
        model_data_uri=model_uri,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        endpoint_name=args.endpoint_name
    )
