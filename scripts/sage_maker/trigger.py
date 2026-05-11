import os
import sagemaker
from sagemaker.huggingface import HuggingFace, TrainingCompilerConfig

bucket = os.getenv("DATA_BUCKET_NAME", "mlops-sentiment-analysis-data")
role = os.getenv("SAGEMAKER_ROLE_ARN")

if not role:
    try:
        role = sagemaker.get_execution_role()
    except Exception:
        raise ValueError(
            "Could not determine SageMaker execution role. "
            "Please set the SAGEMAKER_ROLE_ARN environment variable."
        )

huggingface_estimator = HuggingFace(
    entry_point="train.py",
    source_dir=os.path.dirname(__file__),
    instance_type="ml.g4dn.xlarge",
    instance_count=1,
    role=role,
    transformers_version="4.17.0",
    pytorch_version="1.10.2",
    py_version="py38",
    hyperparameters={
        "epochs": 2,
        "train_batch_size": 16
    },
    compiler_config=TrainingCompilerConfig(enabled=True),
    base_job_name="distilbert-sentiment-optimized",
    output_path=f"s3://{bucket}/models/",
)

if __name__ == "__main__":
    print(f"Triggering SageMaker training with role {role}...")
    huggingface_estimator.fit({
        "train": f"s3://{bucket}/Silver"
    })
