import os
import json
import boto3
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import torch
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.transformers

bucket = os.getenv("DATA_BUCKET_NAME", "mlops-sentiment-analysis-data")
test_s3 = f"s3://{bucket}/Silver/test.csv"
metadata_key = "metadata/model_evaluation_summary.json"
tested_key = "Silver/tested.csv"
tracking_uri = "file:/tmp/mlruns"
experiment_name = "distilbert-sentiment"

test_local = "/tmp/test.csv"
tested_csv_path = "/tmp/tested.csv"
local_model_dir = "/tmp/model"
local_tar_path = "/tmp/model.tar.gz"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

s3 = boto3.client("s3")

def download_from_s3(s3_uri, local_path):
    b, k = s3_uri.replace("s3://", "").split("/", 1)
    s3.download_file(b, k, local_path)
    logger.info(f"Downloaded: {s3_uri} -> {local_path}")

download_from_s3(test_s3, test_local)

response = s3.list_objects_v2(Bucket=bucket, Prefix="models/")
all_models = sorted(
    [obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith("model.tar.gz")],
    reverse=True
)

if not all_models:
    raise FileNotFoundError("No model.tar.gz found in models/")

latest_model_key = all_models[0]
logger.info(f"Using model: s3://{bucket}/{latest_model_key}")
s3.download_file(bucket, latest_model_key, local_tar_path)

os.makedirs(local_model_dir, exist_ok=True)
os.system(f"tar -xvf {local_tar_path} -C {local_model_dir}")

df_full = pd.read_csv(test_local).dropna()
df_full = df_full.rename(columns={"review_body": "text", "label": "label"})

positive_df = df_full[df_full["label"] == 1].sample(n=10000, random_state=42)
negative_df = df_full[df_full["label"] == 0].sample(n=10000, random_state=42)
test_df = pd.concat([positive_df, negative_df]).sample(frac=1, random_state=42).reset_index(drop=True)

tokenizer = DistilBertTokenizerFast.from_pretrained(local_model_dir, local_files_only=True)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

test_dataset = Dataset.from_pandas(test_df).map(tokenize, batched=True)
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

model = DistilBertForSequenceClassification.from_pretrained(local_model_dir, local_files_only=True)

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }

training_args = TrainingArguments(
    output_dir="/tmp/tmp_eval_logs",
    per_device_eval_batch_size=16
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name=f"eval-{datetime.utcnow().isoformat()}"):
    logger.info("Running evaluation...")
    metrics = trainer.evaluate()

    with torch.no_grad():
        outputs = trainer.predict(test_dataset)
        probs = torch.nn.functional.softmax(torch.tensor(outputs.predictions), dim=-1).numpy()
        predictions = np.argmax(probs, axis=1)

    test_df["prediction"] = predictions
    test_df["positive_score"] = probs[:, 1]
    test_df["negative_score"] = probs[:, 0]

    actual_pos = int((test_df["label"] == 1).sum())
    actual_neg = int((test_df["label"] == 0).sum())
    pred_pos = int((test_df["prediction"] == 1).sum())
    pred_neg = int((test_df["prediction"] == 0).sum())

    top_pos = test_df[test_df["prediction"] == 1] \
        .sort_values("positive_score", ascending=False).head(5)["text"].tolist()

    top_neg = test_df[test_df["prediction"] == 0] \
        .sort_values("negative_score", ascending=False).head(5)["text"].tolist()

    mlflow.log_metrics(metrics)
    mlflow.log_metrics({
        "actual_positive": actual_pos,
        "actual_negative": actual_neg,
        "predicted_positive": pred_pos,
        "predicted_negative": pred_neg
    })

    with open("/tmp/top_reviews.json", "w") as f:
        json.dump({
            "top_predicted_positive": top_pos,
            "top_predicted_negative": top_neg
        }, f, indent=2)
    mlflow.log_artifact("/tmp/top_reviews.json")

    eval_summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_name": os.path.dirname(latest_model_key),
        "metrics": metrics,
        "class_summary": {
            "actual_positive": actual_pos,
            "actual_negative": actual_neg,
            "predicted_positive": pred_pos,
            "predicted_negative": pred_neg
        },
        "sample_reviews": {
            "top_predicted_positive": top_pos,
            "top_predicted_negative": top_neg
        }
    }

    with open("/tmp/model_evaluation_summary.json", "w") as f:
        json.dump(eval_summary, f, indent=4)

    s3.upload_file("/tmp/model_evaluation_summary.json", bucket, metadata_key)
    logger.info(f"Dashboard summary -> s3://{bucket}/{metadata_key}")

    test_df[["text", "label", "prediction"]].to_csv(tested_csv_path, index=False)
    s3.upload_file(tested_csv_path, bucket, tested_key)
    logger.info(f"Tested results -> s3://{bucket}/{tested_key}")
