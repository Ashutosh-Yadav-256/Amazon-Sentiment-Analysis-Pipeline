import os
import json
import boto3
import pandas as pd
import tarfile
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

bucket = os.environ.get("BUCKET_NAME")
user = os.environ.get("USER_NAME")
filename = os.environ.get("FILENAME")

processed_key = f"processed/{user}/processed.csv"
output_file_key = f"output/{user}/served.csv"
status_key = f"metadata/{user}/pipeline_status.json"
summary_key = f"metadata/{user}/inference_summary.json"
model_key = "models/model.tar.gz"

s3 = boto3.client("s3")

print("Downloading cleaned reviews...")
obj = s3.get_object(Bucket=bucket, Key=processed_key)
df = pd.read_csv(obj["Body"])
texts = df["review_clean"].astype(str).tolist()

print("Downloading model from S3...")
local_tar = "/tmp/model.tar.gz"
local_model_dir = "/tmp/model"

s3.download_file(bucket, model_key, local_tar)

print("Extracting model...")
with tarfile.open(local_tar, "r:gz") as tar:
    tar.extractall(path=local_model_dir)

print(f"Loading model from {local_model_dir}...")
tokenizer = AutoTokenizer.from_pretrained(local_model_dir, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(local_model_dir, local_files_only=True)

pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

print("Running inference...")
preds = pipe(texts, truncation=True, padding=True)

label_map = {"LABEL_0": "negative", "LABEL_1": "positive"}
df["prediction"] = [label_map.get(p["label"], "unknown") for p in preds]

tmp_csv_path = "/tmp/served.csv"
df.to_csv(tmp_csv_path, index=False)
s3.upload_file(tmp_csv_path, bucket, output_file_key)
print(f"Uploaded served predictions to s3://{bucket}/{output_file_key}")

try:
    print("Creating inference summary...")
    summary = {
        "total_reviews": len(df),
        "positive_reviews": int((df["prediction"] == "positive").sum()),
        "negative_reviews": int((df["prediction"] == "negative").sum()),
        "neutral_reviews": int((df["prediction"] == "neutral").sum()) if "neutral" in df["prediction"].unique() else 0,
        "average_review_length": round(df["review_clean"].astype(str).str.len().mean(), 2),
        "short_reviews_flagged": int(df.get("short_review", pd.Series([False] * len(df))).sum()),
        "top_positive_reviews": df[df["prediction"] == "positive"]["review_clean"].head(5).tolist(),
        "top_negative_reviews": df[df["prediction"] == "negative"]["review_clean"].head(5).tolist()
    }

    s3.put_object(
        Bucket=bucket,
        Key=summary_key,
        Body=json.dumps(summary, indent=2),
        ContentType="application/json"
    )
    print(f"Dashboard summary saved to s3://{bucket}/{summary_key}")
except Exception as e:
    print(f"Failed to create dashboard summary: {e}")

s3.put_object(
    Bucket=bucket,
    Key=status_key,
    Body=json.dumps({"status": "inference_complete"}),
    ContentType="application/json"
)
print("Inference complete.")
