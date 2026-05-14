import os
import json
import boto3
import logging
import pandas as pd
from datetime import datetime
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    import mlflow
    import mlflow.transformers
except ImportError:
    mlflow = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bucket = os.getenv("DATA_BUCKET_NAME", "mlops-sentiment-analysis-data")
train_s3 = f"s3://{bucket}/Silver/sampled.csv"
test_s3 = f"s3://{bucket}/Silver/test.csv"
metadata_key = "metadata/model_evaluation_summary.json"
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"s3://{bucket}/mlflow")
experiment_name = "distilbert-sentiment"

local_train = "/tmp/train.csv"
local_test = "/tmp/test.csv"
model_output_dir = os.getenv("SM_MODEL_DIR", "/tmp/trained_model")
metadata_file = "/tmp/model_evaluation_summary.json"

s3 = boto3.client("s3")

def download_from_s3(s3_uri, local_path):
    b, k = s3_uri.replace("s3://", "").split("/", 1)
    s3.download_file(b, k, local_path)
    logger.info(f"Downloaded: {s3_uri} -> {local_path}")

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }

def run_training():
    sm_train_dir = os.environ.get("SM_CHANNEL_TRAIN")
    if sm_train_dir and os.path.exists(os.path.join(sm_train_dir, "sampled.csv")):
        train_path = os.path.join(sm_train_dir, "sampled.csv")
        test_path = os.path.join(sm_train_dir, "test.csv")
    else:
        download_from_s3(train_s3, local_train)
        download_from_s3(test_s3, local_test)
        train_path = local_train
        test_path = local_test

    train_df = pd.read_csv(train_path).dropna()
    test_df = pd.read_csv(test_path).dropna()

    text_col_train = "review_body" if "review_body" in train_df.columns else "text"
    text_col_test = "review_body" if "review_body" in test_df.columns else "text"
    train_df = train_df.rename(columns={text_col_train: "text"})
    test_df = test_df.rename(columns={text_col_test: "text"})

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

    train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
    test_dataset = Dataset.from_pandas(test_df).map(tokenize, batched=True)
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )

    epochs = int(os.environ.get("EPOCHS", 2))
    batch_size = int(os.environ.get("TRAIN_BATCH_SIZE", 16))

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_dir="./logs",
        logging_steps=10,
        fp16=False,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    if mlflow:
        try:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            logger.warning(f"MLflow setup warning: {e}")

    logger.info("Starting training...")
    trainer.train()

    logger.info("Running evaluation...")
    metrics = trainer.evaluate()

    true_labels = test_df["label"].tolist()
    predictions = trainer.predict(test_dataset).predictions.argmax(-1).tolist()

    actual_pos = sum(1 for l in true_labels if l == 1)
    actual_neg = sum(1 for l in true_labels if l == 0)
    pred_pos = sum(1 for l in predictions if l == 1)
    pred_neg = sum(1 for l in predictions if l == 0)

    test_df["prediction"] = predictions
    top_pos = test_df[test_df["prediction"] == 1].head(5)["text"].tolist()
    top_neg = test_df[test_df["prediction"] == 0].head(5)["text"].tolist()

    if mlflow:
        try:
            with mlflow.start_run(run_name=f"distilbert-{datetime.utcnow().isoformat()}"):
                mlflow.log_metrics(metrics)
                mlflow.log_metrics({
                    "actual_positive": actual_pos,
                    "actual_negative": actual_neg,
                    "predicted_positive": pred_pos,
                    "predicted_negative": pred_neg
                })
        except Exception as e:
            logger.warning(f"MLflow logging error: {e}")

    eval_summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_name": "distilbert-base-uncased",
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

    with open(metadata_file, "w") as f:
        json.dump(eval_summary, f, indent=4)

    s3.upload_file(
        Filename=metadata_file,
        Bucket=bucket,
        Key=metadata_key
    )
    logger.info(f"Uploaded dashboard JSON to: s3://{bucket}/{metadata_key}")

    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    logger.info(f"Model saved to {model_output_dir}")

if __name__ == "__main__":
    run_training()
