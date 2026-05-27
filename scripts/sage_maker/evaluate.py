import argparse
import os
import json
import tarfile
import boto3
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DistilBERT sentiment model")
    parser.add_argument("--test-data", type=str, default="/tmp/test.csv",
                        help="Path to local or S3 test CSV")
    parser.add_argument("--model-dir", type=str, default="/tmp/model",
                        help="Path to model directory")
    parser.add_argument("--output-json", type=str, default="/tmp/eval_metrics.json",
                        help="Path to output metrics JSON")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }

def run_evaluation(model_dir: str, test_csv_path: str, output_json_path: str, batch_size: int = 16):
    print(f"Loading test dataset from {test_csv_path}...")
    df = pd.read_csv(test_csv_path).dropna()
    text_col = "review_body" if "review_body" in df.columns else "text"
    df = df.rename(columns={text_col: "text"})

    print(f"Loading tokenizer and model from {model_dir}...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir, local_files_only=True)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir, local_files_only=True)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

    test_ds = Dataset.from_pandas(df).map(tokenize_fn, batched=True)
    test_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    training_args = TrainingArguments(
        output_dir="/tmp/eval_logs",
        per_device_eval_batch_size=batch_size,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics
    )

    print("Running evaluation...")
    results = trainer.evaluate()
    predictions = trainer.predict(test_ds)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    results["classification_report"] = report

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Evaluation complete. Metrics saved to {output_json_path}")
    print(json.dumps(results, indent=2))
    return results

if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        model_dir=args.model_dir,
        test_csv_path=args.test_data,
        output_json_path=args.output_json,
        batch_size=args.batch_size
    )
