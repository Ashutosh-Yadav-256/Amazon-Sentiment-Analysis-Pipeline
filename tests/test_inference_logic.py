import pytest
import pandas as pd

def map_predictions(preds):
    label_map = {"LABEL_0": "negative", "LABEL_1": "positive"}
    return [label_map.get(p.get("label"), "unknown") for p in preds]

def compute_summary_metrics(df: pd.DataFrame, review_col: str = "review_clean") -> dict:
    pos_mask = df["prediction"].astype(str).str.lower() == "positive"
    neg_mask = df["prediction"].astype(str).str.lower() == "negative"
    neu_mask = df["prediction"].astype(str).str.lower() == "neutral"
    short_col = df["short_review"] if "short_review" in df.columns else pd.Series([False] * len(df))

    return {
        "total_reviews": len(df),
        "positive_reviews": int(pos_mask.sum()),
        "negative_reviews": int(neg_mask.sum()),
        "neutral_reviews": int(neu_mask.sum()),
        "average_review_length": round(float(df[review_col].astype(str).str.len().mean()), 2),
        "short_reviews_flagged": int(short_col.sum()),
        "top_positive_reviews": df[pos_mask][review_col].head(5).tolist(),
        "top_negative_reviews": df[neg_mask][review_col].head(5).tolist()
    }

def test_label_mapping():
    raw_preds = [
        {"label": "LABEL_1", "score": 0.98},
        {"label": "LABEL_0", "score": 0.94},
        {"label": "UNKNOWN", "score": 0.50}
    ]
    labels = map_predictions(raw_preds)
    assert labels == ["positive", "negative", "unknown"]

def test_summary_metrics_calculation():
    data = {
        "review_clean": [
            "excellent quality sound and battery life",
            "terrible broke after one single use",
            "works as advertised perfectly",
            "waste of money defective unit",
            "neutral experience neither good nor bad"
        ],
        "prediction": ["positive", "negative", "positive", "negative", "neutral"],
        "short_review": [False, False, False, True, False]
    }
    df = pd.DataFrame(data)
    summary = compute_summary_metrics(df)

    assert summary["total_reviews"] == 5
    assert summary["positive_reviews"] == 2
    assert summary["negative_reviews"] == 2
    assert summary["neutral_reviews"] == 1
    assert summary["short_reviews_flagged"] == 1
    assert len(summary["top_positive_reviews"]) == 2
    assert len(summary["top_negative_reviews"]) == 2

