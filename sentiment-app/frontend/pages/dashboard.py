import os
import streamlit as st
import pandas as pd
import boto3
import json
import matplotlib.pyplot as plt

BUCKET_NAME = os.getenv("BUCKET_NAME", "mlops-sentiment-app")

def load_csv_from_s3(user):
    s3 = boto3.client("s3")
    key = f"output/{user}/served.csv"
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    return pd.read_csv(obj['Body'])

def load_json_from_s3(user):
    s3 = boto3.client("s3")
    key = f"metadata/{user}/inference_summary.json"
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    return json.load(obj['Body'])

if "user" not in st.session_state:
    st.error("Please log in first.")
    if st.button("Go to Login"):
        st.switch_page("streamlit_app.py")
    st.stop()

user = st.session_state["user"]

st.title("Sentiment Analysis Dashboard")
st.success("Dashboard successfully loaded!")
st.write(f"User: `{user}`")

try:
    df = load_csv_from_s3(user)
    summary = load_json_from_s3(user)
except Exception as e:
    st.error(f"Failed to load inference results or summary from S3: {e}")
    st.info("Ensure the ECS inference pipeline completed and dashboard generation was triggered.")
    st.stop()

candidate_cols = ["review_clean", "review_body", "review", "text"]
review_col = next((c for c in candidate_cols if c in df.columns), df.columns[0])

st.subheader("Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Total Reviews", summary.get("total_reviews", len(df)))
col2.metric("Positive Reviews", summary.get("positive_reviews", 0))
col3.metric("Negative Reviews", summary.get("negative_reviews", 0))

col4, col5 = st.columns(2)
col4.metric("Short Reviews Flagged", summary.get("short_reviews_flagged", 0))
col5.metric("Avg Review Length", summary.get("average_review_length", 0))

st.subheader("Sentiment Distribution")
if "prediction" in df.columns and not df.empty:
    fig1, ax1 = plt.subplots()
    df["prediction"].value_counts().plot(kind="bar", ax=ax1, color="skyblue")
    ax1.set_ylabel("Review Count")
    ax1.set_xlabel("Sentiment")
    ax1.set_title("Sentiment Breakdown")
    st.pyplot(fig1)

st.subheader("Review Length Distribution")
if not df.empty:
    fig2, ax2 = plt.subplots()
    df[review_col].dropna().apply(lambda x: len(str(x))).plot.hist(bins=30, ax=ax2, color="lightgreen")
    ax2.set_xlabel("Review Length (characters)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Histogram of Review Lengths")
    st.pyplot(fig2)

st.subheader("Top 5 Positive Reviews")
pos_reviews = summary.get("top_positive_reviews", [])
if pos_reviews:
    for review in pos_reviews:
        st.success(f"{review}")
else:
    st.write("No positive reviews found.")

st.subheader("Top 5 Negative Reviews")
neg_reviews = summary.get("top_negative_reviews", [])
if neg_reviews:
    for review in neg_reviews:
        st.error(f"{review}")
else:
    st.write("No negative reviews found.")
