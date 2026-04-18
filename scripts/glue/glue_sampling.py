import os
import json
import boto3
import logging
import re
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from pyspark.sql.functions import col, when, rand, regexp_replace
from pyspark.sql.types import IntegerType

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

input_path = "s3://mlops-sentiment-analysis-data/Bronze/anomaly_flagged.parquet"
parquet_output_path = "s3://mlops-sentiment-analysis-data/Bronze/train_final.parquet"
temp_csv_output_path = "s3://mlops-sentiment-analysis-data/tmp/sampled_temp_csv"
final_csv_key = "Silver/sampled.csv"
remaining_data_output_path = "s3://mlops-sentiment-analysis-data/test/anomaly_data_remaining.parquet"
metadata_key = "metadata/class_distribution.json"
bucket = "mlops-sentiment-analysis-data"

df = spark.read.parquet(input_path)
logger.info(f"Loaded anomaly-flagged dataset from: {input_path}")

df_filtered = df.filter((col("star_rating").isNotNull()) & (col("star_rating") != 3))
df_labeled = df_filtered.withColumn("label", when(col("star_rating") > 3, 1).otherwise(0))

df_labeled = df_labeled.withColumn("label", col("label").cast(IntegerType()))

df_labeled = df_labeled.withColumn("review_body", regexp_replace(col("review_body"), r'[\r\n]+', ' '))

dropped_3_star = df.count() - df_labeled.count()
logger.info(f"Dropped {dropped_3_star} neutral (3-star) reviews")

label_dist = df_labeled.groupBy("label").count().collect()
class_counts = {int(row["label"]): row["count"] for row in label_dist}
logger.info(f"Class distribution after dropping 3-star: {class_counts}")

positive_df = df_labeled.filter(col("label") == 1)
negative_df = df_labeled.filter(col("label") == 0)
min_class_count = min(positive_df.count(), negative_df.count(), 25000)
logger.info(f"Sampling {min_class_count} from each class")

positive_sample = positive_df.orderBy(rand()).limit(min_class_count)
negative_sample = negative_df.orderBy(rand()).limit(min_class_count)
df_sampled = positive_sample.union(negative_sample).orderBy(rand())

sampled_ids_df = df_sampled.select("review_body").withColumnRenamed("review_body", "sample_key")
full_ids_df = df_labeled.withColumnRenamed("review_body", "sample_key")

remaining_df = full_ids_df.join(sampled_ids_df, on="sample_key", how="left_anti").withColumnRenamed("sample_key", "review_body")

remaining_df.write.mode("overwrite").parquet(remaining_data_output_path)
logger.info(f"Remaining {remaining_df.count()} records saved to: {remaining_data_output_path}")

df_sampled.write.mode("overwrite").parquet(parquet_output_path)
logger.info(f"Parquet training data saved to: {parquet_output_path}")

df_sampled.select("review_body", "label") \
    .coalesce(1) \
    .write.option("header", True) \
    .option("quote", '"') \
    .option("escape", '"') \
    .mode("overwrite") \
    .csv(temp_csv_output_path)
logger.info(f"Temporary single-part CSV written to: {temp_csv_output_path}")

s3 = boto3.client("s3")
objects = s3.list_objects_v2(Bucket=bucket, Prefix="tmp/sampled_temp_csv/")
for obj in objects.get("Contents", []):
    key = obj["Key"]
    if re.search(r"part-.*\.csv$", key):
        copy_source = {'Bucket': bucket, 'Key': key}
        s3.copy_object(Bucket=bucket, CopySource=copy_source, Key=final_csv_key)
        logger.info(f"Moved final CSV to: s3://{bucket}/{final_csv_key}")
        break

for obj in objects.get("Contents", []):
    s3.delete_object(Bucket=bucket, Key=obj["Key"])
logger.info("Cleaned up temporary CSV files")

class_meta = {
    "dropped_3_star_reviews": dropped_3_star,
    "original_class_distribution": class_counts,
    "sampled_class_counts": {
        "positive": min_class_count,
        "negative": min_class_count
    },
    "remaining_after_sampling": remaining_df.count()
}
s3.put_object(
    Bucket=bucket,
    Key=metadata_key,
    Body=json.dumps(class_meta),
    ContentType='application/json'
)
logger.info(f"Stored class distribution metadata to: s3://{bucket}/{metadata_key}")
