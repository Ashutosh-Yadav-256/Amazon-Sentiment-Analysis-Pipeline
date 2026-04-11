import logging
import json
import boto3
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from pyspark.sql.functions import col, length, avg
from pyspark.sql.types import IntegerType

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

input_path = "s3://mlops-sentiment-analysis-data/Bronze/pre_processed.parquet"
output_path = "s3://mlops-sentiment-analysis-data/Bronze/schema_validated.parquet"
avg_length_metadata_path = "s3://mlops-sentiment-analysis-data/metadata/avg_review_length.json"

df = spark.read.parquet(input_path)
initial_count = df.count()
logger.info(f"Loaded {initial_count} records from: {input_path}")

df = df.filter((col("review_body").isNotNull()) & (col("review_headline").isNotNull()))
text_valid_count = df.count()
logger.info(f"Records with non-null review_body and review_headline: {text_valid_count}")
logger.info(f"Dropped {initial_count - text_valid_count} rows with missing text")

df = df.withColumn("star_rating", col("star_rating").cast(IntegerType()))

invalid_low = df.filter(col("star_rating") < 0).count()
invalid_high = df.filter(col("star_rating") > 5).count()
if invalid_low > 0:
    logger.warning(f"Found {invalid_low} entries with star_rating < 0")
if invalid_high > 0:
    logger.warning(f"Found {invalid_high} entries with star_rating > 5")

df = df.withColumn("review_length", length(col("review_body")))
avg_len = df.select(avg("review_length")).first()[0]
logger.info(f"Average review_body length: {avg_len:.2f}")

avg_data = {"avg_review_length": avg_len}
s3 = boto3.client("s3")
s3.put_object(
    Bucket="mlops-sentiment-analysis-data",
    Key="metadata/avg_review_length.json",
    Body=json.dumps(avg_data),
    ContentType='application/json'
)
logger.info("Stored average review length to: s3://mlops-sentiment-analysis-data/metadata/avg_review_length.json")

df.write.mode("overwrite").parquet(output_path)
logger.info(f"Schema validation complete. Output saved to: {output_path}")
