import logging
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from pyspark.sql.functions import col, lower, trim, regexp_replace

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

input_path = "s3://mlops-sentiment-analysis-data/Bronze/train.parquet"
output_path = "s3://mlops-sentiment-analysis-data/Bronze/pre_processed.parquet"

logger.info(f"Reading input data from: {input_path}")
df = spark.read.parquet(input_path)
initial_count = df.count()
logger.info(f"Total records read: {initial_count}")

df = df.filter((col("review_body").isNotNull()) & (col("review_body") != ""))
filtered_count = df.count()
logger.info(f"Records after null/empty review_body drop: {filtered_count}")
logger.info(f"Records dropped: {initial_count - filtered_count}")

for text_col in ["review_body", "review_headline"]:
    df = df.withColumn(text_col, regexp_replace(trim(lower(col(text_col))), r"\s+", " "))
logger.info("Text normalization completed for: review_body, review_headline")

before_dedup = df.count()
df = df.dropDuplicates(["review_body"])
after_dedup = df.count()
logger.info(f"Dropped {before_dedup - after_dedup} duplicate reviews based on cleaned review_body")

df.write.mode("overwrite").parquet(output_path)
logger.info(f"Cleaned data written to: {output_path}")
logger.info("Data Preprocessing job completed successfully.")
