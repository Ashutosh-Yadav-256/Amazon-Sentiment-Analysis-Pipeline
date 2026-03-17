import os
import argparse
import boto3
from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_BUCKET = os.getenv("DATA_BUCKET_NAME", "mlops-sentiment-analysis-data")
DEFAULT_FILE_PATH = "data/raw/reviews.csv"
DEFAULT_KEY = "raw/reviews.csv"
PART_SIZE = 50 * 1024 * 1024

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-threaded multipart S3 uploader")
    parser.add_argument("--file-path", type=str, default=DEFAULT_FILE_PATH,
                        help="Path to the local file to upload")
    parser.add_argument("--bucket", type=str, default=DEFAULT_BUCKET,
                        help="Destination S3 bucket name")
    parser.add_argument("--key", type=str, default=DEFAULT_KEY,
                        help="Destination S3 object key")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Concurrent upload threads")
    return parser.parse_args()

def upload_part(s3_client, bucket, key, upload_id, part_number, data):
    response = s3_client.upload_part(
        Bucket=bucket,
        Key=key,
        UploadId=upload_id,
        PartNumber=part_number,
        Body=data,
    )
    return {"PartNumber": part_number, "ETag": response["ETag"]}

def multi_part_upload(file_path: str, bucket: str, key: str, max_workers: int = 4):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Local file not found: {file_path}")

    s3_client = boto3.client("s3")
    file_size = os.path.getsize(file_path)
    print(f"Starting multipart upload: {file_path} ({file_size / (1024*1024):.2f} MB) -> s3://{bucket}/{key}")

    upload_id = s3_client.create_multipart_upload(Bucket=bucket, Key=key)["UploadId"]
    parts = []

    try:
        futures = []
        with open(file_path, "rb") as f:
            part_number = 1
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                while True:
                    chunk = f.read(PART_SIZE)
                    if not chunk:
                        break
                    future = executor.submit(
                        upload_part, s3_client, bucket, key, upload_id, part_number, chunk
                    )
                    futures.append(future)
                    part_number += 1

                for future in as_completed(futures):
                    part_result = future.result()
                    parts.append(part_result)
                    print(f"Uploaded part {part_result['PartNumber']}")

        sorted_parts = sorted(parts, key=lambda x: x["PartNumber"])
        s3_client.complete_multipart_upload(
            Bucket=bucket,
            Key=key,
            UploadId=upload_id,
            MultipartUpload={"Parts": sorted_parts},
        )
        print(f"Successfully uploaded to s3://{bucket}/{key}")

    except Exception as e:
        print(f"Upload failed: {e}. Aborting multipart upload {upload_id}...")
        try:
            s3_client.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
            print("Multipart upload aborted successfully.")
        except Exception as abort_err:
            print(f"Failed to abort multipart upload: {abort_err}")
        raise e

if __name__ == "__main__":
    args = parse_args()
    multi_part_upload(
        file_path=args.file_path,
        bucket=args.bucket,
        key=args.key,
        max_workers=args.max_workers
    )
