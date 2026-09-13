import os
import boto3

INSTANCE_ID = os.getenv('EC2_INSTANCE_ID', 'i-0d9483271f7cacc06')
REGION = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')

def stop_instance():
    ec2 = boto3.client('ec2', region_name=REGION)
    print(f"Stopping EC2 instance {INSTANCE_ID}...")
    ec2.stop_instances(InstanceIds=[INSTANCE_ID])
    print("Instance stopped.")

if __name__ == "__main__":
    stop_instance()
