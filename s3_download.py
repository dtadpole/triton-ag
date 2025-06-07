import os
import boto3
import yaml
import argparse
from urllib.parse import urlparse

def load_config(config_path="finetune.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_s3_url(s3_url):
    """Parse S3 URL into bucket and prefix."""
    parsed = urlparse(s3_url)
    if parsed.scheme != 's3':
        raise ValueError(f"Invalid S3 URL: {s3_url}")
    bucket = parsed.netloc
    prefix = parsed.path.lstrip('/')
    return bucket, prefix

def download_from_s3(bucket_name, prefix, local_dir):
    """Download files from S3 bucket to local directory, maintaining folder structure."""
    s3_client = boto3.client('s3')
    os.makedirs(local_dir, exist_ok=True)
    
    # List all objects under the prefix, including those in subfolders
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            
            # Skip if it's the prefix itself
            if key == prefix:
                continue
                
            # Get the relative path from the prefix
            relative_path = key[len(prefix):].lstrip('/')
            if not relative_path:
                continue
                
            # Create the full local path, maintaining the S3 folder structure
            local_path = os.path.join(local_dir, relative_path)
            
            # Create all necessary subdirectories
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download the file
            s3_client.download_file(bucket_name, key, local_path)
            print(f"Downloaded {key} to {local_path}")

def download_all_targets(config):
    """Download all targets specified in the configuration, maintaining S3 folder structure."""
    local_base_dir = config['data']['local_dir']
    
    for s3_url in config['data']['s3_folders']:
        bucket, prefix = parse_s3_url(s3_url)
        
        # Create a subdirectory structure that mirrors the S3 path
        path_parts = prefix.split('/')
        if len(path_parts) > 1:
            # Use the full path structure after the bucket name
            local_dir = os.path.join(local_base_dir, *path_parts)
        else:
            # If it's just a single folder, use it directly
            local_dir = os.path.join(local_base_dir, prefix)
        
        print(f"\nProcessing S3 folder: {s3_url}")
        print(f"Local directory: {local_dir}")
        download_from_s3(bucket, prefix, local_dir)

def main():
    # Load configuration
    config = load_config()
    
    # Download experiences from S3
    print("Starting S3 download test...")
    download_all_targets(config)
    print("\nS3 download test completed!")

if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='Download S3 folders')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()
    
    main(args.config) 
