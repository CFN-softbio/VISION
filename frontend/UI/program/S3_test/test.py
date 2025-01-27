from minio import Minio
from minio.error import S3Error
import logging

# Enable logging
logging.basicConfig(level=logging.DEBUG)

# Initialize the Minio client
client = Minio(
    "69.164.214.19:9000",
    access_key="admin",
    secret_key="oomei4IsAcoh3roo",
    secure=False
)


print('listing...')
# Example operation: List buckets
try:
    buckets = client.list_buckets()
    for bucket in buckets:
        print(bucket.name)
except S3Error as e:
    print("Error:", e)

print('done list.')

# Define bucket and object details
bucket_name = "transmissions"
object_name = "test-object"
file_path = "./userS3-sent.npy"

# Ensure the bucket exists
if not client.bucket_exists(bucket_name):
    client.make_bucket(bucket_name)

# Upload the file
try:
    result = client.fput_object(
        bucket_name=bucket_name,
        object_name=object_name,
        file_path=file_path,
        content_type="application/octet-stream"
    )
    print("File uploaded successfully.")
except S3Error as err:
    print("File upload error:", err)

