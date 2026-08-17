import os
import boto3
import json
from datetime import datetime

# Initialize the S3 client lazily or using credentials from env
def get_s3_client():
    return boto3.client(
        's3',
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_REGION")
    )

def save_recommendation_report(student_id, session_id, goal, recommendation_text):
    bucket_name = os.environ.get("AWS_S3_BUCKET_NAME")
    if not bucket_name:
        return "AWS_S3_BUCKET_NAME not set"
        
    s3_client = get_s3_client()
    
    report = {
        "student_id": student_id,
        "session_id": session_id,
        "goal": goal,
        "recommendation": recommendation_text,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    object_key = f"reports/student_{student_id}/session_{session_id}.json"
    
    try:
        s3_client.put_object(
            Bucket=bucket_name,
            Key=object_key,
            Body=json.dumps(report, indent=2),
            ContentType='application/json'
        )
        return f"s3://{bucket_name}/{object_key}"
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        return None
