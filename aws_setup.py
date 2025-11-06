#!/usr/bin/env python3
"""AWS MCP Server Setup - NFL Prediction System"""
import boto3
import json
import os
from pathlib import Path

AWS_REGION = "us-east-1"
BUCKET_NAME = "nfl-conspiracy-predictions"
LAMBDA_ROLE = "nfl-lambda-role"

def create_iam_role(iam_client):
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": ["lambda.amazonaws.com", "sagemaker.amazonaws.com"]},
            "Action": "sts:AssumeRole"
        }]
    }
    
    try:
        role = iam_client.create_role(
            RoleName=LAMBDA_ROLE,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="NFL prediction system role"
        )
        
        iam_client.attach_role_policy(
            RoleName=LAMBDA_ROLE,
            PolicyArn="arn:aws:iam::aws:policy/AWSLambdaExecute"
        )
        iam_client.attach_role_policy(
            RoleName=LAMBDA_ROLE,
            PolicyArn="arn:aws:iam::aws:policy/AmazonS3FullAccess"
        )
        iam_client.attach_role_policy(
            RoleName=LAMBDA_ROLE,
            PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
        )
        
        return role['Role']['Arn']
    except iam_client.exceptions.EntityAlreadyExistsException:
        return iam_client.get_role(RoleName=LAMBDA_ROLE)['Role']['Arn']

def create_s3_bucket(s3_client):
    try:
        s3_client.create_bucket(Bucket=BUCKET_NAME)
        
        s3_client.put_bucket_encryption(
            Bucket=BUCKET_NAME,
            ServerSideEncryptionConfiguration={
                'Rules': [{'ApplyServerSideEncryptionByDefault': {'SSEAlgorithm': 'AES256'}}]
            }
        )
        
        s3_client.put_bucket_versioning(
            Bucket=BUCKET_NAME,
            VersioningConfiguration={'Status': 'Enabled'}
        )
        
        print(f"✅ S3 bucket created: {BUCKET_NAME}")
    except s3_client.exceptions.BucketAlreadyOwnedByYou:
        print(f"✅ S3 bucket exists: {BUCKET_NAME}")

def upload_data_to_s3(s3_client):
    data_dir = Path("data/referee_conspiracy")
    
    if not data_dir.exists():
        print("⚠️  No local data directory")
        return
    
    for file in data_dir.glob("**/*.parquet"):
        key = f"data/{file.relative_to(data_dir)}"
        s3_client.upload_file(str(file), BUCKET_NAME, key)
        print(f"📤 Uploaded: {key}")
    
    for file in data_dir.glob("**/*.json"):
        key = f"predictions/{file.name}"
        s3_client.upload_file(str(file), BUCKET_NAME, key)
        print(f"📤 Uploaded: {key}")

def create_lambda_function(lambda_client, role_arn):
    lambda_code = """
import json
import boto3
from datetime import datetime
from urllib import request, error

API_TIMEOUT = 5

s3 = boto3.client('s3')
BUCKET = 'nfl-conspiracy-predictions'

def lambda_handler(event, context):
    # Fetch NFL crew data
    crews = fetch_nfl_crews()
    
    # Fetch NOAA weather
    weather = fetch_weather()
    
    # Scrape Reddit for noise
    reddit_noise = scrape_reddit()
    
    # Run prediction
    prediction = run_prediction(crews, weather, reddit_noise)
    
    # Save to S3
    timestamp = datetime.now().isoformat()
    key = f"live_predictions/{timestamp}.json"
    s3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=json.dumps(prediction)
    )
    
    return {'statusCode': 200, 'body': json.dumps(prediction)}

def fetch_nfl_crews():
    return {}

def fetch_weather():
    stadium_coords = {
        'KC': (39.0489, -94.4839),
        'LAC': (33.9534, -118.3390)
    }
    weather_data = {}
    for team, (lat, lon) in stadium_coords.items():
        url = f"https://api.weather.gov/points/{lat},{lon}"
        try:
            with request.urlopen(url, timeout=API_TIMEOUT) as resp:
                if resp.status == 200:
                    payload = json.loads(resp.read().decode('utf-8'))
                    weather_data[team] = payload
        except error.URLError:
            pass
    return weather_data

def scrape_reddit():
    return {}

def run_prediction(crews, weather, reddit):
    return {
        'timestamp': datetime.now().isoformat(),
        'crews': crews,
        'weather': weather,
        'reddit_noise': reddit,
        'prediction': 'UNDER'
    }
"""
    
    import zipfile
    from io import BytesIO
    
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        zip_file.writestr('lambda_function.py', lambda_code)
    
    try:
        lambda_client.create_function(
            FunctionName='nfl-live-predictions',
            Runtime='python3.11',
            Role=role_arn,
            Handler='lambda_function.lambda_handler',
            Code={'ZipFile': zip_buffer.getvalue()},
            Timeout=300,
            MemorySize=512,
            Environment={'Variables': {'BUCKET_NAME': BUCKET_NAME}}
        )
        print("✅ Lambda function created")
    except lambda_client.exceptions.ResourceConflictException:
        print("✅ Lambda function exists")

def create_eventbridge_rule(events_client, account_id: str):
    try:
        target_arn = f'arn:aws:lambda:{AWS_REGION}:{account_id}:function:nfl-live-predictions'
        events_client.put_rule(
            Name='nfl-prediction-schedule',
            ScheduleExpression='rate(15 minutes)',
            State='ENABLED'
        )
        
        events_client.put_targets(
            Rule='nfl-prediction-schedule',
            Targets=[{
                'Id': '1',
                'Arn': target_arn
            }]
        )
        print("✅ EventBridge rule created")
    except Exception as e:
        print(f"⚠️  EventBridge: {e}")

def create_budget(budgets_client, account_id: str):
    try:
        budgets_client.create_budget(
            AccountId=account_id,
            Budget={
                'BudgetName': 'nfl-prediction-budget',
                'BudgetLimit': {'Amount': '10', 'Unit': 'USD'},
                'TimeUnit': 'MONTHLY',
                'BudgetType': 'COST'
            },
            NotificationsWithSubscribers=[{
                'Notification': {
                    'NotificationType': 'ACTUAL',
                    'ComparisonOperator': 'GREATER_THAN',
                    'Threshold': 80,
                    'ThresholdType': 'PERCENTAGE'
                },
                'Subscribers': [{
                    'SubscriptionType': 'EMAIL',
                    'Address': os.getenv('AWS_BUDGET_EMAIL', 'admin@example.com')
                }]
            }]
        )
        print("✅ Budget alert created ($10/month)")
    except Exception as e:
        print(f"⚠️  Budget: {e}")

def main():
    session = boto3.Session(region_name=AWS_REGION)
    
    iam = session.client('iam')
    s3 = session.client('s3')
    lambda_client = session.client('lambda')
    events = session.client('events')
    budgets = session.client('budgets', region_name='us-east-1')
    sts = session.client('sts')
    account_id = sts.get_caller_identity()['Account']
    
    print("🚀 Setting up AWS MCP for NFL Prediction System\n")
    
    role_arn = create_iam_role(iam)
    print(f"✅ IAM role: {role_arn}")
    
    create_s3_bucket(s3)
    upload_data_to_s3(s3)
    
    create_lambda_function(lambda_client, role_arn)
    create_eventbridge_rule(events, account_id)
    create_budget(budgets, account_id)
    
    print("\n✅ AWS MCP Setup Complete")
    print(f"📦 S3 Bucket: {BUCKET_NAME}")
    print("⚡ Lambda: nfl-live-predictions (runs every 15min)")
    print("💰 Budget: $10/month with alerts")

if __name__ == "__main__":
    main()
