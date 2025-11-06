#!/usr/bin/env python3
"""Direct AWS Lambda deployment without MCP"""

import boto3
import os
import json
from pathlib import Path

def deploy():
    """Deploy AI Council to Lambda"""
    
    lambda_client = boto3.client('lambda', region_name='us-east-1')
    iam_client = boto3.client('iam', region_name='us-east-1')
    
    # Get or create Lambda role
    role_name = 'football-ai-council-lambda-role'
    try:
        role = iam_client.get_role(RoleName=role_name)
        role_arn = role['Role']['Arn']
        print(f"✓ Using existing role: {role_arn}")
    except:
        print(f"Creating IAM role: {role_name}")
        assume_role = {
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        }
        role = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(assume_role)
        )
        role_arn = role['Role']['Arn']
        iam_client.attach_role_policy(
            RoleName=role_name,
            PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
        )
        print(f"✓ Created role: {role_arn}")
    
    # Create deployment package
    print("\n📦 Packaging Lambda function...")
    os.system('cd /home/ryan/code/football_betting_system && python deploy_enhanced_ai_council_lambda.py')
    
    zip_path = Path('/home/ryan/code/football_betting_system/lambda_package/enhanced_ai_council.zip')
    if not zip_path.exists():
        print(f"✗ Package not found: {zip_path}")
        return
    
    with open(zip_path, 'rb') as f:
        zip_content = f.read()
    
    # Deploy to Lambda
    function_name = 'enhanced_ai_council_predictions'
    print(f"\n🚀 Deploying to Lambda: {function_name}")
    
    try:
        lambda_client.get_function(FunctionName=function_name)
        print(f"Updating existing function...")
        lambda_client.update_function_code(
            FunctionName=function_name,
            ZipFile=zip_content
        )
    except:
        print(f"Creating new function...")
        lambda_client.create_function(
            FunctionName=function_name,
            Runtime='python3.11',
            Role=role_arn,
            Handler='lambda_function.lambda_handler',
            Code={'ZipFile': zip_content},
            Timeout=300,
            MemorySize=1024,
            Description='Enhanced AI Council NFL predictions'
        )
    
    print(f"✓ Deployed successfully!")
    print(f"\nLambda Function: {function_name}")
    print(f"Region: us-east-1")

if __name__ == "__main__":
    deploy()
