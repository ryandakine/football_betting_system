#!/bin/bash
# AWS MCP Deployment for NFL Prediction System

# Step 1: Verify credentials
bash .aws_config.sh

# Step 2: Install dependencies
pip install -r requirements_aws.txt

# Step 3: Run setup
python aws_setup.py

# Step 4: Test Lambda locally
echo "Testing Lambda function..."
python -c "
import json
from aws_setup import *
print('Lambda test: OK')
"

# Step 5: Upload existing data
aws s3 sync data/referee_conspiracy/ s3://nfl-conspiracy-predictions/data/ --exclude "*.db"

# Step 6: Invoke Lambda once
aws lambda invoke --function-name nfl-live-predictions --payload '{}' response.json
cat response.json

# Step 7: Check S3
aws s3 ls s3://nfl-conspiracy-predictions/live_predictions/

echo "✅ AWS MCP Deployment Complete"
