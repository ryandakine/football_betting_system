#!/bin/bash
set -e

echo "🏈 Deploying NFL Analysis System to AWS Lambda"

# Configuration
FUNCTION_NAME="NFL-GameAnalyzer"
REGION="us-east-1"
MEMORY=3008
TIMEOUT=300
BUCKET="football-betting-system-data"

# Create deployment package
echo "📦 Creating deployment package..."
rm -rf lambda_package
mkdir -p lambda_package

# Copy Lambda handler
cp nfl_lambda_handler.py lambda_package/lambda_function.py
cp nfl_live_data_fetcher.py lambda_package/
cp -r models/*.pkl lambda_package/ 2>/dev/null || echo "No local models to copy"

# Install dependencies
cd lambda_package
pip install -t . \
    boto3 \
    numpy \
    pandas \
    scikit-learn \
    xgboost \
    lightgbm \
    aiohttp \
    requests \
    -q

# Remove unnecessary files to reduce size
find . -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete

# Create ZIP
echo "🗜️ Creating ZIP file..."
zip -r ../nfl_lambda.zip . -q

cd ..

# Upload models to S3 if they exist locally
echo "📤 Uploading models to S3..."
if [ -d "models" ]; then
    aws s3 cp models/ s3://${BUCKET}/models/ --recursive --exclude "*" --include "*.pkl"
fi

# Check if function exists
if aws lambda get-function --function-name ${FUNCTION_NAME} --region ${REGION} 2>/dev/null; then
    echo "♻️ Updating existing function..."
    aws lambda update-function-code \
        --function-name ${FUNCTION_NAME} \
        --zip-file fileb://nfl_lambda.zip \
        --region ${REGION}
    
    aws lambda update-function-configuration \
        --function-name ${FUNCTION_NAME} \
        --timeout ${TIMEOUT} \
        --memory-size ${MEMORY} \
        --region ${REGION} \
        --environment "Variables={
            S3_BUCKET=${BUCKET},
            ODDS_API_KEY=$(grep ODDS_API_KEY .env | cut -d'=' -f2),
            OPENWEATHER_API_KEY=$(grep OPENWEATHER_API_KEY .env | cut -d'=' -f2)
        }"
else
    echo "🆕 Creating new function..."
    
    # Create IAM role if it doesn't exist
    ROLE_NAME="nfl-lambda-execution-role"
    if ! aws iam get-role --role-name ${ROLE_NAME} 2>/dev/null; then
        echo "Creating IAM role..."
        aws iam create-role \
            --role-name ${ROLE_NAME} \
            --assume-role-policy-document '{
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }]
            }'
        
        aws iam attach-role-policy \
            --role-name ${ROLE_NAME} \
            --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
        
        aws iam put-role-policy \
            --role-name ${ROLE_NAME} \
            --policy-name S3Access \
            --policy-document '{
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Action": ["s3:GetObject", "s3:PutObject"],
                    "Resource": "arn:aws:s3:::'${BUCKET}'/*"
                }]
            }'
        
        echo "Waiting for IAM role to propagate..."
        sleep 10
    fi
    
    ROLE_ARN=$(aws iam get-role --role-name ${ROLE_NAME} --query 'Role.Arn' --output text)
    
    aws lambda create-function \
        --function-name ${FUNCTION_NAME} \
        --runtime python3.11 \
        --role ${ROLE_ARN} \
        --handler lambda_function.lambda_handler \
        --zip-file fileb://nfl_lambda.zip \
        --timeout ${TIMEOUT} \
        --memory-size ${MEMORY} \
        --region ${REGION} \
        --environment "Variables={
            S3_BUCKET=${BUCKET},
            ODDS_API_KEY=$(grep ODDS_API_KEY .env | cut -d'=' -f2),
            OPENWEATHER_API_KEY=$(grep OPENWEATHER_API_KEY .env | cut -d'=' -f2)
        }"
fi

echo "✅ Deployment complete!"
echo ""
echo "Test with:"
echo "aws lambda invoke --function-name ${FUNCTION_NAME} --region ${REGION} response.json && cat response.json"
echo ""
echo "Set up hourly schedule with:"
echo "aws events put-rule --name nfl-hourly --schedule-expression 'cron(0 * * * ? *)' --region ${REGION}"
echo "aws events put-targets --rule nfl-hourly --targets 'Id=1,Arn=arn:aws:lambda:${REGION}:$(aws sts get-caller-identity --query Account --output text):function:${FUNCTION_NAME}' --region ${REGION}"
echo "aws lambda add-permission --function-name ${FUNCTION_NAME} --statement-id nfl-hourly --action 'lambda:InvokeFunction' --principal events.amazonaws.com --source-arn arn:aws:events:${REGION}:$(aws sts get-caller-identity --query Account --output text):rule/nfl-hourly --region ${REGION}"
