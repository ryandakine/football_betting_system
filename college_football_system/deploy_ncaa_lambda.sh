#!/bin/bash
set -e

echo "🏈 Deploying NCAA/College Football Analysis System to AWS Lambda"

# Configuration
FUNCTION_NAME="NCAA-GameAnalyzer"
REGION="us-east-1"
MEMORY=3008
TIMEOUT=300
BUCKET="football-betting-system-data"

# Create deployment package
echo "📦 Creating deployment package..."
rm -rf lambda_package
mkdir -p lambda_package

# Copy Lambda handler and NCAA system files
cp college_football_system/lambda_handler.py lambda_package/lambda_function.py
cp college_football_system/main_analyzer.py lambda_package/
cp college_football_system/game_prioritization.py lambda_package/
cp college_football_system/parlay_optimizer.py lambda_package/
cp college_football_system/social_weather_analyzer.py lambda_package/
cp college_football_system/ncaa_live_data_fetcher.py lambda_package/
cp college_football_system/ncaa_injury_tracker.py lambda_package/
cp college_football_system/backtester.py lambda_package/
cp college_football_system/gold_standard_ncaaf_config.py lambda_package/
cp college_football_system/zephyr_integration.py lambda_package/

# Copy shared components
cp football_odds_fetcher.py lambda_package/ 2>/dev/null || true
cp feature_engineer.py lambda_package/ 2>/dev/null || true
cp advanced_feature_engineering.py lambda_package/ 2>/dev/null || true

# Copy models
cp -r models/*.pkl lambda_package/ 2>/dev/null || echo "No local models to copy"

# Install dependencies
cd lambda_package
echo "📥 Installing dependencies..."
pip install -t . \
    boto3 \
    numpy \
    pandas \
    scikit-learn \
    xgboost \
    lightgbm \
    aiohttp \
    requests \
    pydantic \
    tenacity \
    -q

# Remove unnecessary files to reduce size
echo "🧹 Cleaning up unnecessary files..."
find . -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "*.dist-info" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete
find . -name "*.so" -type f -delete 2>/dev/null || true

# Create ZIP
echo "🗜️ Creating ZIP file..."
zip -r ../ncaa_lambda.zip . -q

cd ..

# Upload models to S3 if they exist locally
echo "📤 Uploading models to S3..."
if [ -d "models" ]; then
    aws s3 cp models/ s3://${BUCKET}/models/ncaa/ --recursive --exclude "*" --include "*.pkl"
fi

# Check if function exists
if aws lambda get-function --function-name ${FUNCTION_NAME} --region ${REGION} 2>/dev/null; then
    echo "♻️ Updating existing function..."
    aws lambda update-function-code \
        --function-name ${FUNCTION_NAME} \
        --zip-file fileb://ncaa_lambda.zip \
        --region ${REGION}

    aws lambda update-function-configuration \
        --function-name ${FUNCTION_NAME} \
        --timeout ${TIMEOUT} \
        --memory-size ${MEMORY} \
        --region ${REGION} \
        --environment "Variables={
            S3_BUCKET=${BUCKET},
            ODDS_API_KEY=$(grep ODDS_API_KEY .env | cut -d'=' -f2 2>/dev/null || echo ''),
            NCAA_ODDS_API_KEY=$(grep NCAA_ODDS_API_KEY .env | cut -d'=' -f2 2>/dev/null || echo ''),
            OPENWEATHER_API_KEY=$(grep OPENWEATHER_API_KEY .env | cut -d'=' -f2 2>/dev/null || echo ''),
            ANTHROPIC_API_KEY=$(grep ANTHROPIC_API_KEY .env | cut -d'=' -f2 2>/dev/null || echo ''),
            CFB_DATA_API_KEY=$(grep CFB_DATA_API_KEY .env | cut -d'=' -f2 2>/dev/null || echo ''),
            ZEPHYR_BASE_URL=$(grep ZEPHYR_BASE_URL .env | cut -d'=' -f2 2>/dev/null || echo ''),
            ZEPHYR_API_KEY=$(grep ZEPHYR_API_KEY .env | cut -d'=' -f2 2>/dev/null || echo '')
        }"
else
    echo "🆕 Creating new function..."

    # Create IAM role if it doesn't exist
    ROLE_NAME="ncaa-lambda-execution-role"
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
                    "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
                    "Resource": [
                        "arn:aws:s3:::'${BUCKET}'",
                        "arn:aws:s3:::'${BUCKET}'/*"
                    ]
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
        --zip-file fileb://ncaa_lambda.zip \
        --timeout ${TIMEOUT} \
        --memory-size ${MEMORY} \
        --region ${REGION} \
        --environment "Variables={
            S3_BUCKET=${BUCKET},
            ODDS_API_KEY=$(grep ODDS_API_KEY .env | cut -d'=' -f2 2>/dev/null || echo ''),
            NCAA_ODDS_API_KEY=$(grep NCAA_ODDS_API_KEY .env | cut -d'=' -f2 2>/dev/null || echo ''),
            OPENWEATHER_API_KEY=$(grep OPENWEATHER_API_KEY .env | cut -d'=' -f2 2>/dev/null || echo ''),
            ANTHROPIC_API_KEY=$(grep ANTHROPIC_API_KEY .env | cut -d'=' -f2 2>/dev/null || echo ''),
            CFB_DATA_API_KEY=$(grep CFB_DATA_API_KEY .env | cut -d'=' -f2 2>/dev/null || echo ''),
            ZEPHYR_BASE_URL=$(grep ZEPHYR_BASE_URL .env | cut -d'=' -f2 2>/dev/null || echo ''),
            ZEPHYR_API_KEY=$(grep ZEPHYR_API_KEY .env | cut -d'=' -f2 2>/dev/null || echo '')
        }"
fi

# Cleanup
echo "🧹 Cleaning up temporary files..."
rm -rf lambda_package
rm -f ncaa_lambda.zip

echo ""
echo "✅ NCAA Lambda deployment complete!"
echo ""
echo "Test with:"
echo "aws lambda invoke --function-name ${FUNCTION_NAME} --region ${REGION} response.json && cat response.json"
echo ""
echo "Set up Saturday schedule (for NCAA games) with:"
echo "aws events put-rule --name ncaa-saturday --schedule-expression 'cron(0 */2 * * 6 *)' --region ${REGION}"
echo "aws events put-targets --rule ncaa-saturday --targets 'Id=1,Arn=arn:aws:lambda:${REGION}:$(aws sts get-caller-identity --query Account --output text):function:${FUNCTION_NAME}' --region ${REGION}"
echo "aws lambda add-permission --function-name ${FUNCTION_NAME} --statement-id ncaa-saturday --action 'lambda:InvokeFunction' --principal events.amazonaws.com --source-arn arn:aws:events:${REGION}:$(aws sts get-caller-identity --query Account --output text):rule/ncaa-saturday --region ${REGION}"
echo ""
echo "For weekday MACtion, add:"
echo "aws events put-rule --name ncaa-weekday --schedule-expression 'cron(0 18-23 * * 2-4 *)' --region ${REGION}"
