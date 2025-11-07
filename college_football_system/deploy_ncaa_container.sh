#!/bin/bash
set -e

echo "🏈 Deploying NCAA/College Football Analysis System as Container"

# Configuration
FUNCTION_NAME="NCAA-GameAnalyzer"
REGION="us-east-1"
REPO_NAME="ncaa-analyzer"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Create ECR repository
echo "📦 Creating ECR repository..."
aws ecr create-repository --repository-name ${REPO_NAME} --region ${REGION} 2>/dev/null || echo "Repository already exists"

# Get login token and login to ECR
echo "🔐 Logging in to ECR..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# Build Dockerfile for NCAA if it doesn't exist
if [ ! -f "college_football_system/Dockerfile" ]; then
    echo "📝 Creating Dockerfile for NCAA system..."
    cat > college_football_system/Dockerfile <<'EOF'
FROM public.ecr.aws/lambda/python:3.11

# Copy NCAA system files
COPY college_football_system/ ${LAMBDA_TASK_ROOT}/college_football_system/
COPY lambda_handler.py ${LAMBDA_TASK_ROOT}/
COPY main_analyzer.py ${LAMBDA_TASK_ROOT}/
COPY game_prioritization.py ${LAMBDA_TASK_ROOT}/
COPY parlay_optimizer.py ${LAMBDA_TASK_ROOT}/
COPY social_weather_analyzer.py ${LAMBDA_TASK_ROOT}/
COPY ncaa_live_data_fetcher.py ${LAMBDA_TASK_ROOT}/
COPY ncaa_injury_tracker.py ${LAMBDA_TASK_ROOT}/
COPY backtester.py ${LAMBDA_TASK_ROOT}/
COPY gold_standard_ncaaf_config.py ${LAMBDA_TASK_ROOT}/
COPY zephyr_integration.py ${LAMBDA_TASK_ROOT}/

# Copy shared files if they exist
COPY football_odds_fetcher.py ${LAMBDA_TASK_ROOT}/ || true
COPY feature_engineer.py ${LAMBDA_TASK_ROOT}/ || true
COPY advanced_feature_engineering.py ${LAMBDA_TASK_ROOT}/ || true

# Install dependencies
RUN pip install --no-cache-dir \
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
    --upgrade

# Set handler
CMD [ "lambda_handler.lambda_handler" ]
EOF
fi

# Build and tag image
echo "🔨 Building Docker image..."
cd college_football_system
docker build -t ${REPO_NAME} .
docker tag ${REPO_NAME}:latest ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:latest
cd ..

# Push image
echo "📤 Pushing to ECR..."
docker push ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:latest

# Upload models to S3
echo "📤 Uploading models to S3..."
BUCKET="football-betting-system-data"
if [ -d "models" ]; then
    aws s3 cp models/ s3://${BUCKET}/models/ncaa/ --recursive --exclude "*" --include "*.pkl"
fi

# Create or update Lambda function
ROLE_NAME="ncaa-lambda-execution-role"
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

# Create role if it doesn't exist
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

if aws lambda get-function --function-name ${FUNCTION_NAME} --region ${REGION} 2>/dev/null; then
    echo "♻️ Updating function..."
    aws lambda update-function-code \
        --function-name ${FUNCTION_NAME} \
        --image-uri ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:latest \
        --region ${REGION}
else
    echo "🆕 Creating function..."
    aws lambda create-function \
        --function-name ${FUNCTION_NAME} \
        --package-type Image \
        --code ImageUri=${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:latest \
        --role ${ROLE_ARN} \
        --timeout 300 \
        --memory-size 3008 \
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

echo ""
echo "✅ NCAA Container deployment complete!"
echo ""
echo "Test with:"
echo "aws lambda invoke --function-name ${FUNCTION_NAME} --region ${REGION} response.json && cat response.json"
echo ""
echo "Image URI: ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:latest"
