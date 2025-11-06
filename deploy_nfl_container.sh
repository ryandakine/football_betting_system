#!/bin/bash
set -e

echo "🏈 Deploying NFL Analysis System as Container"

# Configuration
FUNCTION_NAME="NFL-GameAnalyzer"
REGION="us-west-1"
REPO_NAME="nfl-analyzer"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Create ECR repository
echo "📦 Creating ECR repository..."
aws ecr create-repository --repository-name ${REPO_NAME} --region ${REGION} || echo "Repository exists"

# Get login token and login to ECR
echo "🔐 Logging in to ECR..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# Build and tag image
echo "🔨 Building Docker image..."
docker build -t ${REPO_NAME} .
docker tag ${REPO_NAME}:latest ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:latest

# Push image
echo "📤 Pushing to ECR..."
docker push ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:latest

# Upload models to S3
echo "📤 Uploading models to S3..."
BUCKET="football-betting-system-data"
if [ -d "models" ]; then
    aws s3 cp models/ s3://${BUCKET}/models/ --recursive --exclude "*" --include "*.pkl"
fi

# Create or update Lambda function
ROLE_NAME="nfl-lambda-execution-role"
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

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
            ODDS_API_KEY=$(grep ODDS_API_KEY .env | cut -d'=' -f2),
            OPENWEATHER_API_KEY=$(grep OPENWEATHER_API_KEY .env | cut -d'=' -f2)
        }"
fi

echo "✅ Deployment complete!"
echo ""
echo "Test with:"
echo "aws lambda invoke --function-name ${FUNCTION_NAME} --region ${REGION} response.json && cat response.json"