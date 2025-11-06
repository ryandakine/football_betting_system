#!/bin/bash
# AWS Deployment Script for College Football System
# ==================================================

set -e  # Exit on error

# Configuration
AWS_REGION="${AWS_REGION:-us-east-1}"
ECR_REPO_NAME="${ECR_REPO_NAME:-college-football-analyzer}"
LAMBDA_FUNCTION_NAME="${LAMBDA_FUNCTION_NAME:-college-football-analyzer}"
LAMBDA_ROLE_NAME="${LAMBDA_ROLE_NAME:-college-football-lambda-role}"
MEMORY_SIZE="${MEMORY_SIZE:-2048}"
TIMEOUT="${TIMEOUT:-300}"

echo "🚀 Deploying College Football System to AWS"
echo "================================================"
echo "Region: $AWS_REGION"
echo "ECR Repo: $ECR_REPO_NAME"
echo "Lambda: $LAMBDA_FUNCTION_NAME"
echo ""

# Step 1: Build Docker image
echo "📦 Building Docker image..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"
docker build -f college_football_system/Dockerfile.lambda -t $ECR_REPO_NAME:latest .
echo "✅ Docker image built"

# Step 2: Create ECR repository if it doesn't exist
echo "🏗️  Setting up ECR repository..."
aws ecr describe-repositories --repository-names $ECR_REPO_NAME --region $AWS_REGION 2>/dev/null || \
    aws ecr create-repository --repository-name $ECR_REPO_NAME --region $AWS_REGION
echo "✅ ECR repository ready"

# Step 3: Login to ECR
echo "🔐 Logging into ECR..."
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
aws ecr get-login-password --region $AWS_REGION | \
    docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
echo "✅ ECR login successful"

# Step 4: Tag and push image
echo "⬆️  Pushing image to ECR..."
docker tag $ECR_REPO_NAME:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:latest
echo "✅ Image pushed to ECR"

# Step 5: Create/Update Lambda function
echo "⚡ Deploying Lambda function..."

# Check if Lambda exists
if aws lambda get-function --function-name $LAMBDA_FUNCTION_NAME --region $AWS_REGION 2>/dev/null; then
    echo "📝 Updating existing Lambda function..."
    aws lambda update-function-code \
        --function-name $LAMBDA_FUNCTION_NAME \
        --image-uri $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:latest \
        --region $AWS_REGION
    
    aws lambda update-function-configuration \
        --function-name $LAMBDA_FUNCTION_NAME \
        --timeout $TIMEOUT \
        --memory-size $MEMORY_SIZE \
        --region $AWS_REGION
else
    echo "🆕 Creating new Lambda function..."
    
    # Get or create IAM role
    ROLE_ARN=$(aws iam get-role --role-name $LAMBDA_ROLE_NAME --query 'Role.Arn' --output text 2>/dev/null || echo "")
    
    if [ -z "$ROLE_ARN" ]; then
        echo "🔧 Creating Lambda IAM role..."
        aws iam create-role \
            --role-name $LAMBDA_ROLE_NAME \
            --assume-role-policy-document '{
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }]
            }' \
            --region $AWS_REGION
        
        aws iam attach-role-policy \
            --role-name $LAMBDA_ROLE_NAME \
            --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
        
        # Wait for role to propagate
        sleep 10
        ROLE_ARN=$(aws iam get-role --role-name $LAMBDA_ROLE_NAME --query 'Role.Arn' --output text)
    fi
    
    aws lambda create-function \
        --function-name $LAMBDA_FUNCTION_NAME \
        --package-type Image \
        --code ImageUri=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:latest \
        --role $ROLE_ARN \
        --timeout $TIMEOUT \
        --memory-size $MEMORY_SIZE \
        --region $AWS_REGION
fi

echo "✅ Lambda function deployed"

# Step 6: Set environment variables (if .env exists)
if [ -f ".env" ]; then
    echo "🔧 Setting environment variables..."
    # Parse .env and set Lambda env vars
    # Note: This is basic - enhance as needed
    aws lambda update-function-configuration \
        --function-name $LAMBDA_FUNCTION_NAME \
        --environment Variables={BANKROLL=50000} \
        --region $AWS_REGION
    echo "✅ Environment variables set"
fi

echo ""
echo "🎉 Deployment complete!"
echo "================================================"
echo "Lambda ARN: $(aws lambda get-function --function-name $LAMBDA_FUNCTION_NAME --region $AWS_REGION --query 'Configuration.FunctionArn' --output text)"
echo ""
echo "Test your function:"
echo "  aws lambda invoke --function-name $LAMBDA_FUNCTION_NAME --region $AWS_REGION response.json"
echo ""
