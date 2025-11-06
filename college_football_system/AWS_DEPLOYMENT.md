# AWS Deployment Guide - College Football System

## 🚀 Quick Deploy

### Prerequisites
1. AWS CLI installed and configured: `aws configure`
2. Docker installed and running
3. AWS account with Lambda and ECR permissions

### Deploy in 3 Commands

```bash
# 1. Copy and configure environment
cp .env.example .env
# Edit .env with your API keys

# 2. Deploy to AWS
./deploy_aws.sh

# 3. Test the deployment
aws lambda invoke --function-name college-football-analyzer response.json
cat response.json
```

## 📋 What Gets Deployed

### AWS Resources Created
- **ECR Repository**: Docker image storage
- **Lambda Function**: Serverless analyzer (2GB RAM, 5min timeout)
- **IAM Role**: Lambda execution permissions
- **CloudWatch Logs**: Automatic logging

### Cost Estimate
- **Lambda**: ~$0.20 per analysis run (2GB for 60-120 seconds)
- **ECR**: ~$0.10/month for image storage
- **CloudWatch Logs**: ~$0.05/month
- **Total**: ~$2-5/month for daily analysis

## 🎯 Usage

### Manual Invocation
```bash
# Run analysis on-demand
aws lambda invoke \
  --function-name college-football-analyzer \
  --region us-east-1 \
  response.json

# Check results
cat response.json | jq .
```

### Scheduled Execution (EventBridge)
```bash
# Create schedule rule (runs daily at 6 PM)
aws events put-rule \
  --name college-football-daily \
  --schedule-expression "cron(0 18 * * ? *)" \
  --state ENABLED

# Add Lambda as target
aws events put-targets \
  --rule college-football-daily \
  --targets "Id"="1","Arn"="arn:aws:lambda:REGION:ACCOUNT:function:college-football-analyzer"

# Grant EventBridge permission to invoke Lambda
aws lambda add-permission \
  --function-name college-football-analyzer \
  --statement-id EventBridgeInvoke \
  --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn arn:aws:events:REGION:ACCOUNT:rule/college-football-daily
```

### API Gateway (HTTP API)
```bash
# Create REST API for on-demand analysis
aws apigatewayv2 create-api \
  --name college-football-api \
  --protocol-type HTTP \
  --target arn:aws:lambda:REGION:ACCOUNT:function:college-football-analyzer

# Get API endpoint
aws apigatewayv2 get-apis --query 'Items[?Name==`college-football-api`].ApiEndpoint'

# Test via HTTP
curl -X POST https://YOUR_API_ID.execute-api.REGION.amazonaws.com/
```

## 🔧 Configuration

### Update Environment Variables
```bash
aws lambda update-function-configuration \
  --function-name college-football-analyzer \
  --environment Variables="{BANKROLL=75000,ENABLE_CLOUDGPU=true}"
```

### Update Code
```bash
# Make code changes, then redeploy
./deploy_aws.sh
```

### Scale Resources
```bash
# Increase memory/timeout
export MEMORY_SIZE=4096
export TIMEOUT=600
./deploy_aws.sh
```

## 📊 Monitoring

### View Logs
```bash
# Stream live logs
aws logs tail /aws/lambda/college-football-analyzer --follow

# Filter errors
aws logs filter-log-events \
  --log-group-name /aws/lambda/college-football-analyzer \
  --filter-pattern "ERROR"
```

### Check Performance
```bash
# Get function metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Duration \
  --dimensions Name=FunctionName,Value=college-football-analyzer \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-12-31T23:59:59Z \
  --period 3600 \
  --statistics Average,Maximum
```

## 🛠️ Troubleshooting

### Lambda Timeouts
- Increase timeout: `export TIMEOUT=600` and redeploy
- Reduce games analyzed: Edit `main_analyzer.py` line 110

### Memory Issues
- Increase memory: `export MEMORY_SIZE=4096` and redeploy
- Check CloudWatch logs for OOM errors

### API Key Errors
- Verify .env file has correct keys
- Redeploy after changing environment variables

### Cold Start Slow
- Enable provisioned concurrency (costs more):
```bash
aws lambda put-provisioned-concurrency-config \
  --function-name college-football-analyzer \
  --provisioned-concurrent-executions 1
```

## 🗑️ Cleanup

### Delete Everything
```bash
# Delete Lambda function
aws lambda delete-function --function-name college-football-analyzer

# Delete ECR repository
aws ecr delete-repository --repository-name college-football-analyzer --force

# Delete IAM role
aws iam detach-role-policy \
  --role-name college-football-lambda-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
aws iam delete-role --role-name college-football-lambda-role

# Delete EventBridge rule (if created)
aws events remove-targets --rule college-football-daily --ids 1
aws events delete-rule --name college-football-daily
```

## 🎯 Next Steps

1. **Set up alerting**: Add SNS topic for high-edge game alerts
2. **Add S3 storage**: Save analysis results to S3 for historical tracking
3. **Enable API Gateway**: Create public API for mobile access
4. **Add DynamoDB**: Store bet history and track performance
5. **CloudFront CDN**: Add caching layer for faster API responses

## 💡 Pro Tips

- Use Lambda layers for large dependencies (torch, transformers)
- Enable X-Ray tracing for performance analysis
- Set up cost alerts in AWS Budgets
- Use Secrets Manager for API keys instead of env vars
- Consider Step Functions for complex multi-stage analysis
