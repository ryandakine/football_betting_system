# GitHub Repository Secrets Setup Guide

## 🔐 Required Secrets for CI/CD

### **Docker Hub Secrets**
- `DOCKER_USERNAME` - Your Docker Hub username
- `DOCKER_PASSWORD` - Your Docker Hub password/token

### **Security Scanning Secrets**
- `SNYK_TOKEN` - Snyk API token for vulnerability scanning

### **Notification Secrets (Optional)**
- `SLACK_WEBHOOK_URL` - Slack webhook for notifications
- `DISCORD_WEBHOOK_URL` - Discord webhook for notifications

### **Deployment Secrets (Optional)**
- `AWS_ACCESS_KEY_ID` - AWS access key for deployment
- `AWS_SECRET_ACCESS_KEY` - AWS secret key for deployment
- `KUBECONFIG` - Kubernetes configuration for deployment

## 📋 Step-by-Step Setup Instructions

### **1. Access Repository Settings**
1. Go to your GitHub repository
2. Click on **Settings** tab
3. In the left sidebar, click **Secrets and variables** → **Actions**

### **2. Add Each Secret**
1. Click **New repository secret**
2. Enter the secret name (e.g., `DOCKER_USERNAME`)
3. Enter the secret value
4. Click **Add secret**

### **3. Docker Hub Setup**
```bash
# Create Docker Hub account if you don't have one
# Go to https://hub.docker.com/

# Generate access token:
# 1. Go to Account Settings → Security
# 2. Click "New Access Token"
# 3. Give it a name (e.g., "MLB Betting System CI/CD")
# 4. Copy the token
```

### **4. Snyk Setup**
```bash
# Install Snyk CLI
npm install -g snyk

# Login to Snyk
snyk auth

# Get your API token from:
# https://app.snyk.io/account
```

## 🔧 Quick Setup Script

```bash
# This script will help you set up the secrets
# Run this in your local environment to prepare values

echo "=== GitHub Secrets Setup ==="
echo ""
echo "Please prepare the following values:"
echo ""
echo "1. Docker Hub:"
echo "   - Username: [your-docker-username]"
echo "   - Password/Token: [your-docker-token]"
echo ""
echo "2. Snyk:"
echo "   - API Token: [your-snyk-token]"
echo ""
echo "3. Optional - Slack/Discord:"
echo "   - Webhook URL: [your-webhook-url]"
echo ""
echo "Then add them to GitHub repository secrets:"
echo "https://github.com/[username]/[repo]/settings/secrets/actions"
```

## ✅ Verification

After adding secrets, you can verify they're working by:

1. **Trigger a test workflow** - Push a small change to trigger CI/CD
2. **Check workflow logs** - Ensure secrets are being used correctly
3. **Monitor deployments** - Verify Docker builds and deployments work

## 🔒 Security Best Practices

- **Never commit secrets** to your repository
- **Use environment-specific secrets** for different environments
- **Rotate secrets regularly** (especially API tokens)
- **Use least privilege** - Only grant necessary permissions
- **Monitor secret usage** - Check for unauthorized access
