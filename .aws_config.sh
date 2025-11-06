#!/bin/bash
# AWS configuration prerequisite checker.
# Make sure you run `aws configure` or export AWS_PROFILE / credentials before invoking this helper.

set -euo pipefail

if ! command -v aws >/dev/null 2>&1; then
  echo "AWS CLI not found. Install it (https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) before continuing."
  exit 1
fi

if ! aws sts get-caller-identity >/dev/null 2>&1; then
  cat <<'EOF'
AWS credentials are not configured for this environment.
Use one of the following before rerunning:
  • aws configure                # stores credentials via AWS CLI
  • export AWS_PROFILE=your-profile
  • export AWS_ACCESS_KEY_ID=... and AWS_SECRET_ACCESS_KEY=...
EOF
  exit 1
fi

echo "AWS credentials detected for account: $(aws sts get-caller-identity --query Account --output text)"
echo "Ready to run deployment scripts."
