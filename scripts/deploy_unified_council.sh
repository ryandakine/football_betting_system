#!/usr/bin/env bash

# Unified Betting Intelligence AWS deployment orchestrator.
# Supports Lambda, SageMaker, and ECS targets while optionally
# bundling backtesting datasets for archival or training jobs.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_ROOT="${PROJECT_ROOT}/dist"
BUILD_ROOT="${DIST_ROOT}/unified_council_build"
ARTIFACT_NAME="unified_council_lambda.zip"
ARTIFACT_PATH="${DIST_ROOT}/${ARTIFACT_NAME}"
BACKTEST_BUNDLE=""

# Defaults with environment overrides.
DEFAULT_BUCKET="${AWS_UNIFIED_BUCKET:-nfl-conspiracy-predictions}"
DEFAULT_REGION="${AWS_UNIFIED_REGION:-${AWS_REGION:-us-east-1}}"
DEFAULT_FUNCTION="${AWS_UNIFIED_FUNCTION:-unified-betting-intelligence}"
DEFAULT_S3_KEY="${AWS_UNIFIED_KEY:-artifacts/${ARTIFACT_NAME}}"
DEFAULT_TARGET="${DEPLOY_TARGET:-lambda}"
DEFAULT_BACKTEST_BUCKET="${AWS_BACKTEST_BUCKET:-$DEFAULT_BUCKET}"
DEFAULT_BACKTEST_KEY="${AWS_BACKTEST_KEY:-backtesting/backtesting_datasets.tar.gz}"

S3_BUCKET="$DEFAULT_BUCKET"
AWS_REGION="$DEFAULT_REGION"
LAMBDA_FUNCTION="$DEFAULT_FUNCTION"
S3_KEY="$DEFAULT_S3_KEY"
TARGET="$DEFAULT_TARGET"
INCLUDE_BACKTESTS="${INCLUDE_BACKTESTS:-false}"
RUN_SAGEMAKER_JOB="${RUN_SAGEMAKER_JOB:-false}"
UPDATE_ECS_SERVICE="${UPDATE_ECS_SERVICE:-false}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:-${PROJECT_ROOT}/Dockerfile}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --target <lambda|sagemaker|ecs>  Deployment target (default: $TARGET)
  --bucket <name>                 S3 bucket for artifacts (default: $S3_BUCKET)
  --region <name>                 AWS region (default: $AWS_REGION)
  --function <name>               Lambda function name (lambda target)
  --s3-key <key>                  Artifact key within the bucket (default: $S3_KEY)
  --include-backtests             Bundle backtesting datasets and upload to S3
  --backtest-bucket <name>        Bucket for backtesting bundle (default: $DEFAULT_BACKTEST_BUCKET)
  --backtest-key <key>            S3 key for backtesting bundle (default: $DEFAULT_BACKTEST_KEY)
  --launch-sagemaker-job          Start SageMaker training job (requires env vars)
  --update-ecs-service            Force ECS service deployment (requires env vars)
  --dockerfile <path>             Dockerfile for ECS image (default: $DOCKERFILE_PATH)
  --image-tag <tag>               Image tag for ECS deploy (default: $IMAGE_TAG)
  --help                          Show this message

Environment variables:
  AWS_* overrides (AWS_PROFILE, AWS_ACCESS_KEY_ID, etc.)
  SAGEMAKER_ROLE_ARN, SAGEMAKER_IMAGE_URI, SAGEMAKER_INSTANCE_TYPE,
  SAGEMAKER_TRAINING_DATA_S3_URI, SAGEMAKER_OUTPUT_S3_URI,
  SAGEMAKER_INSTANCE_COUNT, SAGEMAKER_VOLUME_SIZE, SAGEMAKER_MAX_RUNTIME,
  SAGEMAKER_JOB_NAME (required when launching training job)
  ECR_REPOSITORY, ECS_CLUSTER, ECS_SERVICE for ECS deploys.
EOF
}

log() {
  printf '==> %s\n' "$*"
}

abort() {
  printf '[ERROR] %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || abort "$1 not found on PATH."
}

# Parse CLI arguments.
BACKTEST_BUCKET="$DEFAULT_BACKTEST_BUCKET"
BACKTEST_KEY="$DEFAULT_BACKTEST_KEY"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="$2"; shift 2 ;;
    --bucket) S3_BUCKET="$2"; shift 2 ;;
    --region) AWS_REGION="$2"; shift 2 ;;
    --function) LAMBDA_FUNCTION="$2"; shift 2 ;;
    --s3-key) S3_KEY="$2"; shift 2 ;;
    --include-backtests) INCLUDE_BACKTESTS="true"; shift ;;
    --backtest-bucket) BACKTEST_BUCKET="$2"; shift 2 ;;
    --backtest-key) BACKTEST_KEY="$2"; shift 2 ;;
    --launch-sagemaker-job) RUN_SAGEMAKER_JOB="true"; shift ;;
    --update-ecs-service) UPDATE_ECS_SERVICE="true"; shift ;;
    --dockerfile) DOCKERFILE_PATH="$2"; shift 2 ;;
    --image-tag) IMAGE_TAG="$2"; shift 2 ;;
    --help) usage; exit 0 ;;
    *) usage; abort "Unknown option: $1" ;;
  esac
done

TARGET="$(echo "$TARGET" | tr '[:upper:]' '[:lower:]')"

require_cmd aws
require_cmd python3
require_cmd zip
require_cmd tar

build_package() {
  log "Preparing build directory at ${BUILD_ROOT}"
  rm -rf "${BUILD_ROOT}"
  mkdir -p "${BUILD_ROOT}"

  log "Copying council sources"
  cp "${PROJECT_ROOT}/unified_betting_intelligence.py" "${BUILD_ROOT}/"
  cp "${PROJECT_ROOT}/betting_types.py" "${BUILD_ROOT}/"
  cp "${PROJECT_ROOT}/config_loader.py" "${BUILD_ROOT}/"
  cp "${PROJECT_ROOT}/ai_council_narrative_unified.py" "${BUILD_ROOT}/"
  cp "${PROJECT_ROOT}/config.yaml" "${BUILD_ROOT}/"
  [[ -f "${PROJECT_ROOT}/logging_config.yaml" ]] && cp "${PROJECT_ROOT}/logging_config.yaml" "${BUILD_ROOT}/"

  local req_file="${PROJECT_ROOT}/requirements_lambda.txt"
  if [[ ! -f "$req_file" ]]; then
    log "requirements_lambda.txt not found, falling back to requirements.txt"
    req_file="${PROJECT_ROOT}/requirements.txt"
  fi

  log "Installing dependencies into build directory"
  python3 -m venv "${BUILD_ROOT}/.venv"
  # shellcheck disable=SC1091
  source "${BUILD_ROOT}/.venv/bin/activate"
  pip install --upgrade pip >/dev/null
  pip install -r "${req_file}" --target "${BUILD_ROOT}" >/dev/null
  deactivate
  rm -rf "${BUILD_ROOT}/.venv"

  log "Cleaning cached artifacts"
  find "${BUILD_ROOT}" -name "__pycache__" -type d -prune -exec rm -rf {} +
  find "${BUILD_ROOT}" -name "*.pyc" -delete
  rm -rf "${BUILD_ROOT}/tests" "${BUILD_ROOT}/test" 2>/dev/null || true

  mkdir -p "${DIST_ROOT}"
  log "Creating deployment zip ${ARTIFACT_NAME}"
  (cd "${BUILD_ROOT}" && zip -qr "${ARTIFACT_PATH}" .)
}

bundle_backtests() {
  local bundle="${DIST_ROOT}/backtesting_datasets.tar.gz"
  local -a rel_paths=()

  shopt -s nullglob
  for path in "${PROJECT_ROOT}"/backtest_results*.json "${PROJECT_ROOT}"/backtest_season_*.json; do
    rel_paths+=("${path#"${PROJECT_ROOT}/"}")
  done
  shopt -u nullglob

  if [[ -d "${PROJECT_ROOT}/data/referee_conspiracy" ]]; then
    rel_paths+=("data/referee_conspiracy")
  fi

  if [[ ${#rel_paths[@]} -eq 0 ]]; then
    log "No backtesting artifacts found; skipping bundle."
    return 1
  fi

  log "Bundling ${#rel_paths[@]} backtesting artifacts into ${bundle}"
  rm -f "${bundle}"
  (cd "${PROJECT_ROOT}" && tar -czf "${bundle}" "${rel_paths[@]}")
  BACKTEST_BUNDLE="${bundle}"
}

upload_artifact() {
  log "Uploading artifact to s3://${S3_BUCKET}/${S3_KEY}"
  aws s3 cp "${ARTIFACT_PATH}" "s3://${S3_BUCKET}/${S3_KEY}" --region "${AWS_REGION}"
}

upload_backtests() {
  [[ -z "${BACKTEST_BUNDLE}" ]] && return 0
  log "Uploading backtesting bundle to s3://${BACKTEST_BUCKET}/${BACKTEST_KEY}"
  aws s3 cp "${BACKTEST_BUNDLE}" "s3://${BACKTEST_BUCKET}/${BACKTEST_KEY}" --region "${AWS_REGION}"
}

deploy_lambda() {
  log "Updating Lambda function ${LAMBDA_FUNCTION}"
  aws lambda update-function-code \
    --function-name "${LAMBDA_FUNCTION}" \
    --s3-bucket "${S3_BUCKET}" \
    --s3-key "${S3_KEY}" \
    --publish \
    --region "${AWS_REGION}" \
    >/tmp/unified_deploy_update.json

  log "Waiting for Lambda update to finish"
  aws lambda wait function-updated \
    --function-name "${LAMBDA_FUNCTION}" \
    --region "${AWS_REGION}"

  log "Invoking Lambda smoke test"
  aws lambda invoke \
    --function-name "${LAMBDA_FUNCTION}" \
    --payload '{}' \
    --region "${AWS_REGION}" \
    /tmp/unified_deploy_invoke.json >/dev/null

  log "Lambda response:"
  cat /tmp/unified_deploy_invoke.json
}

prepare_sagemaker() {
  local code_uri="s3://${S3_BUCKET}/${S3_KEY}"
  log "SageMaker code artifact ready at ${code_uri}"

  if [[ "${INCLUDE_BACKTESTS}" == "true" && -n "${BACKTEST_BUNDLE}" ]]; then
    upload_backtests
  fi

  if [[ "${RUN_SAGEMAKER_JOB}" != "true" ]]; then
    log "Skipping SageMaker training job (set --launch-sagemaker-job to enable)."
    return 0
  fi

  : "${SAGEMAKER_JOB_NAME:?Set SAGEMAKER_JOB_NAME to launch a training job}"
  : "${SAGEMAKER_ROLE_ARN:?Set SAGEMAKER_ROLE_ARN for the training job}"
  : "${SAGEMAKER_IMAGE_URI:?Set SAGEMAKER_IMAGE_URI (ECR image for training)}"
  : "${SAGEMAKER_INSTANCE_TYPE:?Set SAGEMAKER_INSTANCE_TYPE (e.g., ml.m5.xlarge)}"
  : "${SAGEMAKER_OUTPUT_S3_URI:?Set SAGEMAKER_OUTPUT_S3_URI (e.g., s3://bucket/output/)}"

  local input_s3_uri="${SAGEMAKER_TRAINING_DATA_S3_URI:-s3://${BACKTEST_BUCKET}/${BACKTEST_KEY}}"
  local instance_count="${SAGEMAKER_INSTANCE_COUNT:-1}"
  local volume_size="${SAGEMAKER_VOLUME_SIZE:-30}"
  local max_runtime="${SAGEMAKER_MAX_RUNTIME:-7200}"

  log "Launching SageMaker training job ${SAGEMAKER_JOB_NAME}"
  aws sagemaker create-training-job \
    --region "${AWS_REGION}" \
    --training-job-name "${SAGEMAKER_JOB_NAME}" \
    --algorithm-specification "TrainingImage=${SAGEMAKER_IMAGE_URI},TrainingInputMode=File" \
    --role-arn "${SAGEMAKER_ROLE_ARN}" \
    --input-data-config "[{\"ChannelName\":\"training\",\"DataSource\":{\"S3DataSource\":{\"S3DataType\":\"S3Prefix\",\"S3Uri\":\"${input_s3_uri}\",\"S3DataDistributionType\":\"FullyReplicated\"}}}]" \
    --output-data-config "S3OutputPath=${SAGEMAKER_OUTPUT_S3_URI}" \
    --resource-config "InstanceType=${SAGEMAKER_INSTANCE_TYPE},InstanceCount=${instance_count},VolumeSizeInGB=${volume_size}" \
    --stopping-condition "MaxRuntimeInSeconds=${max_runtime}" \
    --hyper-parameters "{\"code_artifact\":\"${code_uri}\"}"

  log "Training job submitted. Monitor via AWS console or CloudWatch."
}

deploy_ecs() {
  require_cmd docker

  [[ -f "${DOCKERFILE_PATH}" ]] || abort "Dockerfile not found at ${DOCKERFILE_PATH}"
  : "${ECR_REPOSITORY:?Set ECR_REPOSITORY (e.g., 123456789012.dkr.ecr.us-east-1.amazonaws.com/unified-council)}"

  local repository_uri="${ECR_REPOSITORY}"
  local image="${repository_uri}:${IMAGE_TAG}"
  local account_id="${repository_uri%%.*}"
  local registry="${repository_uri%%/*}"

  log "Building Docker image ${image}"
  docker build -f "${DOCKERFILE_PATH}" -t "${image}" "${PROJECT_ROOT}"

  log "Logging in to ECR ${registry}"
  aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${registry}"

  log "Pushing image ${image}"
  docker push "${image}"

  if [[ "${UPDATE_ECS_SERVICE}" == "true" ]]; then
    : "${ECS_CLUSTER:?Set ECS_CLUSTER to your ECS cluster name}"
    : "${ECS_SERVICE:?Set ECS_SERVICE to your ECS service name}"

    log "Forcing ECS service deployment ${ECS_SERVICE}"
    aws ecs update-service \
      --cluster "${ECS_CLUSTER}" \
      --service "${ECS_SERVICE}" \
      --force-new-deployment \
      --region "${AWS_REGION}" >/dev/null
    log "ECS service update initiated."
  else
    log "ECS image pushed. Set --update-ecs-service to roll service deployments."
  fi
}

main() {
  log "Validating AWS credentials"
  aws sts get-caller-identity --region "${AWS_REGION}" >/dev/null || abort "Unable to authenticate with AWS."

  build_package
  upload_artifact

  if [[ "${INCLUDE_BACKTESTS}" == "true" ]]; then
    if bundle_backtests; then
      upload_backtests
    else
      log "Proceeding without backtesting bundle."
    fi
  fi

  case "${TARGET}" in
    lambda)
      deploy_lambda
      ;;
    sagemaker)
      prepare_sagemaker
      ;;
    ecs)
      deploy_ecs
      ;;
    *)
      abort "Unsupported target '${TARGET}'. Use lambda, sagemaker, or ecs."
      ;;
  esac

  log "Deployment workflow finished for target ${TARGET}."
}

main "$@"
