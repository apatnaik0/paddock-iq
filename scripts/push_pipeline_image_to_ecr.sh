#!/usr/bin/env bash
set -euo pipefail

STACK_NAME="${STACK_NAME:-PaddockIqStaticSiteStack}"
AWS_PROFILE_NAME="${AWS_PROFILE:-paddockiq}"
AWS_REGION_NAME="${AWS_REGION:-us-east-1}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DOCKERFILE="${DOCKERFILE:-Dockerfile}"
CONTEXT_DIR="${CONTEXT_DIR:-.}"

get_output() {
  local key="$1"
  aws cloudformation describe-stacks \
    --profile "$AWS_PROFILE_NAME" \
    --region "$AWS_REGION_NAME" \
    --stack-name "$STACK_NAME" \
    --query "Stacks[0].Outputs[?OutputKey=='${key}'].OutputValue | [0]" \
    --output text
}

REPOSITORY_URI="${ECR_REPOSITORY_URI:-$(get_output PipelineImageRepositoryUri)}"

if [ -z "$REPOSITORY_URI" ] || [ "$REPOSITORY_URI" = "None" ]; then
  echo "Could not resolve PipelineImageRepositoryUri from stack ${STACK_NAME}." >&2
  exit 1
fi

REGISTRY_HOST="${REPOSITORY_URI%%/*}"
IMAGE_URI="${REPOSITORY_URI}:${IMAGE_TAG}"

echo "Logging in to ${REGISTRY_HOST}"
aws ecr get-login-password \
  --profile "$AWS_PROFILE_NAME" \
  --region "$AWS_REGION_NAME" \
  | docker login --username AWS --password-stdin "$REGISTRY_HOST"

echo "Building ${IMAGE_URI}"
docker build -f "$DOCKERFILE" -t "$IMAGE_URI" "$CONTEXT_DIR"

echo "Pushing ${IMAGE_URI}"
docker push "$IMAGE_URI"

echo "Pushed ${IMAGE_URI}"
