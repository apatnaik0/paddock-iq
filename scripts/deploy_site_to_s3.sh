#!/usr/bin/env bash
set -euo pipefail

STACK_NAME="${STACK_NAME:-PaddockIqStaticSiteStack}"
AWS_PROFILE_NAME="${AWS_PROFILE:-paddockiq}"
AWS_REGION_NAME="${AWS_REGION:-us-east-1}"
SITE_DIR="${SITE_DIR:-site}"

if [ ! -f "${SITE_DIR}/index.html" ]; then
  echo "Missing ${SITE_DIR}/index.html. Run the site pipeline before deploying." >&2
  exit 1
fi

get_output() {
  local key="$1"
  aws cloudformation describe-stacks \
    --profile "$AWS_PROFILE_NAME" \
    --region "$AWS_REGION_NAME" \
    --stack-name "$STACK_NAME" \
    --query "Stacks[0].Outputs[?OutputKey=='${key}'].OutputValue | [0]" \
    --output text
}

BUCKET_NAME="${AWS_S3_BUCKET:-$(get_output SiteBucketName)}"
DISTRIBUTION_ID="${AWS_CLOUDFRONT_DISTRIBUTION_ID:-$(get_output CloudFrontDistributionId)}"

if [ -z "$BUCKET_NAME" ] || [ "$BUCKET_NAME" = "None" ]; then
  echo "Could not resolve SiteBucketName from stack ${STACK_NAME}." >&2
  exit 1
fi
if [ -z "$DISTRIBUTION_ID" ] || [ "$DISTRIBUTION_ID" = "None" ]; then
  echo "Could not resolve CloudFrontDistributionId from stack ${STACK_NAME}." >&2
  exit 1
fi

echo "Deploying ${SITE_DIR}/ to s3://${BUCKET_NAME}"
aws s3 sync "$SITE_DIR/" "s3://${BUCKET_NAME}/" \
  --profile "$AWS_PROFILE_NAME" \
  --region "$AWS_REGION_NAME" \
  --delete \
  --cache-control "public,max-age=300"

echo "Invalidating CloudFront distribution ${DISTRIBUTION_ID}"
aws cloudfront create-invalidation \
  --profile "$AWS_PROFILE_NAME" \
  --distribution-id "$DISTRIBUTION_ID" \
  --paths "/*" \
  --output table
