# PaddockIQ AWS Static Site Infrastructure

This CDK app creates the Phase 3 static hosting layer:

- Private S3 bucket for committed `site/` files
- CloudFront distribution in front of the private bucket
- CloudFront Origin Access Control (OAC)
- S3 bucket policy that only allows CloudFront reads
- GitHub Actions OIDC deploy role scoped to `apatnaik0/paddock-iq` on `main`

It does not run the FastF1 data pipeline. That comes in Phase 4.

## Prerequisites

```bash
AWS_PROFILE=paddockiq aws sts get-caller-identity
node --version
npm --version
cdk --version
```

CDK bootstrap must already be complete for `us-east-1`.

## Install

```bash
cd infra/aws-cdk
npm install
```

## Synthesize

```bash
AWS_PROFILE=paddockiq npx cdk synth
```

## Deploy

```bash
AWS_PROFILE=paddockiq npx cdk deploy \
  -c projectName=paddock-iq \
  -c environmentName=prod \
  -c githubRepository=apatnaik0/paddock-iq \
  -c githubBranch=main
```

After deployment, note these outputs:

- `CloudFrontDomainName`
- `GitHubDeployRoleArn`
- `SiteBucketName`
- `CloudFrontDistributionId`

## GitHub Actions OIDC Setup

Add this repository secret after CDK deploys:

```text
AWS_SITE_DEPLOY_ROLE_ARN=<GitHubDeployRoleArn output value>
```

The workflow `.github/workflows/deploy-aws-site.yml` uses this role to sync `site/` to S3 and invalidate CloudFront.

## Local Deploy

From the repo root:

```bash
AWS_PROFILE=paddockiq ./scripts/deploy_site_to_s3.sh
```

The script reads `SiteBucketName` and `CloudFrontDistributionId` from the CloudFormation stack outputs.

## Destroying

The S3 bucket has `RemovalPolicy.RETAIN` and `autoDeleteObjects=false`, so destroying the stack will not automatically delete site files. This is intentional to avoid accidental data loss.
