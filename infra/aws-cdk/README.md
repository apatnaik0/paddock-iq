# PaddockIQ AWS Static Site Infrastructure

This CDK app creates the Phase 3 static hosting layer:

- Private S3 bucket for committed `site/` files
- CloudFront distribution in front of the private bucket
- CloudFront Origin Access Control (OAC)
- S3 bucket policy that only allows CloudFront reads
- GitHub Actions OIDC deploy role scoped to `apatnaik0/paddock-iq` on `main`
- ECR repository for the future automated pipeline container
- ECS/Fargate cluster and task definition for the race-weekend updater
- Disabled EventBridge schedule for automated session checks
- CloudWatch log group for pipeline runs

CloudFront is controlled by the `enableCloudFront` CDK context value. It defaults to
`false` so a normal synth or deploy will keep the public distribution paused unless
you explicitly opt in. The pipeline schedule is controlled separately by
`enablePipelineSchedule`, which also defaults to `false`.

The scheduled pipeline infrastructure is present but paused by default. It does not
run FastF1 jobs unless `enablePipelineSchedule=true` is deployed and a container
image has been pushed to ECR.

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

Deploy with CloudFront paused:

```bash
AWS_PROFILE=paddockiq npx cdk deploy \
  -c projectName=paddock-iq \
  -c environmentName=prod \
  -c githubRepository=apatnaik0/paddock-iq \
  -c githubBranch=main \
  -c enableCloudFront=false \
  -c enablePipelineSchedule=false
```

Deploy with CloudFront enabled:

```bash
AWS_PROFILE=paddockiq npx cdk deploy \
  -c projectName=paddock-iq \
  -c environmentName=prod \
  -c githubRepository=apatnaik0/paddock-iq \
  -c githubBranch=main \
  -c enableCloudFront=true \
  -c enablePipelineSchedule=false
```

After deployment, note these outputs:

- `CloudFrontDomainName`
- `GitHubDeployRoleArn`
- `SiteBucketName`
- `CloudFrontDistributionId`
- `CloudFrontEnabled`
- `PipelineImageRepositoryUri`
- `PipelineClusterName`
- `PipelineScheduleEnabled`

## GitHub Actions OIDC Setup

Add this repository secret after CDK deploys:

```text
AWS_SITE_DEPLOY_ROLE_ARN=<GitHubDeployRoleArn output value>
```

The workflow `.github/workflows/deploy-aws-site.yml` uses this role to sync `site/` to S3 and invalidate CloudFront.
If this secret is missing, the workflow skips the AWS deploy steps cleanly. That
does not affect local site generation or the committed site files.

## Pause / Resume Public Hosting

To pause public hosting while keeping the bucket, role, and site files intact:

```bash
AWS_PROFILE=paddockiq npx cdk deploy \
  -c projectName=paddock-iq \
  -c environmentName=prod \
  -c githubRepository=apatnaik0/paddock-iq \
  -c githubBranch=main \
  -c enableCloudFront=false \
  -c enablePipelineSchedule=false
```

To resume public hosting:

```bash
AWS_PROFILE=paddockiq npx cdk deploy \
  -c projectName=paddock-iq \
  -c environmentName=prod \
  -c githubRepository=apatnaik0/paddock-iq \
  -c githubBranch=main \
  -c enableCloudFront=true \
  -c enablePipelineSchedule=false
```

## Pipeline Automation Skeleton

The Phase 4 skeleton creates the AWS resources needed to run the Dockerized
pipeline later:

- ECR repo: stores the PaddockIQ pipeline image
- ECS cluster: serverless Fargate execution environment
- Fargate task definition: runs the existing Docker entrypoint in `PADDOCK_MODE=auto`
- EventBridge rule: scheduled trigger, disabled unless explicitly enabled
- CloudWatch logs: stores container logs for debugging pipeline runs

The VPC uses public subnets and no NAT gateways. This keeps idle cost low because
NAT gateways are one of the easiest ways to accidentally create always-on charges.

Build and push the pipeline image after the stack exists:

```bash
AWS_PROFILE=paddockiq ./scripts/push_pipeline_image_to_ecr.sh
```

Enable scheduled checks only when ready:

```bash
AWS_PROFILE=paddockiq npx cdk deploy \
  -c projectName=paddock-iq \
  -c environmentName=prod \
  -c githubRepository=apatnaik0/paddock-iq \
  -c githubBranch=main \
  -c enableCloudFront=true \
  -c enablePipelineSchedule=true \
  -c pipelineSeason=2026
```

The default schedule is:

```text
cron(0 6,12,18,23 ? * FRI,SAT,SUN *)
```

Override it with `-c pipelineScheduleExpression='<eventbridge expression>'`.

Current limitation: the Fargate task can run the auto-update pipeline, but the
cloud-run publish step still needs to be implemented before fully enabling the
schedule. Until then, keep `enablePipelineSchedule=false` for production use.

## Local Deploy

From the repo root:

```bash
AWS_PROFILE=paddockiq ./scripts/deploy_site_to_s3.sh
```

The script reads `SiteBucketName` and `CloudFrontDistributionId` from the CloudFormation stack outputs.

## Destroying

The S3 bucket has `RemovalPolicy.RETAIN` and `autoDeleteObjects=false`, so destroying the stack will not automatically delete site files. This is intentional to avoid accidental data loss.
