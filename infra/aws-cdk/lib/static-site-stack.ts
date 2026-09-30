import * as cdk from 'aws-cdk-lib';
import { CfnOutput, RemovalPolicy, Stack, StackProps } from 'aws-cdk-lib';
import * as cloudfront from 'aws-cdk-lib/aws-cloudfront';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as events from 'aws-cdk-lib/aws-events';
import * as targets from 'aws-cdk-lib/aws-events-targets';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as s3 from 'aws-cdk-lib/aws-s3';
import { Construct } from 'constructs';

export interface StaticSiteStackProps extends StackProps {
  projectName: string;
  environmentName: string;
  githubRepository: string;
  githubBranch: string;
  enableCloudFront: boolean;
  enablePipelineSchedule: boolean;
  pipelineScheduleExpression: string;
  pipelineImageTag: string;
  pipelineSeason: string;
}

export class StaticSiteStack extends Stack {
  constructor(scope: Construct, id: string, props: StaticSiteStackProps) {
    super(scope, id, props);

    const normalizedProject = props.projectName.toLowerCase().replace(/[^a-z0-9-]/g, '-');
    const normalizedEnv = props.environmentName.toLowerCase().replace(/[^a-z0-9-]/g, '-');
    const bucketName = `${normalizedProject}-site-${this.account}-${normalizedEnv}`;

    const siteBucket = new s3.Bucket(this, 'SiteBucket', {
      bucketName,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      enforceSSL: true,
      versioned: true,
      removalPolicy: RemovalPolicy.RETAIN,
      autoDeleteObjects: false,
    });

    const originAccessControl = new cloudfront.CfnOriginAccessControl(this, 'SiteOriginAccessControl', {
      originAccessControlConfig: {
        name: `${props.projectName}-${props.environmentName}-site-oac`,
        description: 'CloudFront access to the private PaddockIQ static site bucket',
        originAccessControlOriginType: 's3',
        signingBehavior: 'always',
        signingProtocol: 'sigv4',
      },
    });

    const distribution = new cloudfront.CfnDistribution(this, 'SiteDistribution', {
      distributionConfig: {
        enabled: props.enableCloudFront,
        comment: `${props.projectName} ${props.environmentName} static site`,
        defaultRootObject: 'index.html',
        httpVersion: 'http2and3',
        priceClass: 'PriceClass_100',
        origins: [
          {
            id: 'SiteBucketOrigin',
            domainName: siteBucket.bucketRegionalDomainName,
            originAccessControlId: originAccessControl.attrId,
            s3OriginConfig: {
              originAccessIdentity: '',
            },
          },
        ],
        defaultCacheBehavior: {
          targetOriginId: 'SiteBucketOrigin',
          viewerProtocolPolicy: 'redirect-to-https',
          allowedMethods: ['GET', 'HEAD', 'OPTIONS'],
          cachedMethods: ['GET', 'HEAD', 'OPTIONS'],
          compress: true,
          cachePolicyId: cloudfront.CachePolicy.CACHING_OPTIMIZED.cachePolicyId,
          originRequestPolicyId: cloudfront.OriginRequestPolicy.CORS_S3_ORIGIN.originRequestPolicyId,
        },
        customErrorResponses: [
          {
            errorCode: 403,
            responseCode: 404,
            responsePagePath: '/index.html',
          },
          {
            errorCode: 404,
            responseCode: 404,
            responsePagePath: '/index.html',
          },
        ],
      },
    });

    siteBucket.addToResourcePolicy(
      new iam.PolicyStatement({
        sid: 'AllowCloudFrontServicePrincipalReadOnly',
        effect: iam.Effect.ALLOW,
        principals: [new iam.ServicePrincipal('cloudfront.amazonaws.com')],
        actions: ['s3:GetObject'],
        resources: [siteBucket.arnForObjects('*')],
        conditions: {
          StringEquals: {
            'AWS:SourceArn': `arn:aws:cloudfront::${this.account}:distribution/${distribution.ref}`,
          },
        },
      }),
    );

    const oidcProvider = new iam.CfnOIDCProvider(this, 'GitHubOidcProvider', {
      url: 'https://token.actions.githubusercontent.com',
      clientIdList: ['sts.amazonaws.com'],
      // GitHub Actions OIDC root CA thumbprint. AWS may ignore this for trusted providers,
      // but CloudFormation still accepts it and avoids a Lambda-backed CDK custom resource.
      thumbprintList: ['6938fd4d98bab03faadb97b34396831e3780aea1'],
    });

    const deployRole = new iam.Role(this, 'GitHubDeployRole', {
      roleName: `${props.projectName}-${props.environmentName}-github-site-deploy`,
      assumedBy: new iam.FederatedPrincipal(
        oidcProvider.attrArn,
        {
          StringEquals: {
            'token.actions.githubusercontent.com:aud': 'sts.amazonaws.com',
          },
          StringLike: {
            'token.actions.githubusercontent.com:sub': `repo:${props.githubRepository}:ref:refs/heads/${props.githubBranch}`,
          },
        },
        'sts:AssumeRoleWithWebIdentity',
      ),
      description: 'Deploys committed PaddockIQ static site files from GitHub Actions to S3/CloudFront',
    });

    deployRole.addToPolicy(
      new iam.PolicyStatement({
        actions: ['s3:ListBucket'],
        resources: [siteBucket.bucketArn],
      }),
    );
    deployRole.addToPolicy(
      new iam.PolicyStatement({
        actions: ['s3:GetObject', 's3:PutObject', 's3:DeleteObject'],
        resources: [siteBucket.arnForObjects('*')],
      }),
    );
    deployRole.addToPolicy(
      new iam.PolicyStatement({
        actions: ['cloudfront:CreateInvalidation'],
        resources: [`arn:aws:cloudfront::${this.account}:distribution/${distribution.ref}`],
      }),
    );

    const pipelineRepository = new ecr.Repository(this, 'PipelineImageRepository', {
      repositoryName: `${normalizedProject}-pipeline-${normalizedEnv}`,
      imageScanOnPush: true,
      removalPolicy: RemovalPolicy.RETAIN,
      lifecycleRules: [
        {
          description: 'Keep the latest pipeline images only',
          maxImageCount: 5,
        },
      ],
    });

    const pipelineVpc = new ec2.Vpc(this, 'PipelineVpc', {
      maxAzs: 2,
      natGateways: 0,
      subnetConfiguration: [
        {
          name: 'public',
          subnetType: ec2.SubnetType.PUBLIC,
        },
      ],
    });

    const pipelineCluster = new ecs.Cluster(this, 'PipelineCluster', {
      clusterName: `${props.projectName}-${props.environmentName}-pipeline`,
      vpc: pipelineVpc,
    });

    const pipelineLogGroup = new logs.LogGroup(this, 'PipelineLogGroup', {
      logGroupName: `/aws/ecs/${props.projectName}/${props.environmentName}/pipeline`,
      retention: logs.RetentionDays.ONE_MONTH,
      removalPolicy: RemovalPolicy.DESTROY,
    });

    const pipelineTask = new ecs.FargateTaskDefinition(this, 'PipelineTaskDefinition', {
      family: `${props.projectName}-${props.environmentName}-pipeline`,
      cpu: 512,
      memoryLimitMiB: 2048,
      ephemeralStorageGiB: 40,
    });

    pipelineTask.addToTaskRolePolicy(
      new iam.PolicyStatement({
        actions: ['s3:ListBucket'],
        resources: [siteBucket.bucketArn],
      }),
    );
    pipelineTask.addToTaskRolePolicy(
      new iam.PolicyStatement({
        actions: ['s3:GetObject', 's3:PutObject', 's3:DeleteObject'],
        resources: [siteBucket.arnForObjects('*')],
      }),
    );
    pipelineTask.addToTaskRolePolicy(
      new iam.PolicyStatement({
        actions: ['cloudfront:CreateInvalidation'],
        resources: [`arn:aws:cloudfront::${this.account}:distribution/${distribution.ref}`],
      }),
    );

    pipelineTask.addContainer('PipelineContainer', {
      containerName: 'pipeline',
      image: ecs.ContainerImage.fromEcrRepository(pipelineRepository, props.pipelineImageTag),
      logging: ecs.LogDrivers.awsLogs({
        streamPrefix: 'pipeline',
        logGroup: pipelineLogGroup,
      }),
      environment: {
        PADDOCK_MODE: 'auto',
        SEASON: props.pipelineSeason,
        QUICK: 'true',
        SITE_BUCKET_NAME: siteBucket.bucketName,
        CLOUDFRONT_DISTRIBUTION_ID: distribution.ref,
        AWS_REGION: Stack.of(this).region,
      },
    });

    const pipelineSchedule = new events.Rule(this, 'PipelineSchedule', {
      ruleName: `${props.projectName}-${props.environmentName}-pipeline-schedule`,
      description: 'Runs the PaddockIQ race-weekend update container. Disabled unless explicitly enabled in CDK context.',
      schedule: events.Schedule.expression(props.pipelineScheduleExpression),
      enabled: props.enablePipelineSchedule,
    });

    pipelineSchedule.addTarget(
      new targets.EcsTask({
        cluster: pipelineCluster,
        taskDefinition: pipelineTask,
        taskCount: 1,
        assignPublicIp: true,
        subnetSelection: {
          subnetType: ec2.SubnetType.PUBLIC,
        },
      }),
    );

    new CfnOutput(this, 'SiteBucketName', {
      value: siteBucket.bucketName,
      description: 'Private S3 bucket that stores the generated static site',
    });
    new CfnOutput(this, 'CloudFrontDistributionId', {
      value: distribution.ref,
      description: 'CloudFront distribution ID for cache invalidations',
    });
    new CfnOutput(this, 'CloudFrontDomainName', {
      value: distribution.attrDomainName,
      description: 'Public CloudFront domain name for the dashboard',
    });
    new CfnOutput(this, 'CloudFrontEnabled', {
      value: props.enableCloudFront ? 'true' : 'false',
      description: 'Whether the CloudFront distribution is enabled by the CDK config',
    });
    new CfnOutput(this, 'GitHubDeployRoleArn', {
      value: deployRole.roleArn,
      description: 'GitHub Actions role ARN for static site deployments',
    });
    new CfnOutput(this, 'PipelineImageRepositoryUri', {
      value: pipelineRepository.repositoryUri,
      description: 'ECR repository URI for the scheduled pipeline container image',
    });
    new CfnOutput(this, 'PipelineClusterName', {
      value: pipelineCluster.clusterName,
      description: 'ECS cluster used by the race-weekend update pipeline',
    });
    new CfnOutput(this, 'PipelineScheduleEnabled', {
      value: props.enablePipelineSchedule ? 'true' : 'false',
      description: 'Whether the EventBridge pipeline schedule is enabled by CDK config',
    });
  }
}
