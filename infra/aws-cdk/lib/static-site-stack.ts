import * as cdk from 'aws-cdk-lib';
import { CfnOutput, RemovalPolicy, Stack, StackProps } from 'aws-cdk-lib';
import * as cloudfront from 'aws-cdk-lib/aws-cloudfront';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as s3 from 'aws-cdk-lib/aws-s3';
import { Construct } from 'constructs';

export interface StaticSiteStackProps extends StackProps {
  projectName: string;
  environmentName: string;
  githubRepository: string;
  githubBranch: string;
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
        enabled: true,
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
    new CfnOutput(this, 'GitHubDeployRoleArn', {
      value: deployRole.roleArn,
      description: 'GitHub Actions role ARN for static site deployments',
    });
  }
}
