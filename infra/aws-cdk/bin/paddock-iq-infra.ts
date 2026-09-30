#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib';
import { StaticSiteStack } from '../lib/static-site-stack';

const app = new cdk.App();

const projectName = app.node.tryGetContext('projectName') ?? 'paddock-iq';
const environmentName = app.node.tryGetContext('environmentName') ?? 'prod';
const githubRepository = app.node.tryGetContext('githubRepository') ?? 'apatnaik0/paddock-iq';
const githubBranch = app.node.tryGetContext('githubBranch') ?? 'main';
const enableCloudFrontContext = app.node.tryGetContext('enableCloudFront') ?? 'false';
const enableCloudFront = String(enableCloudFrontContext).toLowerCase() === 'true';
const enablePipelineScheduleContext = app.node.tryGetContext('enablePipelineSchedule') ?? 'false';
const enablePipelineSchedule = String(enablePipelineScheduleContext).toLowerCase() === 'true';
const pipelineScheduleExpression = app.node.tryGetContext('pipelineScheduleExpression') ?? 'cron(0 6,12,18,23 ? * FRI,SAT,SUN *)';
const pipelineImageTag = app.node.tryGetContext('pipelineImageTag') ?? 'latest';
const pipelineSeason = String(app.node.tryGetContext('pipelineSeason') ?? new Date().getUTCFullYear());

new StaticSiteStack(app, 'PaddockIqStaticSiteStack', {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION ?? 'us-east-1',
  },
  projectName,
  environmentName,
  githubRepository,
  githubBranch,
  enableCloudFront,
  enablePipelineSchedule,
  pipelineScheduleExpression,
  pipelineImageTag,
  pipelineSeason,
});
