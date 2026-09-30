# F1 Analysis + Prediction Demo (Local)

This project builds a local end-to-end demo for:
- Session-wise analysis (FP1, FP2, FP3, Qualifying, Race)
- Basic ML predictions (Qualifying and Race positions)
- Static website output that can later be pushed to GitHub Pages

Reference event used now: **2025 season, Round 1**.

## 1) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Run full pipeline

```bash
python -m src.f1demo.pipeline --season 2025 --round 1 --train-round-end 2
```

Outputs:
- `outputs/` -> CSV/JSON summaries, plots, model artifacts
- `site/` -> static website files (`index.html` is the Race Hub)

## 2a) Automatic current-race update

Phase 1 automation is available through:

```bash
python -m src.f1demo.auto_update --season 2026 --quick
```

What it does:
- Detects the current/next race from the FastF1 calendar.
- Detects which sessions have usable lap data.
- Skips cleanly when no new session data is available.
- Regenerates the race pages only when new data is detected or `--force` is used.
- Tracks processed sessions in `site/races/state.json`.
- Keeps `site/races/manifest.json` aligned with the current/completed race state.

Useful options:

```bash
python -m src.f1demo.auto_update --season 2026 --dry-run
python -m src.f1demo.auto_update --season 2026 --round 7 --force --quick
python -m src.f1demo.auto_update --season 2026 --no-complete-after-race
```

## 2b) Dockerized pipeline

Phase 2 adds a container entrypoint so the same update logic can run locally, in CI, or later in ECS Fargate.

Build:

```bash
docker build -t paddock-iq-pipeline .
```

Automatic current-race update:

```bash
docker run --rm \
  -e SEASON=2026 \
  -e QUICK=true \
  paddock-iq-pipeline
```

Dry run without regenerating files:

```bash
docker run --rm \
  -e SEASON=2026 \
  -e DRY_RUN=true \
  paddock-iq-pipeline
```

Manual fixed-round pipeline run:

```bash
docker run --rm \
  -e PADDOCK_MODE=pipeline \
  -e SEASON=2026 \
  -e ROUND=7 \
  -e QUICK=true \
  paddock-iq-pipeline
```

Serve the committed static site from the container:

```bash
docker run --rm -p 8000:8000 \
  -e PADDOCK_MODE=serve \
  paddock-iq-pipeline
```

Useful container environment variables:
- `PADDOCK_MODE`: `auto` (default), `pipeline`, or `serve`
- `SEASON`: season year, defaults to current UTC year
- `ROUND`: optional in `auto`, required in `pipeline`
- `TRAIN_ROUND_END`: training backfill round, defaults to `2`
- `QUICK`: `true` by default for faster cloud/runtime updates
- `FORCE_UPDATE`: force regeneration in `auto` mode
- `DRY_RUN`: detect available sessions without writing site/state files
- `MIN_LAPS`, `MIN_DRIVERS`, `LOOKAHEAD_DAYS`: session availability thresholds
- `NO_COMPLETE_AFTER_RACE`: keep a race marked current even when race data exists
- `PADDOCK_GA4_MEASUREMENT_ID`: optional GA4 analytics ID

## 3) View local website

```bash
cd site
python -m http.server 8000
```

Open: `http://localhost:8000`

Navigation:
- `Home / Race Hub`: `index.html`
- Race pages: `races/<season>_round_<nn>/index.html` (Overview), plus `round.html` and `strategy.html`

## 4) Notes
- FastF1 cache is written to `data/cache`.
- First run can take longer due to data downloads.
- This demo is intentionally free-resource only.
- Manual GitHub Action is included at `.github/workflows/update-site.yml`.

## 5) Private viewer analytics (optional)

You can track:
- Page views
- Time spent on each page/tab (Overview, Weekend Analysis, Practice/Q1/Q2/Q3/Race sub-tabs, Strategy Lab)

This uses **Google Analytics 4 (GA4)**, which is free. Metrics are visible in your GA account dashboard only.

Run with GA4 enabled:

```bash
python -m src.f1demo.pipeline --season 2025 --round 1 --quick --ga4-measurement-id G-XXXXXXXXXX
```

Or set once as an env var:

```bash
export PADDOCK_GA4_MEASUREMENT_ID=G-XXXXXXXXXX
python -m src.f1demo.pipeline --season 2025 --round 1 --quick
```

## 6) AWS Static Hosting (Phase 3)

Phase 3 hosts the committed static `site/` output on AWS using CDK TypeScript:

- private S3 bucket
- CloudFront CDN
- CloudFront Origin Access Control
- GitHub Actions OIDC deploy role

Install and synthesize:

```bash
cd infra/aws-cdk
npm install
AWS_PROFILE=paddockiq npx cdk synth
```

Deploy infrastructure:

```bash
AWS_PROFILE=paddockiq npx cdk deploy \
  -c projectName=paddock-iq \
  -c environmentName=prod \
  -c githubRepository=apatnaik0/paddock-iq \
  -c githubBranch=main
```

Deploy the current local `site/` folder manually:

```bash
AWS_PROFILE=paddockiq ./scripts/deploy_site_to_s3.sh
```

After CDK deploys, add the `GitHubDeployRoleArn` stack output as a GitHub repository secret named:

```text
AWS_SITE_DEPLOY_ROLE_ARN
```
