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
