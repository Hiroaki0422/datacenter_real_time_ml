# dc_real_time — Spatial-Temporal Carbon & Price Forecasting for Data Center Infrastructure

> **One-line thesis**: Predict 1–4 hour ahead wholesale electricity price surges and carbon intensity in CAISO, mapped onto California data center sites from the v1 dataset, to power a public "carbon-aware compute scheduling" advisory.

## Status
- **Phase**: 1 (data exploration) ✅ DONE — 1y backfill received from Colab
- **Started**: 2026-07-04
- **Owner**: Hiroaki
- **License**: TBD (probably MIT to match v1)

## Quick Links
- [Project Summary](docs/PROJECT_SUMMARY.md) — full spec, problem, architecture
- [Spike Classifier Design](docs/SPIKE_CLASSIFIER.md) — multi-class label scheme
- [Phase Plan](docs/PHASE_PLAN.md) — 6-week milestone breakdown
- [Data Sources](docs/DATA_SOURCES.md) — what we pull, size, auth, access notes
- [Decision Log](docs/DECISIONS.md) — locked decisions, in-flight questions
- [Phase 1 Findings](docs/PHASE1_FINDINGS.md) — empirical EDA results (spike class frequencies, GHG semantics, etc.)

## Project Structure

```
dc_real_time/
├── README.md                 # this file
├── docs/                     # design docs, decisions, phase plan
│   ├── PROJECT_SUMMARY.md
│   ├── SPIKE_CLASSIFIER.md
│   ├── PHASE_PLAN.md
│   ├── DATA_SOURCES.md
│   └── DECISIONS.md
├── data/                     # downloaded raw + processed data
│   ├── raw/                  # immutable downloads (gitignored)
│   ├── processed/            # cleaned/joined parquet
│   └── external/             # v1 dataset cross-link (read-only)
├── notebooks/                # exploratory notebooks + colab handoff
│   ├── 01_explore_caiso_lmp.ipynb
│   ├── 02_explore_openmeteo.ipynb
│   ├── 03_explore_status_feeds.ipynb
│   └── colab_handoff/        # noteboooks meant for Colab Pro
├── src/                      # importable library code
│   ├── data/                 # ingestion modules
│   ├── features/             # feature engineering
│   ├── models/               # training + inference
│   └── api/                  # FastAPI service
├── scripts/                  # CLI / cron entry points
├── models/                   # serialized model artifacts (gitignored)
├── artifacts/                # plots, eval reports, logs
└── requirements.txt
```

## Relationship To v1 Project

This is **v2 of the DC water stress work**, not a standalone project:

| v1 (existing) | v2 (this project) |
|---|---|
| Static WUE × climate × basin stress | + Real-time LMP, carbon, water draw per DC |
| "This DC uses 1.18 L/kWh, basin is High stress" | "This DC is currently consuming 50 MW on CAISO zone SP15, carbon is 380 gCO2/kWh, basin is at 60% drought severity" |
| Researcher tool | Public transparency tool + scheduler |

The v1 dataset (`/root/project/datacenter_water_stress/data/processed/us_dc_with_stress.csv`) is the static anchor for this project.
