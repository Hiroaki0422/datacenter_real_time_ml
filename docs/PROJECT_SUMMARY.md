# Project Summary — dc_real_time

> Predict 1–4 hour ahead wholesale electricity price surges and carbon intensity in CAISO, mapped onto California data center sites from the v1 dataset, to power a public "carbon-aware compute scheduling" advisory.

## 1. Problem Statement

**What's missing in the world today:**
- Cloud providers run *internal* carbon-aware schedulers (Google, Microsoft, AWS) but those signals are closed-source
- Existing public tools (WattTime, Electricity Maps) are *paid* and coarse — regional, not substation-level
- "Carbon-aware compute" discourse lacks a **publicly auditable model** with full feature lineage

**What this project delivers:**
- 1–4h-ahead forecast of LMP and marginal carbon intensity for CAISO zones
- Mapped to **227 California data centers** from the v1 dataset
- Public API + civic dashboard + "shift your workload" advisory
- Closes the loop with v1 by adding real-time carbon + price on top of static water

**Use cases served (in priority order):**
1. Carbon-aware workload scheduler — pause training jobs when CAISO is dirty
2. Civic transparency dashboard — "Right now, this DC is consuming N MW from a grid burning N% gas"
3. Compute / HPC users — choose regions/times by predicted carbon
4. Research — public model benchmark for marginal emissions prediction

## 2. Scope

### In Scope (v1)
- CAISO only (best data access, isolated grid, 227 of v1's DCs)
- 1–4 hour forecast horizon, 5-min granularity
- **Multi-class LMP spike classifier** (see [SPIKE_CLASSIFIER.md](SPIKE_CLASSIFIER.md))
- **Continuous marginal carbon forecaster** via XGBoost regression
- Overlay onto 227 CA data center sites
- "Shift advisory" output (rule-based, not autonomous)
- Public API + simple dashboard
- Historic 5y backtest

### Out of Scope (v1)
- Other ISOs (PJM, ERCOT deferred to v2)
- Real-time DC load attribution (deconvolution, hard)
- Autonomous job control (we advise, don't act)
- Sub-hour forecasting
- Crypto-mining load separation
- 24h+ day-ahead forecasts (different model class)

### Deferred to v2
- Other ISOs
- Drought overlay (USGS streamflow → basin stress)
- Per-operator breakdown
- Network signal proxies (replacing dropped RIPE RIS)

## 3. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  STATIC LAYER (v1 + supplements)                                │
│  • 227 CA DC lat/lon, operator, WUE, climate_adj                 │
│  • CAISO substation locations (EIA-930 + OSM Overpass)           │
│  • CAISO zone → DC mapping (geospatial join)                     │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│  REAL-TIME STREAM LAYER (5-min refresh)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐    │
│  │ gridstatus   │  │ Open-Meteo   │  │ AWS/GCP/Azure status │    │
│  │ • 5-min LMP  │  │ • local wx   │  │ • maintenance feeds  │    │
│  │ • fuel mix   │  │ • solar irrad│  │ • region degradation │    │
│  │ • GHG rate   │  │ • cloud cover│  │                      │    │
│  │ • reserve    │  │ • wet-bulb T │  │ (cheap compute proxy)│    │
│  └──────────────┘  └──────────────┘  └──────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│  FEATURE PIPELINE (per CAISO zone, per DC)                       │
│  • LMP rolling stats (60min, 4h, 24h mean, std, slope)           │
│  • Fuel mix deltas (gas %, solar %, imports %)                   │
│  • LMP day-ahead vs real-time spread                             │
│  • Local weather (temp, humidity, solar, wet-bulb)               │
│  • Calendar (hour, day-of-week, holiday, season)                 │
│  • Maintenance flags (per provider × region)                     │
│  • DC overlay (per-site: nearest substation, WUE, basin)         │
└──────────────────────────────────────────────────────────────────┘
                              ↓
              ┌───────────────────────────────┐
              │   FEATURE STORE (SQLite/Redis) │
              │   • offline: 5y historic        │
              │   • online: last 24h rolling    │
              └───────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│  MODEL LAYER (offline + online inference)                        │
│                                                                   │
│  Model A — LMP Multi-Class Spike Classifier                      │
│   • Target: 4-class label (0=normal, 1=moderate, 2=high, 3=extreme)│
│   • Baseline: LMP vs 4h-rolling mean                            │
│   • Algorithm: XGBoost multi-class (softprob)                    │
│   • Loss: multi-class log loss with class weights                │
│   • Retrain: weekly                                               │
│                                                                   │
│  Model B — Marginal Carbon Forecaster (XGBoost Regression)       │
│   • Target: E[gCO2/kWh] in next 1-4h                            │
│   • Algorithm: XGBoost regression                                │
│   • Loss: quantile (q=0.5, q=0.9)                                │
│   • Retrain: weekly                                               │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│  INFERENCE + OUTPUTS (FastAPI, 1-min refresh)                    │
│  • Per-zone: spike_class_probs, E[carbon], E[LMP], uncertainty   │
│  • Per-DC:  expected kWh cost, expected gCO2, expected water     │
│  • "Shift advisory" rule: IF P(high|extreme)>0.5 OR carbon>500  │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│  CONSUMERS                                                        │
│  • FastAPI public endpoint (JSON, free, rate-limited)             │
│  • Civic dashboard (static, hosted)                              │
│  • "Scheduler" CLI / SDK (Python)                                 │
└──────────────────────────────────────────────────────────────────┘
```

## 4. Data Sources

| Source | What | Latency | Cost | Auth |
|---|---|---|---|---|
| **gridstatus.io** | CAISO 5-min LMP, fuel mix, GHG | 5 min | Free tier | API key |
| **Open-Meteo** | Local weather at DC coords | 1 hour | Free | None |
| **CAISO OASIS** | Day-ahead LMP, 5-min RT LMP, gen mix | 5 min | Free | None (slow/HTML) |
| **AWS Health Dashboard** | us-west-1/2 service degradation | Real-time | Free | Public RSS/JSON |
| **GCP Status** | Region status | Real-time | Free | Public JSON |
| **Azure Status** | Region status | Real-time | Free | Public JSON |
| **EIA Open Data** | Historic ISO load, gen by fuel | 1 hour | Free | API key |
| **v1 dataset** | 227 CA DC lat/lon, WUE, BWS | Static | Already have | — |
| **OSM Overpass** | Substations, power infrastructure | Static | Free | None |

**Dropped from prior brainstorm**: RIPE RIS BGP (wrong signal — BGP is control-plane, not traffic; doesn't proxy compute activity).

## 5. Model Choices

### Model A: Multi-Class LMP Spike Classifier
- **Algorithm**: XGBoost multi-class classification (softprob)
- **Output**: 4-class probability vector (normal, moderate, high, extreme)
- **Loss**: Multi-class log loss with class weights (inverse frequency)
- **Why multi-class over binary**: richer downstream signal for visualization (heatmaps, stack probability bars), aligned with "magnitude" framing
- **Baseline**: persistence + hour-of-day mean

### Model B: XGBoost Regression for Marginal Carbon
- **Algorithm**: XGBoost regression
- **Output**: E[gCO2/kWh] next 1-4h
- **Loss**: Quantile regression (q=0.5, q=0.9) for uncertainty
- **Baseline**: persistence + hour-of-day × month × fuel-mix-lookup

See [SPIKE_CLASSIFIER.md](SPIKE_CLASSIFIER.md) for full label scheme.

## 6. The Theory Questions This Project Forces

| Question | Where it bites |
|---|---|
| **Offline/online feature skew** | Open-Meteo historic archive ≠ live API distribution |
| **Concept drift detection** | How do you know the model is stale? (Page-Hinkley, ADWIN, PSI) |
| **Class imbalance in spike detection** | Why focal loss / class weights on multi-class |
| **Marginal vs average emissions** | Why predicting "next marginal generator" is harder than "fleet average" |
| **Calibration vs discrimination** | A 0.7 P(extreme) should mean 70% of the time |
| **End-to-end system latency** | From grid signal → feature → prediction → advisory, what's the 95p latency? |

## 7. Acceptance Criteria (Phase 6)

- Model A: PR-AUC (one-vs-rest) > 0.6 on 2024 holdout
- Model B: MAPE < 30% on 2024 holdout
- Ablation: live features improve Model A PR-AUC by ≥ 0.05
- API: p95 latency < 5s end-to-end
- Dashboard: refreshes every 5 min, no manual intervention
- Public: GitHub repo, blog post, public dashboard live
