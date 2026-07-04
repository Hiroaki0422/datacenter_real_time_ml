# Decision Log

> Locked decisions, in-flight questions, and the reasoning behind them. Updated as we go.

## Locked Decisions

### D1: Spike classifier is multi-class, not binary (2026-07-04)
- **Decision**: 4 classes (normal, moderate, high, extreme) based on LMP ratio vs 4h-rolling baseline
- **Thresholds**: 1.5x, 3.0x, 6.0x (subject to Phase 1 EDA adjustment)
- **Why**: richer downstream signal, better visualization, more naturally calibrated
- **Doc**: [SPIKE_CLASSIFIER.md](SPIKE_CLASSIFIER.md)

### D2: Marginal carbon forecaster uses XGBoost regression (2026-07-04)
- **Decision**: XGBoost regression with quantile loss
- **Why**: tabular data + concept-drift resilience + retrain speed
- **Loss**: Quantile (q=0.5, q=0.9) for uncertainty quantification

### D3: Single ISO for v1 — CAISO (2026-07-04)
- **Decision**: CAISO only for v1
- **Why**: best data access, isolated grid, 227 of v1's DCs, civic angle (drought, wildfire, rolling blackouts)
- **Deferred**: PJM, ERCOT, etc. to v2

### D4: Drop RIPE RIS BGP from feature set (2026-07-04)
- **Decision**: no network signal proxy
- **Why**: BGP is control-plane, not traffic. Doesn't proxy compute activity.
- **Replacement**: status feeds (AWS/GCP/Azure) + LMP DA-vs-RT spread as cheap proxies

### D5: 1-4 hour forecast horizon, 5-min granularity (2026-07-04)
- **Decision**: not sub-hour, not day-ahead
- **Why**: matches typical LMP ramp timescale; long enough for advisory action, short enough for stream features to dominate

### D6: Weekly retrain cadence (2026-07-04)
- **Decision**: not daily, not online
- **Why**: concept drift in CAISO is slow (solar capacity grows ~10%/year)
- **Upgrade path**: River / online learning in v1.1 if needed

### D7: VPS-friendly stack, no Kafka/Flink/Spark (2026-07-04)
- **Decision**: SQLite (offline) + Redis (online) + FastAPI (serving) + cron (scheduling)
- **Why**: proven, debuggable, fits 4 vCPU / 8GB VPS

## In-Flight Questions

### Q1: Exact spike thresholds
- **Current**: 1.5x, 3.0x, 6.0x
- **Resolve**: Phase 1 (EDA on CAISO LMP)
- **Action**: compute observed class frequencies, adjust to get target distribution

### Q2: 4h baseline window vs 6h or 8h
- **Current**: 4h
- **Resolve**: Phase 1 (sensitivity analysis)
- **Action**: try 2h, 4h, 6h, 8h; pick most stable class distribution

### Q3: Per-zone vs global thresholds
- **Current**: global (same for all CAISO zones)
- **Resolve**: Phase 1 (EDA)
- **Action**: compare volatility across zones; per-zone if heterogeneous

### Q4: Marginal emissions target definition
- **Current**: predicting `gCO2/kWh` from CAISO generation mix
- **Alternative**: predicting "next marginal generator" (more research-grade)
- **Resolve**: Phase 1 (verify CAISO publishes MER directly)
- **Action**: if yes, use it as label; if no, derive from gen mix + gas price spread

### Q5: Carbon + LMP joint model vs separate
- **Current**: separate models
- **Resolve**: Phase 3 (if correlation is high, consider multi-output)
- **Action**: measure label correlation; decide

## D8: Spike thresholds locked at 1.5x/3.0x/6.0x with 4h baseline (2026-07-04, end of Phase 1)
- **Decision**: confirmed by 7-day EDA on 3 CAISO trading hubs
- **Observed frequencies**: Normal 81-84%, Moderate 12%, High 3-5%, Extreme 1.5-2.4%
- **Window sensitivity**: 4h best (2h undercounts, 8h overcounts whole-day shifts)
- **Per-zone heterogeneity**: SP15 is most volatile (2.44% extreme), NP15 calmest (1.48%)

## D9: Use `caiso.get_lmp(date=...)` for backfill (2026-07-04, Phase 1)
- **Decision**: OASIS `get_lmp` only works one day at a time with `date=` arg
- **Range mode is broken** in current OASIS API (returns "No data")
- **Today's LMP not available** (only yesterday and earlier)
- **Live inference**: needs different endpoint, TBD Phase 4

## D10: Carbon intensity is system-wide, not per-zone (2026-07-04, Phase 1)
- **Decision**: GHG field is identical across 3 CAISO trading hubs at any timestamp
- **Implication**: treat carbon as system-wide feature; per-DC variation comes from local carbon-from-grid-mix assumptions, not from real-time ISO data
- **Backup**: derive carbon intensity from fuel mix when GHG=0 (zero-carbon renewables at margin)

## D11: Status feeds (AWS/GCP/Azure) for compute proxy (2026-07-04, Phase 1)
- **Decision**: use as binary feature ("is this provider's region degraded?")
- **Observation**: 0 events for us-west-1/2 in current snapshot; sparse signal in normal times
- **Value**: becomes informative only during incidents; OK for "abnormal operating condition" flag

## Rejected Ideas (with reason)

| Idea | Reason |
|---|---|
| LSTM / Transformer for time series | 5-min LMP archive quality is too recent (~2y); not enough data; XGBoost wins on tabular |
| River / online learning for v1 | Added complexity; weekly retrain is sufficient for slow CAISO drift |
| Day-ahead forecast | Different model class; not "stream feature" project |
| Real-time DC load attribution | Substation-level deconvolution is hard; v1 honesty is "zone-level forecast, not per-DC attribution" |
| Authenticated public API | Adds operational overhead; v1 is open, rate-limited |
| Build own map of substations | OSM Overpass + EIA-930 sufficient |
| Drop Open-Meteo | Already in v1, zero ramp cost |
| **Multi-class spike classifier** | **Replaced with regression on lmp_target_4h (continuous LMP ratio) — no threshold debate, no class-imbalance pain, more useful downstream** |
| **Binary GHG classifier** | **Replaced with regression on ghg_target_4h (continuous) — binary was a workaround for data sparsity, regression is the real ask** |

## Revised ML Problems (2026-07-04, mid-Phase 3)

**Original plan**: 4-class classifier for spike (Normal/Moderate/High/Extreme) + binary GHG + multi-output
**Revised plan**: Two regression models
- **Model A — LMP ratio regressor**: predict `lmp_target_4h` = mean LMP / 4h-rolling mean in next 4h (continuous)
  - "When will the grid be expensive relative to baseline?"
  - Used directly for "shift workload" advisory at any threshold
- **Model B — Carbon regressor**: predict `ghg_target_4h` = mean GHG in next 4h (continuous, short tons/MWh)
  - "When will the grid be carbon-heavy?"
  - Used for "carbon-aware scheduling" advisory

**Why we switched**:
1. No class imbalance to fight (no need for class weights)
2. No threshold debate (operator picks their own cutoff)
3. More useful for downstream consumers (continuous value, not categorical label)
4. The "advisory" rule becomes a simple threshold on the prediction, applied downstream
5. Multi-class log loss baseline (11.18) was uninformative — regression gives clearer metrics (MAE, RMSE, R²)

**Constraint**: Model B (carbon) only has positive labels in May–Jul 2026 due to CAISO API publication scope. We'll train on what's available and document the limitation.

## Outlier Handling (2026-07-04, mid-Phase 3)

**Problem**: `lmp_pct_change_60m` had values up to 278,390 (from LMP near 0 in prior period). Broke XGBoost training with "Input data contains inf or a value too large."

**Solution**: Winsorize at 1st/99.9th percentile, applied per-column in the feature engineering pipeline. Applied to:
- All numeric features (cap at 0.1% / 99.9% percentiles)
- Special: `pct_change` columns hard-clipped to ±100 (1,000,000% change is meaningless even as a signal)

**Why not filter**: A 278,390% LMP jump IS a real market event (oversupply transition). Winsorizing preserves the row but caps the magnitude so it doesn't dominate the model.

**Why not log-transform**: Tried implicitly via `lmp_ratio_4h` as target — but features themselves are still in raw scale. Could add log1p on highly skewed columns if model performance suffers.
