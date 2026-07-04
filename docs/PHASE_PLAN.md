# Phase Plan — 6 Weeks

> Each phase has a single, shippable outcome. Don't move on until the deliverable works.

## Phase 0 — Scaffolding ✅ DONE (2026-07-04)
**Goal**: Project structure, docs, data exploration plan

**Deliverables**:
- [x] `dc_real_time/` folder structure
- [x] Project summary, spike classifier design, phase plan
- [x] Data sources doc with size/access notes
- [x] Decisions log
- [x] `.gitignore` for raw data + model artifacts
- [x] `requirements.txt`

**Exit criteria met**: All docs written; can hand off project cold to another agent / future-self

---

## Phase 1 — Data Exploration ✅ DONE (2026-07-04, 1y backfill received)
**Goal**: Understand what's actually available, sizes, quirks, before designing features

### Status
- [x] CAISO 5-min LMP pulled (1y, 3 trading hubs, 315,141 rows, 12 MB)
- [x] CAISO fuel mix pulled (1y, 14 fuel types, 105,108 rows, 4.6 MB)
- [x] Open-Meteo pulled (1y partial, 50/227 sites, 441,600 hourly rows, 5.4 MB)
- [x] Status feeds verified (AWS, GCP, Azure JSON/RSS all accessible)
- [x] Spike class frequencies computed on 1y — thresholds CONFIRMED at 1.5x/3.0x/6.0x with 4h baseline
- [x] GHG semantics clarified (system-wide, short tons/MWh of marginal generator)
- [x] Colab backfill executed; output saved + committed to repo

### Deliverables (this phase)
- `data/processed/caiso_lmp_1y.parquet` — 315,141 rows × 11 cols, 12 MB
- `data/processed/caiso_fuel_mix_1y.parquet` — 105,108 rows × 16 cols, 4.6 MB
- `data/processed/openmeteo_ca_dc_1y.parquet` — 441,600 rows × 10 cols, 5.4 MB (50 of 227 sites)
- `data/processed/caiso_lmp_7d_sample.parquet` — initial sample kept for fast iteration
- `data/processed/caiso_fuel_mix_7d_sample.parquet` — initial sample kept
- `data/processed/openmeteo_santaclara_7d_sample.parquet` — initial sample kept
- `data/external/ca_dc_sites.csv` — 227 CA DCs (committed, small)
- `notebooks/01_explore_caiso_lmp.ipynb` — EDA notebook (6 cells)
- `notebooks/01_explore_caiso_lmp.py` — JupyText-style companion
- `notebooks/colab_handoff/01_caiso_1y_backfill.md` — Colab instructions (executed)
- `notebooks/colab_handoff/01_colab_run_log.ipynb` — actual Colab run with outputs (saved for reference)
- `docs/PHASE1_FINDINGS.md` — empirical results, decisions, caveats

### Exit criteria
- [x] Can answer "what's a typical LMP distribution for SP15?" with 1y of data
- [x] Spike class frequencies known, thresholds confirmed on full year
- [x] Size budget clear: 1y fits in 17 MB; 5y would be ~85 MB
- [ ] Per-site weather pulled for all 227 CA DC sites (50 done, 177 pending rate-limit reset)

### Key Findings on 1y Data (see PHASE1_FINDINGS.md for full report)
- NP15 mean LMP: $31.94, max $1,150 (vs $14.51 mean, $127 max in 7d)
- SP15 mean LMP: $26.30, max $1,148 (vs $9.21, $40 in 7d)
- ZP26 mean LMP: $26.91, max $1,130 (vs $11.06, $69 in 7d)
- 7d sample under-represented winter peaks; 1y shows real scarcity events
- Class 3 (extreme) frequency settled: 0.88% NP15, 1.65% SP15, 1.55% ZP26
- Solar mean: 6,651 MW (vs 9,386 in 7d summer sample) — seasonal dip expected
- Natural gas mean: 5,294 MW (vs 1,133 in 7d) — winter heating kicks in

### Key Findings (see PHASE1_FINDINGS.md for details)
- CAISO 5-min LMP works via `get_lmp(date=...)`; range mode is broken
- "Today's LMP" not available; only ~24h delayed — affects live inference design
- GHG (carbon) is system-wide, not per-zone
- Open-Meteo works; size budget is essentially free
- AWS/GCP/Azure status feeds work, mostly empty for us-west-* regions

---

## Phase 2 — Feature Pipeline (Week 2)
**Goal**: Offline feature store from Phase 1 data

### Tasks
1. Build per-zone feature engineering:
   - LMP rolling stats (60min, 4h, 24h mean/std/slope)
   - Fuel mix features (gas %, solar %, imports %)
   - DA-vs-RT spread
   - Weather features (rolling means for stability)
2. Build per-DC overlay (zone lookup, WUE, climate_adj, BWS)
3. Build target variable (multi-class spike label per zone per timestamp)
4. Train/val/test split: 2019-2023 train, 2024 val, 2025- test
5. Class weights computed and saved

### Deliverables
- `src/features/build_features.py` — feature engineering module
- `data/processed/features_offline.parquet` — full feature set
- `data/processed/targets.parquet` — labels
- `artifacts/feature_schema.json` — column definitions

### Exit criteria
- Feature pipeline runs end-to-end on a sample
- Schema is stable (no NaN explosion)
- Train/val/test sizes documented

---

## Phase 3 — Baseline Models (Week 3)
**Goal**: Beat naive baselines with XGBoost

### Tasks
1. **Model A**: Multi-class XGBoost spike classifier
   - Train on 2019-2023, validate on 2024
   - Compare to baselines: persistence, hour-of-day mean
   - Report: log loss, per-class PR-AUC, confusion matrix, reliability diagram
2. **Model B**: XGBoost regression for marginal carbon
   - Train on 2019-2023, validate on 2024
   - Compare to baselines: persistence, hour-of-day × month × fuel-mix-lookup
   - Report: MAPE, RMSE, residual plots
3. **Ablation study**: with vs without live features (Open-Meteo, fuel mix)
   - **Confirm stream features do real work** (PR-AUC delta ≥ 0.05)

### Deliverables
- `src/models/train_lmp.py` — Model A training script
- `src/models/train_carbon.py` — Model B training script
- `models/v0.1/lmp_spike.json` — Model A checkpoint
- `models/v0.1/carbon.json` — Model B checkpoint
- `artifacts/eval_report_v0.1.md` — metrics, plots
- `artifacts/ablation_table.csv` — with/without live features

### Exit criteria
- Both models beat naive baselines on validation
- Stream features ablation shows PR-AUC delta ≥ 0.05 (else redesign)
- Models serializable, re-loadable, deterministic

---

## Phase 4 — Live Inference + API (Week 4)
**Goal**: Serve real-time forecasts via FastAPI

### Tasks
1. Build ingestion cron jobs:
   - `scripts/fetch_gridstatus.py` (every 5 min)
   - `scripts/fetch_openmeteo.py` (every 1 hour)
   - `scripts/fetch_status_feeds.py` (every 5 min)
2. Online feature store (Redis hash per zone)
3. FastAPI service:
   - `GET /forecast/{zone_id}` — spike class probs, E[carbon], E[lmp]
   - `GET /dc/{dc_id}/forecast` — per-DC overlay
   - `GET /advisory` — shift advisory
   - `GET /healthz`
4. p95 latency < 5s end-to-end

### Deliverables
- `src/api/app.py` — FastAPI service
- `scripts/cron_*.sh` — cron entries
- `artifacts/api_load_test.md` — latency report

### Exit criteria
- API serves real predictions, refreshed every 5 min
- All endpoints return valid JSON
- No memory leaks over 24h run

---

## Phase 5 — DC Overlay + Civic Dashboard (Week 5)
**Goal**: Public-facing visualization, integrates with v1 data

### Tasks
1. Per-DC advisory endpoint (combines Model A + B + WUE)
2. Static dashboard (HTML + JS, no server):
   - Map of 227 CA DCs
   - Per-DC current state (LMP, carbon, water, advisory)
   - 24h forecast heatmap
3. "Shift advisory" rule engine documented
4. Public SDK stub (Python)

### Deliverables
- `src/api/dc_overlay.py` — per-DC logic
- `web/index.html` — civic dashboard (deployable to S3/Netlify/static)
- `sdk/dc_forecast.py` — Python client
- `artifacts/dashboard_screenshots/` — visual proof

### Exit criteria
- Dashboard refreshes automatically
- Per-DC forecast is sensible (sanity check on 5-10 sites)
- "Shift advisory" rules are explainable

---

## Phase 6 — Public Launch (Week 6)
**Goal**: Public release, evaluation, blog post

### Tasks
1. Backtest report (full 2024-2025)
2. README polish, architecture diagram
3. Public API rate limits + ToS
4. Blog post: "Building a public carbon-aware compute scheduler"
5. (Optional) Submit to HN, ML newsletters

### Deliverables
- `docs/BACKTEST_REPORT.md`
- `README.md` final
- `docs/BLOG_POST.md` (draft)
- Public API live, dashboard live

### Exit criteria
- Project is self-explanatory cold-read
- All 6 phase deliverables verified
- v2 of v1's DC water story is complete (real-time carbon + water + grid)

---

## Risks & Pivots

| Risk | Pivot |
|---|---|
| CAISO API breaks | Switch to EIA API as backup; degrade features |
| Open-Meteo rate limits | Cache aggressively; use historic archive for training only |
| Stream features don't beat static | Switch to "weather forecast fusion only" model; document why |
| VPS too small for training | Colab Pro handoff; VPS only for inference |
| Public interest is zero | Pivot framing to "DC grid stress dashboard for journalists" |
| Concept drift too fast | Online learning (river library) in v1.1 |

## Decision Points

| Phase | Decision |
|---|---|
| End of Phase 1 | Lock spike thresholds based on observed frequency |
| End of Phase 3 | Lock model architecture; decide if ablation is enough |
| End of Phase 4 | Lock API design; commit to no auth, rate-limited public |
| End of Phase 5 | Lock dashboard scope; what's MVP vs nice-to-have |
| End of Phase 6 | v1 done. Start v2 (drought overlay + other ISOs)? |
