# Resume Prompt — Next Session for dc_real_time

> **Copy this entire file into the next chat session** to restore full context. The agent will know exactly where we left off and what to do next.

---

## Project: `dc_real_time` (real-time ML for California data centers)

**Repo**: `git@github.com:Hiroaki0422/datacenter_real_time_ml.git` (branch: master)
**VPS path**: `/root/project/dc_real_time/`
**Active project**: Phase 4 of a 6-phase plan. Civic-advocacy tool. MIT license (planned).

**One-line thesis**: Predict 1-4h ahead wholesale electricity price surges and carbon intensity in CAISO, mapped onto California data center sites from the v1 dataset, to power a public "carbon-aware compute scheduling" advisory.

**Status as of end of last session (2026-07-05 01:18 UTC)**: Phases 0-3 done, Phase 4 mostly done (D1-D8), Phase 5/6 not started.

---

## What's Working (verified end-to-end)

```
Phase 0: Scaffolding ✓
Phase 1: Data exploration (1y LMP + fuel mix, partial weather) ✓
Phase 2: Feature engineering (45 features, train/val/test split) ✓
Phase 3: Model training (multi-horizon, 5-min wins for both) ✓
  - LMP ratio model: val R²=0.706, test R²=0.590
  - Carbon model: val R²=0.812, test R²=0.765
Phase 4 (in progress):
  D1  Dockerfile ✓
  D2  docker-compose (api + redis) ✓
  D3  trainer service in compose ✓
  D4  nginx service in compose ✓
  D5  retrain_scheduler.py + registry.py ✓
  D6  live_fetcher.py (5-min cycle, Redis cache) ✓
  D7  nginx.conf (blue/green + canary) ✓
  D8  Real model calls in app.py ✓ (just completed)
```

**Live forecast verified**:
```bash
curl http://localhost/forecast/NP15
# Returns: lmp_ratio_pred=2.55, lmp_dollar_estimate=$63.89, advisory="defer"
```

---

## What's NOT Working / Placeholders Remaining

| Item | Why it matters | Difficulty |
|---|---|---|
| **D8 feature parity** | Model expects 51 features; we provide 48 (3 padded with zeros). Real feature parity with training pipeline needs historical feature window from Redis. | Medium |
| **Per-zone features** | All 3 zones show same value because system-wide load + fuel mix is used. Need zone-specific LMP features (cross-zone spreads, weather). | Medium |
| **D9 - actual canary** | nginx.conf has location-based canary only (manual). Real upstream weight-based canary (5% automatic) needs two API containers with different model versions. | Low |
| **D10 - drift detector** | `src/monitoring/drift_detector.py` doesn't exist. `retrain_scheduler.py` references `artifacts/drift_log.json` but no producer. | Medium |
| **D11 - frontend** | No HTML/JS yet. Just API endpoints. | High (but separate track) |
| **D12 - E2E test** | Full cron + canary + retrain loop not yet wired | Low |

---

## Key Architecture (in place)

```
                  :80
                   ↓
              ┌──────────┐
              │  Nginx   │  blue/green + canary routes
              └────┬─────┘
                   ↓
              ┌──────────┐
              │   API    │  FastAPI, loads models/champion/*.json
              │  :8000   │  GET /forecast, /dc/{id}/forecast, /admin/reload
              └────┬─────┘
                   ↓
         ┌─────────┴──────────┐
         ↓                    ↓
    ┌─────────┐         ┌──────────┐
    │  Redis  │ ←───────│ Trainer  │  cron, --profile cron
    │  :6379  │ fetcher │ (sleep)  │  python -m src.models.retrain_scheduler
    └─────────┘         └──────────┘
         ↑
   live_fetcher.py
   every 5 min, populates:
   - features:zone:{NP15,SP15,ZP26}:now
   - features:dc:DC-XXXXX:now
   - meta:last_fetch
```

---

## File Structure

```
dc_real_time/
├── README.md
├── DOCKER.md                    # Docker usage docs
├── Dockerfile                   # multi-stage, 854MB
├── docker-compose.yml           # api + redis + trainer + nginx
├── .dockerignore
├── .gitignore                   # tracks registry.json, .gitkeep; ignores *.parquet
├── requirements.txt             # pinned deps
├── models/
│   ├── registry.json            # model registry (committed)
│   ├── .gitkeep
│   ├── champion -> v0.1         # symlink to active model version
│   └── v0.1/                    # contains lmp_ratio.json, carbon.json, etc.
├── data/
│   ├── raw/                     # (gitignored, not used)
│   ├── processed/               # (gitignored)
│   │   ├── caiso_lmp_1y.parquet
│   │   ├── caiso_fuel_mix_1y.parquet
│   │   ├── openmeteo_ca_dc_1y.parquet (50 of 227 sites)
│   │   ├── features_offline.parquet
│   │   ├── train.parquet, val.parquet, test.parquet
│   ├── external/
│   │   ├── ca_dc_sites.csv      # 227 CA DC sites (committed)
│   │   └── README.md
├── artifacts/                   # (gitignored JSON/CSV/PNG outputs)
│   ├── feature_schema.json      # 45 feature column names
│   ├── lmp_horizon_comparison.csv
│   ├── carbon_horizon_comparison.csv
│   └── ...
├── docs/
│   ├── PROJECT_SUMMARY.md
│   ├── SPIKE_CLASSIFIER.md
│   ├── PHASE_PLAN.md
│   ├── DATA_SOURCES.md
│   ├── DECISIONS.md
│   ├── PHASE1_FINDINGS.md
│   └── FEATURE_SCHEMA.md
├── nginx/
│   ├── nginx.conf               # blue/green (location-based) + canary
│   └── README.md
├── scripts/
│   ├── backfill_1y.py
│   ├── check_ghg_availability.py
│   ├── check_realtime_availability.py
│   ├── explore_ghg_windows.py
│   ├── weather_per_site.py
│   └── docker-entrypoint.sh
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── app.py               # D8: real model calls
│   ├── data/
│   │   ├── __init__.py
│   │   └── live_fetcher.py      # 5-min CAISO + weather cycle
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py     # 45 features + targets
│   └── models/
│       ├── __init__.py
│       ├── baselines.py
│       ├── registry.py           # model registry, atomic promotion
│       ├── retrain_scheduler.py  # drift/schedule check
│       ├── train_lmp.py
│       ├── train_lmp_multi_horizon.py
│       ├── train_carbon.py
│       └── train_carbon_multi_horizon.py
├── notebooks/
│   ├── 01_explore_caiso_lmp.ipynb
│   ├── 01_explore_caiso_lmp.py
│   └── colab_handoff/
│       ├── 01_caiso_1y_backfill.md
│       └── 01_colab_run_log.ipynb
```

---

## Recent Commits

```
44dfbd3  D8: Real model calls in app.py
110371a  D6: live fetcher with Redis cache + 5-min cycle
ab765cb  D4 + D7: nginx reverse proxy with blue/green + canary
b1e9246  D3 + D5: trainer service + retrain scheduler + model registry
6e357a4  D2: docker-compose with api + redis services
1e922ae  D1: Dockerfile + FastAPI stub + entrypoint
... (Phase 0-3 commits before these)
```

---

## Key Decisions (locked, see docs/DECISIONS.md for full list)

- **D1**: Multi-class spike classifier → regression on `lmp_ratio_target_4h` and `ghg_target_4h`
- **D8**: 5-min forward horizon wins for both LMP and carbon
- **D9**: Multi-step wins by 4x in R² for LMP, 3x for carbon vs 4h horizon
- **D12**: Outlier handling = winsorize at 0.1%/99.9%, pct_change hard-clipped to ±100
- **D13**: CAISO GHG only available for last ~90 days; we train on May-Jul 2026 only for carbon
- **Deployment**: Docker + docker-compose, one image, multiple services via CMD override
- **Blue/green**: model-level via symlink swap (not container-level yet)

---

## How To Start The Stack

```bash
cd /root/project/dc_real_time

# Start everything
docker compose up -d

# Run a live fetch (5-min cycle)
docker compose --profile cron up -d trainer
docker exec dc_real_time_trainer python -m src.data.live_fetcher

# Check the forecast
curl http://localhost/forecast/NP15 | python3 -m json.tool

# View per-DC advisory (uses cached data from fetcher)
curl http://localhost/dc/DC-00088/forecast

# Trigger model reload (after retrain)
curl -X POST http://localhost/admin/reload

# Stop everything
docker compose down
```

---

## What To Do Next (Priority Order)

### 1. **Fix feature parity (D8 full)** — make predictions truly live
**Why**: All 3 zones return same value because we use system-wide load + fuel mix
**How**:
- Cache a rolling 24h feature history per zone in Redis (e.g., `features:zone:NP15:history` as a sorted set)
- In `live_fetcher.py`, after each fetch, compute the 60m/4h/24h rolling stats and store in Redis
- In `app.py` forecast endpoint, fetch the history from Redis, compute features properly
**Result**: Per-zone-specific forecasts, real LMP ratio from rolling stats, not just load

### 2. **Drift detector (D10)**
**Why**: `retrain_scheduler.py` reads from a log that doesn't exist
**How**:
- Create `src/monitoring/drift_detector.py` that:
  - Reads recent features from Redis
  - Reads reference distribution from training (saved during `build_features.py`)
  - Computes PSI per feature
  - Writes `artifacts/drift_log.json`
- Add to docker-compose as a profile="cron" service with hourly schedule
**Result**: Auto-retrain triggers when drift detected

### 3. **Frontend (D11)**
**Why**: API has data, no visualization
**How**:
- `web/index.html` + JS + Plotly.js
- Map: 227 CA DC sites via Leaflet, colored by zone status, sized by MW
- Time series: LMP history + forecast band (uses /forecast endpoint)
- Advisory card: current grid state (clean/warn/dirty)
- Host via FastAPI static files

### 4. **Actual canary (D9)**
**Why**: nginx canary is location-based, not weight-based
**How**: Run two API containers with different MODEL_VERSION env vars, use upstream weights

---

## Critical Constraints / Gotchas

1. **User-Agent header required for CAISO CSV**: pandas default is blocked
2. **Last rows of demand.csv may have empty `Current demand`**: filter to most recent non-empty
3. **Model expects 51 features, schema has 45 + 3 zone dummies = 48**: pad with zeros (current workaround)
4. **No `docker-compose` v2 pre-installed**: package is `docker-compose-v2`
5. **CAISO OASIS LMP has 1-2h lag**: use predicted LMP from current load + fuel mix for real-time
6. **GHG data only published for last ~90 days**: train carbon model on May-Jul 2026 only
7. **Volume mount permissions**: artifacts dir must be owned by uid 1000 (app user in container)
8. **For each new code file, rebuild the image**: `docker build -t dc_real_time_api:v0.1 .`
9. **Image HEALTHCHECK probes :8000** (baked into Dockerfile). For services that DON'T run uvicorn (e.g. trainer with `sleep infinity`), override the healthcheck in compose — see trainer block for the Redis-ping pattern.
10. **Alpine `wget` resolves `localhost` to `::1` first**: nginx healthchecks using `http://localhost/...` may get "Connection refused" even when the proxy is fine. Use `http://127.0.0.1/...` instead.

---

## MLOPs Boundary (your earlier question)

- **Trainer is OUTSIDE the API**: separate Docker service, run on demand or via cron
- **Model swap is INSIDE the deployment**: atomic symlink swap, API reload via /admin/reload
- **For full blue/green**: run two API containers with different MODEL_VERSION, route via nginx upstream weights (D9)

---

## Where Data Lives

| Path | Status | Notes |
|---|---|---|
| `/root/project/dc_real_time/data/processed/caiso_lmp_1y.parquet` | ✓ exists | 315k rows, 12MB |
| `/root/project/dc_real_time/data/processed/caiso_fuel_mix_1y.parquet` | ✓ exists | 105k rows, 4.6MB |
| `/root/project/dc_real_time/data/processed/features_offline.parquet` | ✓ exists | 315k rows, 63MB (training input) |
| `/root/project/dc_real_time/data/processed/train.parquet` etc. | ✓ exists | Time splits |
| `/root/project/dc_real_time/data/external/ca_dc_sites.csv` | ✓ exists | 227 CA DCs |
| `/root/project/dc_real_time/models/v0.1/lmp_ratio.json` | ✓ exists | 5-min LMP model |
| `/root/project/dc_real_time/models/v0.1/carbon.json` | ✓ exists | 5-min carbon model |
| `/root/project/dc_real_time/models/champion -> v0.1` | ✓ symlink | Active model |

---

## What Success Looks Like (Phase 4 completion)

- [x] D1 Dockerfile
- [x] D2 compose api+redis
- [x] D3 trainer in compose
- [x] D4 nginx in compose
- [x] D5 retrain scheduler
- [x] D6 live fetcher
- [x] D7 nginx blue/green config
- [x] D8 API with real model calls
- [ ] D8 full: per-zone feature parity (zone-specific values)
- [ ] D9: actual weight-based canary
- [ ] D10: drift detector producing artifacts/drift_log.json
- [ ] D11: frontend HTML/JS
- [ ] D12: E2E test (cron + retrain + canary)

---

## Tips For The Next Agent

1. **Don't rebuild the Docker image unless you change Python code** — `docker compose restart` is enough
2. **Test fetcher output first** before testing API:
   ```bash
   docker exec dc_real_time_redis redis-cli GET 'features:zone:NP15:now'
   ```
3. **Linter is wrong about pandas/xgboost** — they're installed in the venv, ignore the "missing import" errors
4. **`docker compose` (v2) is the right command**, not `docker-compose` (v1)
5. **For a fresh rebuild**:
   ```bash
   docker compose down
   docker build -t dc_real_time_api:v0.1 .
   docker compose up -d
   ```
6. **Model files are NOT in the Docker image** — they're mounted from `./models` volume. The image loads them at startup.

---

**End of resume prompt.**
