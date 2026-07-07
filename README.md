# dc_real_time — Spatial-Temporal Carbon & Price Forecasting for Data Center Infrastructure

> **One-line thesis**: Predict 30-minute to 4-hour ahead wholesale electricity price and carbon intensity in CAISO, mapped onto California data center sites, to power a public "carbon-aware compute scheduling" advisory.

> **Status**: Phase 4 (D1–D12) — production stack live, multi-horizon ML models trained and auto-promoting, frontend dashboard serving at `http://localhost/`.

## What it does

The system ingests live data from three sources, trains multi-horizon XGBoost models, and serves a public dashboard that tells data center operators and the public:

- **For each of 3 CAISO zones** (NP15, SP15, ZP26): predicted LMP and carbon intensity at 30m / 1h / 2h / 4h horizons
- **For each of 227 California data centers**: worst-case advisory across all 4 horizons, with per-horizon breakdown
- **For all 3 zones**: 24h (or windowed) LMP history from CAISO OASIS
- **Auto-retrains** when new carbon data becomes available, when drift is detected, or when the champion ages past 7 days

Advisories follow carbon-aware-compute etiquette: **ok / watch / defer / pause** based on LMP thresholds (>$30/$50/$100/MWh).

## Quick start

### Prerequisites

- Linux with Docker + Docker Compose v2 (`docker compose`, not `docker-compose`)
- 4 vCPU, 8 GB RAM, 50 GB disk (VPS-class)
- Port 80 (nginx) and 6379 (redis) available

### Launch

```bash
git clone git@github.com:Hiroaki0422/datacenter_real_time_ml.git
cd datacenter_real_time_ml

# Start the full stack (fetcher, API, Redis, nginx)
docker compose up -d

# Open the dashboard
open http://localhost/
```

### Verify it's working

```bash
# All services healthy
docker compose ps

# Forecast endpoint returns live LMP
curl 'http://localhost/forecast/NP15?horizon=1h' | python3 -m json.tool

# Per-DC advisory with 4 horizons
curl 'http://localhost/dc/DC-00088/forecast?horizon=4h' | python3 -m json.tool

# LMP history (last 12 entries per zone)
curl 'http://localhost/zones/history?limit=12' | python3 -m json.tool

# All 227 DC sites
curl 'http://localhost/sites' | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'{d[\"count\"]} sites')"
```

Expected output: 5 services `healthy` (fetcher, api, redis, trainer, nginx), forecast returns LMP in the $20–$60 range depending on grid conditions.

### Stop

```bash
# Graceful shutdown — stops all 5 services, removes containers and networks.
# Volumes (Redis persistence, model files) are kept on the host.
docker compose down

# If you also started the trainer (cron profile), bring it down too:
docker compose --profile cron down

# Hard stop — same as above, but kills containers with SIGKILL (use only if
# the graceful shutdown hangs on a stuck fetcher cycle).
docker compose down --timeout 10

# Nuclear option — also delete Redis persistence volume (loses any
# unflushed LMP history and the carbon retrain queue). Use only when
# you want a truly clean slate.
docker compose down -v
```

### What survives a shutdown

| Data | Survives? | Where |
|---|---|---|
| Trained model files (v0.1, v0.2, ...) | ✅ Yes | `./models/` (host volume) |
| Model registry (champion, candidates, history) | ✅ Yes | `./models/registry.json` |
| Redis LMP history (24h of 5-min intervals) | ⚠️ Only with named volume | `redis_data` Docker volume |
| Trained-on datasets (1y LMP, fuel mix) | ✅ Yes | `./data/processed/` (host volume) |
| Training artifacts (eval reports, plots) | ✅ Yes | `./artifacts/` (host volume) |
| Current fetcher cycle output | ❌ No | In-memory only |

To re-launch after a clean shutdown, just `docker compose up -d` again — all state is restored.

### Restarting just one service

```bash
# Restart nginx (e.g. after config change or stale IP cache)
docker compose restart nginx

# Restart the API (e.g. after model swap + /admin/reload)
docker compose restart api

# Restart the fetcher (e.g. if it crashed and missed cycles)
docker compose restart fetcher
```

The fetcher in particular benefits from `restart fetcher` if it crashed mid-cycle — the next 5-min cycle will repopulate Redis from the latest CAISO snapshot.

### Removing all traces

```bash
# Stop everything, remove containers, networks, AND the redis volume
docker compose down -v

# Remove the Docker images too (forces a full rebuild on next up)
docker rmi dc_real_time_api:v0.1 nginx:1.27-alpine redis:7-alpine

# Optional: remove the host-side model registry and training artifacts
# (DESTRUCTIVE — only do this if you want to start from a clean slate)
rm -rf models/registry.json models/v0.* artifacts/eval_*.json
```

After this, the fetcher auto-starts with `docker compose up -d`. To retrain and rebuild the registry from scratch:
```bash
docker compose run --rm trainer python -m src.models.retrain_scheduler --train --auto-promote
```

## Architecture

```
                  :80                          (nginx: blue/green + canary routes)
                   ↓
              ┌──────────┐
              │  Nginx   │
              └────┬─────┘
                   ↓
              ┌──────────┐
              │   API    │  FastAPI, loads models/champion/*.json (8 files: 4 horizons × 2 targets)
              │  :8000   │  GET /forecast, /dc/{id}/forecast, /sites, /zones/history, /admin/reload
              └────┬─────┘
                   ↓
              ┌──────────┐
              │  Redis   │  Online feature store (per-zone features, per-DC advisories, LMP history)
              │  :6379   │  Populated by fetcher, read by API
              └────┬─────┘
                   ↓
         ┌─────────┴──────────┐
         ↓                    ↓
    ┌──────────┐        ┌──────────┐
    │ Fetcher  │        │ Trainer  │  python -m src.models.retrain_scheduler (drift/queue/age)
    │ :--loop  │        │ :on-call │  python -m src.models.train_lmp_multi_horizon --version vN
    │ 5-min    │        │          │  python -m src.models.train_carbon_multi_horizon --version vN
    └──────────┘        └──────────┘
         │
   Redis keys written (fetcher):
   features:zone:{Z}:now        — per-zone live features + predicted LMP
   features:dc:{id}:now         — per-DC advisory (all 4 horizons)
   features:zone:{Z}:lmp_history — rolling 24h of real CAISO LMP
   meta:last_fetch              — cycle timestamp
   meta:carbon_retrain_queued   — carbon retrain trigger
```

**MLOPs boundary**: fetcher and trainer run OUTSIDE the API (separate services). Model swap is INSIDE — atomic symlink `models/champion -> models/{version}` plus `/admin/reload` endpoint. The fetcher runs continuously in `--loop` mode and is auto-started by `docker compose up -d`.

## Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Frontend dashboard (Plotly.js SPA) |
| `GET /healthz` | Liveness check |
| `GET /readyz` | Readiness check (models loaded) |
| `GET /model/info` | Currently loaded model paths |
| `POST /admin/reload` | Reload models from disk (after symlink swap) |
| `GET /forecast/{zone}?horizon={30m\|1h\|2h\|4h}` | Per-zone forecast at given horizon |
| `GET /dc/{dc_id}/forecast?horizon={30m\|1h\|2h\|4h}` | Per-DC advisory with all 4 horizons |
| `GET /sites` | All 227 California DC sites (for the map) |
| `GET /zones/history?since=ISO&limit=N` | Per-zone LMP history (paginated) |

## Multi-horizon models

Each version directory (e.g. `models/v0.2/`) contains 8 model files:

```
lmp_ratio_30m.json    lmp_ratio_1h.json    lmp_ratio_2h.json    lmp_ratio_4h.json
carbon_30m.json       carbon_1h.json       carbon_2h.json       carbon_4h.json
```

The fetcher writes all 4 horizon values per DC, the API serves any of them via `?horizon=X`. v0.3 is the current champion with val R²:

| Horizon | LMP R² | Carbon R² |
|---|---|---|
| 30m | 0.64 | 0.74 |
| 1h  | 0.56 | 0.71 |
| 2h  | 0.40 | 0.65 |
| 4h  | 0.18 | 0.65 |

(5-min forward horizon was the original winner per `docs/DECISIONS.md` D8, but the user spec for averages asked for 30m/1h/2h/4h. The API keeps the 5-min as the "30m" anchor.)

## Retraining

The retrain scheduler checks 3 triggers and retrains if any fires:

1. **Drift**: `artifacts/drift_log.json` reports `max_psi > 0.2` (D10 — drift log producer not yet built, so currently always 0)
2. **Schedule**: champion is older than 7 days (`RETRAIN_MAX_AGE_DAYS`)
3. **Carbon window**: fetcher queued a retrain because CAISO published new GHG data

```bash
# Check if a retrain is needed
docker exec dc_real_time_trainer python -m src.models.retrain_scheduler --check

# Force a retrain and auto-promote
docker exec dc_real_time_trainer python -m src.models.retrain_scheduler --train --auto-promote
```

The fetcher runs continuously as a dedicated service (auto-started with `docker compose up -d`). The trainer is on-demand via `docker compose run --rm trainer` or the cron profile.

## Training pipeline

```bash
# Retrain from scratch (LMP + carbon, all 4 horizons)
docker compose run --rm trainer python -m src.models.retrain_scheduler --train

# Just LMP
docker compose run --rm trainer python -m src.models.train_lmp_multi_horizon --version v0.4

# Just carbon
docker compose run --rm trainer python -m src.models.train_carbon_multi_horizon --version v0.4
```

Training takes ~6 minutes (LMP ~4 min, carbon ~2 min). Output: comparison CSV + plot in `artifacts/`, per-version models in `models/{version}/`.

## Files

```
dc_real_time/
├── README.md                          # this file
├── RESUME.md                          # handoff doc for next session
├── DOCKER.md                          # Docker usage notes
├── docker-compose.yml                 # 5 services: fetcher, api, redis, trainer, nginx
├── Dockerfile                         # multi-stage, ~30s cold build
├── nginx/
│   ├── nginx.conf                     # blue/green + canary routes
│   └── conf.d/                        # empty (suppresses image's default.conf)
├── web/
│   └── index.html                     # Plotly.js SPA, no build step
├── src/
│   ├── api/app.py                     # FastAPI: 11 endpoints
│   ├── data/live_fetcher.py           # 5-min cycle, per-zone LMP, carbon retrain queue
│   ├── features/build_features.py    # 45 features + targets
│   └── models/
│       ├── registry.py                # atomic version + promote
│       ├── train_lmp_multi_horizon.py # 4-horizon LMP trainer
│       ├── train_carbon_multi_horizon.py # 4-horizon carbon trainer
│       └── retrain_scheduler.py       # drift + schedule + carbon queue triggers
├── data/
│   ├── processed/                     # 1y CAISO LMP, fuel mix, Open-Meteo (parquet)
│   └── external/ca_dc_sites.csv       # 227 CA DC sites
├── models/                            # registry: champion symlink + versioned dirs
│   ├── registry.json                  # v0.1 (history), v0.3 (champion)
│   ├── champion -> v0.3
│   ├── v0.1/  (legacy)
│   ├── v0.2/  (archived)
│   └── v0.3/  # 8 model files
├── artifacts/                          # eval reports, comparison plots
├── scripts/
│   └── backfill_registry_v01.py       # one-time backfill of v0.1 in registry
└── docs/
    ├── PROJECT_SUMMARY.md             # full spec
    ├── DECISIONS.md                   # locked design decisions
    ├── PHASE_PLAN.md                   # 6-phase milestone plan
    ├── FEATURE_SCHEMA.md              # 45 features + 3 targets
    ├── DATA_SOURCES.md                # what we pull and why
    ├── PHASE1_FINDINGS.md             # EDA results
    ├── SPIKE_CLASSIFIER.md            # legacy multi-class design (rejected in D1)
    └── CARBON_DATA_WINDOW.md          # Phase 3 carbon retrain mechanism
```

## Operational notes

### Logs

```bash
# Tail all services
docker compose logs -f

# Just the fetcher (every 5 min cycle)
docker compose logs -f fetcher
```

### Health checks

All services have Docker healthchecks. Run `docker compose ps` to see status. Common statuses:
- `healthy` — passing healthcheck
- `unhealthy` — recent healthcheck failed (check logs)
- `starting` — healthcheck in start_period (e.g. waiting for Redis)

### Data freshness

The 5-min fetcher cycle writes to `meta:last_fetch` (TTL 600s). If this key is missing, the fetcher is stuck or crashed. The frontend's "Last fetch" line shows this.

### Permissions

Volume-mounted files in the API container are owned by the host user, not by `app` (uid 1000). If you see `PermissionError` on logs, run:

```bash
chown -R 1000:1000 web/ data/processed/
```

### Nginx upstream IP changes

When the API container restarts, it gets a new IP. Nginx caches the old IP. If you see 502s, run:

```bash
docker compose restart nginx
```

(Proper fix: configure nginx with `resolver 127.0.0.11` so it re-resolves on each request. Tracked as a TODO.)

## Remaining work (per RESUME.md)

| Item | Status |
|---|---|
| D1–D8 (Docker, compose, fetcher, blue/green, real model calls, dedicated fetcher service) | ✅ Done |
| D8 full (per-zone LMP history + 22 LMP features at inference) | ✅ Done |
| D9 (real weight-based canary) | ❌ Not started — nginx canary is location-based only |
| D10 (drift detector producing `artifacts/drift_log.json`) | ❌ Not started — retrain scheduler reads the log but no producer exists |
| D11 (Plotly.js frontend) | ✅ Done |
| D12 (E2E test: cron + retrain + canary) | ❌ Not started |
| Multi-horizon training (30m/1h/2h/4h) | ✅ Done |
| Per-DC multi-horizon advisory | ✅ Done |
| /zones/history pagination | ✅ Done |
| Carbon auto-retrain on window expansion | ✅ Done |

## License

MIT (matches v1 dataset license). See `LICENSE` (TBD).

## Related projects

- **v1 (datacenter_water_stress)**: Static WUE × climate × basin stress for 1,575 US DCs. Used as the static anchor for the 227 California DC sites in this project.
