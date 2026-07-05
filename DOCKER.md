# dc_real_time — Docker Setup

## Quick Start

```bash
# Build the image
docker build -t dc_real_time_api:v0.1 .

# Start the stack (api + redis)
docker compose up -d

# Check it's running
docker compose ps

# Tail logs
docker compose logs -f api

# Test the API
curl http://localhost:8000/healthz

# Stop everything
docker compose down
```

## Services

| Service | Port | What it does |
|---|---|---|
| `api` | 8000 | FastAPI service (Phase 4 stub for now) |
| `redis` | 6379 | Online feature store + cache |
| `trainer` | — | Cron-driven retraining (D3) |
| `nginx` | 80 | Blue/green reverse proxy (D4) |

## Volume Mounts

| Host | Container | Notes |
|---|---|---|
| `./models` | `/app/models` | Read-write, model files |
| `./data` | `/app/data:ro` | Read-only, large datasets |
| `./artifacts` | `/app/artifacts` | Read-write, eval results |

The `:ro` on data is intentional — training reads it but never writes.
Models and artifacts are read-write because the trainer writes to models/.

## Model Loading

The API loads models from `models/champion/` (a symlink to `models/v0.1/`,
`models/v0.2/`, etc.). To deploy a new model:

```bash
# Train a new version (in trainer container)
docker compose run --rm trainer python -m src.models.train_lmp

# Promote to champion (atomic symlink swap)
cd models
ln -sfn v0.2 champion

# Trigger API reload (SIGHUP or HTTP endpoint)
curl -X POST http://localhost:8000/admin/reload
```

The API will pick up the new model without dropping in-flight requests.

## Healthcheck

```bash
docker compose ps   # shows "healthy" or "starting" status
```

The API image has a built-in healthcheck that pings `/healthz` every 30s.
