# Nginx blue/green + canary config

## Architecture

```
Internet
   ↓
Nginx (:80) — single entry point, zero-downtime reload
   ↓
   api:8000  ← single service, model is swapped via symlink
```

We use **model-level** blue/green (different XGBoost JSON files swapped via
symlink), not **container-level** blue/green. The API loads `models/champion.json`
on startup; updating the symlink + hitting `/admin/reload` is a zero-downtime
model swap.

For **automatic traffic canary** at 5%, you'd need two `api` services running
different model versions. That's Phase 5 work — for now, canary is via the
`/canary/` route with manual traffic.

## Routes

| Route | Backend | Use |
|---|---|---|
| `http://localhost/` | api:8000 (blue / production) | Default traffic |
| `http://localhost/canary/` | api:8000 (also, but tagged) | Manual canary testing |
| `http://localhost/nginx-health` | (Nginx itself) | Nginx liveness |

## How To Deploy A New Model (Zero-Downtime)

```bash
# 1. Train new model in trainer container
docker compose --profile cron run --rm trainer \
    python -m src.models.retrain_scheduler --train --auto-promote

# 2. Trigger API reload (atomic, no downtime)
curl -X POST http://localhost/admin/reload
```

The trainer writes a new model directory (e.g., `models/v0.2/`), updates the
`models/champion` symlink atomically, and the API reloads the new model on
the next request. In-flight requests use the old model (no interruption).

## How To Canary Test

```bash
# Send a request to the canary route
curl http://localhost/canary/forecast/NP15

# The response includes X-Canary: green header
```

For now, this just adds a header. For real traffic splitting, you'd run
two API containers and use upstream weights.

## How To Roll Back (Without Redeploying)

```bash
# Find the previous model version
ls -la models/

# Restore the previous champion symlink
ln -sfn v0.1 models/champion

# Reload
curl -X POST http://localhost/admin/reload
```

## Zero-Downtime Nginx Reload

If you edit `nginx.conf`:

```bash
docker compose exec nginx nginx -s reload
```

This validates the config and re-spawns workers without dropping connections.

## Health Checks

- `http://localhost/nginx-health` — Nginx is up
- `http://localhost/healthz` — API is up
- `http://localhost/readyz` — API has loaded its model and dependencies

## Future: Real Container-Level Blue/Green

When you outgrow model-level blue/green, the upgrade path is:

```yaml
# docker-compose.yml
services:
  api-blue:
    image: dc_real_time_api:v0.1
    environment:
      - MODEL_VERSION=v0.1
  api-green:
    image: dc_real_time_api:v0.1
    environment:
      - MODEL_VERSION=v0.2
  nginx:
    # upstream weights can be flipped to 100/0 -> 0/100 atomically
```

The Nginx config stays the same; just the backend service names change.
