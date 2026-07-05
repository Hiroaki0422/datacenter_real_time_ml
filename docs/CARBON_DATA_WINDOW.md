# Carbon model data window

The carbon (GHG / marginal emissions) field from CAISO is only published
for the most recent ~90 days. Our 1y backfill contains non-zero GHG
data from **2026-05-01** through the most-recent CAISO publish date.

This is a moving target — the latest GHG data in our LMP parquet
grows by ~1 day per day. The carbon model was initially trained on a
small window (May 1 → Jul 4 = 64 days, ~30% non-zero) but will improve
as the window expands.

## Auto-retrain mechanism (Phase 3)

The fetcher now:
1. Probes `data/processed/caiso_lmp_1y.parquet` to find the current
   carbon data window (oldest and latest non-zero GHG timestamps).
2. If the window has grown by `>= CARBON_MIN_NEW_DAYS=7` since the
   last training, queues a retrain by writing to `meta:carbon_retrain_queued`
   in Redis (TTL 7 days).
3. Updates `meta:last_carbon_data_date` to the latest GHG date for
   next-cycle throttling.

The retrain scheduler picks up the queue flag via `check_carbon_queue()`
and runs `train_new_model()` if set, then clears the flag.

## Why we don't retrain on every cycle

- `CARBON_MIN_NEW_DAYS=7`: only retrain if 7+ new days of data
- `CARBON_MIN_WINDOW_DAYS=30`: only retrain if total window is
  at least 30 days (avoid overfitting on tiny windows)
- These env vars are configurable; defaults are in `live_fetcher.py`

## Verifying the auto-retrain works

```bash
# 1. Trigger the fetcher
docker exec dc_real_time_trainer python -m src.data.live_fetcher

# 2. Check the queue flag (should be set on first run)
docker exec dc_real_time_redis redis-cli GET 'meta:carbon_retrain_queued'

# 3. Check the scheduler picks it up
docker exec dc_real_time_trainer python -m src.models.retrain_scheduler --check
# Output: "Should retrain: True (carbon: window expanded)"

# 4. Run the actual retrain
docker exec dc_real_time_trainer python -m src.models.retrain_scheduler --train --auto-promote
```

## Future work

- The carbon training script (`train_carbon_multi_horizon.py`) already
  auto-detects the window from the parquet. No code change needed when
  the window grows.
- The fetcher's `predict_lmp_from_state` doesn't use carbon data yet
  (carbon is its own model). The API serves both via `/forecast/{zone}?horizon=X`
  which returns both `lmp_dollar_estimate` and `carbon_pred_short_ton_per_mwh`.
- Drift detection (D10) will use the carbon queue as one of its
  retrain triggers alongside scheduled retrain and feature drift.
