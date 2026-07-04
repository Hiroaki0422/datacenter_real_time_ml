# Feature Schema — Phase 2

> Defines the exact feature set for both models. This is the contract between Phase 2 (features) and Phase 3 (models).

## Two Targets, Two Feature Sets

| Model | Target | Granularity | Horizon | What we predict |
|---|---|---|---|---|
| **A — Multi-class spike** | `spike_class ∈ {0,1,2,3}` | per zone, per 5-min | t+1h (next hour aggregated) | Probability vector for advisory |
| **B — Carbon regression** | `ghg_short_ton_per_mwh` | per timestamp (system-wide) | t+1h | Continuous carbon intensity |

**Note on the horizon**: The 5-min LMP is the *current* value. To predict the *next 1-4h*, we engineer features that capture the *current state* of the grid (LMP rolling stats, fuel mix, weather nowcast) and the target is the *aggregated* value 1h later (e.g., max spike class in next 4 5-min intervals, or mean GHG in next 12 intervals).

## Per-Zone Features (Model A inputs)

### Group 1: Same-zone LMP rolling stats (12 features)

For each zone, compute rolling statistics of `LMP` over multiple windows:

| Feature | Window | Notes |
|---|---|---|
| `lmp_mean_60m` | 60 min | Short-term trend |
| `lmp_std_60m` | 60 min | Recent volatility |
| `lmp_mean_4h` | 4h (240 min) | Matches baseline window |
| `lmp_std_4h` | 4h | 4h volatility |
| `lmp_mean_24h` | 24h | Daily baseline |
| `lmp_max_24h` | 24h | Daily peak |
| `lmp_min_24h` | 24h | Daily trough |
| `lmp_slope_60m` | 60 min linear fit | Direction of travel |
| `lmp_ratio_to_4h` | current LMP / 4h mean | Already-classified! **LEAKAGE — DO NOT INCLUDE** |
| `lmp_pct_change_5m` | 5 min | Recent jump |
| `lmp_pct_change_60m` | 60 min | Hour-over-hour |
| `lmp_range_4h` | max - min in 4h | Intraday range |

**Important**: I had to remove `lmp_ratio_to_4h` because that's literally how we define the target. Including it would be data leakage (target leakage). The model needs to predict the spike class without seeing the ratio.

### Group 2: Energy / Congestion / Loss components (3 features)

From the LMP decomposition:
- `energy_60m_mean` — same-zone, 60 min mean
- `congestion_60m_mean` — same-zone, 60 min mean
- `loss_60m_mean` — same-zone, 60 min mean

### Group 3: Cross-zone features (3 features)

How this zone compares to neighbors:
- `lmp_spread_np_sp` — `lmp_NP15 - lmp_SP15` (transmission congestion proxy)
- `lmp_spread_np_zp` — `lmp_NP15 - lmp_ZP26`
- `lmp_max_across_zones_60m` — highest LMP across all 3 zones in last 60 min

### Group 4: Fuel mix (system-wide, 4 features)

Joined from `caiso_fuel_mix_1y.parquet`:
- `solar_mw_60m_mean` — current solar output
- `wind_mw_60m_mean` — current wind output
- `natgas_mw_60m_mean` — gas generation (marginal fuel indicator)
- `imports_mw_60m_mean` — import share

### Group 5: Carbon (system-wide, 2 features)

- `ghg_60m_mean` — current carbon intensity (rolling mean)
- `ghg_nonzero_pct_60m` — fraction of last 60 min where GHG > 0

### Group 6: Calendar (8 features)

- `hour_of_day` — 0-23
- `day_of_week` — 0-6
- `month` — 1-12
- `is_weekend` — 0/1
- `hour_sin`, `hour_cos` — cyclic encoding of hour
- `month_sin`, `month_cos` — cyclic encoding of month

### Group 7: Lag features (3 features)

Previous-interval LMP at different lookbacks (captures momentum):
- `lmp_lag_1` — 5 min ago
- `lmp_lag_12` — 1h ago
- `lmp_lag_48` — 4h ago

### Group 8: Per-zone identifier (1 feature)

- `zone` — categorical (NP15, SP15, ZP26)

**Total per-zone features: 12 + 3 + 3 + 4 + 2 + 8 + 3 + 1 = 36**

## Per-DC Features (joined on top of per-zone)

After predicting the zone-level spike + carbon, we map to each DC site:

| Feature | Source |
|---|---|
| `caiso_zone` | from `ca_dc_sites.csv` (already assigned) |
| `latitude`, `longitude` | from `ca_dc_sites.csv` |
| `mw_capacity` | from `ca_dc_sites.csv` |
| `wue_default` | from `ca_dc_sites.csv` (water usage effectiveness) |
| `bws_score` | from `ca_dc_sites.csv` (water stress) |
| `climate_adj` | from `ca_dc_sites.csv` |
| `operator` | from `ca_dc_sites.csv` (categorical) |

**Per-DC per-timestamp rows = 227 DCs × 5-min intervals = ~600k+ rows** for 1y.

## Target Variables

### `spike_class` (Model A)

Multi-class, computed at training time per zone per timestamp:
```python
baseline = LMP.rolling('4h').mean()
ratio = LMP / baseline
spike_class = pd.cut(ratio, bins=[0, 1.5, 3.0, 6.0, inf], labels=[0,1,2,3])
```

### `ghg_short_ton_per_mwh` (Model B)

Continuous, from CAISO directly. Forward-filled where missing.

### Future-state target (for "predict next 1h")

To make this a *forecasting* problem (not just "predict current state"):
- `spike_class_target` = `argmax(P(spike in next 4 intervals))` — what class will we be in 1h from now?
- `ghg_target` = mean GHG in next 12 intervals (1h ahead, 5-min × 12)

This is what separates "feature engineering" from "the model can use any LMP it wants."

## Train / Val / Test Split

Time-based, no shuffle (prevent leakage from future to past):

| Split | Date range | Months | Use |
|---|---|---|---|
| Train | 2025-07-04 → 2025-12-31 | 6 | Model fitting |
| Val | 2026-01-01 → 2026-03-31 | 3 | Hyperparameter tuning, threshold selection |
| Test | 2026-04-01 → 2026-07-03 | 3 | Final evaluation (held out) |

**Leakage prevention**: When building rolling features, use `shift(1)` so the feature at time t uses data only from t-window to t-1 (not t itself).

## Class Weights (for Model A)

Computed from training-set class frequencies:
```python
class_weights = {0: 1.0, 1: 5.0, 2: 25.0, 3: 100.0}  # inverse frequency, capped
```

This is passed to XGBoost via `sample_weight` or `scale_pos_weight` per class.

## Validation Checks (before saving features)

1. **No NaN in features** (except for leading warm-up period, which is dropped)
2. **No inf in features**
3. **No leakage**: every feature at time t uses data from t-window to t-1
4. **Shape sanity**: ~315k rows × 36 features for per-zone; ~71M rows for per-DC
5. **Per-zone counts match**: NP15 = SP15 = ZP26 in row count (after timezone alignment)
6. **Target distribution matches Phase 1 findings** (0.88-1.65% Class 3)

## Output Files

- `data/processed/features_offline.parquet` — per-zone features + target (~315k rows × 38 cols)
- `data/processed/features_offline_dc.parquet` — per-DC overlay (~71M rows × ~45 cols) — optional, may be too large
- `data/processed/train.parquet`, `val.parquet`, `test.parquet` — split versions
- `artifacts/feature_schema.json` — column metadata
- `artifacts/class_weights.json` — class weights for Model A

## What's NOT In v1 (Deferred)

- Network signal proxies (RIPE RIS was dropped)
- Day-ahead LMP features (need separate query, defer to v1.1)
- Per-DC weather overlay (50/227 sites available, defer full overlay)
- Real-time status feeds (sparse, defer to v1.1)
