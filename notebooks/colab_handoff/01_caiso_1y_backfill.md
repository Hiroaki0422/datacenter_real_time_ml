# Colab Handoff: CAISO 1-Year Historical Backfill

> **When to use this**: Phase 1 of the dc_real_time project. Pulls 1 year of CAISO 5-min LMP + fuel mix for offline training. Too slow for a single interactive session on a VPS — use Colab Pro.

## Quick Start

1. Open this notebook in Colab Pro
2. Run all cells
3. Download the resulting parquet files
4. Upload to `/root/project/dc_real_time/data/processed/` on your VPS

## Configuration

```python
# How much history to pull
HISTORY_DAYS = 365  # Full year
# HISTORY_DAYS = 180  # Half year (CAISO retains ~90-180d reliably)
# HISTORY_DAYS = 90   # Just the last 3 months (fastest, but limits training)

# Which CAISO trading hubs
ZONES = ['TH_NP15_GEN-APND', 'TH_SP15_GEN-APND', 'TH_ZP26_GEN-APND']
# TH_NP15 = Northern California (PG&E)
# TH_SP15 = Southern California (SCE/SDG&E)  
# TH_ZP26 = Central California (avoided cost zone)
```

## Why Colab

- **Time**: ~10-20 min to pull 1y of 5-min LMP via OASIS (one HTTP call per day, ~10s each, 365 calls)
- **Storage**: ~80 MB parquet for 1y
- **VPS is fine for inference** but the initial backfill is faster on Colab's faster network

## Size Budget

| Window | Rows | Parquet Size |
|--------|------|--------------|
| 7d | 5,799 | 250 KB |
| 30d | ~25,000 | ~1 MB |
| 90d | ~75,000 | ~3 MB |
| 180d | ~150,000 | ~6 MB |
| 365d | ~300,000 | ~12 MB |

(Per location: 288 5-min intervals/day × N days)

## Code

```python
!pip install gridstatus pyarrow --quiet

import gridstatus
import pandas as pd
from datetime import datetime, timedelta
import time

caiso = gridstatus.CAISO()

# === 1. Pull LMP for 1 year ===
print("Pulling LMP...")
end = datetime.now() - timedelta(days=1)
lmp_dfs = []
t0 = time.time()
for i in range(HISTORY_DAYS, 0, -1):
    d = end - timedelta(days=i-1)
    try:
        lmp = caiso.get_lmp(date=d, market='REAL_TIME_5_MIN')
        lmp_dfs.append(lmp)
        if i % 30 == 0:
            print(f"  {HISTORY_DAYS - i + 1}/{HISTORY_DAYS} days pulled ({time.time()-t0:.0f}s)")
    except Exception as e:
        print(f"  {d.date()}: skipped ({e})")

lmp_df = pd.concat(lmp_dfs, ignore_index=True)
print(f"LMP: {lmp_df.shape}, memory: {lmp_df.memory_usage(deep=True).sum()/1024**2:.1f} MB")
lmp_df.to_parquet('caiso_lmp_1y.parquet')
print(f"Saved caiso_lmp_1y.parquet ({lmp_df.memory_usage(deep=True).sum()/1024**2:.1f} MB)")

# === 2. Pull fuel mix for 1 year ===
print("\nPulling fuel mix...")
fm_dfs = []
for i in range(HISTORY_DAYS, 0, -1):
    d = end - timedelta(days=i-1)
    try:
        fm = caiso.get_fuel_mix(date=d)
        fm_dfs.append(fm)
    except Exception as e:
        print(f"  {d.date()}: skipped ({e})")

fm_df = pd.concat(fm_dfs, ignore_index=True)
print(f"Fuel mix: {fm_df.shape}")
fm_df.to_parquet('caiso_fuel_mix_1y.parquet')

# === 3. Verify data ===
print("\n=== LMP stats ===")
print(lmp_df.groupby('Location')['LMP'].describe())
print("\n=== Fuel mix means (MW) ===")
fuel_cols = [c for c in fm_df.columns if c not in ['Time','Interval Start','Interval End']]
print(fm_df[fuel_cols].mean().sort_values(ascending=False))

# === 4. Download ===
from google.colab import files
files.download('caiso_lmp_1y.parquet')
files.download('caiso_fuel_mix_1y.parquet')
```

## Expected Output

After running, you'll have:
- `caiso_lmp_1y.parquet` (~12 MB, ~300k rows × 11 cols)
- `caiso_fuel_mix_1y.parquet` (~5 MB, ~100k rows × 16 cols)

Upload to: `/root/project/dc_real_time/data/processed/`

## Verification

Run on VPS after upload:
```python
import pandas as pd
lmp = pd.read_parquet('data/processed/caiso_lmp_1y.parquet')
print(f"LMP: {lmp.shape}, {lmp['Time'].min()} → {lmp['Time'].max()}")
print(lmp.groupby('Location')['LMP'].describe())
```
