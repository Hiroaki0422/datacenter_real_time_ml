"""Explore shorter forward windows for GHG target."""
import pandas as pd
import numpy as np
from pathlib import Path

P = Path('/root/project/dc_real_time/data/processed')
lmp = pd.read_parquet(P / 'caiso_lmp_1y.parquet')

# Look at May-Jul 2026 only (where GHG exists)
may_start = pd.Timestamp('2026-05-01', tz='US/Pacific')
end = lmp['Time'].max()
sub = lmp[(lmp['Time'] >= may_start) & (lmp['Time'] <= end)].copy()
print(f"May-Jul 2026: {len(sub):,} rows, {sub['Location'].nunique()} zones")
print()

# For each forward window, compute forward mean GHG
print("=== Forward MEAN (target: mean GHG in next X min) ===")
for window_min in [5, 15, 30, 60, 120, 240]:
    window_int = window_min // 5
    sub[f'ghg_fwd_{window_min}m'] = sub['GHG'].shift(-window_int).rolling(window_int).mean()
    fwd = sub[f'ghg_fwd_{window_min}m'].dropna()
    nz_count = (fwd > 0).sum()
    nz_pct = nz_count / len(fwd) * 100 if len(fwd) else 0
    mean_nz = fwd[fwd > 0].mean() if nz_count else 0
    print(f"  {window_min:3d}-min forward mean:")
    print(f"    Rows: {len(fwd):,}, Non-zero: {nz_count:,} ({nz_pct:.1f}%)")
    print(f"    Mean (non-zero): {mean_nz:.2f}")
    print()

# Also: max in window
print("=== Forward MAX (target: any carbon in next X min) ===")
for window_min in [5, 15, 30, 60, 120, 240]:
    window_int = window_min // 5
    sub[f'ghg_max_{window_min}m'] = sub['GHG'].shift(-window_int).rolling(window_int).max()
    fwd = sub[f'ghg_max_{window_min}m'].dropna()
    nz_count = (fwd > 0).sum()
    nz_pct = nz_count / len(fwd) * 100 if len(fwd) else 0
    print(f"  {window_min:3d}-min forward max: Non-zero: {nz_count:,} ({nz_pct:.1f}%)")
PY
