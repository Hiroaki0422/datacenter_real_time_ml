"""
Feature engineering pipeline for dc_real_time.

Reads:
  - data/processed/caiso_lmp_1y.parquet  (5-min LMP for 3 trading hubs)
  - data/processed/caiso_fuel_mix_1y.parquet  (5-min fuel mix)
  - data/external/ca_dc_sites.csv  (227 CA DC sites)

Writes:
  - data/processed/features_offline.parquet  (per-zone, 5-min features + target)
  - data/processed/train.parquet, val.parquet, test.parquet  (time-based splits)
  - artifacts/feature_schema.json  (column metadata)
  - artifacts/class_weights.json  (for Model A)

See docs/FEATURE_SCHEMA.md for the full feature spec.
"""
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get('PROJECT_ROOT', '/root/project/dc_real_time'))
PROCESSED = PROJECT_ROOT / 'data' / 'processed'
EXTERNAL = PROJECT_ROOT / 'data' / 'external'
ARTIFACTS = PROJECT_ROOT / 'artifacts'
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# Time-based split boundaries
TRAIN_END = pd.Timestamp('2026-01-01', tz='US/Pacific')
VAL_END = pd.Timestamp('2026-04-01', tz='US/Pacific')

# Spike thresholds (LOCKED in Phase 1, see DECISIONS.md)
SPIKE_THRESHOLDS = (1.5, 3.0, 6.0)
SPIKE_WINDOW = '4h'
ROLLING_WINDOWS = {
    '60m': '60min',
    '4h': '4h',
    '24h': '24h',
}


def load_data():
    """Load LMP, fuel mix, and DC sites."""
    print("Loading data...")
    lmp = pd.read_parquet(PROCESSED / 'caiso_lmp_1y.parquet')
    fm = pd.read_parquet(PROCESSED / 'caiso_fuel_mix_1y.parquet')
    dc = pd.read_csv(EXTERNAL / 'ca_dc_sites.csv')
    print(f"  LMP: {lmp.shape}, fuel mix: {fm.shape}, DC sites: {len(dc)}")
    return lmp, fm, dc


def add_calendar(df: pd.DataFrame, time_col='Time') -> pd.DataFrame:
    """Add hour-of-day, day-of-week, month, weekend, cyclic encodings."""
    t = df[time_col]
    df['hour_of_day'] = t.dt.hour
    df['day_of_week'] = t.dt.dayofweek
    df['month'] = t.dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    # Cyclic encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return df


def add_lmp_features(lmp: pd.DataFrame) -> pd.DataFrame:
    """Per-zone LMP rolling features. CRITICAL: shift(1) to prevent target leakage."""
    print("Adding per-zone LMP features...")
    lmp = lmp.sort_values(['Location', 'Time']).copy()
    # Clean short-zone id
    lmp['zone'] = lmp['Location'].str.replace('TH_', '').str.replace('_GEN-APND', '')

    # Per-zone rolling features
    parts = []
    for zone, sub in lmp.groupby('zone', sort=False):
        sub = sub.set_index('Time').sort_index()
        s = sub['LMP']

        out = pd.DataFrame(index=sub.index)
        out['zone'] = zone
        out['LMP'] = s
        out['Energy'] = sub['Energy']
        out['Congestion'] = sub['Congestion']
        out['Loss'] = sub['Loss']
        out['GHG'] = sub['GHG']

        # Rolling stats — shift(1) to prevent leakage (feature at t uses data from t-window to t-1)
        for label, window in ROLLING_WINDOWS.items():
            shifted = s.shift(1)
            out[f'lmp_mean_{label}'] = shifted.rolling(window).mean()
            out[f'lmp_std_{label}'] = shifted.rolling(window).std()
            if label == '24h':
                out[f'lmp_max_{label}'] = shifted.rolling(window).max()
                out[f'lmp_min_{label}'] = shifted.rolling(window).min()

        # 60min rolling for energy/congestion/loss/GHG
        for col in ['Energy', 'Congestion', 'Loss', 'GHG']:
            out[f'{col.lower()}_60m_mean'] = sub[col].shift(1).rolling('60min').mean()

        # GHG non-zero fraction in last 60min
        ghg_shifted = sub['GHG'].shift(1)
        out['ghg_nonzero_pct_60m'] = (ghg_shifted.gt(0)).rolling('60min').mean()

        # Lag features (LMP at t-1, t-12, t-48)
        out['lmp_lag_1'] = s.shift(1)         # 5 min ago
        out['lmp_lag_12'] = s.shift(12)       # 1h ago
        out['lmp_lag_48'] = s.shift(48)       # 4h ago

        # 60min slope (linear fit)
        def linreg_slope(arr):
            if len(arr) < 2 or np.all(np.isnan(arr)):
                return np.nan
            y = arr
            x = np.arange(len(y))
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                return np.nan
            return np.polyfit(x[mask], y[mask], 1)[0]

        out['lmp_slope_60m'] = s.shift(1).rolling('60min').apply(linreg_slope, raw=True)

        # 5min and 60min percent change
        out['lmp_pct_change_5m'] = s.pct_change(1)
        out['lmp_pct_change_60m'] = s.pct_change(12)

        # 4h range
        out['lmp_range_4h'] = s.shift(1).rolling('4h').max() - s.shift(1).rolling('4h').min()

        # Calendar
        out = add_calendar(out.reset_index(), 'Time').set_index('Time')

        parts.append(out)

    out = pd.concat(parts).reset_index()
    print(f"  Per-zone features: {out.shape}")
    return out


def add_fuel_mix_features(features: pd.DataFrame, fm: pd.DataFrame) -> pd.DataFrame:
    """Join system-wide fuel mix features (rolling means, shift(1) for no leakage)."""
    print("Adding fuel mix features...")
    fm = fm.sort_values('Time').copy()
    fm = fm.set_index('Time')

    fuel_cols = ['Solar', 'Wind', 'Natural Gas', 'Imports', 'Nuclear', 'Large Hydro', 'Batteries']
    fm_roll = pd.DataFrame(index=fm.index)
    for col in fuel_cols:
        shifted = fm[col].shift(1)
        fm_roll[f'{col.lower().replace(" ", "_")}_mw_60m_mean'] = shifted.rolling('60min').mean()

    fm_roll = fm_roll.reset_index()
    # Join on Time (system-wide, same for all zones)
    out = features.merge(fm_roll, on='Time', how='left')
    print(f"  After fuel mix join: {out.shape}")
    return out


def add_cross_zone_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add features that span multiple zones (LMP spreads, max-across-zones)."""
    print("Adding cross-zone features...")
    # Pivot LMP to one column per zone
    pivot = features.pivot_table(index='Time', columns='zone', values='LMP', aggfunc='first')
    pivot.columns = [f'lmp_{c}' for c in pivot.columns]

    # Spreads
    pivot['lmp_spread_np_sp'] = pivot['lmp_NP15'] - pivot['lmp_SP15']
    pivot['lmp_spread_np_zp'] = pivot['lmp_NP15'] - pivot['lmp_ZP26']
    pivot['lmp_spread_sp_zp'] = pivot['lmp_SP15'] - pivot['lmp_ZP26']
    pivot['lmp_max_across_zones'] = pivot[['lmp_NP15', 'lmp_SP15', 'lmp_ZP26']].max(axis=1)
    pivot['lmp_min_across_zones'] = pivot[['lmp_NP15', 'lmp_SP15', 'lmp_ZP26']].min(axis=1)
    pivot['lmp_mean_across_zones'] = pivot[['lmp_NP15', 'lmp_SP15', 'lmp_ZP26']].mean(axis=1)

    # Rolling max across zones in last 60min (shift(1) for no leakage)
    pivot['lmp_max_across_zones_60m'] = pivot['lmp_max_across_zones'].shift(1).rolling('60min').max()

    # Drop per-zone LMP columns (we already have them per-zone in features)
    pivot = pivot.drop(columns=['lmp_NP15', 'lmp_SP15', 'lmp_ZP26'])

    pivot = pivot.reset_index()
    out = features.merge(pivot, on='Time', how='left')
    print(f"  After cross-zone join: {out.shape}")
    return out


def winsorize_features(features: pd.DataFrame, exclude_cols: list) -> pd.DataFrame:
    """Winsorize numeric features at 0.1% / 99.9% percentiles.

    Per DECISIONS.md: handles extreme values from edge cases (e.g., LMP jumps
    from near-zero to high value, producing 100,000%+ pct_change).

    - All numeric features: clip at 0.1th and 99.9th percentile
    - pct_change columns: hard-clipped to ±100 (1,000,000% is not meaningful)
    - Preserves all rows; only caps magnitudes
    """
    print("Winsorizing features at 0.1% / 99.9% percentiles...")
    features = features.copy()
    n_clipped = 0
    for col in features.columns:
        if col in exclude_cols:
            continue
        if not pd.api.types.is_numeric_dtype(features[col]):
            continue
        try:
            if 'pct_change' in col:
                # Hard cap: 100x change is already extreme
                n_low = (features[col] < -100).sum()
                n_high = (features[col] > 100).sum()
                features[col] = features[col].clip(-100, 100)
                n_clipped += int(n_low + n_high)
            else:
                # Percentile cap
                q_lo = float(features[col].quantile(0.001))
                q_hi = float(features[col].quantile(0.999))
                if abs(q_hi) < 1e10 and abs(q_lo) < 1e10:
                    n_low = (features[col] < q_lo).sum()
                    n_high = (features[col] > q_hi).sum()
                    features[col] = features[col].clip(q_lo, q_hi)
                    n_clipped += int(n_low + n_high)
        except (TypeError, ValueError):
            pass
    print(f"  Values clipped: {n_clipped:,}")
    return features


def compute_target_label(features: pd.DataFrame) -> pd.DataFrame:
    """Compute continuous regression targets.

    Targets (all continuous, all 4h forward horizon):
      - lmp_target_4h: mean LMP in next 4h (level, $/MWh)
      - lmp_ratio_target_4h: mean (LMP / 4h baseline) in next 4h (ratio)
      - ghg_target_4h: mean GHG in next 4h (carbon, short tons/MWh)

    The 4h forward horizon is used because 1h is too short for meaningful
    forward-looking signal (mostly zeros for GHG; LMP forward signal is short).

    We do NOT classify (no spike_class column). Multi-class was rejected
    in DECISIONS.md in favor of regression.
    """
    print("Computing continuous regression targets...")
    WINDOW_INT = 48  # 4h × 12 (5-min intervals per hour) — for baseline computation
    FORWARD_HORIZON_LONG = 48  # 4h forward
    parts = []
    for zone, sub in features.groupby('zone', sort=False):
        sub = sub.sort_values('Time').copy()

        # Baseline for ratio target (shifted: only past data)
        baseline = sub['LMP'].rolling(WINDOW_INT, min_periods=1).mean()

        # === Forward-looking targets (4h) ===
        # LMP level: mean LMP in next 4h
        sub['lmp_target_4h'] = sub['LMP'].shift(-FORWARD_HORIZON_LONG).rolling(FORWARD_HORIZON_LONG).mean()
        # LMP ratio: mean (LMP / 4h baseline) in next 4h — directly comparable to current
        future_lmp = sub['LMP'].shift(-FORWARD_HORIZON_LONG).rolling(FORWARD_HORIZON_LONG).mean()
        future_baseline = baseline.shift(-FORWARD_HORIZON_LONG).rolling(FORWARD_HORIZON_LONG).mean()
        sub['lmp_ratio_target_4h'] = future_lmp / future_baseline.replace(0, np.nan)
        # Clip extreme ratio targets (rare cases where forward baseline is near 0)
        sub['lmp_ratio_target_4h'] = sub['lmp_ratio_target_4h'].clip(0, 10)
        # GHG: mean carbon intensity in next 4h
        sub['ghg_target_4h'] = sub['GHG'].shift(-FORWARD_HORIZON_LONG).rolling(FORWARD_HORIZON_LONG).mean()

        parts.append(sub)

    out = pd.concat(parts, ignore_index=True)
    n_lmp = out['lmp_target_4h'].notna().sum()
    n_ratio = out['lmp_ratio_target_4h'].notna().sum()
    n_ghg = out['ghg_target_4h'].notna().sum()
    print(f"  Total rows: {len(out):,}")
    print(f"  With lmp_target_4h: {n_lmp:,}")
    print(f"  With lmp_ratio_target_4h: {n_ratio:,}")
    print(f"  With ghg_target_4h: {n_ghg:,}")
    print(f"  Non-zero ghg_target_4h: {int((out['ghg_target_4h'] > 0).sum()):,}")
    return out


def compute_class_weights(features: pd.DataFrame) -> dict:
    """Inverse-frequency class weights, capped at 100x for stability."""
    counts = features['spike_class'].value_counts(normalize=True)
    weights = {}
    for cls, freq in counts.items():
        weights[int(cls)] = float(min(1.0 / freq, 100.0))
    # Normalize so class 0 has weight 1
    base = weights[0]
    weights = {k: v / base for k, v in weights.items()}
    return weights


def validate_features(features: pd.DataFrame) -> dict:
    """Sanity checks before saving."""
    print("Validating features...")
    report = {}

    # 1. Shape
    report['shape'] = list(features.shape)
    print(f"  Shape: {features.shape}")

    # 2. NaN counts per column
    nan_counts = features.isna().sum()
    report['nan_counts_per_col'] = {k: int(v) for k, v in nan_counts.items() if v > 0}
    total_nans = int(nan_counts.sum())
    report['total_nans'] = total_nans
    print(f"  Total NaN values: {total_nans:,}")

    # 3. Per-zone row counts
    zone_counts = features.groupby('zone').size()
    report['zone_counts'] = {k: int(v) for k, v in zone_counts.items()}
    print(f"  Per-zone counts: {report['zone_counts']}")

    # 4. Date range per zone
    date_ranges = features.groupby('zone').agg(
        start=('Time', 'min'),
        end=('Time', 'max')
    )
    report['date_ranges'] = {k: {'start': str(v['start']), 'end': str(v['end'])}
                             for k, v in date_ranges.iterrows()}
    print(f"  Date ranges: {report['date_ranges']}")

    # 5. Target distributions (continuous)
    if 'spike_class' in features.columns:
        target_dist = features['spike_class'].value_counts().sort_index()
        report['spike_class_dist'] = {int(k): int(v) for k, v in target_dist.items()}
        print(f"  Spike class distribution: {report['spike_class_dist']}")
        total = features['spike_class'].notna().sum()
        report['spike_class_pct'] = {k: round(v / total * 100, 2)
                                      for k, v in report['spike_class_dist'].items()}
        print(f"  Spike class pct: {report['spike_class_pct']}")

    # Continuous target stats
    for tgt in ['lmp_target_4h', 'lmp_ratio_target_4h', 'ghg_target_4h']:
        if tgt in features.columns:
            arr = features[tgt].dropna()
            report[f'{tgt}_stats'] = {
                'count': int(len(arr)),
                'mean': float(arr.mean()) if len(arr) else None,
                'std': float(arr.std()) if len(arr) else None,
                'min': float(arr.min()) if len(arr) else None,
                'p50': float(arr.median()) if len(arr) else None,
                'p95': float(arr.quantile(0.95)) if len(arr) else None,
                'p99': float(arr.quantile(0.99)) if len(arr) else None,
                'max': float(arr.max()) if len(arr) else None,
                'pct_nonzero': float((arr > 0).mean() * 100) if len(arr) else 0.0,
            }
            print(f"  {tgt}: mean={arr.mean():.2f}, p95={arr.quantile(0.95):.2f}, max={arr.max():.2f}")

    # 7. Forward target distribution
    if 'spike_class_target_1h' in features.columns:
        fwd_dist = features['spike_class_target_1h'].value_counts().sort_index()
        report['spike_class_target_1h_dist'] = {int(k): int(v) for k, v in fwd_dist.items()}

    return report


def time_split(features: pd.DataFrame):
    """Time-based train/val/test split (no shuffle)."""
    print("Splitting train/val/test by time...")
    features = features.copy()
    features['split'] = np.where(
        features['Time'] < TRAIN_END, 'train',
        np.where(features['Time'] < VAL_END, 'val', 'test')
    )
    counts = features['split'].value_counts()
    print(f"  Split sizes: {counts.to_dict()}")
    return features


def main():
    lmp, fm, dc = load_data()

    # Build per-zone features
    features = add_lmp_features(lmp)
    features = add_fuel_mix_features(features, fm)
    features = add_cross_zone_features(features)

    # Winsorize (outlier handling)
    exclude = ['Time', 'zone', 'LMP', 'GHG',
               'lmp_target_4h', 'lmp_ratio_target_4h', 'ghg_target_4h']
    features = winsorize_features(features, exclude_cols=exclude)

    # Compute targets (do this AFTER winsorize so target isn't clipped)
    features = compute_target_label(features)

    # Class weights (use only training-period class frequencies for realistic weights)
    # Class weights are no longer needed for regression but keep for reference
    if 'spike_class' in features.columns:
        train_mask = features['Time'] < TRAIN_END
        class_weights = compute_class_weights(features[train_mask])
        print(f"  Class weights (legacy): {class_weights}")
    else:
        class_weights = {}
        print("  No class weights (regression mode)")

    # Validate
    report = validate_features(features)
    report['class_weights'] = class_weights

    # Save features (full, with all rows)
    out_path = PROCESSED / 'features_offline.parquet'
    features.to_parquet(out_path)
    print(f"\nSaved features: {out_path} ({os.path.getsize(out_path)/1024/1024:.1f} MB)")
    # Time split + save train/val/test
    features = time_split(features)
    for split in ['train', 'val', 'test']:
        sub = features[features['split'] == split].drop(columns=['split'])
        path = PROCESSED / f'{split}.parquet'
        sub.to_parquet(path)
        print(f"  {split}: {sub.shape} → {path} ({os.path.getsize(path)/1024/1024:.1f} MB)")

    # Save schema + class weights + validation report
    schema = {
        'feature_columns': [c for c in features.columns
                            if c not in ['Time', 'zone', 'split',
                                         'LMP', 'GHG',
                                         'lmp_target_4h', 'lmp_ratio_target_4h', 'ghg_target_4h']],
        'target_columns': ['lmp_target_4h', 'lmp_ratio_target_4h', 'ghg_target_4h'],
        'identifier_columns': ['Time', 'zone'],
        'spike_thresholds': SPIKE_THRESHOLDS,
        'spike_window': SPIKE_WINDOW,
        'train_end': str(TRAIN_END),
        'val_end': str(VAL_END),
        'winsorize_percentiles': '0.1% / 99.9%',
        'pct_change_hard_clip': 100,
    }
    with open(ARTIFACTS / 'feature_schema.json', 'w') as f:
        json.dump(schema, f, indent=2)
    with open(ARTIFACTS / 'class_weights.json', 'w') as f:
        json.dump(class_weights, f, indent=2)
    with open(ARTIFACTS / 'feature_validation_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print("\n✓ Feature pipeline complete")
    print(f"  features_offline.parquet: {features.shape[0]:,} rows × {features.shape[1]} cols")
    print(f"  Feature columns: {len(schema['feature_columns'])}")
    print(f"  Target columns: {schema['target_columns']}")


if __name__ == '__main__':
    main()
