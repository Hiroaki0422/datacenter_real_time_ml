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

PROJECT_ROOT = Path('/root/project/dc_real_time')
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


def compute_target_label(features: pd.DataFrame) -> pd.DataFrame:
    """Compute multi-class spike label per zone per timestamp.

    Important: this uses RAW LMP (not shifted) because we want the *current* state
    as the target. Features should be shifted to prevent leakage.

    Baseline is computed over a 4-hour rolling window. At 5-min granularity,
    4h = 48 intervals.
    """
    print("Computing spike class target...")
    WINDOW_INT = 48  # 4h × 12 (5-min intervals per hour)
    parts = []
    for zone, sub in features.groupby('zone', sort=False):
        sub = sub.sort_values('Time').copy()
        # Use integer-based rolling on the raw LMP (not shifted) to compute the
        # target ratio. Min periods = 1 so the first 47 intervals get a baseline.
        baseline = sub['LMP'].rolling(WINDOW_INT, min_periods=1).mean()
        ratio = sub['LMP'] / baseline
        labels = pd.cut(ratio,
                        bins=[0, *SPIKE_THRESHOLDS, float('inf')],
                        labels=[0, 1, 2, 3],
                        right=False)
        sub['spike_class'] = labels.astype('Int64')
        # Forward-looking target: max spike class in next 12 intervals (1h ahead)
        sub['spike_class_target_1h'] = sub['spike_class'].shift(-12).rolling(12).max()
        sub['spike_class_target_1h'] = sub['spike_class_target_1h'].astype('Int64')
        # Carbon target: mean GHG in next 12 intervals
        sub['ghg_target_1h'] = sub['GHG'].shift(-12).rolling(12).mean()
        parts.append(sub)

    out = pd.concat(parts, ignore_index=True)
    n_with_target = out['spike_class_target_1h'].notna().sum()
    print(f"  Total rows: {len(out):,}, with 1h target: {n_with_target:,}")
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

    # 5. Target distribution
    target_dist = features['spike_class'].value_counts().sort_index()
    report['spike_class_dist'] = {int(k): int(v) for k, v in target_dist.items()}
    print(f"  Spike class distribution: {report['spike_class_dist']}")

    # 6. Class distribution as percent
    total = features['spike_class'].notna().sum()
    report['spike_class_pct'] = {k: round(v / total * 100, 2)
                                  for k, v in report['spike_class_dist'].items()}
    print(f"  Spike class pct: {report['spike_class_pct']}")

    # 7. Forward target distribution
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

    # Compute targets
    features = compute_target_label(features)

    # Class weights (use only training-period class frequencies for realistic weights)
    train_mask = features['Time'] < TRAIN_END
    class_weights = compute_class_weights(features[train_mask])
    print(f"  Class weights: {class_weights}")

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
                                         'LMP', 'spike_class', 'spike_class_target_1h',
                                         'ghg_target_1h', 'GHG']],
        'target_columns': ['spike_class', 'spike_class_target_1h', 'ghg_target_1h'],
        'identifier_columns': ['Time', 'zone'],
        'spike_thresholds': SPIKE_THRESHOLDS,
        'spike_window': SPIKE_WINDOW,
        'train_end': str(TRAIN_END),
        'val_end': str(VAL_END),
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
    print(f"  Class weights: {class_weights}")


if __name__ == '__main__':
    main()
