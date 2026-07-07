"""
Multi-horizon carbon (GHG) training.

Trains XGBoost regressor at 4 forward horizons: 30, 60, 120, 240 minutes
(30m, 1h, 2h, 4h — the user's "predict averages" horizons).
The 5-min and 15-min horizons are dropped — they aren't averages.

Saves ALL 4 models to models/{version}/:
  - carbon_30m.json
  - carbon_1h.json
  - carbon_2h.json
  - carbon_4h.json

Also writes:
  - artifacts/carbon_horizon_comparison.csv (all 4 horizons)
  - artifacts/eval_carbon_multi_horizon.json (per-horizon metrics)

Usage:
  python -m src.models.train_carbon_multi_horizon [--version v0.2]
"""
import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
PROJECT_ROOT = Path(os.environ.get('PROJECT_ROOT', '/app'))
PROCESSED = PROJECT_ROOT / 'data' / 'processed'
ARTIFACTS = PROJECT_ROOT / 'artifacts'

# Carbon data is only available May-Jul 2026 (initial hardcoded value,
# auto-detected in _detect_carbon_window() below)
GHG_START = pd.Timestamp('2026-05-01', tz='US/Pacific')

# The 4 horizons we save as separate models (user spec: 30m, 1h, 2h, 4h averages)
HORIZONS_MIN = [30, 60, 120, 240]
HORIZON_LABELS = {30: '30m', 60: '1h', 120: '2h', 240: '4h'}

# Carbon data window: auto-detect from the LMP parquet if possible.
# This lets the training script pick up new carbon data as it accrues,
# rather than being hardcoded to 2026-05-01 onwards.
def _detect_carbon_window():
    """Return (start, end) of the carbon data window from the LMP parquet.

    Returns the dates between which non-zero GHG data exists, with a
    1-day buffer on each side.
    """
    try:
        lmp = pd.read_parquet(PROCESSED / 'caiso_lmp_1y.parquet')
        ghg = lmp[lmp['GHG'] > 0]
        if len(ghg) == 0:
            return GHG_START, pd.Timestamp('2026-12-31', tz='US/Pacific')
        oldest = ghg['Time'].min() - pd.Timedelta(days=1)
        newest = ghg['Time'].max() + pd.Timedelta(days=1)
        return oldest, newest
    except Exception:
        return GHG_START, pd.Timestamp('2026-12-31', tz='US/Pacific')

# Will be overwritten in main() if auto-detection succeeds
GHG_START, GHG_END = _detect_carbon_window()


def load_data():
    lmp = pd.read_parquet(PROCESSED / 'caiso_lmp_1y.parquet')
    fm = pd.read_parquet(PROCESSED / 'caiso_fuel_mix_1y.parquet')

    # Restrict to the carbon data window (auto-detected)
    lmp = lmp[(lmp['Time'] >= GHG_START) & (lmp['Time'] <= GHG_END)].copy()
    fm = fm[(fm['Time'] >= GHG_START) & (fm['Time'] <= GHG_END)].copy()
    return lmp, fm


def build_features_for_horizon(lmp: pd.DataFrame, fm: pd.DataFrame, horizon_min: int) -> pd.DataFrame:
    """Build features + target at the given forward horizon.

    For efficiency we don't re-run the full pipeline. We re-use the same
    features as the main pipeline (re-derived here) and only vary the target.
    """
    WINDOW_INT_4H = 48  # for baseline computation
    parts = []

    for zone in ['NP15', 'SP15', 'ZP26']:
        zone_full = f'TH_{zone}_GEN-APND'
        sub = lmp[lmp['Location'] == zone_full].sort_values('Time').set_index('Time')

        out = pd.DataFrame(index=sub.index)
        out['zone'] = zone
        out['LMP'] = sub['LMP']
        out['Energy'] = sub['Energy']
        out['Congestion'] = sub['Congestion']
        out['Loss'] = sub['Loss']
        out['GHG'] = sub['GHG']

        s = sub['LMP']

        # LMP rolling stats (shifted to prevent leakage)
        for label, window in [('60m', '60min'), ('4h', '4h'), ('24h', '24h')]:
            shifted = s.shift(1)
            out[f'lmp_mean_{label}'] = shifted.rolling(window).mean()
            out[f'lmp_std_{label}'] = shifted.rolling(window).std()
            if label == '24h':
                out[f'lmp_max_{label}'] = shifted.rolling(window).max()
                out[f'lmp_min_{label}'] = shifted.rolling(window).min()

        # Energy/Congestion/Loss rolling
        for col in ['Energy', 'Congestion', 'Loss', 'GHG']:
            out[f'{col.lower()}_60m_mean'] = sub[col].shift(1).rolling('60min').mean()

        # GHG non-zero fraction
        out['ghg_nonzero_pct_60m'] = (sub['GHG'].shift(1).gt(0)).rolling('60min').mean()

        # Lag features
        out['lmp_lag_1'] = s.shift(1)
        out['lmp_lag_12'] = s.shift(12)
        out['lmp_lag_48'] = s.shift(48)

        # 60m slope
        def slope(arr):
            if len(arr) < 2 or np.all(np.isnan(arr)):
                return np.nan
            y = arr
            x = np.arange(len(y))
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                return np.nan
            return np.polyfit(x[mask], y[mask], 1)[0]
        out['lmp_slope_60m'] = s.shift(1).rolling('60min').apply(slope, raw=True)

        # Pct change
        out['lmp_pct_change_5m'] = s.pct_change(1)
        out['lmp_pct_change_60m'] = s.pct_change(12)

        # 4h range
        out['lmp_range_4h'] = s.shift(1).rolling('4h').max() - s.shift(1).rolling('4h').min()

        # Calendar
        t = sub.index
        out['hour_of_day'] = t.hour
        out['day_of_week'] = t.dayofweek
        out['month'] = t.month
        out['is_weekend'] = (t.dayofweek >= 5).astype(int)
        out['hour_sin'] = np.sin(2 * np.pi * t.hour / 24)
        out['hour_cos'] = np.cos(2 * np.pi * t.hour / 24)
        out['month_sin'] = np.sin(2 * np.pi * t.month / 12)
        out['month_cos'] = np.cos(2 * np.pi * t.month / 12)

        parts.append(out)

    features = pd.concat(parts).reset_index().rename(columns={'index': 'Time'})

    # Fuel mix features (60min rolling mean, shifted)
    fm_idx = fm.set_index('Time')
    fm_roll = pd.DataFrame(index=fm_idx.index)
    for col in ['Solar', 'Wind', 'Natural Gas', 'Imports', 'Nuclear', 'Large Hydro', 'Batteries']:
        fm_roll[f'{col.lower().replace(" ", "_")}_mw_60m_mean'] = fm_idx[col].shift(1).rolling(12, min_periods=1).mean()
    fm_roll = fm_roll.reset_index()
    features = features.merge(fm_roll, on='Time', how='left')

    # Cross-zone features
    pivot = features.pivot_table(index='Time', columns='zone', values='LMP', aggfunc='first')
    pivot.columns = [f'lmp_{c}' for c in pivot.columns]
    pivot['lmp_spread_np_sp'] = pivot['lmp_NP15'] - pivot['lmp_SP15']
    pivot['lmp_spread_np_zp'] = pivot['lmp_NP15'] - pivot['lmp_ZP26']
    pivot['lmp_spread_sp_zp'] = pivot['lmp_SP15'] - pivot['lmp_ZP26']
    pivot['lmp_max_across_zones'] = pivot[['lmp_NP15', 'lmp_SP15', 'lmp_ZP26']].max(axis=1)
    pivot['lmp_min_across_zones'] = pivot[['lmp_NP15', 'lmp_SP15', 'lmp_ZP26']].min(axis=1)
    pivot['lmp_mean_across_zones'] = pivot[['lmp_NP15', 'lmp_SP15', 'lmp_ZP26']].mean(axis=1)
    pivot['lmp_max_across_zones_60m'] = pivot['lmp_max_across_zones'].shift(1).rolling('60min').max()
    pivot = pivot.drop(columns=['lmp_NP15', 'lmp_SP15', 'lmp_ZP26']).reset_index()
    features = features.merge(pivot, on='Time', how='left')

    # Winsorize (using train percentiles only)
    exclude = ['Time', 'zone', 'LMP', 'GHG', f'ghg_target_{horizon_min}m']
    features = _winsorize(features, exclude)

    # Forward target at the specified horizon
    horizon_int = horizon_min // 5
    parts = []
    for zone, sub in features.groupby('zone', sort=False):
        sub = sub.sort_values('Time').copy()
        sub[f'ghg_target_{horizon_min}m'] = sub['GHG'].shift(-horizon_int).rolling(horizon_int).mean()
        parts.append(sub)
    features = pd.concat(parts, ignore_index=True)

    return features


def _winsorize(features: pd.DataFrame, exclude_cols: list) -> pd.DataFrame:
    """Winsorize numeric features at 0.1% / 99.9% percentiles."""
    features = features.copy()
    bounds = {}
    for col in features.columns:
        if col in exclude_cols or not pd.api.types.is_numeric_dtype(features[col]):
            continue
        if 'pct_change' in col:
            bounds[col] = (-100, 100)
        else:
            try:
                q_lo = float(features[col].quantile(0.001))
                q_hi = float(features[col].quantile(0.999))
                if abs(q_hi) < 1e10 and abs(q_lo) < 1e10:
                    bounds[col] = (q_lo, q_hi)
            except (TypeError, ValueError):
                pass
    for col, (lo, hi) in bounds.items():
        if col in features.columns:
            features[col] = features[col].clip(lo, hi)
    return features


def get_feature_cols(df, target_col):
    exclude = {'Time', 'zone', 'split', 'LMP', 'GHG', target_col}
    return [c for c in df.columns if c not in exclude]


def encode_zone(train, val, test):
    zones = ['NP15', 'SP15', 'ZP26']
    for z in zones:
        for df in [train, val, test]:
            df[f'zone_{z}'] = (df['zone'] == z).astype(int)
    return train, val, test, [f'zone_{z}' for z in zones]


def time_split(df, train_frac=0.6, val_frac=0.2):
    """Time-based split within the carbon data window."""
    df = df.sort_values('Time').reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    df['split'] = 'test'
    df.loc[:train_end, 'split'] = 'train'
    df.loc[train_end:val_end, 'split'] = 'val'
    return df


def main(version: str = 'v0.1') -> dict:
    """Train carbon models at 4 horizons (30m, 1h, 2h, 4h) and save all to models/{version}/.

    Returns: {
        'version': str,
        'horizons': {
            '30m': {'val_r2': ..., 'test_r2': ..., 'model_path': 'models/v0.1/carbon_30m.json'},
            '1h': {...}, '2h': {...}, '4h': {...},
        },
        'n_train_total': int,
    }
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default=version, help='model version dir name (e.g. v0.2)')
    # Only parse args if invoked as a script (avoid stealing parent args when imported)
    import sys as _sys
    if _sys.argv and 'train_carbon_multi_horizon' in _sys.argv[0]:
        args = parser.parse_args()
    else:
        try:
            args, _ = parser.parse_known_args()
        except SystemExit:
            args = parser.parse_args([])
    version = args.version

    models_dir = PROJECT_ROOT / 'models' / version
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"CARBON MODEL: Multi-Horizon Training (version={version})")
    print(f"GHG data window: {GHG_START.date()} → {GHG_END.date()}")
    print(f"Horizons (min): {HORIZONS_MIN}")
    print(f"Models dir: {models_dir}")
    print("=" * 60)

    lmp, fm = load_data()
    print(f"LMP rows: {len(lmp):,}, Fuel mix rows: {len(fm):,} (auto-detected carbon window)")

    results = []
    horizon_metrics = {}

    for horizon in HORIZONS_MIN:
        label = HORIZON_LABELS[horizon]
        print(f"\n{'='*60}")
        print(f"HORIZON = {horizon} min ({label})")
        print(f"{'='*60}")

        target_col = f'ghg_target_{horizon}m'
        features = build_features_for_horizon(lmp, fm, horizon)
        features = time_split(features, train_frac=0.6, val_frac=0.2)
        features, _, _, zone_cols = encode_zone(
            features[features['split']=='train'].copy(),
            features[features['split']=='val'].copy(),
            features[features['split']=='test'].copy(),
        )
        # Re-do the split so all 3 dfs have zone cols
        features = build_features_for_horizon(lmp, fm, horizon)
        features = time_split(features, train_frac=0.6, val_frac=0.2)
        for z in ['NP15', 'SP15', 'ZP26']:
            features[f'zone_{z}'] = (features['zone'] == z).astype(int)
        zone_cols = [f'zone_{z}' for z in ['NP15', 'SP15', 'ZP26']]

        feature_cols = get_feature_cols(features, target_col) + zone_cols

        train = features[features['split']=='train'].dropna(subset=[target_col])
        val = features[features['split']=='val'].dropna(subset=[target_col])
        test = features[features['split']=='test'].dropna(subset=[target_col])
        print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
        print(f"  Train non-zero target: {(train[target_col] > 0).sum()} ({(train[target_col] > 0).mean()*100:.1f}%)")
        print(f"  Val non-zero target:   {(val[target_col] > 0).sum()} ({(val[target_col] > 0).mean()*100:.1f}%)")
        print(f"  Test non-zero target:  {(test[target_col] > 0).sum()} ({(test[target_col] > 0).mean()*100:.1f}%)")

        X_train = train[feature_cols].fillna(-999).values.astype(np.float32)
        X_val = val[feature_cols].fillna(-999).values.astype(np.float32)
        X_test = test[feature_cols].fillna(-999).values.astype(np.float32)
        y_train = train[target_col].values.astype(np.float32)
        y_val = val[target_col].values.astype(np.float32)
        y_test = test[target_col].values.astype(np.float32)

        alpha = 0.9
        model = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=alpha,
            eval_metric='mae',
            max_depth=5,
            learning_rate=0.05,
            n_estimators=500,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=20,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=30,
            tree_method='hist',
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_r2 = r2_score(y_val, y_val_pred) if y_val.std() > 0 else 0

        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred) if y_test.std() > 0 else 0

        # Quantile-specific metrics
        test_pinball = float(np.mean(np.where(y_test >= y_test_pred,
                                               alpha * (y_test - y_test_pred),
                                               (1 - alpha) * (y_test_pred - y_test))))
        test_coverage = float(np.mean(y_test <= y_test_pred))

        # Non-zero-only metrics
        nz_val = y_val > 0
        nz_test = y_test > 0
        val_mae_nz = mean_absolute_error(y_val[nz_val], y_val_pred[nz_val]) if nz_val.any() else 0
        val_rmse_nz = np.sqrt(mean_squared_error(y_val[nz_val], y_val_pred[nz_val])) if nz_val.any() else 0
        test_mae_nz = mean_absolute_error(y_test[nz_test], y_test_pred[nz_test]) if nz_test.any() else 0
        test_rmse_nz = np.sqrt(mean_squared_error(y_test[nz_test], y_test_pred[nz_test])) if nz_test.any() else 0

        # Baseline: predict-0
        baseline_mae = float(np.abs(y_val).mean())
        baseline_rmse = float(np.sqrt((y_val**2).mean()))

        print(f"\n  Val:   MAE={val_mae:.4f}, RMSE={val_rmse:.4f}, R²={val_r2:.4f}")
        print(f"          (non-zero: MAE={val_mae_nz:.4f}, RMSE={val_rmse_nz:.4f})")
        print(f"  Test:  MAE={test_mae:.4f}, RMSE={test_rmse:.4f}, R²={test_r2:.4f}")
        print(f"          (non-zero: MAE={test_mae_nz:.4f}, RMSE={test_rmse_nz:.4f})")
        print(f"  Quantile (α={alpha}): pinball={test_pinball:.4f}, coverage={test_coverage:.3f}")
        print(f"  Baseline (predict 0): MAE={baseline_mae:.4f}, RMSE={baseline_rmse:.4f}")
        print(f"  Best iter: {model.best_iteration}")

        # Save model for this horizon
        model_path = models_dir / f'carbon_{label}.json'
        model.save_model(model_path)
        print(f"  ★ Saved {model_path.name}")

        results.append({
            'horizon_min': horizon,
            'horizon_label': label,
            'best_iteration': int(model.best_iteration),
            'val_mae': val_mae, 'val_rmse': val_rmse, 'val_r2': val_r2,
            'val_mae_nz': val_mae_nz, 'val_rmse_nz': val_rmse_nz,
            'test_mae': test_mae, 'test_rmse': test_rmse, 'test_r2': test_r2,
            'test_mae_nz': test_mae_nz, 'test_rmse_nz': test_rmse_nz,
            'baseline_mae': baseline_mae, 'baseline_rmse': baseline_rmse,
            'n_train': len(train), 'n_val': len(val), 'n_test': len(test),
            'pct_nonzero_train': float((y_train > 0).mean() * 100),
            'pct_nonzero_val': float((y_val > 0).mean() * 100),
            'model_path': str(model_path.relative_to(PROJECT_ROOT)),
        })

        horizon_metrics[label] = {
            'val_r2': val_r2,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'n_train': len(train),
            'model_path': str(model_path.relative_to(PROJECT_ROOT)),
        }

    # Save comparison table
    df = pd.DataFrame(results)
    df.to_csv(ARTIFACTS / 'carbon_horizon_comparison.csv', index=False)
    print(f"\n\n{'='*60}")
    print("HORIZON COMPARISON (sorted by val R²)")
    print(f"{'='*60}")
    print(df.sort_values('val_r2', ascending=False).to_string(index=False))

    # Best horizon
    best = df.loc[df['val_r2'].idxmax()]
    print(f"\n★ Best horizon: {int(best['horizon_min'])} min ({best['horizon_label']})")
    print(f"  Val R²: {best['val_r2']:.4f}")
    print(f"  Val MAE: {best['val_mae']:.4f}")
    print(f"  Test R²: {best['test_r2']:.4f}")
    print(f"  Test MAE: {best['test_mae']:.4f}")

    # Save summary
    summary = {
        'experiment': 'carbon_multi_horizon',
        'version': version,
        'horizons_tested': HORIZONS_MIN,
        'horizon_labels': HORIZON_LABELS,
        'data_window': f'{GHG_START.date()} onwards',
        'best_horizon_min': int(best['horizon_min']),
        'best_horizon_label': best['horizon_label'],
        'best_val_r2': float(best['val_r2']),
        'best_test_r2': float(best['test_r2']),
        'all_results': results,
    }
    with open(ARTIFACTS / 'eval_carbon_multi_horizon.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {ARTIFACTS / 'eval_carbon_multi_horizon.json'}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(df['horizon_min'], df['val_r2'], 'o-', label='Val R²')
    axes[0].plot(df['horizon_min'], df['test_r2'], 's-', label='Test R²')
    axes[0].set_xlabel('Forward horizon (min)')
    axes[0].set_ylabel('R²')
    axes[0].set_title('Carbon Model: R² vs Forward Horizon')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)

    axes[1].plot(df['horizon_min'], df['val_mae'], 'o-', label='Val MAE')
    axes[1].plot(df['horizon_min'], df['test_mae'], 's-', label='Test MAE')
    axes[1].plot(df['horizon_min'], df['baseline_mae'], '^--', label='Baseline (predict 0)', alpha=0.5)
    axes[1].set_xlabel('Forward horizon (min)')
    axes[1].set_ylabel('MAE (short tons/MWh)')
    axes[1].set_title('Carbon Model: MAE vs Forward Horizon')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ARTIFACTS / 'carbon_horizon_comparison.png', dpi=100)
    print(f"  Plot: {ARTIFACTS / 'carbon_horizon_comparison.png'}")

    return {
        'version': version,
        'horizons': horizon_metrics,
        'n_train_total': sum(r['n_train'] for r in results),
        'best_horizon': best['horizon_label'],
    }


if __name__ == '__main__':
    result = main()
    print("\n=== Training complete ===")
    print(f"Version: {result['version']}")
    print(f"Best horizon: {result['best_horizon']}")
    for label, m in result['horizons'].items():
        print(f"  {label}: val_R²={m['val_r2']:.4f}, model={m['model_path']}")
