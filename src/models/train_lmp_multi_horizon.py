"""
Multi-horizon LMP ratio training.

Trains XGBoost regressor at 4 forward horizons: 30, 60, 120, 240 minutes
(30m, 1h, 2h, 4h — the user's "predict averages" horizons).
The 5-min and 15-min horizons are dropped — they aren't averages.

Saves ALL 4 models to models/{version}/:
  - lmp_ratio_30m.json
  - lmp_ratio_1h.json
  - lmp_ratio_2h.json
  - lmp_ratio_4h.json

Also writes:
  - artifacts/lmp_horizon_comparison.csv (all 4 horizons)
  - artifacts/eval_lmp_multi_horizon.json (per-horizon metrics)

Usage:
  python -m src.models.train_lmp_multi_horizon [--version v0.2]
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

# The 4 horizons we save as separate models (user spec: 30m, 1h, 2h, 4h averages)
HORIZONS_MIN = [30, 60, 120, 240]
HORIZON_LABELS = {30: '30m', 60: '1h', 120: '2h', 240: '4h'}


def load_data():
    """Load pre-built features and add per-horizon targets."""
    # Re-use the existing features_offline.parquet
    features = pd.read_parquet(PROCESSED / 'features_offline.parquet')
    lmp = pd.read_parquet(PROCESSED / 'caiso_lmp_1y.parquet')
    return features, lmp


def add_horizon_target(features: pd.DataFrame, lmp: pd.DataFrame, horizon_min: int) -> pd.DataFrame:
    """Add the LMP ratio target at the given forward horizon.

    For each (zone, time), compute mean of (LMP / 4h baseline) in next `horizon_int` intervals.
    """
    horizon_int = horizon_min // 5
    target_col = f'lmp_ratio_target_{horizon_min}m'

    # Compute 4h baseline for the raw LMP
    parts = []
    for zone in ['NP15', 'SP15', 'ZP26']:
        zone_full = f'TH_{zone}_GEN-APND'
        sub = lmp[lmp['Location'] == zone_full].sort_values('Time').set_index('Time')
        lmp_s = sub['LMP']
        baseline = lmp_s.rolling(48, min_periods=1).mean()

        # Forward ratio: mean (LMP / baseline) in next N intervals
        future_lmp = lmp_s.shift(-horizon_int).rolling(horizon_int).mean()
        future_baseline = baseline.shift(-horizon_int).rolling(horizon_int).mean()
        ratio = future_lmp / future_baseline.replace(0, np.nan)
        ratio = ratio.clip(0, 10)  # handle near-zero baselines

        out = pd.DataFrame({
            'Time': sub.index,
            'zone': zone,
            target_col: ratio.values
        })
        parts.append(out)

    target_df = pd.concat(parts, ignore_index=True)
    return features.merge(target_df, on=['Time', 'zone'], how='left')


def get_feature_cols(df, target_col):
    exclude = {'Time', 'zone', 'split', 'LMP', 'GHG',
               'lmp_target_4h', 'lmp_ratio_target_4h', 'ghg_target_4h'}
    # Also exclude any other target columns to keep this clean
    exclude.update(c for c in df.columns if c.startswith('lmp_ratio_target_') or c.startswith('lmp_target_') or c.startswith('ghg_target_'))
    return [c for c in df.columns if c not in exclude]


def encode_zone(train, val, test):
    zones = ['NP15', 'SP15', 'ZP26']
    for z in zones:
        for df in [train, val, test]:
            df[f'zone_{z}'] = (df['zone'] == z).astype(int)
    return train, val, test, [f'zone_{z}' for z in zones]


def main(version: str = 'v0.1') -> dict:
    """Train LMP ratio models at 4 horizons (30m, 1h, 2h, 4h) and save all to models/{version}/.

    Returns: {
        'version': str,
        'horizons': {
            '30m': {'val_r2': ..., 'test_r2': ..., 'model_path': 'models/v0.1/lmp_ratio_30m.json'},
            '1h': {...},
            '2h': {...},
            '4h': {...},
        },
        'n_train_total': int,
    }
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default=version, help='model version dir name (e.g. v0.2)')
    # Only parse args if invoked as a script (sys.argv[0] ends in train_lmp_multi_horizon.py
    # or .py, but if imported from another module, skip parsing to avoid stealing parent args).
    import sys as _sys
    if _sys.argv and 'train_lmp_multi_horizon' in _sys.argv[0]:
        args = parser.parse_args()
    else:
        # Called as a library; try to parse only known args, fall back to default
        try:
            args, _ = parser.parse_known_args()
        except SystemExit:
            args = parser.parse_args([])  # all defaults
    version = args.version

    models_dir = PROJECT_ROOT / 'models' / version
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"LMP RATIO: Multi-Horizon Training (version={version})")
    print(f"Horizons (min): {HORIZONS_MIN}")
    print(f"Models dir: {models_dir}")
    print("=" * 60)

    features, lmp = load_data()
    print(f"Features rows: {len(features):,}")

    results = []
    horizon_metrics = {}

    for horizon in HORIZONS_MIN:
        label = HORIZON_LABELS[horizon]
        print(f"\n{'='*60}")
        print(f"HORIZON = {horizon} min ({label})")
        print(f"{'='*60}")

        # Add target at this horizon
        target_col = f'lmp_ratio_target_{horizon}m'
        features_h = add_horizon_target(features, lmp, horizon)
        # Restrict to rows where target is defined
        sub = features_h.dropna(subset=[target_col]).copy()
        # Use the same time split as before
        train_end = pd.Timestamp('2026-01-01', tz='US/Pacific')
        val_end = pd.Timestamp('2026-04-01', tz='US/Pacific')
        train = sub[sub['Time'] < train_end].copy()
        val = sub[(sub['Time'] >= train_end) & (sub['Time'] < val_end)].copy()
        # Use last 20% of available data as test
        test_start = sub['Time'].quantile(0.8)
        test = sub[sub['Time'] >= test_start].copy()

        # Encode zone
        train, val, test, zone_cols = encode_zone(train, val, test)
        feature_cols = get_feature_cols(sub, target_col) + zone_cols

        print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
        print(f"  Train target mean: {train[target_col].mean():.3f}, std: {train[target_col].std():.3f}")

        # Drop NaN target
        train = train.dropna(subset=[target_col])
        val = val.dropna(subset=[target_col])
        test = test.dropna(subset=[target_col])

        # X, y
        X_train = train[feature_cols].fillna(-999).values.astype(np.float32)
        X_val = val[feature_cols].fillna(-999).values.astype(np.float32)
        X_test = test[feature_cols].fillna(-999).values.astype(np.float32)
        y_train = train[target_col].values.astype(np.float32)
        y_val = val[target_col].values.astype(np.float32)
        y_test = test[target_col].values.astype(np.float32)

        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            eval_metric='rmse',
            max_depth=6,
            learning_rate=0.05,
            n_estimators=500,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
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

        # Predict-mean baseline
        mean_pred = float(y_train.mean())
        val_mae_mean = float(np.abs(y_val - mean_pred).mean())
        val_rmse_mean = float(np.sqrt(((y_val - mean_pred)**2).mean()))
        val_r2_mean = r2_score(y_val, np.full_like(y_val, mean_pred))

        print(f"\n  Val:   MAE={val_mae:.4f}, RMSE={val_rmse:.4f}, R²={val_r2:.4f}")
        print(f"  Test:  MAE={test_mae:.4f}, RMSE={test_rmse:.4f}, R²={test_r2:.4f}")
        print(f"  Baseline (predict mean): Val MAE={val_mae_mean:.4f}, R²={val_r2_mean:.4f}")
        print(f"  Best iter: {model.best_iteration}")

        # Save model for this horizon
        model_path = models_dir / f'lmp_ratio_{label}.json'
        model.save_model(model_path)
        print(f"  ★ Saved {model_path.name}")

        results.append({
            'horizon_min': horizon,
            'horizon_label': label,
            'best_iteration': int(model.best_iteration),
            'val_mae': val_mae, 'val_rmse': val_rmse, 'val_r2': val_r2,
            'test_mae': test_mae, 'test_rmse': test_rmse, 'test_r2': test_r2,
            'baseline_predict_mean_mae': val_mae_mean,
            'baseline_predict_mean_r2': val_r2_mean,
            'n_train': len(train), 'n_val': len(val), 'n_test': len(test),
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

    # Save comparison
    df = pd.DataFrame(results)
    df.to_csv(ARTIFACTS / 'lmp_horizon_comparison.csv', index=False)
    print(f"\n\n{'='*60}")
    print("HORIZON COMPARISON (sorted by val R²)")
    print(f"{'='*60}")
    print(df.sort_values('val_r2', ascending=False).to_string(index=False))

    best = df.loc[df['val_r2'].idxmax()]
    print(f"\n★ Best horizon: {int(best['horizon_min'])} min ({best['horizon_label']})")
    print(f"  Val R²: {best['val_r2']:.4f}, MAE: {best['val_mae']:.4f}")
    print(f"  Test R²: {best['test_r2']:.4f}, MAE: {best['test_mae']:.4f}")

    summary = {
        'experiment': 'lmp_ratio_multi_horizon',
        'version': version,
        'horizons_tested': HORIZONS_MIN,
        'horizon_labels': HORIZON_LABELS,
        'best_horizon_min': int(best['horizon_min']),
        'best_horizon_label': best['horizon_label'],
        'best_val_r2': float(best['val_r2']),
        'best_test_r2': float(best['test_r2']),
        'all_results': results,
    }
    with open(ARTIFACTS / 'eval_lmp_multi_horizon.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {ARTIFACTS / 'eval_lmp_multi_horizon.json'}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(df['horizon_min'], df['val_r2'], 'o-', label='Val R²')
    axes[0].plot(df['horizon_min'], df['test_r2'], 's-', label='Test R²')
    axes[0].set_xlabel('Forward horizon (min)')
    axes[0].set_ylabel('R²')
    axes[0].set_title('LMP Model: R² vs Forward Horizon')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)

    axes[1].plot(df['horizon_min'], df['val_mae'], 'o-', label='Val MAE')
    axes[1].plot(df['horizon_min'], df['test_mae'], 's-', label='Test MAE')
    axes[1].plot(df['horizon_min'], df['baseline_predict_mean_mae'], '^--', label='Baseline (predict mean)', alpha=0.5)
    axes[1].set_xlabel('Forward horizon (min)')
    axes[1].set_ylabel('MAE (LMP ratio)')
    axes[1].set_title('LMP Model: MAE vs Forward Horizon')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ARTIFACTS / 'lmp_horizon_comparison.png', dpi=100)
    print(f"  Plot: {ARTIFACTS / 'lmp_horizon_comparison.png'}")

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
