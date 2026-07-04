"""
Baseline models for dc_real_time.

These are the floor — any XGBoost model must beat them.
- For LMP ratio: hour-of-day, day-of-week mean per zone
- For GHG: predict 0 (since 90% of intervals are clean)
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

PROCESSED = Path('/root/project/dc_real_time/data/processed')
ARTIFACTS = Path('/root/project/dc_real_time/artifacts')


def fit_persistence_regression(train: pd.DataFrame, target_col: str):
    """Persistence baseline for regression: predict mean of last value per (zone, dow, hour)."""
    df = train.copy().sort_values('Time')
    df['dow'] = df['Time'].dt.dayofweek
    df['hour'] = df['Time'].dt.hour
    last = df.groupby(['zone', 'dow', 'hour'])[target_col].mean().reset_index()
    last.columns = ['zone', 'dow', 'hour', 'persistence_pred']
    return last


def predict_persistence_regression(persistence_df, eval_df, target_col):
    df = eval_df.copy()
    df['dow'] = df['Time'].dt.dayofweek
    df['hour'] = df['Time'].dt.hour
    out = df.merge(persistence_df, on=['zone', 'dow', 'hour'], how='left')
    out['persistence_pred'] = out['persistence_pred'].fillna(eval_df[target_col].mean() if target_col in eval_df.columns else 0)
    return out[['persistence_pred']]


def fit_hour_mean(train: pd.DataFrame, target_col: str):
    df = train.copy()
    df['hour'] = df['Time'].dt.hour
    return df.groupby(['zone', 'hour'])[target_col].mean().reset_index()


def predict_hour_mean(means, eval_df, target_col):
    df = eval_df.copy()
    df['hour'] = df['Time'].dt.hour
    out = df.merge(means, on=['zone', 'hour'], how='left', suffixes=('_true', '_mean'))
    pred_col = f'{target_col}_pred'
    out[pred_col] = out[f'{target_col}_mean'].fillna(0)
    return out[[pred_col]]


def regression_metrics(y_true, y_pred):
    """Compute MAE, RMSE, R²."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    mae = np.abs(y_true - y_pred).mean()
    rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return float(mae), float(rmse), float(r2)


def main():
    print("=" * 60)
    print("BASELINES (regression)")
    print("=" * 60)

    train = pd.read_parquet(PROCESSED / 'train.parquet')
    val = pd.read_parquet(PROCESSED / 'val.parquet')

    metrics = {}

    # === LMP ratio (4h target) ===
    print("\n--- LMP ratio 4h-target (regression) ---")
    for tgt, label in [('lmp_ratio_target_4h', 'LMP ratio'),
                        ('lmp_target_4h', 'LMP level ($/MWh)'),
                        ('ghg_target_4h', 'GHG (short tons/MWh)')]:
        train_clean = train.dropna(subset=[tgt])
        val_clean = val.dropna(subset=[tgt])
        if len(train_clean) == 0 or len(val_clean) == 0:
            print(f"\n  {label}: SKIP (no data)")
            continue

        print(f"\n  {label}:")
        print(f"    Train: {len(train_clean):,} | Val: {len(val_clean):,}")

        # Baseline 1: hour-of-day mean
        means = fit_hour_mean(train_clean, tgt)
        pred = predict_hour_mean(means, val_clean, tgt)
        y_true = val_clean[tgt].values
        y_pred = pred[f'{tgt}_pred'].values
        mae, rmse, r2 = regression_metrics(y_true, y_pred)
        print(f"    Hour-mean: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

        # Baseline 2: predict 0
        y_pred_zero = np.zeros_like(y_true)
        mae0, rmse0, r20 = regression_metrics(y_true, y_pred_zero)
        print(f"    Predict-0: MAE={mae0:.4f}, RMSE={rmse0:.4f}, R²={r20:.4f}")

        # Baseline 3: predict mean
        mean_val = float(train_clean[tgt].mean())
        y_pred_mean = np.full_like(y_true, mean_val)
        mae_m, rmse_m, r2_m = regression_metrics(y_true, y_pred_mean)
        print(f"    Predict-mean ({mean_val:.3f}): MAE={mae_m:.4f}, RMSE={rmse_m:.4f}, R²={r2_m:.4f}")

        metrics[tgt] = {
            'n_val': int(len(val_clean)),
            'n_train': int(len(train_clean)),
            'hour_mean': {'mae': mae, 'rmse': rmse, 'r2': r2},
            'predict_zero': {'mae': mae0, 'rmse': rmse0, 'r2': r20},
            'predict_mean': {'mae': mae_m, 'rmse': rmse_m, 'r2': r2_m, 'value': mean_val},
        }

    with open(ARTIFACTS / 'baseline_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved baseline metrics to {ARTIFACTS / 'baseline_metrics.json'}")


if __name__ == '__main__':
    main()
