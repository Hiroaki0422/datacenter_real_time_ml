"""
Model A: XGBoost regression for LMP ratio.

Predicts mean(LMP / 4h baseline) for the next 4h, given current grid state.
Target: lmp_ratio_target_4h (continuous).

Output:
  - models/v0.1/lmp_ratio.json (XGBoost model)
  - artifacts/eval_lmp.json (metrics)
  - artifacts/lmp_feature_importance.csv
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path('/root/project/dc_real_time')
PROCESSED = PROJECT_ROOT / 'data' / 'processed'
ARTIFACTS = PROJECT_ROOT / 'artifacts'
MODELS = PROJECT_ROOT / 'models' / 'v0.1'
MODELS.mkdir(parents=True, exist_ok=True)


def load_data():
    return (
        pd.read_parquet(PROCESSED / 'train.parquet'),
        pd.read_parquet(PROCESSED / 'val.parquet'),
        pd.read_parquet(PROCESSED / 'test.parquet'),
    )


def get_feature_cols(df):
    exclude = {'Time', 'zone', 'split',
               'LMP', 'GHG',
               'lmp_target_4h', 'lmp_ratio_target_4h', 'ghg_target_4h'}
    return [c for c in df.columns if c not in exclude]


def encode_zone(train, val, test):
    zones = ['NP15', 'SP15', 'ZP26']
    for z in zones:
        for df in [train, val, test]:
            df[f'zone_{z}'] = (df['zone'] == z).astype(int)
    return train, val, test, [f'zone_{z}' for z in zones]


def winsorize_dfs(dfs, exclude):
    """Apply same winsorize logic to all dataframes (using train percentiles for consistency)."""
    # Compute percentiles from train
    train = dfs[0]
    bounds = {}
    for col in train.columns:
        if col in exclude or not pd.api.types.is_numeric_dtype(train[col]):
            continue
        if 'pct_change' in col:
            bounds[col] = (-100, 100)
        else:
            try:
                q_lo = float(train[col].quantile(0.001))
                q_hi = float(train[col].quantile(0.999))
                if abs(q_hi) < 1e10 and abs(q_lo) < 1e10:
                    bounds[col] = (q_lo, q_hi)
            except (TypeError, ValueError):
                pass

    out = []
    for df in dfs:
        df = df.copy()
        for col, (lo, hi) in bounds.items():
            if col in df.columns:
                df[col] = df[col].clip(lo, hi)
        out.append(df)
    return out


def main():
    print("=" * 60)
    print("MODEL A: XGBoost regression for LMP ratio (4h target)")
    print("=" * 60)

    train, val, test = load_data()
    train, val, test, zone_cols = encode_zone(train, val, test)

    feature_cols = get_feature_cols(train) + zone_cols
    print(f"Feature columns: {len(feature_cols)}")

    target_col = 'lmp_ratio_target_4h'

    # Drop NaN target
    train = train.dropna(subset=[target_col])
    val = val.dropna(subset=[target_col])
    test = test.dropna(subset=[target_col])
    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    # Winsorize features
    exclude = ['Time', 'zone', 'LMP', 'GHG',
               'lmp_target_4h', 'lmp_ratio_target_4h', 'ghg_target_4h'] + zone_cols
    train, val, test = winsorize_dfs([train, val, test], exclude)

    # Prepare X, y
    X_train = train[feature_cols].fillna(-999).values.astype(np.float32)
    X_val = val[feature_cols].fillna(-999).values.astype(np.float32)
    X_test = test[feature_cols].fillna(-999).values.astype(np.float32)
    y_train = train[target_col].values.astype(np.float32)
    y_val = val[target_col].values.astype(np.float32)
    y_test = test[target_col].values.astype(np.float32)

    print(f"\nTarget stats (train):")
    print(f"  Mean: {y_train.mean():.3f}, Std: {y_train.std():.3f}")
    print(f"  Min: {y_train.min():.3f}, Max: {y_train.max():.3f}")

    # XGBoost regressor
    print("\nTraining XGBoost regressor...")
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
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Best val RMSE: {model.best_score:.4f}")

    # Save model
    model_path = MODELS / 'lmp_ratio.json'
    model.save_model(model_path)
    print(f"  Saved: {model_path}")

    # Predict
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Metrics
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)

    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    print("\n--- Validation ---")
    print(f"  MAE:  {val_mae:.4f}")
    print(f"  RMSE: {val_rmse:.4f}")
    print(f"  R²:   {val_r2:.4f}")

    print("\n--- Test ---")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  R²:   {test_r2:.4f}")

    # Load baseline for comparison
    with open(ARTIFACTS / 'baseline_metrics.json') as f:
        bl = json.load(f)
    if 'lmp_ratio_target_4h' in bl:
        bl_r2 = bl['lmp_ratio_target_4h']['hour_mean']['r2']
        bl_mae = bl['lmp_ratio_target_4h']['hour_mean']['mae']
        print(f"\nBaseline (hour-mean) R²: {bl_r2:.4f}, MAE: {bl_mae:.4f}")
        improvement = val_r2 - bl_r2
        print(f"Improvement over baseline: {improvement:+.4f} R²")

    # Feature importance
    importance = model.feature_importances_
    fi_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    fi_df.to_csv(ARTIFACTS / 'lmp_feature_importance.csv', index=False)
    print(f"\nTop 15 features by importance:")
    print(fi_df.head(15).to_string())

    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 8))
    top20 = fi_df.head(20).iloc[::-1]
    ax.barh(top20['feature'], top20['importance'])
    ax.set_xlabel('Importance')
    ax.set_title('Top 20 Features — Model A (LMP ratio regression)')
    plt.tight_layout()
    plt.savefig(ARTIFACTS / 'lmp_feature_importance.png', dpi=100)
    print(f"  Feature importance plot: {ARTIFACTS / 'lmp_feature_importance.png'}")

    # Plot predicted vs actual
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, y_t, y_p, title in [
        (axes[0], y_val, y_val_pred, 'Validation'),
        (axes[1], y_test, y_test_pred, 'Test')
    ]:
        ax.scatter(y_t, y_p, alpha=0.1, s=1)
        ax.plot([0, max(y_t.max(), 10)], [0, max(y_t.max(), 10)], 'r--', linewidth=1)
        ax.set_xlabel('Actual LMP ratio')
        ax.set_ylabel('Predicted LMP ratio')
        ax.set_title(f'{title} (R² = {r2_score(y_t, y_p):.3f})')
        ax.set_xlim(0, max(y_t.max(), 10))
        ax.set_ylim(0, max(y_p.max(), 10))
    plt.tight_layout()
    plt.savefig(ARTIFACTS / 'lmp_predicted_vs_actual.png', dpi=100)
    print(f"  Predicted vs actual: {ARTIFACTS / 'lmp_predicted_vs_actual.png'}")

    # Save metrics
    metrics = {
        'model': 'xgb_regression_lmp_ratio_4h',
        'n_features': len(feature_cols),
        'best_iteration': int(model.best_iteration),
        'val': {'mae': val_mae, 'rmse': val_rmse, 'r2': val_r2},
        'test': {'mae': test_mae, 'rmse': test_rmse, 'r2': test_r2},
        'baseline_hour_mean': bl.get('lmp_ratio_target_4h', {}),
    }
    with open(ARTIFACTS / 'eval_lmp.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {ARTIFACTS / 'eval_lmp.json'}")


if __name__ == '__main__':
    main()
