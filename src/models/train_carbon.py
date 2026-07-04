"""
Model B: XGBoost regression for carbon intensity (GHG).

Predicts mean GHG (short tons CO2 / MWh) for the next 4h.

Data caveat: CAISO only publishes GHG field for recent ~90 days.
In our 1y backfill, non-zero GHG only appears in May-Jul 2026.
So train uses the full window, but mostly zero target.
We still train to learn the "rare event" signal.

Output:
  - models/v0.1/carbon.json (XGBoost model)
  - artifacts/eval_carbon.json
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
    print("MODEL B: XGBoost regression for GHG (4h target)")
    print("=" * 60)

    train, val, test = load_data()
    train, val, test, zone_cols = encode_zone(train, val, test)

    feature_cols = get_feature_cols(train) + zone_cols
    print(f"Feature columns: {len(feature_cols)}")

    target_col = 'ghg_target_4h'

    # Drop NaN target
    train = train.dropna(subset=[target_col])
    val = val.dropna(subset=[target_col])
    test = test.dropna(subset=[target_col])
    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    # Winsorize
    exclude = ['Time', 'zone', 'LMP', 'GHG',
               'lmp_target_4h', 'lmp_ratio_target_4h', 'ghg_target_4h'] + zone_cols
    train, val, test = winsorize_dfs([train, val, test], exclude)

    # X, y
    X_train = train[feature_cols].fillna(-999).values.astype(np.float32)
    X_val = val[feature_cols].fillna(-999).values.astype(np.float32)
    X_test = test[feature_cols].fillna(-999).values.astype(np.float32)
    y_train = train[target_col].values.astype(np.float32)
    y_val = val[target_col].values.astype(np.float32)
    y_test = test[target_col].values.astype(np.float32)

    print(f"\nTarget stats (train):")
    print(f"  Non-zero: {(y_train > 0).sum()} / {len(y_train)} ({(y_train > 0).mean()*100:.2f}%)")
    print(f"  Mean: {y_train.mean():.3f}, Max: {y_train.max():.3f}")

    print(f"\nTarget stats (val):")
    print(f"  Non-zero: {(y_val > 0).sum()} / {len(y_val)} ({(y_val > 0).mean()*100:.2f}%)")

    print(f"\nTarget stats (test):")
    print(f"  Non-zero: {(y_test > 0).sum()} / {len(y_test)} ({(y_test > 0).mean()*100:.2f}%)")
    print(f"  Mean: {y_test.mean():.3f}, Max: {y_test.max():.3f}")

    # Train XGBoost regressor
    print("\nTraining XGBoost regressor for GHG...")
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        max_depth=5,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20,  # higher to handle rare positives
        reg_alpha=0.5,
        reg_lambda=2.0,
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

    # Save
    model_path = MODELS / 'carbon.json'
    model.save_model(model_path)
    print(f"  Saved: {model_path}")

    # Predict
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Metrics
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred) if y_val.std() > 0 else 0

    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred) if y_test.std() > 0 else 0

    # Non-zero-only metrics (where model could actually do something)
    nz_val = y_val > 0
    nz_test = y_test > 0
    if nz_val.any():
        val_mae_nz = mean_absolute_error(y_val[nz_val], y_val_pred[nz_val])
        val_rmse_nz = np.sqrt(mean_squared_error(y_val[nz_val], y_val_pred[nz_val]))
    else:
        val_mae_nz, val_rmse_nz = 0, 0
    if nz_test.any():
        test_mae_nz = mean_absolute_error(y_test[nz_test], y_test_pred[nz_test])
        test_rmse_nz = np.sqrt(mean_squared_error(y_test[nz_test], y_test_pred[nz_test]))
    else:
        test_mae_nz, test_rmse_nz = 0, 0

    print("\n--- Validation ---")
    print(f"  All:    MAE={val_mae:.4f}, RMSE={val_rmse:.4f}, R²={val_r2:.4f}")
    print(f"  Nonzero: MAE={val_mae_nz:.4f}, RMSE={val_rmse_nz:.4f}")
    print("\n--- Test ---")
    print(f"  All:    MAE={test_mae:.4f}, RMSE={test_rmse:.4f}, R²={test_r2:.4f}")
    print(f"  Nonzero: MAE={test_mae_nz:.4f}, RMSE={test_rmse_nz:.4f}")

    # Load baseline
    with open(ARTIFACTS / 'baseline_metrics.json') as f:
        bl = json.load(f)
    if 'ghg_target_4h' in bl:
        bl_r2 = bl['ghg_target_4h']['hour_mean']['r2']
        bl_mae = bl['ghg_target_4h']['hour_mean']['mae']
        print(f"\nBaseline (hour-mean) R²: {bl_r2:.4f}, MAE: {bl_mae:.4f}")

    # Feature importance
    importance = model.feature_importances_
    fi_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    fi_df.to_csv(ARTIFACTS / 'carbon_feature_importance.csv', index=False)
    print(f"\nTop 15 features by importance:")
    print(fi_df.head(15).to_string())

    # Save metrics
    metrics = {
        'model': 'xgb_regression_ghg_4h',
        'n_features': len(feature_cols),
        'best_iteration': int(model.best_iteration),
        'val': {
            'mae': val_mae, 'rmse': val_rmse, 'r2': val_r2,
            'mae_nonzero': val_mae_nz, 'rmse_nonzero': val_rmse_nz,
        },
        'test': {
            'mae': test_mae, 'rmse': test_rmse, 'r2': test_r2,
            'mae_nonzero': test_mae_nz, 'rmse_nonzero': test_rmse_nz,
        },
        'data_caveat': 'CAISO only publishes GHG for last ~90 days. Train/val mostly zero. Test (Apr-Jul 2026) has the actual events.',
    }
    with open(ARTIFACTS / 'eval_carbon.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {ARTIFACTS / 'eval_carbon.json'}")


if __name__ == '__main__':
    main()
