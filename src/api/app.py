"""
FastAPI application for dc_real_time.

Endpoints:
  GET  /healthz             — liveness check
  GET  /readyz              — readiness check (model loaded, deps OK)
  GET  /forecast/{zone}     — LMP ratio + carbon forecast for a zone
  GET  /dc/{dc_id}/forecast — per-DC advisory
  POST /admin/reload        — reload model from disk (for symlink swap)
  GET  /model/info          — current model metadata

Models are loaded from /app/models/champion/{lmp_ratio,carbon}.json
(atomic symlink swap for zero-downtime model deployment).
"""
from fastapi import FastAPI, HTTPException
from datetime import datetime
from pathlib import Path
import json
import logging
import os

import numpy as np
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="dc_real_time API",
    version="0.2.0",
    description="Spatial-temporal carbon & price forecasting for data centers",
)

# === Model loading ===
MODELS_DIR = Path('/app/models/champion')
LMP_MODEL = None
CARBON_MODEL = None
MODEL_METADATA = {
    "lmp_ratio_loaded": False,
    "carbon_loaded": False,
    "loaded_at": None,
    "lmp_path": None,
    "carbon_path": None,
}


def _load_model(model_name: str):
    """Load an XGBoost model from the champion directory."""
    path = MODELS_DIR / model_name
    if not path.exists():
        logger.warning(f"  {model_name} not found at {path}")
        return None, None
    try:
        model = xgb.XGBRegressor()
        model.load_model(str(path))
        logger.info(f"  Loaded {model_name} from {path}")
        return model, str(path)
    except Exception as e:
        logger.error(f"  Failed to load {model_name}: {e}")
        return None, None


def load_models():
    """Load both models. Called on startup and on /admin/reload."""
    global LMP_MODEL, CARBON_MODEL
    LMP_MODEL, MODEL_METADATA['lmp_path'] = _load_model('lmp_ratio.json')
    CARBON_MODEL, MODEL_METADATA['carbon_path'] = _load_model('carbon.json')
    MODEL_METADATA['lmp_ratio_loaded'] = LMP_MODEL is not None
    MODEL_METADATA['carbon_loaded'] = CARBON_MODEL is not None
    MODEL_METADATA['loaded_at'] = datetime.now().isoformat()


# Load on startup
load_models()


# === Feature engineering for inference ===
# At inference time, we have:
#   - Current load (MW) - from live fetcher
#   - Current fuel mix (per fuel) - from live fetcher
#   - Current weather (temp, humidity, etc.) - from Open-Meteo
#   - Calendar features (hour, day_of_week, month)
#
# We build the same 45 features the model was trained on (see
# src/features/build_features.py). For inference we only have ONE
# timestamp's worth of data, so rolling stats are minimal (use last known).
# Real impl: pull historical feature window from Redis.

FEATURE_COLS = None  # loaded from feature_schema.json if available


def load_feature_schema():
    """Load feature column list from artifacts/feature_schema.json."""
    global FEATURE_COLS
    schema_path = Path('/app/artifacts/feature_schema.json')
    if not schema_path.exists():
        logger.warning(f"  feature_schema.json not found at {schema_path}")
        FEATURE_COLS = None
        return
    try:
        with open(schema_path) as f:
            schema = json.load(f)
        FEATURE_COLS = schema.get('feature_columns', [])
        # Drop zone_* dummies from schema (we add them at inference)
        FEATURE_COLS = [c for c in FEATURE_COLS if not c.startswith('zone_')]
        logger.info(f"  Loaded {len(FEATURE_COLS)} feature columns from schema")
    except Exception as e:
        logger.error(f"  Failed to load feature schema: {e}")


load_feature_schema()


def build_inference_features(
    zone: str,
    load_mw: float,
    fuel_mix: dict,
    weather: dict,
    hour: int,
    day_of_week: int,
    month: int,
    is_weekend: bool,
) -> np.ndarray:
    """Build a single-row feature vector for inference.

    For Phase 4 v0.2: builds a SIMPLIFIED feature set matching the model's
    expected inputs. For full 45-feature parity with training, use
    historical features from Redis (D8 full impl).

    Returns: 1D numpy array of feature values, in the same order as
    feature_schema.json (minus zone dummies).
    """
    import pandas as pd

    # Defaults for features we don't have at inference
    defaults = {col: 0.0 for col in (FEATURE_COLS or [])}

    # Direct values from current state
    if FEATURE_COLS and 'LMP' in FEATURE_COLS:
        # LMP not in inference (would need 4h baseline); use load as proxy
        pass
    if FEATURE_COLS and 'Energy' in FEATURE_COLS:
        defaults['Energy'] = load_mw
    if FEATURE_COLS and 'Congestion' in FEATURE_COLS:
        defaults['Congestion'] = 0
    if FEATURE_COLS and 'Loss' in FEATURE_COLS:
        defaults['Loss'] = 0
    if FEATURE_COLS and 'GHG' in FEATURE_COLS:
        defaults['GHG'] = 0  # GHG has 1-2h lag, use 0 for now

    # Calendar features
    import math
    if FEATURE_COLS and 'hour_of_day' in FEATURE_COLS:
        defaults['hour_of_day'] = hour
    if FEATURE_COLS and 'day_of_week' in FEATURE_COLS:
        defaults['day_of_week'] = day_of_week
    if FEATURE_COLS and 'month' in FEATURE_COLS:
        defaults['month'] = month
    if FEATURE_COLS and 'is_weekend' in FEATURE_COLS:
        defaults['is_weekend'] = int(is_weekend)
    if FEATURE_COLS and 'hour_sin' in FEATURE_COLS:
        defaults['hour_sin'] = math.sin(2 * math.pi * hour / 24)
    if FEATURE_COLS and 'hour_cos' in FEATURE_COLS:
        defaults['hour_cos'] = math.cos(2 * math.pi * hour / 24)
    if FEATURE_COLS and 'month_sin' in FEATURE_COLS:
        defaults['month_sin'] = math.sin(2 * math.pi * month / 12)
    if FEATURE_COLS and 'month_cos' in FEATURE_COLS:
        defaults['month_cos'] = math.cos(2 * math.pi * month / 12)

    # Fuel mix features
    fuel_map = {
        'solar_mw_60m_mean': 'Solar',
        'wind_mw_60m_mean': 'Wind',
        'natural_gas_mw_60m_mean': 'NaturalGas',
        'imports_mw_60m_mean': 'Imports',
        'nuclear_mw_60m_mean': 'Nuclear',
        'large_hydro_mw_60m_mean': 'Hydro',  # combined
        'batteries_mw_60m_mean': 'Batteries',
    }
    for feat_col, fuel_name in fuel_map.items():
        if FEATURE_COLS and feat_col in FEATURE_COLS:
            defaults[feat_col] = fuel_mix.get(fuel_name, 0) or fuel_mix.get(fuel_name + '_MW', 0)

    # Weather features
    if weather:
        if FEATURE_COLS and 'lmp_slope_60m' in FEATURE_COLS:
            defaults['lmp_slope_60m'] = 0
        # Open-Meteo doesn't provide wet_bulb; use temp as proxy
        if FEATURE_COLS and 'lmp_range_4h' in FEATURE_COLS:
            defaults['lmp_range_4h'] = 0

    # Assemble in column order
    if FEATURE_COLS is None:
        # Fallback: minimal feature set
        return np.array([[
            load_mw, fuel_mix.get('Solar', 0) or 0, fuel_mix.get('Wind', 0) or 0,
            fuel_mix.get('NaturalGas', 0) or 0, hour, day_of_week, month,
        ]], dtype=np.float32)
    return np.array([[defaults.get(c, 0) for c in FEATURE_COLS]], dtype=np.float32)


def zone_to_dummies(zone: str) -> dict:
    """One-hot encode the zone."""
    return {
        'zone_NP15': int(zone == 'NP15'),
        'zone_SP15': int(zone == 'SP15'),
        'zone_ZP26': int(zone == 'ZP26'),
    }


def advisory_from_lmp(lmp: float) -> str:
    """Map predicted LMP to advisory string."""
    if lmp is None or np.isnan(lmp):
        return 'unknown'
    if lmp > 100:
        return 'pause'
    if lmp > 50:
        return 'defer'
    if lmp > 30:
        return 'watch'
    return 'ok'


# === Endpoints ===

@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "model_loaded_at": MODEL_METADATA['loaded_at'],
        "lmp_model": MODEL_METADATA['lmp_path'],
        "carbon_model": MODEL_METADATA['carbon_path'],
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/readyz")
def readyz():
    ready = MODEL_METADATA['lmp_ratio_loaded'] or MODEL_METADATA['carbon_loaded']
    return {
        "ready": ready,
        "service": "dc_real_time_api",
        "version": "0.2.0",
        "lmp_loaded": MODEL_METADATA['lmp_ratio_loaded'],
        "carbon_loaded": MODEL_METADATA['carbon_loaded'],
    }


@app.get("/model/info")
def model_info():
    return MODEL_METADATA


@app.get("/forecast/{zone}")
def forecast(zone: str):
    """Get forecast for a CAISO zone (NP15, SP15, ZP26)."""
    if zone not in ['NP15', 'SP15', 'ZP26']:
        raise HTTPException(status_code=404, detail=f"Unknown zone: {zone}")

    # Try to get live data from Redis
    try:
        import redis
        r = redis.from_url(os.environ.get('REDIS_URL', 'redis://redis:6379'),
                          decode_responses=True, socket_connect_timeout=2)
        zone_key = f"features:zone:{zone}:now"
        cached = r.get(zone_key)
        if cached:
            data = json.loads(cached)
            load_mw = data.get('load_mw', 25000) or 25000
            fuel_mix = data.get('fuel_mix', {})
            weather = data.get('weather', {})
        else:
            # No live data, use defaults
            load_mw, fuel_mix, weather = 25000, {}, {}
    except Exception as e:
        logger.warning(f"  Redis fetch failed for {zone}: {e}; using defaults")
        load_mw, fuel_mix, weather = 25000, {}, {}

    now = datetime.now()
    features = build_inference_features(
        zone=zone, load_mw=load_mw, fuel_mix=fuel_mix, weather=weather,
        hour=now.hour, day_of_week=now.weekday(),
        month=now.month, is_weekend=now.weekday() >= 5,
    )
    zone_dummies = zone_to_dummies(zone)

    # Pad features to include zone dummies (model expects them in train)
    if FEATURE_COLS:
        # Add zone dummies in the order expected by model
        feature_with_zone = np.concatenate([
            features,
            np.array([[zone_dummies['zone_NP15'],
                       zone_dummies['zone_SP15'],
                       zone_dummies['zone_ZP26']]], dtype=np.float32),
        ], axis=1)
        # Model was trained with 51 features; pad with zeros if short
        if feature_with_zone.shape[1] < 51:
            pad = np.zeros((1, 51 - feature_with_zone.shape[1]), dtype=np.float32)
            feature_with_zone = np.concatenate([feature_with_zone, pad], axis=1)
    else:
        # Fallback: provide 51 zeros
        feature_with_zone = np.zeros((1, 51), dtype=np.float32)

    # Predict LMP ratio
    lmp_ratio_pred = None
    if LMP_MODEL is not None and FEATURE_COLS is not None:
        try:
            lmp_ratio_pred = float(LMP_MODEL.predict(feature_with_zone)[0])
        except Exception as e:
            logger.error(f"  LMP prediction failed: {e}")

    # Predict carbon
    carbon_pred = None
    if CARBON_MODEL is not None and FEATURE_COLS is not None:
        try:
            carbon_pred = float(CARBON_MODEL.predict(feature_with_zone)[0])
        except Exception as e:
            logger.error(f"  Carbon prediction failed: {e}")

    # Map LMP ratio to a $/MWh estimate (rough: 25 baseline)
    lmp_dollar_est = lmp_ratio_pred * 25.0 if lmp_ratio_pred else None

    return {
        "zone": zone,
        "horizon_min": 5,
        "predicted_at": now.isoformat(),
        "load_mw": load_mw,
        "lmp_ratio_pred": lmp_ratio_pred,
        "lmp_dollar_estimate": lmp_dollar_est,
        "carbon_pred_short_ton_per_mwh": carbon_pred,
        "advisory": advisory_from_lmp(lmp_dollar_est),
        "model_version": "champion",
        "data_source": "live" if cached else "fallback",
    }


@app.get("/dc/{dc_id}/forecast")
def dc_forecast(dc_id: str):
    """Get per-DC advisory based on zone forecast."""
    # Try to get cached advisory from Redis (computed by live_fetcher)
    try:
        import redis
        r = redis.from_url(os.environ.get('REDIS_URL', 'redis://redis:6379'),
                          decode_responses=True, socket_connect_timeout=2)
        key = f"features:dc:{dc_id}:now"
        cached = r.get(key)
        if cached:
            return json.loads(cached)
    except Exception as e:
        logger.warning(f"  Redis fetch failed for {dc_id}: {e}")

    raise HTTPException(status_code=404, detail=f"No forecast for {dc_id}")


@app.post("/admin/reload")
def admin_reload():
    """Reload models from disk (for atomic symlink swap)."""
    load_models()
    return {
        "reloaded": True,
        "at": MODEL_METADATA['loaded_at'],
        "lmp_loaded": MODEL_METADATA['lmp_ratio_loaded'],
        "carbon_loaded": MODEL_METADATA['carbon_loaded'],
    }
