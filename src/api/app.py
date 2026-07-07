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
from fastapi.staticfiles import StaticFiles
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
MODELS_DIR = Path(os.environ.get('MODELS_DIR', '/app/models/champion'))
# Per-horizon model dict: {horizon: (lmp_model, carbon_model, loaded_at)}
LMP_MODELS = {}     # horizon -> model
CARBON_MODELS = {}  # horizon -> model
DEFAULT_HORIZON = '30m'
SUPPORTED_HORIZONS = ['30m', '1h', '2h', '4h']

MODEL_METADATA = {
    "lmp_ratio_loaded": False,
    "carbon_loaded": False,
    "loaded_at": None,
    "lmp_path": None,
    "carbon_path": None,
    "horizons_loaded": [],
}


def _load_model(model_name: str, horizon: str = '30m'):
    """Load an XGBoost model from the champion directory.

    Args:
        model_name: e.g. 'lmp_ratio' or 'carbon'. Resolves to
                    lmp_ratio_{horizon}.json or carbon_{horizon}.json
        horizon: '30m' | '1h' | '2h' | '4h' (default '30m' — closest to v0.1's 5-min)
    """
    fname = f"{model_name}_{horizon}.json"
    path = MODELS_DIR / fname
    if not path.exists():
        # Fallback to v0.1's flat filename
        legacy = MODELS_DIR / f"{model_name}.json"
        if legacy.exists():
            path = legacy
        else:
            logger.warning(f"  {fname} (and {legacy.name}) not found at {MODELS_DIR}")
            return None, None
    try:
        model = xgb.XGBRegressor()
        model.load_model(str(path))
        logger.info(f"  Loaded {fname} from {path}")
        return model, str(path)
    except Exception as e:
        logger.error(f"  Failed to load {fname}: {e}")
        return None, None


def load_models():
    """Load all horizon models. Called on startup and on /admin/reload."""
    global LMP_MODELS, CARBON_MODELS
    LMP_MODELS = {}
    CARBON_MODELS = {}
    paths_seen = []
    for h in SUPPORTED_HORIZONS:
        lmp_m, lmp_p = _load_model('lmp_ratio', h)
        c_m, c_p = _load_model('carbon', h)
        if lmp_m is not None:
            LMP_MODELS[h] = lmp_m
            paths_seen.append(lmp_p)
        if c_m is not None:
            CARBON_MODELS[h] = c_m
            paths_seen.append(c_p)
    MODEL_METADATA['lmp_ratio_loaded'] = bool(LMP_MODELS)
    MODEL_METADATA['carbon_loaded'] = bool(CARBON_MODELS)
    MODEL_METADATA['loaded_at'] = datetime.now().isoformat()
    MODEL_METADATA['lmp_path'] = list(LMP_MODELS.keys())
    MODEL_METADATA['carbon_path'] = list(CARBON_MODELS.keys())
    MODEL_METADATA['horizons_loaded'] = list(LMP_MODELS.keys())
    logger.info(f"  Loaded horizons: {list(LMP_MODELS.keys())}")


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
    lmp_features: dict = None,
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

    # LMP-derived features (22 features, computed from Redis history — D8 full)
    if lmp_features:
        for k, v in lmp_features.items():
            if FEATURE_COLS and k in FEATURE_COLS:
                defaults[k] = v

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


def get_redis_history(zone: str) -> list:
    """Read per-zone LMP history from Redis (sorted set).

    Returns a list of {t: iso_str, lmp: float} dicts, sorted by time ascending.
    Empty list on any failure.
    """
    try:
        import redis
        r = redis.from_url(os.environ.get('REDIS_URL', 'redis://redis:6379'),
                          decode_responses=True, socket_connect_timeout=2)
        key = f"features:zone:{zone}:lmp_history"
        raw = r.zrange(key, 0, -1)  # all entries, oldest first
        out = []
        for entry in raw:
            try:
                out.append(json.loads(entry))
            except (json.JSONDecodeError, TypeError):
                continue
        return out
    except Exception as e:
        logger.warning(f"  Failed to read LMP history for {zone}: {e}")
        return []


def compute_lmp_features(zone: str, current_lmp: float) -> dict:
    """Compute the 22 LMP-derived features the model expects.

    Mirrors add_lmp_features() + add_cross_zone_features() from training
    (src/features/build_features.py). Returns a dict mapping feature name
    to value, suitable for merging into the inference feature vector.

    Per-zone features (15):
      lmp_mean_60m, lmp_std_60m, lmp_mean_4h, lmp_std_4h,
      lmp_mean_24h, lmp_std_24h, lmp_max_24h, lmp_min_24h,
      lmp_lag_1, lmp_lag_12, lmp_lag_48,
      lmp_slope_60m, lmp_pct_change_5m, lmp_pct_change_60m, lmp_range_4h

    Cross-zone features (6):
      lmp_spread_np_sp, lmp_spread_np_zp, lmp_spread_sp_zp,
      lmp_max_across_zones, lmp_min_across_zones, lmp_mean_across_zones,
      lmp_max_across_zones_60m
    """
    import math
    import numpy as np
    from datetime import datetime as _dt

    # Read history for this zone + all other zones (for cross-zone)
    zone_hist = get_redis_history(zone)
    all_zone_hists = {z: get_redis_history(z) for z in ['NP15', 'SP15', 'ZP26']}

    # Parse timestamps once
    def parse_ts(entries):
        return [(entry['lmp'], _dt.fromisoformat(entry['t'].replace('Z', '+00:00')))
                for entry in entries if 'lmp' in entry and 't' in entry]

    parsed = parse_ts(zone_hist)
    if not parsed:
        # No history — return zeros for all LMP features
        return {col: 0.0 for col in [
            'lmp_mean_60m', 'lmp_std_60m', 'lmp_mean_4h', 'lmp_std_4h',
            'lmp_mean_24h', 'lmp_std_24h', 'lmp_max_24h', 'lmp_min_24h',
            'lmp_lag_1', 'lmp_lag_12', 'lmp_lag_48',
            'lmp_slope_60m', 'lmp_pct_change_5m', 'lmp_pct_change_60m', 'lmp_range_4h',
            'lmp_spread_np_sp', 'lmp_spread_np_zp', 'lmp_spread_sp_zp',
            'lmp_max_across_zones', 'lmp_min_across_zones', 'lmp_mean_across_zones',
            'lmp_max_across_zones_60m',
        ]}

    # Convert to numpy arrays
    lmps = np.array([p[0] for p in parsed], dtype=np.float64)
    times = [p[1] for p in parsed]
    now = _dt.now(_dt.now().astimezone().tzinfo)  # local-aware "now"

    # Per-zone features
    def in_window(window_sec, until_idx=None):
        """Get LMPs within the last window_sec seconds, up to index until_idx."""
        end = times[until_idx] if until_idx is not None else times[-1]
        return np.array([l for l, t in zip(lmps, times)
                         if (end - t).total_seconds() <= window_sec], dtype=np.float64)

    last_60m = in_window(60 * 60)
    last_4h = in_window(4 * 3600)
    last_24h = in_window(24 * 3600)

    out = {}

    # Rolling means/stds
    if len(last_60m) > 0:
        out['lmp_mean_60m'] = float(np.mean(last_60m))
        out['lmp_std_60m'] = float(np.std(last_60m)) if len(last_60m) > 1 else 0.0
    else:
        out['lmp_mean_60m'] = float(lmps[-1])
        out['lmp_std_60m'] = 0.0

    if len(last_4h) > 0:
        out['lmp_mean_4h'] = float(np.mean(last_4h))
        out['lmp_std_4h'] = float(np.std(last_4h)) if len(last_4h) > 1 else 0.0
    else:
        out['lmp_mean_4h'] = float(lmps[-1])
        out['lmp_std_4h'] = 0.0

    if len(last_24h) > 0:
        out['lmp_mean_24h'] = float(np.mean(last_24h))
        out['lmp_std_24h'] = float(np.std(last_24h)) if len(last_24h) > 1 else 0.0
        out['lmp_max_24h'] = float(np.max(last_24h))
        out['lmp_min_24h'] = float(np.min(last_24h))
    else:
        out['lmp_mean_24h'] = float(lmps[-1])
        out['lmp_std_24h'] = 0.0
        out['lmp_max_24h'] = float(lmps[-1])
        out['lmp_min_24h'] = float(lmps[-1])

    # Lags (5min, 1h, 4h ago)
    out['lmp_lag_1'] = float(lmps[-1]) if len(lmps) >= 1 else float(current_lmp)
    out['lmp_lag_12'] = float(lmps[-12]) if len(lmps) >= 12 else float(lmps[0])
    out['lmp_lag_48'] = float(lmps[-48]) if len(lmps) >= 48 else float(lmps[0])

    # Slope: linear regression of LMP on time in last 60min
    if len(last_60m) >= 2:
        x = np.arange(len(last_60m), dtype=np.float64)
        y = last_60m
        # slope = covariance(x,y) / variance(x)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        cov = np.mean((x - x_mean) * (y - y_mean))
        var = np.mean((x - x_mean) ** 2)
        out['lmp_slope_60m'] = float(cov / var) if var > 0 else 0.0
    else:
        out['lmp_slope_60m'] = 0.0

    # Pct changes (5min ago and 60min ago)
    if len(lmps) >= 2:
        out['lmp_pct_change_5m'] = float((lmps[-1] - lmps[-2]) / lmps[-2]) if lmps[-2] != 0 else 0.0
    else:
        out['lmp_pct_change_5m'] = 0.0
    if len(lmps) >= 13:
        out['lmp_pct_change_60m'] = float((lmps[-1] - lmps[-13]) / lmps[-13]) if lmps[-13] != 0 else 0.0
    else:
        out['lmp_pct_change_60m'] = 0.0

    # Range over last 4h
    if len(last_4h) > 0:
        out['lmp_range_4h'] = float(np.max(last_4h) - np.min(last_4h))
    else:
        out['lmp_range_4h'] = 0.0

    # Cross-zone features (latest LMP from each zone's history)
    latest_per_zone = {}
    for z, hist in all_zone_hists.items():
        if hist:
            latest_per_zone[z] = float(hist[-1]['lmp'])
        else:
            latest_per_zone[z] = current_lmp  # fallback to this zone's current

    n_latest = [latest_per_zone.get('NP15', 0), latest_per_zone.get('SP15', 0), latest_per_zone.get('ZP26', 0)]
    out['lmp_spread_np_sp'] = latest_per_zone.get('NP15', 0) - latest_per_zone.get('SP15', 0)
    out['lmp_spread_np_zp'] = latest_per_zone.get('NP15', 0) - latest_per_zone.get('ZP26', 0)
    out['lmp_spread_sp_zp'] = latest_per_zone.get('SP15', 0) - latest_per_zone.get('ZP26', 0)
    out['lmp_max_across_zones'] = float(max(n_latest))
    out['lmp_min_across_zones'] = float(min(n_latest))
    out['lmp_mean_across_zones'] = float(np.mean(n_latest))

    # Max across zones in last 60min — use this zone's history's max
    out['lmp_max_across_zones_60m'] = out['lmp_max_across_zones']  # approximation

    return out


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
def forecast(zone: str, horizon: str = '30m'):
    """Get forecast for a CAISO zone at the given forward horizon.

    Args:
        zone: NP15 | SP15 | ZP26
        horizon: 30m (default) | 1h | 2h | 4h
    """
    if zone not in ['NP15', 'SP15', 'ZP26']:
        raise HTTPException(status_code=404, detail=f"Unknown zone: {zone}")
    if horizon not in SUPPORTED_HORIZONS:
        raise HTTPException(status_code=400, detail=f"Unknown horizon: {horizon}. Supported: {SUPPORTED_HORIZONS}")

    # Try to get live data from Redis
    cached = None
    data = {}
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
    # Compute LMP features from Redis history (D8 full)
    current_lmp = float(data.get('predicted_lmp', 25) or 25) if cached else 25.0
    lmp_features = compute_lmp_features(zone, current_lmp)
    features = build_inference_features(
        zone=zone, load_mw=load_mw, fuel_mix=fuel_mix, weather=weather,
        hour=now.hour, day_of_week=now.weekday(),
        month=now.month, is_weekend=now.weekday() >= 5,
        lmp_features=lmp_features,
    )
    zone_dummies = zone_to_dummies(zone)

    # Pick the model for the requested horizon (do this before building features
    # so we know how many features the model expects).
    lmp_model = LMP_MODELS.get(horizon)
    carbon_model = CARBON_MODELS.get(horizon)
    # LMP and carbon models may have different expected feature counts
    # (LMP v0.2 = 48, Carbon v0.2 = 51). Pad to the max so both work.
    lmp_expected = lmp_model.get_booster().num_features() if lmp_model is not None else 51
    carbon_expected = carbon_model.get_booster().num_features() if carbon_model is not None else 51
    expected_features = max(lmp_expected, carbon_expected)

    # Pad features to include zone dummies (model expects them in train)
    if FEATURE_COLS:
        # Add zone dummies in the order expected by model
        base = np.concatenate([
            features,
            np.array([[zone_dummies['zone_NP15'],
                       zone_dummies['zone_SP15'],
                       zone_dummies['zone_ZP26']]], dtype=np.float32),
        ], axis=1)
        # Build per-model feature vectors. LMP and carbon models may have
        # different expected feature counts (LMP v0.2 = 48, Carbon v0.2 = 51)
        # because the two training scripts have different feature pipelines.
        def fit_to(model, vec):
            if model is None:
                return None
            n = model.get_booster().num_features()
            if vec.shape[1] < n:
                pad = np.zeros((1, n - vec.shape[1]), dtype=np.float32)
                return np.concatenate([vec, pad], axis=1)
            elif vec.shape[1] > n:
                return vec[:, :n]
            return vec
        feature_with_zone_lmp = fit_to(lmp_model, base)
        feature_with_zone_carbon = fit_to(carbon_model, base)
        # For backwards compat (DC endpoint, etc.) use the LMP-shaped vector
        feature_with_zone = feature_with_zone_lmp if feature_with_zone_lmp is not None else base
    else:
        feature_with_zone = None
        feature_with_zone_lmp = None
        feature_with_zone_carbon = None

    # Predict LMP ratio
    lmp_ratio_pred = None
    if lmp_model is not None and feature_with_zone_lmp is not None:
        try:
            lmp_ratio_pred = float(lmp_model.predict(feature_with_zone_lmp)[0])
        except Exception as e:
            logger.error(f"  LMP prediction failed: {e}")

    # Predict carbon
    carbon_pred = None
    if carbon_model is not None and feature_with_zone_carbon is not None:
        try:
            carbon_pred = float(carbon_model.predict(feature_with_zone_carbon)[0])
        except Exception as e:
            logger.error(f"  Carbon prediction failed: {e}")

    # Map LMP ratio to a $/MWh estimate (rough: 25 baseline)
    lmp_dollar_est = lmp_ratio_pred * 25.0 if lmp_ratio_pred else None

    return {
        "zone": zone,
        "horizon": horizon,
        "horizon_min": {'30m': 30, '1h': 60, '2h': 120, '4h': 240}[horizon],
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
def dc_forecast(dc_id: str, horizon: str = '30m'):
    """Get per-DC advisory at the given forward horizon.

    Args:
        dc_id: e.g. DC-00088
        horizon: 30m (default) | 1h | 2h | 4h

    Returns the cached advisory (all 4 horizons) flattened to the requested
    horizon, plus metadata. If the requested horizon isn't cached, falls
    back to the 30m value.
    """
    if horizon not in SUPPORTED_HORIZONS:
        raise HTTPException(status_code=400, detail=f"Unknown horizon: {horizon}. Supported: {SUPPORTED_HORIZONS}")

    # Try to get cached advisory from Redis (computed by live_fetcher)
    try:
        import redis
        r = redis.from_url(os.environ.get('REDIS_URL', 'redis://redis:6379'),
                          decode_responses=True, socket_connect_timeout=2)
        key = f"features:dc:{dc_id}:now"
        cached = r.get(key)
        if cached:
            data = json.loads(cached)
            # Flatten to the requested horizon
            horizons = data.get('horizons', {})
            h_data = horizons.get(horizon, {})
            if h_data:
                return {
                    'dc_id': data.get('dc_id'),
                    'name': data.get('name'),
                    'zone': data.get('zone'),
                    'operator': data.get('operator'),
                    'mw_capacity': data.get('mw_capacity'),
                    'wue': data.get('wue'),
                    'bws_score': data.get('bws_score'),
                    'horizon': horizon,
                    'horizon_min': {'30m': 30, '1h': 60, '2h': 120, '4h': 240}[horizon],
                    'lmp_dollar_estimate': h_data.get('lmp_dollar_per_mwh'),
                    'advisory': h_data.get('advisory'),
                    'all_horizons': {
                        h: {
                            'lmp_dollar_estimate': h2.get('lmp_dollar_per_mwh'),
                            'advisory': h2.get('advisory'),
                        } for h, h2 in horizons.items()
                    },
                    'computed_at': data.get('computed_at'),
                    'data_source': 'live',
                }
            # Cached data has no horizons field (old format). Fall through to 404.
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


# === Frontend support (D11) ===

# Load the 227 CA DC sites once at startup for the /sites endpoint.
SITES_CSV_PATH = Path('/app/data/external/ca_dc_sites.csv')
DC_SITES_CACHE: list = []


def load_dc_sites() -> list:
    """Load the 227 CA DC sites from CSV into a list of dicts."""
    if DC_SITES_CACHE:
        return DC_SITES_CACHE
    if not SITES_CSV_PATH.exists():
        logger.warning(f"  DC sites CSV not found at {SITES_CSV_PATH}")
        return []
    import csv
    sites = []
    with open(SITES_CSV_PATH) as f:
        for row in csv.DictReader(f):
            # Coerce numeric fields; skip rows that can't be parsed
            try:
                sites.append({
                    'dc_id': row['dc_id'],
                    'name': row['name'],
                    'provider': row.get('provider', ''),
                    'state': row.get('state', ''),
                    'latitude': float(row['latitude']),
                    'longitude': float(row['longitude']),
                    'mw_capacity': float(row.get('MW_total_power', 0) or 0),
                    'wue': float(row.get('wue_default', 1.18) or 1.18),
                    'bws_score': float(row.get('bws_score', 0) or 0),
                    'caiso_zone': row.get('caiso_zone', ''),
                })
            except (ValueError, KeyError) as e:
                logger.warning(f"  Skipping malformed DC row: {e}")
                continue
    logger.info(f"  Loaded {len(sites)} DC sites from {SITES_CSV_PATH}")
    DC_SITES_CACHE.extend(sites)
    return DC_SITES_CACHE


load_dc_sites()


@app.get("/sites")
def get_sites():
    """Return all 227 CA DC sites as JSON (for the frontend map)."""
    return {'sites': load_dc_sites(), 'count': len(DC_SITES_CACHE)}


@app.get("/zones/history")
def zones_history(since: str = None, limit: int = None):
    """Return per-zone LMP history (sorted-set from fetcher) for the time series.

    Query params:
        since: ISO timestamp (e.g. "2026-07-05T13:00:00Z"). If given, only return
               entries with t >= since. Useful for paginating large histories.
        limit: int. If given, return only the most recent N entries per zone
               (after the since filter).

    Returns: {zones: {NP15: [...], ...}, last_updated: ISO, total_in_redis: {zone: count}}
    Behavior:
        - No params: returns all entries (current behavior, but will grow over time)
        - since only: returns entries since that timestamp
        - limit only: returns the last N entries per zone
        - both: returns last N entries since the timestamp

    Note: This endpoint can grow unbounded over time. Callers should pass
    `limit` for the frontend (which doesn't need more than 6h of data for
    a 24h chart).
    """
    try:
        import redis
        from datetime import datetime as _dt
        r = redis.from_url(os.environ.get('REDIS_URL', 'redis://redis:6379'),
                          decode_responses=True, socket_connect_timeout=2)
        out = {'zones': {}, 'last_updated': None, 'total_in_redis': {}}

        # Parse the 'since' param into a unix timestamp
        since_unix = None
        if since:
            try:
                # Accept both 'Z' and '+00:00' suffixes
                dt = _dt.fromisoformat(since.replace('Z', '+00:00'))
                since_unix = dt.timestamp()
            except (ValueError, AttributeError) as e:
                raise HTTPException(status_code=400, detail=f"Invalid 'since' format: {e}. Use ISO 8601.")

        for zone in ['NP15', 'SP15', 'ZP26']:
            key = f"features:zone:{zone}:lmp_history"
            # Get total count
            total = r.zcard(key)
            out['total_in_redis'][zone] = total

            # Get entries — use ZRANGEBYSCORE if 'since' given, else full range
            if since_unix is not None:
                raw = r.zrangebyscore(key, since_unix, '+inf')
            else:
                raw = r.zrange(key, 0, -1)

            # Apply limit (take most recent N)
            if limit and limit > 0 and len(raw) > limit:
                raw = raw[-limit:]

            entries = []
            for entry in raw:
                try:
                    entries.append(json.loads(entry))
                except (json.JSONDecodeError, TypeError):
                    continue
            out['zones'][zone] = entries

        # Last fetch metadata
        meta = r.get('meta:last_fetch')
        if meta:
            out['last_updated'] = json.loads(meta).get('cycle_at')
        return out
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"  Failed to read zones history: {e}")
        return {'zones': {}, 'last_updated': None, 'error': str(e)}


# === Static file serving (D11) ===
# Mount the web/ directory at / so the frontend is served at the root.
WEB_DIR = Path('/app/web')
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
    logger.info(f"  Serving frontend from {WEB_DIR}")
else:
    logger.info(f"  No web/ directory at {WEB_DIR}; frontend not served")
