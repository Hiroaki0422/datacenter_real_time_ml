"""
Live data fetcher for dc_real_time.

Polls external APIs every 5 minutes, writes current state to Redis
as the "online feature store". Also serves the predicted-LMP service:
uses the LMP model to predict current LMP from current load + fuel mix
(real LMP has 1-2h OASIS lag).

Cached results are keyed by zone in Redis:
  features:zone:NP15:now        -> JSON of {load, fuel_mix, weather, predicted_lmp, ...}
  features:dc:{dc_id}:now      -> per-DC advisory
  meta:last_fetch             -> timestamp of last successful fetch

Environment:
  REDIS_URL=redis://redis:6379
  CACHE_TTL_SEC=300            # how long cached features are valid

Usage:
  python -m src.data.live_fetcher             # single fetch cycle
  python -m src.data.live_fetcher --loop    # run every 5 min forever
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import requests_cache
from retry_requests import retry

# Add src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# Caching for Open-Meteo
_cache = requests_cache.CachedSession('.cache_weather', expire_after=300)
_retry = retry(_cache, retries=2, backoff_factor=0.3)


# === Data sources ===

def fetch_caiso_load_fuelmix() -> pd.DataFrame:
    """Fetch latest CAISO load + fuel mix from caiso.com/outlook.

    Lag: ~7 minutes for current data.
    Returns: DataFrame with Time, Load, Solar, Wind, Natural Gas, ... (1 most-recent row).
    """
    try:
        # caiso.com/outlook has current data as direct CSV
        # Need a User-Agent header — pandas/requests default is blocked
        headers = {"User-Agent": "dc_real_time/0.1 (research; contact@example.com)"}
        load_url = "https://www.caiso.com/outlook/current/demand.csv"
        fm_url = "https://www.caiso.com/outlook/current/fuelsource.csv"
        load_df = pd.read_csv(load_url, storage_options={'User-Agent': headers['User-Agent']})
        # requests fallback (more reliable for CAISO)
        import requests
        load_r = requests.get(load_url, headers=headers, timeout=10)
        load_df = pd.read_csv(__import__('io').StringIO(load_r.text))
        fm_r = requests.get(fm_url, headers=headers, timeout=10)
        fm_df = pd.read_csv(__import__('io').StringIO(fm_r.text))
        # Most recent row that has Current demand populated
        # (last few rows may have empty Current demand while CAISO updates)
        load_with_demand = load_df[load_df['Current demand'].notna() & (load_df['Current demand'] != '')]
        if len(load_with_demand) == 0:
            logger.error("  No CAISO data with Current demand")
            return pd.DataFrame()
        load_latest = load_with_demand.iloc[-1]
        fm_latest = fm_df.iloc[-1]
        # Build a single row
        out = {
            'Time': pd.Timestamp.now(tz='US/Pacific').isoformat(),
            'Load_MW': float(load_latest.get('Current demand', 0)),
            'Solar_MW': float(fm_latest.get('Solar', 0)),
            'Wind_MW': float(fm_latest.get('Wind', 0)),
            'NaturalGas_MW': float(fm_latest.get('Natural Gas', 0)),
            'Nuclear_MW': float(fm_latest.get('Nuclear', 0)),
            'Hydro_MW': float(fm_latest.get('Large Hydro', 0)) + float(fm_latest.get('Small hydro', 0)),
            'Batteries_MW': float(fm_latest.get('Batteries', 0)),
            'Imports_MW': float(fm_latest.get('Imports', 0)),
            'Geothermal_MW': float(fm_latest.get('Geothermal', 0)),
            'Biomass_MW': float(fm_latest.get('Biomass', 0)) + float(fm_latest.get('Biogas', 0)),
        }
        logger.info(f"  CAISO load: {out['Load_MW']:.0f} MW, solar: {out['Solar_MW']:.0f} MW")
        return pd.DataFrame([out])
    except Exception as e:
        logger.error(f"Failed to fetch CAISO data: {e}")
        return pd.DataFrame()


def fetch_openmeteo_current(lat: float, lon: float) -> dict:
    """Fetch current weather from Open-Meteo for one location."""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": ["temperature_2m", "relative_humidity_2m", "cloud_cover", "wind_speed_10m"],
            "timezone": "America/Los_Angeles",
        }
        r = _retry.get(url, params=params, timeout=5)
        r.raise_for_status()
        data = r.json()
        current = data.get('current', {})
        return {
            'temperature_2m': current.get('temperature_2m'),
            'relative_humidity_2m': current.get('relative_humidity_2m'),
            'cloud_cover': current.get('cloud_cover'),
            'wind_speed_10m': current.get('wind_speed_10m'),
        }
    except Exception as e:
        logger.warning(f"  Open-Meteo failed for ({lat},{lon}): {e}")
        return {}


def fetch_nws_current(lat: float, lon: float) -> dict:
    """Fetch current weather alert status from NWS API."""
    try:
        # NWS gridpoint lookup
        r = requests.get(
            f"https://api.weather.gov/points/{lat},{lon}",
            timeout=5,
            headers={"User-Agent": "(dc_real_time, ops@example.com)"},
        )
        if r.status_code != 200:
            return {}
        j = r.json()
        forecast_url = j.get('properties', {}).get('forecast')
        if not forecast_url:
            return {}
        # Get active alerts
        alerts_url = f"https://api.weather.gov/alerts/active?point={lat},{lon}"
        r2 = requests.get(alerts_url, timeout=5,
                          headers={"User-Agent": "(dc_real_time, ops@example.com)"})
        active_alerts = []
        if r2.status_code == 200:
            for alert in r2.json().get('features', []):
                active_alerts.append(alert.get('properties', {}).get('event', 'unknown'))
        return {
            'forecast_url': forecast_url,
            'active_alerts': active_alerts,
        }
    except Exception as e:
        logger.warning(f"  NWS failed for ({lat},{lon}): {e}")
        return {}


# === DC site info ===

DC_SITES_PATH = Path('/app/data/external/ca_dc_sites.csv')


def load_dc_sites() -> pd.DataFrame:
    """Load the 227 CA DC sites."""
    if not DC_SITES_PATH.exists():
        logger.warning(f"DC sites file not found at {DC_SITES_PATH}")
        return pd.DataFrame()
    return pd.read_csv(DC_SITES_PATH)


# === Predicted LMP ===

def predict_lmp_from_state(load_mw: float, fuel_mix: dict) -> float:
    """Predict LMP ($/MWh) from current load + fuel mix using the trained LMP model.

    Real LMP has 1-2h lag from OASIS. We use the LMP model to estimate current
    LMP from features we DO have in real-time (load + fuel mix).

    Returns: predicted LMP in $/MWh
    """
    try:
        import xgboost as xgb
        # Try multiple model paths (best first, then standard)
        for model_name in ['lmp_ratio_best.json', 'lmp_ratio.json']:
            model_path = Path(f'/app/models/champion/{model_name}')
            if model_path.exists():
                break
        else:
            logger.warning("  No champion model; using rough LMP estimate")
            return _rough_lmp_estimate(load_mw, fuel_mix)
        # Build a feature row matching the model's expected features
        # For Phase 4 v0.1 we use a simplified feature set
        # In v0.2+ we'd build the full 51-feature pipeline
        features = _build_lmp_features(load_mw, fuel_mix)
        if features is None:
            return _rough_lmp_estimate(load_mw, fuel_mix)
        model = xgb.XGBRegressor()
        model.load_model(str(model_path))
        # Predict LMP ratio
        ratio = model.predict(features)[0]
        # Get baseline (rough: $25/MWh)
        baseline = 25.0
        return float(ratio * baseline)
    except Exception as e:
        logger.warning(f"  LMP model prediction failed: {e}; using rough estimate")
        return _rough_lmp_estimate(load_mw, fuel_mix)


def _build_lmp_features(load_mw: float, fuel_mix: dict) -> np.ndarray:
    """Build a feature row for the LMP model.

    For Phase 4 v0.1 stub: build a minimal feature set.
    Real implementation (D8) uses the full 51-feature pipeline.
    """
    # Simplified features matching the LMP model expected inputs
    # In a real impl, this would mirror build_features_for_horizon
    return None  # signals to use rough estimate


def _rough_lmp_estimate(load_mw: float, fuel_mix: dict) -> float:
    """Very rough LMP estimate: $20-40/MWh depending on load.

    Used as fallback when the trained model isn't loaded.
    """
    base = 25.0
    load_factor = (load_mw - 15000) / 15000  # 15GW baseline
    solar_supply = fuel_mix.get('Solar_MW', 0) / 10000
    # Simple linear model
    return max(5.0, base * (1 + 0.5 * load_factor - 0.3 * solar_supply))


# === Redis cache ===

def get_redis_client():
    """Get a Redis client from REDIS_URL env var."""
    import redis
    redis_url = os.environ.get('REDIS_URL', 'redis://redis:6379')
    return redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=5)


def cache_features(redis_client, zone: str, features: dict, ttl_sec: int = 300) -> None:
    """Cache features for a zone in Redis with TTL."""
    key = f"features:zone:{zone}:now"
    redis_client.setex(key, ttl_sec, json.dumps(features, default=str))


def cache_dc_advisory(redis_client, dc_id: str, advisory: dict, ttl_sec: int = 300) -> None:
    """Cache per-DC advisory in Redis."""
    key = f"features:dc:{dc_id}:now"
    redis_client.setex(key, ttl_sec, json.dumps(advisory, default=str))


# === Per-DC advisory logic ===

def compute_dc_advisory(dc: pd.Series, zone_features: dict) -> dict:
    """Compute advisory for a single DC based on zone state + DC attributes."""
    predicted_lmp = zone_features.get('predicted_lmp', None)
    advisory = "ok"
    if predicted_lmp is not None:
        if predicted_lmp > 100:
            advisory = "pause"
        elif predicted_lmp > 50:
            advisory = "defer"
        elif predicted_lmp > 30:
            advisory = "watch"
    return {
        'dc_id': dc.get('dc_id'),
        'name': dc.get('name'),
        'zone': dc.get('caiso_zone'),
        'operator': dc.get('provider'),
        'mw_capacity': float(dc.get('MW_total_power', 0)) if pd.notna(dc.get('MW_total_power')) else 0,
        'wue': float(dc.get('wue_default', 1.18)) if pd.notna(dc.get('wue_default')) else 1.18,
        'bws_score': float(dc.get('bws_score', 0)) if pd.notna(dc.get('bws_score')) else 0,
        'predicted_lmp': predicted_lmp,
        'advisory': advisory,
        'computed_at': datetime.now(timezone.utc).isoformat(),
    }


# === Main fetch cycle ===

def fetch_cycle(redis_client, dc_sites: pd.DataFrame) -> dict:
    """One fetch cycle: get live data, predict, cache."""
    cycle_start = datetime.now(timezone.utc)
    logger.info("=== Live fetch cycle started ===")

    # 1. Get system-wide data
    iso_data = fetch_caiso_load_fuelmix()
    if iso_data.empty:
        logger.error("  No CAISO data; skipping cycle")
        return {'status': 'no_data', 'cycle_at': cycle_start.isoformat()}

    row = iso_data.iloc[0]
    load_mw = row['Load_MW']
    fuel_mix = {k.replace('_MW', ''): v for k, v in row.items() if k.endswith('_MW')}

    # 2. Predict LMP from current state
    predicted_lmp = predict_lmp_from_state(load_mw, fuel_mix)
    logger.info(f"  Predicted LMP: ${predicted_lmp:.2f}/MWh")

    # 3. Get weather (per-zone centroid for now)
    zone_centroids = {
        'NP15': (38.5, -121.5),  # Sacramento
        'SP15': (34.0, -118.2),  # LA
        'ZP26': (36.0, -119.5),  # Central CA
    }
    zone_weather = {}
    for zone, (lat, lon) in zone_centroids.items():
        weather = fetch_openmeteo_current(lat, lon)
        zone_weather[zone] = weather
    logger.info(f"  Weather fetched for {len(zone_weather)} zones")

    # 4. Build per-zone features
    cache_summary = {'zones': {}, 'dcs': 0}
    for zone in ['NP15', 'SP15', 'ZP26']:
        zone_features = {
            'zone': zone,
            'time': row['Time'],
            'load_mw': load_mw,
            'fuel_mix': fuel_mix,
            'predicted_lmp': predicted_lmp,
            'weather': zone_weather.get(zone, {}),
            'fetched_at': datetime.now(timezone.utc).isoformat(),
        }
        cache_features(redis_client, zone, zone_features)
        cache_summary['zones'][zone] = 'ok'

    # 5. Compute per-DC advisories
    if not dc_sites.empty:
        for _, dc in dc_sites.iterrows():
            zone = dc.get('caiso_zone')
            zone_features = {
                'predicted_lmp': predicted_lmp,
            }
            advisory = compute_dc_advisory(dc, zone_features)
            cache_dc_advisory(redis_client, dc['dc_id'], advisory)
        cache_summary['dcs'] = len(dc_sites)

    # 6. Update meta
    redis_client.setex(
        'meta:last_fetch', 600,
        json.dumps({
            'cycle_at': cycle_start.isoformat(),
            'load_mw': load_mw,
            'predicted_lmp': predicted_lmp,
            'n_dcs_cached': cache_summary['dcs'],
        })
    )

    elapsed = (datetime.now(timezone.utc) - cycle_start).total_seconds()
    logger.info(f"=== Cycle done in {elapsed:.1f}s ===")
    return {
        'status': 'ok',
        'cycle_at': cycle_start.isoformat(),
        'elapsed_sec': elapsed,
        'load_mw': load_mw,
        'predicted_lmp': predicted_lmp,
        'n_zones': 3,
        'n_dcs': cache_summary['dcs'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loop', action='store_true', help='run every 5 min forever')
    parser.add_argument('--interval', type=int, default=300, help='seconds between cycles')
    args = parser.parse_args()

    redis_client = get_redis_client()
    dc_sites = load_dc_sites()
    logger.info(f"Loaded {len(dc_sites)} DC sites")

    if args.loop:
        logger.info(f"Running loop mode, interval={args.interval}s")
        while True:
            try:
                fetch_cycle(redis_client, dc_sites)
            except Exception as e:
                logger.exception(f"Cycle failed: {e}")
            time.sleep(args.interval)
    else:
        result = fetch_cycle(redis_client, dc_sites)
        print(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    main()
