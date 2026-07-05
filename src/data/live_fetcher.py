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


# === Per-zone LMP fetch (D8 full) ===

# CAISO trading hub IDs — same as used during v0.1 training (see
# notebooks/colab_handoff/01_caiso_1y_backfill.md).
ZONE_TO_LOCATION = {
    'NP15': 'TH_NP15_GEN-APND',
    'SP15': 'TH_SP15_GEN-APND',
    'ZP26': 'TH_ZP26_GEN-APND',
}

# Per-zone LMP history TTL in Redis: 25h (24h of history + 1h buffer).
# Each entry is one (timestamp, lmp) pair; we keep ~288 per zone (5-min × 24h).
LMP_HISTORY_TTL_SEC = 25 * 3600


def fetch_caiso_per_zone_lmp() -> dict:
    """Fetch today's per-zone LMP intervals from CAISO OASIS via gridstatus.

    Real OASIS LMP has 1-2h lag, but the REAL_TIME_5_MIN market is
    available within ~5-15 min. For a 5-min fetcher cycle, this gives
    us per-zone LMP that's at most 15 min stale — good enough for
    rolling stats over 60m/4h/24h windows.

    Returns: {zone: {latest: {lmp, energy, ...}, intervals: [{time, lmp}, ...]}}
    - latest is the most recent interval (used as "current LMP" for that zone)
    - intervals is the full day (used to backfill history on first run)
    Empty dict on failure.
    """
    try:
        import gridstatus
    except ImportError:
        logger.warning("  gridstatus not installed; per-zone LMP unavailable")
        return {}

    try:
        caiso = gridstatus.CAISO()
        # gridstatus calls OASIS with date='today' and returns the full day's
        # data. We take the most recent interval per zone as the "current" LMP,
        # and keep all intervals for history backfill.
        df = caiso.get_lmp(date='today', market='REAL_TIME_5_MIN')
        if df is None or df.empty:
            logger.warning("  gridstatus returned empty per-zone LMP")
            return {}
        if 'Location' not in df.columns or 'Time' not in df.columns or 'LMP' not in df.columns:
            logger.warning(f"  Unexpected gridstatus columns: {df.columns.tolist()[:8]}")
            return {}

        # Build location -> zone map (reverse of ZONE_TO_LOCATION)
        loc_to_zone = {v: k for k, v in ZONE_TO_LOCATION.items()}

        # Filter to only the 3 trading hubs we care about
        df = df[df['Location'].isin(ZONE_TO_LOCATION.values())].copy()
        if df.empty:
            logger.warning("  No rows for our 3 zones in gridstatus response")
            return {}

        df = df.sort_values('Time')

        result = {}
        for zone, location in ZONE_TO_LOCATION.items():
            zone_df = df[df['Location'] == location]
            if zone_df.empty:
                continue
            latest = zone_df.iloc[-1]
            intervals = [
                {
                    'time': row['Time'],
                    'lmp': float(row['LMP']),
                }
                for _, row in zone_df.iterrows()
            ]
            result[zone] = {
                'latest': {
                    'lmp': float(latest['LMP']),
                    'energy': float(latest.get('Energy', 0) or 0),
                    'congestion': float(latest.get('Congestion', 0) or 0),
                    'loss': float(latest.get('Loss', 0) or 0),
                    'ghg': float(latest.get('GHG', 0) or 0),
                    'time': latest['Time'].isoformat() if hasattr(latest['Time'], 'isoformat') else str(latest['Time']),
                },
                'intervals': intervals,
            }
        return result
    except Exception as e:
        logger.warning(f"  Per-zone LMP fetch failed: {e}")
        return {}


def append_lmp_history(redis_client, zone: str, lmp: float, ts_unix: float) -> None:
    """Append a (timestamp, lmp) pair to the per-zone LMP history sorted set.

    Key: features:zone:{zone}:lmp_history
    Score: unix timestamp
    Value: JSON {t: iso_time, lmp: float}

    Old entries (>25h) are removed via ZREMRANGEBYSCORE.
    """
    import time as _time
    key = f"features:zone:{zone}:lmp_history"
    iso_time = datetime.fromtimestamp(ts_unix, tz=timezone.utc).isoformat()
    value = json.dumps({"t": iso_time, "lmp": lmp})
    cutoff = ts_unix - LMP_HISTORY_TTL_SEC
    pipe = redis_client.pipeline()
    pipe.zremrangebyscore(key, "-inf", cutoff)
    pipe.zadd(key, {value: ts_unix})
    pipe.expire(key, LMP_HISTORY_TTL_SEC)
    pipe.execute()


def read_lmp_history(redis_client, zone: str, max_age_sec: int = 24 * 3600) -> list:
    """Read per-zone LMP history from Redis as a list of {t, lmp} dicts.

    Returns entries newer than (now - max_age_sec), sorted by timestamp ascending.
    Empty list if no history (caller should fall back to defaults).
    """
    import time as _time
    key = f"features:zone:{zone}:lmp_history"
    cutoff = _time.time() - max_age_sec
    raw = redis_client.zrangebyscore(key, cutoff, "+inf")
    out = []
    for entry in raw:
        try:
            out.append(json.loads(entry))
        except (json.JSONDecodeError, TypeError):
            continue
    return out


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
    """Cache per-DC advisory (single JSON with all 4 horizons) in Redis.

    The advisory dict has shape:
      {
        'dc_id': ..., 'name': ..., 'zone': ..., 'mw_capacity': ..., 'wue': ..., 'bws_score': ...,
        'horizons': {
          '30m': {'lmp': 28.5, 'advisory': 'ok', 'computed_at': ...},
          '1h':  {...}, '2h': {...}, '4h': {...},
        },
        'computed_at': ...,
      }
    """
    key = f"features:dc:{dc_id}:now"
    redis_client.setex(key, ttl_sec, json.dumps(advisory, default=str))


def _lmp_to_advisory(lmp: float | None) -> str:
    """Map a predicted LMP ($/MWh) to an advisory string."""
    if lmp is None or (isinstance(lmp, float) and np.isnan(lmp)):
        return "unknown"
    if lmp > 100:
        return "pause"
    if lmp > 50:
        return "defer"
    if lmp > 30:
        return "watch"
    return "ok"


def compute_dc_advisory(dc: pd.Series, zone_horizons: dict) -> dict:
    """Compute per-horizon advisory for a single DC.

    Args:
        dc: row from ca_dc_sites.csv (must have dc_id, name, provider, caiso_zone, etc.)
        zone_horizons: {
          '30m': 27.5,    # $/MWh, or None
          '1h':  31.2,
          '2h':  29.0,
          '4h':  33.5,
        }
        The per-horizon LMP for the DC's zone — comes from the zone's per-horizon
        prediction (currently all DCs in a zone get the same value, but the
        schema supports per-DC future personalization).
    """
    horizons = {}
    for h in ['30m', '1h', '2h', '4h']:
        lmp = zone_horizons.get(h)
        horizons[h] = {
            'lmp_dollar_per_mwh': lmp,
            'advisory': _lmp_to_advisory(lmp),
        }
    # Primary advisory = worst across all 4 horizons (most conservative for ops)
    severity = {'ok': 0, 'unknown': 1, 'watch': 2, 'defer': 3, 'pause': 4}
    worst = max(horizons.values(), key=lambda x: severity.get(x['advisory'], 1))['advisory']
    return {
        'dc_id': dc.get('dc_id'),
        'name': dc.get('name'),
        'zone': dc.get('caiso_zone'),
        'operator': dc.get('provider'),
        'mw_capacity': float(dc.get('MW_total_power', 0)) if pd.notna(dc.get('MW_total_power')) else 0,
        'wue': float(dc.get('wue_default', 1.18)) if pd.notna(dc.get('wue_default')) else 1.18,
        'bws_score': float(dc.get('bws_score', 0)) if pd.notna(dc.get('bws_score')) else 0,
        'advisory': worst,           # worst-case across all horizons
        'horizons': horizons,
        'computed_at': datetime.now(timezone.utc).isoformat(),
    }


def predict_lmp_per_horizon(zone: str, per_zone_lmp: dict, system_lmp: 'float | None' = None) -> dict:
    """For a given zone, return predicted LMP at each of the 4 horizons.

    Uses the most recent real per-zone LMP if available, else falls back to
    the system-wide LMP. Until the API exposes per-horizon model calls,
    we use the same anchor for all 4 horizons (best estimate at all
    horizons given current data).

    Args:
        zone: 'NP15' | 'SP15' | 'ZP26'
        per_zone_lmp: dict from fetch_caiso_per_zone_lmp() — has 'latest' per zone
        system_lmp: fallback LMP ($/MWh) if per-zone data is unavailable

    Returns: {'30m': float, '1h': float, '2h': float, '4h': float} (all in $/MWh)
    """
    zone_data = per_zone_lmp.get(zone, {})
    latest = zone_data.get('latest', {})
    lmp_now = latest.get('lmp')
    if lmp_now is None:
        lmp_now = system_lmp  # fallback to system-wide
    if lmp_now is None:
        return {'30m': None, '1h': None, '2h': None, '4h': None}
    return {'30m': lmp_now, '1h': lmp_now, '2h': lmp_now, '4h': lmp_now}


# === Main fetch cycle ===

def fetch_cycle(redis_client, dc_sites: pd.DataFrame) -> dict:
    """One fetch cycle: get live data, predict, cache."""
    cycle_start = datetime.now(timezone.utc)
    logger.info("=== Live fetch cycle started ===")

    # 1. Get system-wide data (best effort — if it fails, fall back to 0 load)
    iso_data = fetch_caiso_load_fuelmix()
    if iso_data.empty:
        logger.warning("  CAISO system load/fuelmix unreachable; using defaults")
        load_mw = 0
        fuel_mix = {}
        row = None
    else:
        row = iso_data.iloc[0]
        load_mw = row['Load_MW']
        fuel_mix = {k.replace('_MW', ''): v for k, v in row.items() if k.endswith('_MW')}

    # 2. Predict LMP from current state (use load_mw=0 if no data — model still gives
    # a reasonable default).
    predicted_lmp = predict_lmp_from_state(load_mw, fuel_mix)
    logger.info(f"  System LMP: ${predicted_lmp:.2f}/MWh (load={load_mw}MW)")

    # 2b. Fetch per-zone LMP from CAISO OASIS via gridstatus (D8 full)
    # Returns real per-zone LMPs (REAL_TIME_5_MIN market, ~5-15min lag).
    # These are used as the "current LMP" per zone, and appended to a
    # rolling history in Redis for the 22 LMP-derived features the
    # model was trained on.
    import time as _time
    cycle_ts = _time.time()
    per_zone_lmp = fetch_caiso_per_zone_lmp()
    if per_zone_lmp:
        n_intervals_total = 0
        for zone, lmp_data in per_zone_lmp.items():
            # Append all of today's intervals (backfills history fast on first run)
            for interval in lmp_data.get('intervals', []):
                ts = interval['time']
                # Convert pandas Timestamp to unix
                if hasattr(ts, 'timestamp'):
                    ts_unix = ts.timestamp()
                else:
                    # Parse the ISO string if needed
                    from datetime import datetime as _dt
                    if isinstance(ts, str):
                        ts_unix = _dt.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
                    else:
                        ts_unix = cycle_ts
                append_lmp_history(redis_client, zone, interval['lmp'], ts_unix)
                n_intervals_total += 1
        latest = per_zone_lmp.get('NP15', {}).get('latest', {})
        logger.info(
            f"  Per-zone LMP: "
            f"NP15=${per_zone_lmp.get('NP15', {}).get('latest', {}).get('lmp', 0):.2f} "
            f"SP15=${per_zone_lmp.get('SP15', {}).get('latest', {}).get('lmp', 0):.2f} "
            f"ZP26=${per_zone_lmp.get('ZP26', {}).get('latest', {}).get('lmp', 0):.2f} "
            f"({n_intervals_total} intervals written to history)"
        )
    else:
        logger.warning("  Per-zone LMP fetch failed; history not updated this cycle")

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
        # Use the per-zone LMP if we have it (D8 full); fall back to the
        # system-wide predicted_lmp so the cache is never empty.
        zone_lmp = per_zone_lmp.get(zone, {}).get('latest', {}).get('lmp', predicted_lmp)
        zone_features = {
            'zone': zone,
            'time': row['Time'] if row is not None else None,
            'load_mw': load_mw,
            'fuel_mix': fuel_mix,
            'predicted_lmp': zone_lmp,
            'predicted_lmp_source': 'oasis_rt5min' if per_zone_lmp.get(zone) else 'model_fallback',
            'weather': zone_weather.get(zone, {}),
            'fetched_at': datetime.now(timezone.utc).isoformat(),
        }
        cache_features(redis_client, zone, zone_features)
        cache_summary['zones'][zone] = 'ok'

    # 5. Compute per-DC advisories (per-horizon)
    if not dc_sites.empty:
        for _, dc in dc_sites.iterrows():
            zone = dc.get('caiso_zone')
            # Per-horizon LMP for this DC's zone
            zone_horizons = predict_lmp_per_horizon(zone, per_zone_lmp, predicted_lmp)
            advisory = compute_dc_advisory(dc, zone_horizons)
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
