"""
Drift detector for dc_real_time.

Computes PSI (Population Stability Index) for model features
by comparing reference distributions (from training data) against
current distributions (from Redis live data).

Usage:
  python -m src.monitoring.drift_detector              # one-shot
  python -m src.monitoring.drift_detector --loop       # every hour
  python -m src.monitoring.drift_detector --build-ref  # rebuild reference only

Writes:
  artifacts/drift_log.json       — consumed by retrain_scheduler.py
  artifacts/drift_reference.json — feature bin edges (generated on first run)
"""
import argparse
import json
import logging
import math
import os
import sys
import time as _time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

N_BINS = 20
DRIFT_LOG_PATH = Path('/app/artifacts/drift_log.json')
DRIFT_REF_PATH = Path('/app/artifacts/drift_reference.json')

ZONES = ['NP15', 'SP15', 'ZP26']

FEATURES_LMP = [
    'lmp_mean_60m', 'lmp_std_60m', 'lmp_mean_4h', 'lmp_std_4h',
    'lmp_mean_24h', 'lmp_std_24h', 'lmp_max_24h', 'lmp_min_24h',
    'lmp_lag_1', 'lmp_lag_12', 'lmp_lag_48',
    'lmp_slope_60m', 'lmp_pct_change_5m', 'lmp_pct_change_60m', 'lmp_range_4h',
]

FEATURES_CROSS_ZONE = [
    'lmp_spread_np_sp', 'lmp_spread_np_zp', 'lmp_spread_sp_zp',
    'lmp_max_across_zones', 'lmp_min_across_zones', 'lmp_mean_across_zones',
    'lmp_max_across_zones_60m',
]

FEATURES_FUEL = [
    'Energy', 'Congestion', 'Loss',
    'solar_mw_60m_mean', 'wind_mw_60m_mean', 'natural_gas_mw_60m_mean',
    'imports_mw_60m_mean', 'nuclear_mw_60m_mean',
    'large_hydro_mw_60m_mean', 'batteries_mw_60m_mean',
]

ALL_FEATURES = FEATURES_LMP + FEATURES_CROSS_ZONE + FEATURES_FUEL


def compute_psi(reference_pcts: list, current_values: np.ndarray, bin_edges: list) -> float:
    """Compute PSI between reference bin distribution and a batch of current values.

    PSI = sum((actual_i - expected_i) * ln(actual_i / expected_i))
    Applied Laplace smoothing to avoid log(0).
    """
    if len(current_values) < 2:
        return 0.0
    counts, _ = np.histogram(current_values, bins=bin_edges)
    n = len(current_values)
    current_pcts = np.array([max(c / n, 1e-6) for c in counts], dtype=np.float64)
    ref_pcts = np.array([max(p, 1e-6) for p in reference_pcts], dtype=np.float64)
    psi = float(np.sum((current_pcts - ref_pcts) * np.log(current_pcts / ref_pcts)))
    return max(psi, 0.0)


def _in_window(values, timestamps, window_sec, end_idx=None):
    """Get values within window_sec seconds before end_idx."""
    if end_idx is None:
        end_idx = len(timestamps) - 1
    end_t = timestamps[end_idx]
    mask = np.array([(end_t - t).total_seconds() <= window_sec and (end_t - t).total_seconds() >= 0
                     for t in timestamps])
    return values[mask]


def compute_lmp_distributions(zone_lmps, zone_timestamps, all_zone_hists):
    """Compute per-feature distributions from LMP history.

    For each 5-min step in the history, computes the 22 LMP-derived features
    using look-back windows (60m, 4h, 24h). Returns a dict mapping feature
    name to list of values.
    """
    lmps = np.array([p[0] for p in zone_lmps], dtype=np.float64)
    times = [p[1] for p in zone_lmps]
    n = len(lmps)

    if n < 48:
        return {}

    out = {feat: [] for feat in FEATURES_LMP + FEATURES_CROSS_ZONE}

    # As a simplifying approximation, compute features at each 5-min step
    # using the data available up to that point. Skip the first 48 (4h)
    # so we have enough history.
    for i in range(48, n):
        last_60m = _in_window(lmps, times, 3600, i)
        last_4h = _in_window(lmps, times, 4 * 3600, i)
        last_24h = _in_window(lmps, times, 24 * 3600, i)
        up_to_i = lmps[:i + 1]

        if len(last_60m) == 0:
            continue

        feats = {}

        feats['lmp_mean_60m'] = float(np.mean(last_60m))
        feats['lmp_std_60m'] = float(np.std(last_60m)) if len(last_60m) > 1 else 0.0
        feats['lmp_mean_4h'] = float(np.mean(last_4h)) if len(last_4h) > 0 else feats['lmp_mean_60m']
        feats['lmp_std_4h'] = float(np.std(last_4h)) if len(last_4h) > 1 else 0.0
        feats['lmp_mean_24h'] = float(np.mean(last_24h)) if len(last_24h) > 0 else feats['lmp_mean_60m']
        feats['lmp_std_24h'] = float(np.std(last_24h)) if len(last_24h) > 1 else 0.0
        feats['lmp_max_24h'] = float(np.max(last_24h)) if len(last_24h) > 0 else float(lmps[i])
        feats['lmp_min_24h'] = float(np.min(last_24h)) if len(last_24h) > 0 else float(lmps[i])

        feats['lmp_lag_1'] = float(up_to_i[-1])
        feats['lmp_lag_12'] = float(up_to_i[-12]) if len(up_to_i) >= 12 else float(up_to_i[0])
        feats['lmp_lag_48'] = float(up_to_i[-48]) if len(up_to_i) >= 48 else float(up_to_i[0])

        if len(last_60m) >= 2:
            x = np.arange(len(last_60m), dtype=np.float64)
            y = last_60m
            x_m, y_m = np.mean(x), np.mean(y)
            cov = np.mean((x - x_m) * (y - y_m))
            var = np.mean((x - x_m) ** 2)
            feats['lmp_slope_60m'] = float(cov / var) if var > 0 else 0.0
        else:
            feats['lmp_slope_60m'] = 0.0

        if len(up_to_i) >= 2:
            feats['lmp_pct_change_5m'] = float((up_to_i[-1] - up_to_i[-2]) / up_to_i[-2]) if up_to_i[-2] != 0 else 0.0
        else:
            feats['lmp_pct_change_5m'] = 0.0
        if len(up_to_i) >= 13:
            feats['lmp_pct_change_60m'] = float((up_to_i[-1] - up_to_i[-13]) / up_to_i[-13]) if up_to_i[-13] != 0 else 0.0
        else:
            feats['lmp_pct_change_60m'] = 0.0

        feats['lmp_range_4h'] = float(np.max(last_4h) - np.min(last_4h)) if len(last_4h) > 0 else 0.0

        # Cross-zone features from latest values
        latest_per_zone = {}
        for z, hists in all_zone_hists.items():
            entries = hists.get('lmps', [])
            idx = min(i, len(entries) - 1) if entries else -1
            latest_per_zone[z] = entries[idx][0] if idx >= 0 else lmps[i]

        n_lmp = [latest_per_zone.get('NP15', 0), latest_per_zone.get('SP15', 0), latest_per_zone.get('ZP26', 0)]
        feats['lmp_spread_np_sp'] = latest_per_zone.get('NP15', 0) - latest_per_zone.get('SP15', 0)
        feats['lmp_spread_np_zp'] = latest_per_zone.get('NP15', 0) - latest_per_zone.get('ZP26', 0)
        feats['lmp_spread_sp_zp'] = latest_per_zone.get('SP15', 0) - latest_per_zone.get('ZP26', 0)
        feats['lmp_max_across_zones'] = float(max(n_lmp))
        feats['lmp_min_across_zones'] = float(min(n_lmp))
        feats['lmp_mean_across_zones'] = float(np.mean(n_lmp))
        feats['lmp_max_across_zones_60m'] = float(max(n_lmp))

        for feat in out:
            out[feat].append(feats.get(feat, 0.0))

    return {feat: np.array(vals) for feat, vals in out.items() if len(vals) > 0}


def load_or_build_reference() -> dict:
    """Load cached reference or build from training data."""
    if DRIFT_REF_PATH.exists():
        with open(DRIFT_REF_PATH) as f:
            return json.load(f)
    return build_reference()


def build_reference() -> dict:
    """Build reference distributions from training data."""
    train_path = Path('/app/data/processed/train.parquet')
    if not train_path.exists():
        logger.warning(f"  Training data not found at {train_path}; drift reference not available")
        return None

    import pandas as pd
    logger.info("  Building reference from training data...")
    df = pd.read_parquet(train_path)
    logger.info(f"  Loaded {len(df):,} training rows")

    available = [c for c in ALL_FEATURES if c in df.columns]
    logger.info(f"  Available features: {len(available)}/{len(ALL_FEATURES)}")

    reference = {
        'built_at': datetime.now(timezone.utc).isoformat(),
        'n_train': len(df),
        'features': {},
    }

    for feat in available:
        values = df[feat].dropna().values.astype(np.float64)
        if len(values) < N_BINS:
            continue

        percentiles = np.linspace(0, 100, N_BINS + 1)
        bin_edges = np.percentile(values, percentiles)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        counts, _ = np.histogram(values, bins=bin_edges)
        bin_pcts = (counts / len(values)).tolist()

        reference['features'][feat] = {
            'bin_edges': bin_edges.tolist(),
            'bin_pcts': bin_pcts,
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'p5': float(np.percentile(values, 5)),
            'p95': float(np.percentile(values, 95)),
        }

    with open(DRIFT_REF_PATH, 'w') as f:
        json.dump(reference, f, indent=2)
    logger.info(f"  Saved reference: {len(reference['features'])} features")
    return reference


def read_redis_history(redis_client, zone: str) -> list:
    """Read per-zone LMP history from Redis as (lmp, timestamp) pairs."""
    key = f"features:zone:{zone}:lmp_history"
    raw = redis_client.zrange(key, 0, -1)
    out = []
    for entry in raw:
        try:
            d = json.loads(entry)
            if 'lmp' in d and 't' in d:
                t = datetime.fromisoformat(d['t'].replace('Z', '+00:00'))
                out.append((float(d['lmp']), t))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return out


def read_zone_snapshot(redis_client, zone: str) -> dict:
    """Read latest zone features from Redis cache."""
    key = f"features:zone:{zone}:now"
    raw = redis_client.get(key)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def collect_current_features() -> dict:
    """Collect current feature distributions from Redis.

    Returns dict of {feature_name: np.array([...values...])}.
    """
    import redis
    r = redis.from_url(os.environ.get('REDIS_URL', 'redis://redis:6379'),
                       decode_responses=True, socket_connect_timeout=5)

    # Read LMP history for all 3 zones
    zone_hists = {}
    for zone in ZONES:
        entries = read_redis_history(r, zone)
        zone_hists[zone] = {
            'lmps': entries,
            'n': len(entries),
        }
        logger.info(f"  {zone}: {len(entries)} LMP history entries")

    # Build per-zone LMP feature distributions
    all_feature_values = {feat: [] for feat in FEATURES_LMP + FEATURES_CROSS_ZONE}

    # Prepare cross-zone history for all zones
    all_zone_hists_full = {}
    for zone in ZONES:
        all_zone_hists_full[zone] = zone_hists[zone]

    for zone in ZONES:
        entries = zone_hists[zone]['lmps']
        if len(entries) < 48:
            logger.warning(f"  {zone}: only {len(entries)} entries, need 48+")
            continue

        dists = compute_lmp_distributions(entries, zone_hists, all_zone_hists_full)
        for feat, vals in dists.items():
            all_feature_values[feat].extend(vals.tolist())

    # Latest snapshot features (fuel mix, load) — single value per zone
    snapshot_features = {feat: [] for feat in FEATURES_FUEL}
    for zone in ZONES:
        snap = read_zone_snapshot(r, zone)
        load_mw = snap.get('load_mw', 0)
        fuel_mix = snap.get('fuel_mix', {})

        if FEATURES_FUEL[0] in ALL_FEATURES:
            snapshot_features['Energy'].append(load_mw if load_mw else 0)
        if 'solar_mw_60m_mean' in ALL_FEATURES:
            snapshot_features['solar_mw_60m_mean'].append(fuel_mix.get('Solar', 0) or 0)
        if 'natural_gas_mw_60m_mean' in ALL_FEATURES:
            snapshot_features['natural_gas_mw_60m_mean'].append(fuel_mix.get('NaturalGas', 0) or 0)
        if 'wind_mw_60m_mean' in ALL_FEATURES:
            snapshot_features['wind_mw_60m_mean'].append(fuel_mix.get('Wind', 0) or 0)
        if 'imports_mw_60m_mean' in ALL_FEATURES:
            snapshot_features['imports_mw_60m_mean'].append(fuel_mix.get('Imports', 0) or 0)
        if 'nuclear_mw_60m_mean' in ALL_FEATURES:
            snapshot_features['nuclear_mw_60m_mean'].append(fuel_mix.get('Nuclear', 0) or 0)
        if 'large_hydro_mw_60m_mean' in ALL_FEATURES:
            snapshot_features['large_hydro_mw_60m_mean'].append(fuel_mix.get('Hydro', 0) or 0)
        if 'batteries_mw_60m_mean' in ALL_FEATURES:
            snapshot_features['batteries_mw_60m_mean'].append(fuel_mix.get('Batteries', 0) or 0)

    # Merge LMP-derived features + snapshot features
    result = {}
    for feat, vals in all_feature_values.items():
        arr = np.array(vals, dtype=np.float64)
        if len(arr) > 0:
            result[feat] = arr
    for feat, vals in snapshot_features.items():
        arr = np.array(vals, dtype=np.float64)
        if len(arr) > 0:
            result[feat] = arr

    return result


def compute_drift() -> dict:
    """Main drift computation."""
    logger.info("=== Drift detector cycle ===")

    reference = load_or_build_reference()
    if reference is None:
        return {"max_psi": 0.0, "should_retrain": False, "reason": "no reference distribution"}
    if not reference.get('features'):
        return {"max_psi": 0.0, "should_retrain": False, "reason": "empty reference"}

    current = collect_current_features()
    if not current:
        return {"max_psi": 0.0, "should_retrain": False, "reason": "no current data"}

    threshold = float(os.environ.get('DRIFT_PSI_THRESHOLD', '0.2'))

    feature_psis = {}
    max_psi = 0.0
    worst_feature = None

    ref_features = reference['features']
    for feat in ref_features:
        if feat not in current:
            continue
        vals = current[feat]
        if len(vals) < N_BINS:
            continue
        psi = compute_psi(
            ref_features[feat]['bin_pcts'],
            vals,
            ref_features[feat]['bin_edges'],
        )
        feature_psis[feat] = round(psi, 6)
        if psi > max_psi:
            max_psi = psi
            worst_feature = feat

    should_retrain = max_psi > threshold

    result = {
        'max_psi': round(max_psi, 6),
        'worst_feature': worst_feature,
        'n_features_compared': len(feature_psis),
        'n_features_reference': len(ref_features),
        'computed_at': datetime.now(timezone.utc).isoformat(),
        'threshold': threshold,
        'should_retrain': should_retrain,
        'feature_psis': feature_psis,
    }

    logger.info(
        f"  max_psi={result['max_psi']:.4f} ({worst_feature}), "
        f"{result['n_features_compared']}/{result['n_features_reference']} features, "
        f"should_retrain={should_retrain}"
    )
    return result


def save_drift_log(result: dict):
    """Append result to drift_log.json."""
    log = {'history': []}
    if DRIFT_LOG_PATH.exists():
        try:
            with open(DRIFT_LOG_PATH) as f:
                existing = json.load(f)
            if 'latest' in existing:
                log['history'].append(existing['latest'])
            if existing.get('history'):
                log['history'].extend(existing['history'])
        except (json.JSONDecodeError, OSError):
            pass

    log['history'] = log['history'][-100:]
    log['latest'] = result

    with open(DRIFT_LOG_PATH, 'w') as f:
        json.dump(log, f, indent=2)
    logger.info(f"  Wrote drift log: max_psi={result['max_psi']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Drift detector')
    parser.add_argument('--loop', action='store_true', help='run in loop mode')
    parser.add_argument('--interval', type=int, default=3600, help='loop interval (seconds)')
    parser.add_argument('--build-ref', action='store_true', help='only build reference distributions')
    args = parser.parse_args()

    if args.build_ref:
        build_reference()
        return

    if args.loop:
        logger.info(f"Drift detector loop mode, interval={args.interval}s")
        while True:
            try:
                result = compute_drift()
                save_drift_log(result)
            except Exception as e:
                logger.exception(f"Cycle failed: {e}")
            _time.sleep(args.interval)
    else:
        result = compute_drift()
        save_drift_log(result)
        if not result.get('should_retrain', False):
            logger.info("No drift detected")
        print(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    main()
