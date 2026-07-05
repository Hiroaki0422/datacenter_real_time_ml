"""
Retrain scheduler for dc_real_time.

Runs as a cron-driven service (or standalone script) that:
  1. Checks for drift alerts (from drift_detector)
  2. Optionally trains a new model version
  3. Validates against current champion
  4. If better: registers as candidate
  5. If auto-promote enabled AND metrics improve: promote to champion (atomic symlink swap)

Usage:
  python -m src.models.retrain_scheduler          # full cycle
  python -m src.models.retrain_scheduler --check   # just check, don't train
  python -m src.models.retrain_scheduler --train  # force train (ignore drift check)

Environment:
  AUTO_PROMOTE=1   to auto-promote candidates to champion (default: 0)
  MIN_IMPROVEMENT   minimum metric improvement to promote (default: 0.01)
"""
import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.registry import (
    load_registry, register_candidate, promote_to_champion, get_champion_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


def check_drift_signal() -> dict:
    """Check if drift has been detected. Returns a dict with 'should_retrain' boolean."""
    # In a full system, this would query the drift detector's output
    # For now, check the latest drift log file
    drift_log = Path('/app/artifacts/drift_log.json')
    if not drift_log.exists():
        return {"should_retrain": False, "reason": "no drift log found"}
    try:
        with open(drift_log) as f:
            log = json.load(f)
        latest = log.get('latest', {})
        max_psi = latest.get('max_psi', 0.0)
        threshold = float(os.environ.get('DRIFT_PSI_THRESHOLD', '0.2'))
        if max_psi > threshold:
            return {
                "should_retrain": True,
                "reason": f"max_psi={max_psi:.3f} > threshold={threshold}",
                "max_psi": max_psi,
            }
        return {
            "should_retrain": False,
            "reason": f"max_psi={max_psi:.3f} <= threshold={threshold}",
            "max_psi": max_psi,
        }
    except Exception as e:
        return {"should_retrain": False, "reason": f"error reading drift log: {e}"}


def check_scheduled_retrain() -> dict:
    """Check if it's time for a scheduled retrain (e.g., weekly)."""
    registry = load_registry()
    if not registry.get("champion"):
        return {"should_retrain": True, "reason": "no champion yet"}
    last_trained = registry["champion"].get("promoted_at") or registry["champion"].get("registered_at")
    if not last_trained:
        return {"should_retrain": True, "reason": "no training timestamp"}
    try:
        last = datetime.fromisoformat(last_trained.replace('Z', '+00:00'))
        age_days = (datetime.now(timezone.utc) - last).days
        max_age = int(os.environ.get('RETRAIN_MAX_AGE_DAYS', '7'))
        if age_days >= max_age:
            return {
                "should_retrain": True,
                "reason": f"champion is {age_days} days old (max {max_age})",
                "age_days": age_days,
            }
        return {
            "should_retrain": False,
            "reason": f"champion is {age_days} days old (max {max_age})",
            "age_days": age_days,
        }
    except Exception as e:
        return {"should_retrain": True, "reason": f"error parsing timestamp: {e}"}


def train_new_model(version: str = None) -> dict:
    """Train a new model version (LMP + carbon at 4 horizons: 30m, 1h, 2h, 4h).

    Calls the multi-horizon training scripts, captures all 8 model files
    (4 LMP + 4 carbon), and returns a result dict structured for the
    registry: {version, horizons: {lmp_ratio: {30m:..., 1h:..., 2h:..., 4h:...},
                                    carbon:    {30m:..., 1h:..., 2h:..., 4h:...}}}

    Returns: dict ready for register_candidate(...)
    """
    # Determine next version if not provided
    if not version:
        registry = load_registry()
        existing_versions = set()
        if registry.get("champion"):
            existing_versions.add(registry["champion"]["version"])
        for c in registry.get("candidates", []):
            existing_versions.add(c["version"])
        for h in registry.get("history", []):
            existing_versions.add(h["version"])
        n = len(existing_versions)
        version = f"v0.{n + 1}"

    logger.info(f"Training new model version: {version}")

    # Ensure version dir exists
    new_dir = Path(os.environ.get('MODELS_DIR', '/app/models')) / version
    new_dir.mkdir(parents=True, exist_ok=True)

    # 1. Train LMP at all 4 horizons
    logger.info("=== Training LMP models (30m, 1h, 2h, 4h) ===")
    from src.models.train_lmp_multi_horizon import main as train_lmp
    lmp_result = train_lmp(version=version)
    logger.info(f"  LMP best horizon: {lmp_result['best_horizon']}")

    # 2. Train carbon at all 4 horizons
    logger.info("=== Training carbon models (30m, 1h, 2h, 4h) ===")
    from src.models.train_carbon_multi_horizon import main as train_carbon
    carbon_result = train_carbon(version=version)
    logger.info(f"  Carbon best horizon: {carbon_result['best_horizon']}")

    # 3. Build the registry metrics structure
    # Nested: {lmp_ratio: {30m: {metrics...}, 1h: {...}}, carbon: {30m: {...}, ...}}
    metrics = {
        'lmp_ratio': {h: lmp_result['horizons'][h] for h in ['30m', '1h', '2h', '4h']},
        'carbon':    {h: carbon_result['horizons'][h] for h in ['30m', '1h', '2h', '4h']},
    }
    model_paths = {
        'lmp_ratio': {h: lmp_result['horizons'][h]['model_path'] for h in ['30m', '1h', '2h', '4h']},
        'carbon':    {h: carbon_result['horizons'][h]['model_path'] for h in ['30m', '1h', '2h', '4h']},
    }

    return {
        "version": version,
        "metrics": metrics,
        "model_paths": model_paths,
        "best_lmp_horizon": lmp_result['best_horizon'],
        "best_carbon_horizon": carbon_result['best_horizon'],
        "n_train_total": lmp_result['n_train_total'] + carbon_result['n_train_total'],
    }


def should_promote(new_metrics: dict, champion_metrics: dict) -> tuple:
    """Decide if new model should be promoted. Returns (should_promote, reason).

    Compares on the 30-min LMP ratio R² (the closest-horizon average
    prediction we have for both versions). If the new model is at least
    MIN_IMPROVEMENT better, promote.
    """
    if not champion_metrics:
        return True, "no champion exists"

    # New metrics are nested: {lmp_ratio: {30m: {val_r2: ...}, ...}, carbon: {...}}
    new_r2 = 0
    if isinstance(new_metrics, dict):
        lmp = new_metrics.get('lmp_ratio', {})
        if isinstance(lmp, dict):
            # Prefer 30m (shortest horizon, most comparable), fall back to 1h
            new_r2 = lmp.get('30m', lmp.get('1h', {})).get('val_r2', 0)

    old_r2 = 0
    if isinstance(champion_metrics, dict):
        lmp = champion_metrics.get('lmp_ratio', {})
        if isinstance(lmp, dict):
            old_r2 = lmp.get('30m', lmp.get('1h', {})).get('val_r2', 0)
        # Backward compat: v0.1 had flat metrics with lmp_ratio_5m key
        if old_r2 == 0:
            old_r2 = champion_metrics.get('lmp_ratio_5m', {}).get('val_r2', 0)

    min_improvement = float(os.environ.get('MIN_IMPROVEMENT', '0.01'))

    if new_r2 >= old_r2 + min_improvement:
        return True, f"new 30m R²={new_r2:.4f} > old R²={old_r2:.4f} + {min_improvement}"
    return False, f"new 30m R²={new_r2:.4f} < old R²={old_r2:.4f} + {min_improvement}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='just check, dont train')
    parser.add_argument('--train', action='store_true', help='force train, ignore drift check')
    parser.add_argument('--auto-promote', action='store_true', help='auto-promote if better')
    args = parser.parse_args()

    logger.info("=== Retrain scheduler starting ===")

    # 1. Check if we should retrain
    if args.train:
        should_retrain = True
        reason = "--train flag set"
    else:
        drift = check_drift_signal()
        schedule = check_scheduled_retrain()
        should_retrain = drift['should_retrain'] or schedule['should_retrain']
        reason = f"drift: {drift.get('reason', '?')}; schedule: {schedule.get('reason', '?')}"

    logger.info(f"Should retrain: {should_retrain} ({reason})")

    if args.check or not should_retrain:
        logger.info("Exiting without training")
        return

    # 2. Train new model
    result = train_new_model()
    new_version = result["version"]
    new_metrics = result["metrics"]

    # 3. Register as candidate
    register_candidate(
        version=new_version,
        metrics=new_metrics,
        model_paths=result["model_paths"],
        notes=f"Auto-trained: {reason}",
    )
    logger.info(f"Registered candidate: {new_version}")

    # 4. Decide whether to promote
    champion = get_champion_metrics() or {}
    promote, promote_reason = should_promote(new_metrics, champion)
    logger.info(f"Should promote: {promote} ({promote_reason})")

    if promote:
        auto = args.auto_promote or os.environ.get('AUTO_PROMOTE', '0') == '1'
        if auto:
            promote_to_champion(new_version)
            logger.info(f"★ AUTO-PROMOTED {new_version} to champion")
        else:
            logger.info(f"  Candidate {new_version} registered; manual promotion required")
    else:
        logger.info(f"  Candidate {new_version} worse than champion; not promoting")

    logger.info("=== Retrain scheduler done ===")


if __name__ == '__main__':
    main()
