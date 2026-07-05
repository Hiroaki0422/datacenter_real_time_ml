"""
Model registry for dc_real_time.

Tracks all model versions with metadata:
- Training date, git SHA, data hash
- Validation metrics
- Promotion status (champion, candidate, archived)
- Hyperparameters and feature schema version

Stored as JSON in models/registry.json. Atomic writes via temp file + rename.

The registry path is /app/models/registry.json inside the container (the
default) and can be overridden via MODELS_DIR for host-side scripts
(e.g. backfill, dev tooling). The atomic symlink swap is also performed
relative to MODELS_DIR.
"""
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Inside container: /app/models. On host: override via env var.
_MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/app/models"))
REGISTRY_PATH = _MODELS_DIR / "registry.json"


def load_registry() -> dict:
    """Load the registry from disk, or return empty schema."""
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {
        "schema_version": "1.0",
        "champion": None,
        "candidates": [],
        "history": [],
        "notes": "Empty registry"
    }


def save_registry(registry: dict) -> None:
    """Atomic write to disk (write to .tmp, then rename)."""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = REGISTRY_PATH.with_suffix('.tmp')
    with open(tmp_path, 'w') as f:
        json.dump(registry, f, indent=2, default=str)
    os.replace(tmp_path, REGISTRY_PATH)


def register_candidate(
    version: str,
    metrics: dict,
    model_paths: dict,
    notes: str = "",
) -> dict:
    """Register a newly trained model as a candidate.

    Args:
        version: e.g. "v0.2"
        metrics: {"lmp_ratio": {"val_r2": 0.71, ...}, "carbon": {...}}
        model_paths: {"lmp_ratio": "models/v0.2/lmp_ratio.json", ...}
        notes: free text

    Returns:
        The updated registry.
    """
    registry = load_registry()

    git_sha = _get_git_sha()
    data_hash = _get_data_hash()

    candidate = {
        "version": version,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "data_hash": data_hash,
        "metrics": metrics,
        "model_paths": model_paths,
        "status": "candidate",
        "notes": notes,
    }
    registry["candidates"].append(candidate)
    save_registry(registry)
    return registry


def promote_to_champion(version: str) -> dict:
    """Promote a candidate to champion (atomic symlink swap).

    Steps:
      1. Find candidate by version
      2. Archive current champion to history
      3. Set new champion
      4. Update symlink models/champion -> models/{version}
      5. Return updated registry

    The symlink swap is atomic on Linux (POSIX rename(2)).
    """
    registry = load_registry()
    candidate = next(
        (c for c in registry["candidates"] if c["version"] == version),
        None
    )
    if not candidate:
        raise ValueError(f"No candidate with version {version}")

    # Archive current champion
    if registry["champion"]:
        archived = dict(registry["champion"])
        archived["archived_at"] = datetime.now(timezone.utc).isoformat()
        registry["history"].append(archived)

    # Promote
    candidate["status"] = "champion"
    candidate["promoted_at"] = datetime.now(timezone.utc).isoformat()
    registry["champion"] = candidate
    save_registry(registry)

    # Atomic symlink swap. Use a RELATIVE target so the symlink resolves
    # identically whether you're on the host (where the volume lives at
    # e.g. /root/project/dc_real_time/models/) or inside the container
    # (where the same dir is mounted at /app/models/). An absolute target
    # would resolve to a host-only path and break the container.
    champion_link = REGISTRY_PATH.parent / "champion"
    if champion_link.is_symlink() or champion_link.exists():
        champion_link.unlink()
    target = REGISTRY_PATH.parent / version
    if not target.exists():
        raise FileNotFoundError(f"Model dir {target} does not exist")
    # Use just the version dir name (relative), not the absolute path.
    champion_link.symlink_to(version)

    return registry


def get_champion_metrics() -> Optional[dict]:
    """Get the current champion's metrics, or None if no champion."""
    registry = load_registry()
    if registry["champion"]:
        return registry["champion"].get("metrics")
    return None


def _get_git_sha() -> str:
    """Get current git SHA, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _get_data_hash() -> str:
    """Get a quick hash of the training data files (mtime + size)."""
    import hashlib
    data_dir = Path(os.environ.get("DATA_DIR", "/app/data/processed"))
    if not data_dir.exists():
        return "no-data"
    h = hashlib.sha256()
    for f in sorted(data_dir.glob('*.parquet')):
        stat = f.stat()
        h.update(f.name.encode())
        h.update(str(stat.st_size).encode())
        h.update(str(int(stat.st_mtime)).encode())
    return h.hexdigest()[:12]
