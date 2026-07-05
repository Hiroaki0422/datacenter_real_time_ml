"""
Backfill the model registry for v0.1.

The v0.1 model was trained before registry.py existed, so it was promoted
manually via `ln -sfn models/v0.1 models/champion`. This script:

1. Reads metrics from artifacts/eval_lmp.json and eval_carbon.json
2. Calls register_candidate('v0.1', ...)  → adds v0.1 to candidates[]
3. Calls promote_to_champion('v0.1')      → moves to champion, archives nothing,
                                            re-asserts the symlink

This makes models/registry.json the authoritative record going forward.
The atomic symlink swap in promote_to_champion() is a no-op (the link
already points to v0.1) but it exercises the full code path.

Run once:
    python scripts/backfill_registry_v01.py
"""
import json
import os
import sys
from pathlib import Path

# Make src/ importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Point registry.py at the host paths (it defaults to /app/models in-container)
os.environ.setdefault("MODELS_DIR", str(ROOT / "models"))
os.environ.setdefault("DATA_DIR", str(ROOT / "data" / "processed"))

from src.models.registry import register_candidate, promote_to_champion, load_registry

ARTIFACTS = ROOT / "artifacts"
MODELS = ROOT / "models"


def main() -> None:
    # Load metrics from training-time eval files.
    # NOTE: eval_lmp.json contains 4h-horizon metrics; the 5-min model is what
    # actually serves. We backfill the metrics we have and flag the gap in notes
    # rather than blocking on regenerating them.
    metrics = {}
    lmp_eval = ARTIFACTS / "eval_lmp.json"
    carbon_eval = ARTIFACTS / "eval_carbon.json"

    if lmp_eval.exists():
        with open(lmp_eval) as f:
            metrics["lmp_ratio"] = json.load(f)
    else:
        print(f"WARNING: {lmp_eval} missing — using empty metrics for lmp_ratio")

    if carbon_eval.exists():
        with open(carbon_eval) as f:
            metrics["carbon"] = json.load(f)
    else:
        print(f"WARNING: {carbon_eval} missing — using empty metrics for carbon")

    model_paths = {
        "lmp_ratio": "models/v0.1/lmp_ratio.json",
        "carbon": "models/v0.1/carbon.json",
    }

    notes = (
        "Backfilled: v0.1 was the initial training run before registry.py existed. "
        "Symlink was set manually with `ln -sfn models/v0.1 models/champion`. "
        "eval_lmp.json contains 4h-horizon metrics; the 5-min model is what serves."
    )

    print("=== Registering v0.1 as candidate ===")
    register_candidate(
        version="v0.1",
        metrics=metrics,
        model_paths=model_paths,
        notes=notes,
    )

    print("\n=== Promoting v0.1 to champion ===")
    promote_to_champion("v0.1")

    print("\n=== Final registry state ===")
    registry = load_registry()
    print(json.dumps(registry, indent=2, default=str))

    print("\n=== Symlink check ===")
    champion_link = MODELS / "champion"
    if champion_link.is_symlink():
        print(f"  {champion_link} -> {champion_link.readlink()}")
    else:
        print(f"  {champion_link} is NOT a symlink!")


if __name__ == "__main__":
    main()
