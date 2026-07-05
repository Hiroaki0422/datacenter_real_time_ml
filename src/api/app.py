"""
FastAPI application for dc_real_time.

Endpoints:
  GET  /healthz             — liveness check
  GET  /readyz              — readiness check (model loaded, deps OK)
  GET  /forecast/{zone}     — LMP ratio + carbon forecast for a zone
  GET  /dc/{dc_id}/forecast — per-DC advisory
  POST /admin/reload        — reload model from disk (for SIGHUP-less reload)

This is a Phase 4 stub. Full implementation in D8.
"""
from fastapi import FastAPI
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="dc_real_time API",
    version="0.1.0",
    description="Spatial-temporal carbon & price forecasting for data centers",
)

# Track when the model was last loaded (for the /healthz response)
_model_loaded_at: str = "stub"


@app.get("/healthz")
def healthz():
    """Liveness check — returns 200 if the process is running."""
    return {
        "status": "ok",
        "model_loaded_at": _model_loaded_at,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/readyz")
def readyz():
    """Readiness check — returns 200 when the service can serve traffic."""
    return {"ready": True, "service": "dc_real_time_api", "version": "0.1.0"}


@app.get("/forecast/{zone}")
def forecast(zone: str):
    """Stub: returns placeholder forecast for a CAISO zone.

    Real implementation in Phase 4 D8: load 5-min LMP + carbon models,
    call live_fetcher, return prediction with confidence interval.
    """
    return {
        "zone": zone,
        "horizon_min": 5,
        "lmp_ratio_pred": None,
        "carbon_pred_short_ton_per_mwh": None,
        "advisory": "stub",
        "note": "Phase 4 stub. Implementation in D8.",
    }


@app.get("/dc/{dc_id}/forecast")
def dc_forecast(dc_id: str):
    """Stub: per-DC advisory."""
    return {
        "dc_id": dc_id,
        "zone": None,
        "advisory": "stub",
        "note": "Phase 4 stub. Implementation in D8.",
    }


@app.post("/admin/reload")
def admin_reload():
    """Trigger model reload from disk (for atomic symlink swap)."""
    global _model_loaded_at
    _model_loaded_at = datetime.now().isoformat()
    logger.info("Model reload triggered")
    return {"reloaded": True, "at": _model_loaded_at}
