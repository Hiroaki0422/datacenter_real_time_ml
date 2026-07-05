# =============================================================================
# dc_real_time — Multi-stage Dockerfile
# =============================================================================
# One image, used by multiple services (api, trainer, drift-monitor) via CMD.
# Sized for VPS deployment (4 vCPU / 8GB) but portable to laptop or cloud.
#
# Build:    docker build -t dc_real_time_api:v0.1 .
# Run API:  docker run --rm -p 8000:8000 -v $(pwd)/models:/app/models:ro dc_real_time_api:v0.1
# Run trainer: docker run --rm -v $(pwd)/models:/app/models dc_real_time_api:v0.1 \
#                   python -m src.models.retrain_scheduler
# =============================================================================

# ---- Stage 1: build dependencies in a throwaway image ----
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools needed for some pip packages (e.g., pandas wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install into a virtual env we can copy later
COPY requirements.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ---- Stage 2: lean production image ----
FROM python:3.11-slim AS runtime

# Labels for traceability
LABEL org.opencontainers.image.title="dc_real_time" \
      org.opencontainers.image.description="Spatial-temporal carbon & price forecasting for data center infrastructure" \
      org.opencontainers.image.source="https://github.com/Hiroaki0422/datacenter_real_time_ml" \
      org.opencontainers.image.licenses="MIT"

# Create a non-root user for security
RUN groupadd -r app && useradd -r -g app -u 1000 app

WORKDIR /app

# Copy the virtual env from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install only runtime system deps (curl for healthcheck, tini for signal handling)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY README.md ./
# Always copy these directories (they exist, may be empty if not yet populated)
COPY artifacts/ ./artifacts/
COPY models/ ./models/

# Ensure the app user owns the app directory
RUN chown -R app:app /app

USER app

# Expose the API port (8000 for FastAPI; trainer/drift-monitor don't need it)
EXPOSE 8000

# Health check (only meaningful for the API service)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

# Use tini for proper signal handling (SIGTERM for graceful shutdown, SIGHUP for reload)
ENTRYPOINT ["/usr/bin/tini", "--", "bash", "scripts/docker-entrypoint.sh"]

# Default command: run the API
# Override per-service in docker-compose.yml:
#   trainer:   ["python", "-m", "src.models.retrain_scheduler"]
#   drift:     ["python", "-m", "src.monitoring.drift_detector"]
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
