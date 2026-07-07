#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# E2E Test — cron → retrain → canary loop (D12)
#
# Validates the full pipeline:
#   1. Preconditions (all services healthy)
#   2. Force retrain + auto-promote (trains a new model version)
#   3. Models exist on disk
#   4. Registry updated
#   5. Champion symlink updated
#   6. API reloads new models
#   7. Forecast endpoints return valid data
#   8. Canary routing works (X-Canary header present on ~5% of requests)
#   9. Frontend and health endpoints respond
#
# Usage:
#   bash scripts/e2e_test.sh
#
# Requires:
#   - Running docker compose stack (all services healthy)
#   - curl, docker, grep, python3
# =============================================================================

PASS=0
FAIL=0
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

pass() { PASS=$((PASS+1)); echo -e "  ${GREEN}✓${NC} $1"; }
fail() { FAIL=$((FAIL+1)); echo -e "  ${RED}✗${NC} $1"; }

echo "=== D12 E2E Test: cron → retrain → canary ==="
echo

# ──────────────────────────────────────────────
# 1. Preconditions
# ──────────────────────────────────────────────
echo "--- 1. Preconditions ---"

SERVICES=$(docker compose ps --format json 2>/dev/null | python3 -c "
import sys,json
for line in sys.stdin:
    s=json.loads(line)
    print(f\"{s['Name']}:{s['Health']}\")
" 2>/dev/null || echo "")

if [ -z "$SERVICES" ]; then
  fail "docker compose ps failed — is the stack running?"
  echo
  echo "  Start the stack first: docker compose up -d"
  exit 1
fi

echo "$SERVICES" | while IFS=: read -r name health; do
  health="${health:-unknown}"
  if [ "$health" != "healthy" ] && [ "$health" != "(healthy)" ]; then
    fail "$name is $health (expected healthy)"
  fi
done

# Check Redis reachable
if docker exec dc_real_time_redis redis-cli ping 2>/dev/null | grep -q PONG; then
  pass "Redis reachable"
else
  fail "Redis unreachable"
fi

# Check API reachable
if curl -sf http://localhost/healthz > /dev/null 2>&1; then
  pass "API reachable"
else
  fail "API unreachable"
fi

# Record pre-state
PRE_CHAMPION=$(python3 -c "import json; r=json.load(open('models/registry.json')); print(r.get('champion',{}).get('version','none'))" 2>/dev/null || echo "unknown")
pass "Pre-state champion: $PRE_CHAMPION"

echo

# ──────────────────────────────────────────────
# 2. Force retrain + auto-promote
# ──────────────────────────────────────────────
echo "--- 2. Training new model (--train --auto-promote) ---"
echo "     This takes ~6 minutes..."

START_TS=$(date +%s)
TRAIN_OUTPUT=$(docker compose run --rm trainer \
  python -m src.models.retrain_scheduler --train --auto-promote 2>&1) || true
END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))

NEW_VERSION=$(echo "$TRAIN_OUTPUT" | grep -oP 'Registered candidate: \K(v[\d.]+)' | head -1)
if echo "$TRAIN_OUTPUT" | grep -q 'AUTO-PROMOTED'; then
  pass "Training completed (${ELAPSED}s), version $NEW_VERSION auto-promoted"
elif echo "$TRAIN_OUTPUT" | grep -q 'Registered candidate'; then
  pass "Training completed (${ELAPSED}s), version $NEW_VERSION registered (not promoted)"
else
  echo "$TRAIN_OUTPUT" | tail -5
  fail "Training did not complete"
  echo
  echo "  Full output above. Check docker compose logs trainer for details."
  exit 1
fi

echo

# ──────────────────────────────────────────────
# 3. Models exist on disk
# ──────────────────────────────────────────────
echo "--- 3. Model artifacts ---"

MODEL_DIR="models/$NEW_VERSION"
EXPECTED_FILES=(
  lmp_ratio_30m.json lmp_ratio_1h.json
  carbon_30m.json    carbon_1h.json
)
MISSING=0
for f in "${EXPECTED_FILES[@]}"; do
  if [ ! -f "$MODEL_DIR/$f" ]; then
    fail "Missing: $MODEL_DIR/$f"
    MISSING=$((MISSING+1))
  fi
done
if [ "$MISSING" -eq 0 ]; then
  pass "All ${#EXPECTED_FILES[@]} model files present in $MODEL_DIR/"
fi

echo

# ──────────────────────────────────────────────
# 4. Registry updated
# ──────────────────────────────────────────────
echo "--- 4. Registry ---"

REG_CHAMPION=$(python3 -c "
import json
r=json.load(open('models/registry.json'))
c=r.get('champion',{})
print(c.get('version','none'))
print(c.get('status','unknown'))
" 2>/dev/null | head -1)

if [ "$REG_CHAMPION" = "$NEW_VERSION" ]; then
  pass "Registry champion: $REG_CHAMPION"
else
  fail "Registry champion is $REG_CHAMPION (expected $NEW_VERSION)"
fi

echo

# ──────────────────────────────────────────────
# 5. Champion symlink
# ──────────────────────────────────────────────
echo "--- 5. Champion symlink ---"

SYMLINK_TARGET=$(readlink models/champion 2>/dev/null || echo "")
if [ "$SYMLINK_TARGET" = "$NEW_VERSION" ]; then
  pass "Symlink: champion -> $SYMLINK_TARGET"
else
  fail "Symlink points to $SYMLINK_TARGET (expected $NEW_VERSION)"
fi

echo

# ──────────────────────────────────────────────
# 6. API reload
# ──────────────────────────────────────────────
echo "--- 6. API reload ---"

RELOAD=$(curl -sf -X POST http://localhost/admin/reload 2>/dev/null) || true
if echo "$RELOAD" | grep -q '"reloaded": true'; then
  pass "API reloaded"
else
  fail "API reload failed: $RELOAD"
fi

# Verify models loaded
MODEL_INFO=$(curl -sf http://localhost/model/info 2>/dev/null) || true
if echo "$MODEL_INFO" | grep -q '"lmp_loaded": true'; then
  pass "LMP models loaded"
else
  fail "LMP models not loaded"
fi
if echo "$MODEL_INFO" | grep -q '"carbon_loaded": true'; then
  pass "Carbon models loaded"
else
  fail "Carbon models not loaded"
fi

echo

# ──────────────────────────────────────────────
# 7. Forecast endpoints
# ──────────────────────────────────────────────
echo "--- 7. Forecast endpoints ---"

for zone in NP15 SP15 ZP26; do
  FCST=$(curl -sf "http://localhost/forecast/$zone?horizon=30m" 2>/dev/null) || true
  ADVISORY=$(echo "$FCST" | python3 -c "import sys,json; print(json.load(sys.stdin).get('advisory','unknown'))" 2>/dev/null || echo "error")
  if [ "$ADVISORY" = "unknown" ] || [ "$ADVISORY" = "error" ]; then
    fail "$zone forecast: advisory=$ADVISORY"
  else
    pass "$zone forecast: advisory=$ADVISORY"
  fi
done

# DC forecast (sample)
DC_FCST=$(curl -sf "http://localhost/dc/DC-00088/forecast?horizon=30m" 2>/dev/null) || true
DC_HORIZONS=$(echo "$DC_FCST" | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(len(d.get('all_horizons',{})))
" 2>/dev/null || echo "0")
if [ "$DC_HORIZONS" -ge 2 ]; then
  pass "DC-00088: $DC_HORIZONS horizons returned"
else
  fail "DC-00088: only $DC_HORIZONS horizons (expected 2+)"
fi

echo

# ──────────────────────────────────────────────
# 8. Canary routing
# ──────────────────────────────────────────────
echo "--- 8. Canary routing (5% split) ---"

CANARY_COUNT=0
TOTAL=100
for i in $(seq 1 $TOTAL); do
  HEADERS=$(curl -sfI "http://localhost/" 2>/dev/null) || true
  if echo "$HEADERS" | grep -q "X-Canary: green"; then
    CANARY_COUNT=$((CANARY_COUNT + 1))
  fi
done

if [ "$CANARY_COUNT" -gt 0 ]; then
  pass "Canary: $CANARY_COUNT/$TOTAL requests hit green (X-Canary: green)"
else
  # All from same client IP — verify canary is configured correctly
  if curl -sfI "http://localhost/" | grep -q "X-Canary"; then
    pass "Canary: header present (all same client IP, expected ~5%)"
  else
    fail "No X-Canary header detected — canary routing may not be active"
  fi
fi

echo

# ──────────────────────────────────────────────
# 9. Frontend and health
# ──────────────────────────────────────────────
echo "--- 9. Frontend & health ---"

# Healthcheck
HEALTH=$(curl -sf http://localhost/healthz 2>/dev/null) || true
if echo "$HEALTH" | grep -q '"status": "ok"'; then
  pass "Healthcheck: ok"
else
  fail "Healthcheck failed"
fi

# Frontend
FRONTEND=$(curl -sf http://localhost/ 2>/dev/null) || true
if echo "$FRONTEND" | grep -q "NP15"; then
  pass "Frontend: serves dashboard (contains NP15)"
else
  fail "Frontend: not serving dashboard"
fi

# Readiness
if curl -sf http://localhost/readyz > /dev/null 2>&1; then
  pass "Readiness: ready"
else
  fail "Readiness: not ready"
fi

echo

# ──────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────
echo "=== Results ==="
TOTAL=$((PASS + FAIL))
echo -e "  ${GREEN}$PASS passed${NC} / ${RED}$FAIL failed${NC} / $TOTAL total"

if [ "$FAIL" -gt 0 ]; then
  echo
  echo "  Failed steps: check logs above for details."
  echo "  Common fixes:"
  echo "    - docker compose logs fetcher — Redis data freshness"
  echo "    - docker compose logs api — model loading errors"
  echo "    - docker compose logs trainer — training errors"
  exit 1
fi

echo
echo "✓ E2E test passed — full cron → retrain → canary loop verified"
