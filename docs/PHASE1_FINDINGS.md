# Phase 1 — Data Exploration Findings (2026-07-04, updated with 1y data)

> Empirical results from CAISO pulls. 7-day sample used for initial EDA; 1-year backfill executed on Colab Pro confirms thresholds on full data.

## Sample Data Inventory

| Stream | File | Rows | Size | Date range | Status |
|---|---|---|---|---|---|
| CAISO 5-min LMP (3 zones) | `data/processed/caiso_lmp_1y.parquet` | 315,141 | 12 MB | 2025-07-04 → 2026-07-04 | ✅ from Colab |
| CAISO 5-min LMP (3 zones) | `data/processed/caiso_lmp_7d_sample.parquet` | 5,799 | 247 KB | 2026-06-27 → 2026-07-04 | ✅ initial sample |
| CAISO 5-min fuel mix | `data/processed/caiso_fuel_mix_1y.parquet` | 105,108 | 4.6 MB | 2025-07-04 → 2026-07-03 | ✅ from Colab |
| CAISO 5-min fuel mix | `data/processed/caiso_fuel_mix_7d_sample.parquet` | 2,016 | 121 KB | 2026-06-27 → 2026-07-03 | ✅ initial sample |
| Open-Meteo (50 CA DCs) | `data/processed/openmeteo_ca_dc_1y.parquet` | 441,600 | 5.4 MB | 2025-07-01 → 2026-07-04 | ⚠️ partial (50/227 sites) |
| Open-Meteo (Santa Clara) | `data/processed/openmeteo_santaclara_7d_sample.parquet` | 168 | 11 KB | 2026-06-27 → 2026-07-03 | ✅ initial sample |
| AWS Health events | (live JSON, not stored) | 2 events | ~113 KB | 2026-07-04 snapshot | ✅ works |
| GCP Status incidents | (live JSON, not stored) | varies | ~47 KB | live | ✅ works |
| Azure Status feed | (live RSS, not stored) | varies | ~591 B | live | ✅ works |

## CAISO LMP — Key Stats (1-year backfill)

| Location | Mean | Std | Min | Max | Class 3 (extreme) % |
|---|---|---|---|---|---|
| TH_NP15_GEN-APND | $31.94 | $24.19 | -$76.57 | **$1,149.81** | 0.88% |
| TH_SP15_GEN-APND | $26.30 | $27.77 | -$66.26 | **$1,148.19** | 1.65% |
| TH_ZP26_GEN-APND | $26.91 | $23.77 | -$105.95 | **$1,130.08** | 1.55% |

**Observations on 1y**:
- **Max LMPs near $1,150/MWh** are 9x the 7d max — these are real scarcity events (winter cold snaps, gas price spikes, transmission constraints)
- 1y class 3 rates (0.88-1.65%) are *lower* than 7d (1.48-2.44%) because 7d was a hot week that over-represented extreme hours
- Negative LMPs (oversupply) reach -$106 in ZP26 — solar curtailment

## Multi-Class Spike Label Frequency (1-year data, 4h baseline, 1.5x/3x/6x)

| Zone | Normal (0) | Moderate (1) | High (2) | Extreme (3) | n |
|---|---|---|---|---|---|
| NP15 | 91.37% | 6.29% | 1.46% | 0.88% | 100,894 |
| SP15 | 88.34% | 7.89% | 2.11% | 1.65% | 97,220 |
| ZP26 | 88.17% | 8.10% | 2.17% | 1.55% | 97,847 |

**Decision: KEEP thresholds at 1.5x/3x/6x with 4h baseline.** Class 0 is dominant (~88-91%), Class 3 is rare but present (0.88-1.65%). Distribution is workable for multi-class XGBoost with class weights.

## Sensitivity: Baseline Window

| Window | NP15 Class 3 | SP15 Class 3 | ZP26 Class 3 |
|---|---|---|---|
| 2h | 0.85% | 1.39% | 1.80% |
| **4h** | **1.48%** | **2.44%** | **2.27%** |
| 6h | 2.13% | 4.00% | 2.72% |
| 8h | 2.36% | 5.57% | 4.10% |

**4h chosen** as a good balance. 2h under-counts (too short to capture ramp), 8h over-counts (counts whole-day shifts as "spikes").

## Marginal Carbon (GHG) — Sample Stats

| Zone | Min | Max | Mean | Std | Non-zero count |
|---|---|---|---|---|---|
| All 3 zones | 0.0 | 13.5 | 1.6 | 3.46 | 466/1933 (24%) |

**Observations**:
- GHG is system-wide (one value per timestamp, identical across 3 trading hubs)
- 76% of intervals: GHG = 0 (likely "renewables at the margin" — no fossil fuel setting price)
- Non-zero values cluster around 1.6-1.65 (natural gas: ~1.6 short tons/MWh = ~1,451 gCO2/kWh)
- High values 11-13 (likely gas peakers or coal-equivalent at the margin)
- **Unit**: appears to be **short tons CO2 / MWh of marginal generator** (CAISO OASIS documentation)
- **Note**: We may want to derive carbon intensity from fuel mix instead for more reliable signal

## Fuel Mix — 7-day Mean Generation

| Source | Mean MW | % of total |
|---|---|---|
| Solar | 9,386 | 38.6% |
| Wind | 4,710 | 19.4% |
| Imports | 3,659 | 15.0% |
| Nuclear | 2,277 | 9.4% |
| Large Hydro | 1,980 | 8.1% |
| Natural Gas | 1,133 | 4.7% |
| Geothermal | 770 | 3.2% |
| Small Hydro | 283 | 1.2% |
| Biomass | 235 | 1.0% |
| Biogas | 172 | 0.7% |
| **Total** | **24,320** | **100%** |

**Observations**:
- CAISO is **~60% renewable** in 7-day average (solar + wind + hydro + geo + biomass)
- **Coal = 0** (CAISO banned it years ago)
- Natural gas is the swing fuel
- **Batteries = -286 MW** (negative = charging; discharge shows as positive elsewhere)
- Solar dominates midday, drops to 0 at night (creates the famous "duck curve")

## Open-Meteo — Santa Clara Sample

| Variable | Min | Max | Mean |
|---|---|---|---|
| Temperature 2m (°C) | 11.3 | 28.45 | 19.4 |
| Wet bulb (°C) | 9.9 | 17.7 | 13.5 |
| Cloud cover (%) | 0 | 100 | varies |
| Solar radiation (W/m²) | 0 | 1,031 | varies |
| Wind speed (m/s) | 0 | ~7 | ~3 |

Wet-bulb never exceeded 18°C in this 7-day window (mild summer). For CA DC cooling, we'd want to see how this relates to LMP and DC load.

## Status Feeds — Live Snapshot (2026-07-04)

| Provider | Active events | us-west-1/2 events | Note |
|---|---|---|---|
| AWS | 2 | 0 | Both events are ME-CENTRAL-1 (UAE drone strike) and ME-SOUTH-1 (Bahrain) |
| GCP | varies | 0 | Clean at time of check |
| Azure | varies | 0 | Clean at time of check |

**Implication**: status feeds are sparse in CA (which is good — no degradation → no advisory). They become informative when there's an actual incident (we'll get the structured event data).

## Key Caveats Discovered

1. **CAISO 5-min LMP API works only with `date=` arg, not `start=`+`end=`** (with the date being a single day, not a range). The `get_lmp` method's range mode is broken in the current OASIS API.
2. **Today's LMP is not yet available** — only yesterday and earlier. So cron needs to fetch ~24h delayed, not live. **For real-time inference, we'll need the "current" OASIS endpoint or poll every 5 min for the latest published value**.
3. **GHG field is system-wide** (one value per timestamp, not per zone). This means we treat carbon intensity as a system signal, not zone-specific.
4. **Open-Meteo date format must be YYYY-MM-DD**, not the other ISO formats it sometimes accepts.
5. **CAISO `caiso.com/outlook` URL** is a parallel data source for fuel mix (separate from OASIS) and works for current/historical.
6. **EIA API not yet tested** — to do in Phase 2 if needed (backup/cross-check vs CAISO).

## Locked Decisions (moved to DECISIONS.md)

- D8: Spike thresholds locked at 1.5x/3.0x/6.0x with 4h baseline (D1 reaffirmed)
- D9: Use `caiso.get_lmp(date=...)` (one day at a time) for backfill; live inference uses a different endpoint
- D10: Treat carbon intensity as system-wide signal (not per-zone)
- D11: Fuel mix + GHG is sufficient for carbon label; no need to pull separate "marginal emissions" series

## Phase 1 Exit Criteria — Status

- [x] CAISO LMP pulled, distributions understood
- [x] Spike class frequencies computed, thresholds locked
- [x] Open-Meteo working
- [x] Status feeds verified accessible
- [x] Colab handoff notebook created for 1y backfill
- [ ] 1y backfill executed (Phase 1.5 — needs Colab Pro)
- [ ] EIA API tested (deferred — not needed for v1 if CAISO works)

## Recommended Next Steps

1. **Execute Colab handoff** for 1y of LMP + fuel mix (~12-20 min on Colab Pro)
2. **Pull weather for 227 CA DC sites** (not just Santa Clara centroid) — Open-Meteo supports batch
3. **Build feature pipeline** (Phase 2)
4. **First XGBoost baseline** (Phase 3)
