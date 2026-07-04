# Data Sources — Access & Size Notes

## Summary Table

| Source | What | Size Estimate | Auth | Refresh | Status |
|---|---|---|---|---|---|
| gridstatus.io (CAISO) | 5-min LMP, fuel mix, GHG | ~50-200MB/yr/zone | API key (free) | 5 min | To explore Phase 1 |
| CAISO OASIS | Day-ahead + RT LMP, gen mix | ~100-500MB/yr | None (HTML scraping) | 5 min | Backup; gridstatus preferred |
| Open-Meteo | Hourly weather per DC | ~5-20MB/yr/DC | None | 1 hour | Already used in v1 |
| AWS Health | us-west-1/2 status JSON | KB-scale | None | Real-time | Cheap signal |
| GCP Status | Region status JSON | KB-scale | None | Real-time | Cheap signal |
| Azure Status | Region status JSON | KB-scale | None | Real-time | Cheap signal |
| EIA Open Data | Historic ISO load, gen by fuel | 10-100MB/yr/series | API key (free, instant) | 1 hour | To explore Phase 1 |
| v1 dataset | 227 CA DC lat/lon, WUE, BWS | <1MB | None | Static | Already have |
| OSM Overpass | Substations, power infrastructure | 1-50MB per query | None | Static | To explore Phase 2 |

## Size Budget

- **VPS total**: ~50 GB
- **Per-stream cap**: 5 GB
- **Rule of thumb**: if 1y of 5-min LMP for all 24 CAISO zones > 2 GB → Colab handoff for historic training, VPS for inference

## Detailed Notes

### gridstatus.io
- **URL**: https://www.gridstatus.io/
- **Free tier**: hobbyist; daily request limit
- **Coverage**: CAISO, ERCOT, PJM, MISO, NYISO, ISO-NE, SPP
- **What we get**: 5-min LMP, fuel mix, GHG emissions, reserve margin, outages
- **Best practice**: batch requests, cache aggressively, retrain weekly not daily

### Open-Meteo
- **URL**: https://open-meteo.com/
- **Free tier**: 10k requests/day for non-commercial
- **Historic archive**: 80+ years, hourly
- **Forecast**: 7-16 day, hourly
- **Variables we'll use**: temperature_2m, relative_humidity_2m, wet_bulb_temperature_2m, cloud_cover, shortwave_radiation
- **Already familiar from v1**

### AWS / GCP / Azure Status Feeds
- **AWS Health**: https://health.aws.amazon.com/public/currentevents (JSON via RSS-to-JSON or scrape)
- **GCP Status**: https://status.cloud.google.com/ (JSON)
- **Azure Status**: https://status.azure.com/ (JSON)
- **Use**: binary feature "is this provider's region in degraded state?" Cheap proxy for compute slowdown.

### EIA Open Data
- **URL**: https://www.eia.gov/opendata/
- **API key**: free, instant at https://www.eia.gov/opendata/register.php
- **Series we'll use**: 
  - CAISO total load (hourly, 10y+)
  - CAISO generation by fuel (hourly, 10y+)
  - Used as historic backup / cross-check vs gridstatus
- **Rate limit**: 5000 req/h

### v1 Dataset (read-only)
- **Path**: `/root/project/datacenter_water_stress/data/processed/us_dc_with_stress.csv`
- **Records**: 1,575 US DCs (227 in CA)
- **Cols we need**: `state`, `latitude`, `longitude`, `operator`, `est_mw`, `wue_default`, `bws_score`, `climate_adj`
- **Note**: we use it as static anchor; v1 is not modified

### OSM Overpass
- **URL**: https://overpass-api.de/
- **Query**: power infrastructure near CA DC cluster centroids
- **Use**: map CA DC lat/lon → nearest CAISO substation
- **Refresh**: static; one-time batch

## Data Not Used (and Why)

| Source | Why skipped |
|---|---|
| RIPE RIS BGP | Control-plane signal, not traffic; doesn't proxy compute |
| Twitter/X sentiment | NLP-heavy, drift, no civic angle |
| FRED economic | Daily/weekly, not real-time enough |
| Crypto prices | Not grid-related despite "Texas crypto" hype |
| Phasor Measurement Unit (PMU) data | 30Hz; storage/processing pain on VPS |

## Where To Get API Keys

- **EIA**: https://www.eia.gov/opendata/register.php (instant, email only)
- **gridstatus**: https://www.gridstatus.io/ (free hobbyist tier signup)
- **Open-Meteo**: no key needed
- **Status feeds**: no key needed
