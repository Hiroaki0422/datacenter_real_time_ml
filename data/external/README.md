# External Data — Symlinks / Cross-Project References

## v1 Dataset (read-only)

Source: `/root/project/datacenter_water_stress/data/processed/us_dc_with_stress.csv`

What we use from it:
- `state` = 'CA' filter → 227 California DCs
- `latitude`, `longitude` — for geospatial join to CAISO zones
- `operator` — for operator-specific features
- `est_mw` — capacity
- `wue_default`, `climate_adj`, `bws_score` — for water overlay

## OSM / Substation Data (Phase 2)
To be downloaded to: `data/external/osm_substations.geojson`

## EIA API Key
Save your key to: `data/external/.env` (gitignored)
```
EIA_API_KEY=your_key_here
GRIDSTATUS_API_KEY=your_key_here
```
