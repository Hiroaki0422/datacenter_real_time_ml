"""
Pull per-site weather for all 227 CA DCs (1 year hourly).
Uses Open-Meteo archive API which is free, no key needed.
"""
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
import time
import os

OUT_DIR = '/root/project/dc_real_time/data/processed'
os.makedirs(OUT_DIR, exist_ok=True)

# Load CA DC sites
sites = pd.read_csv('/root/project/dc_real_time/data/external/ca_dc_sites.csv')
print(f"Sites to pull: {len(sites)}")

cache = requests_cache.CachedSession('.cache_weather', expire_after=86400)
retry_session = retry(cache, retries=3, backoff_factor=0.3)
client = openmeteo_requests.Client(session=retry_session)

url = "https://archive-api.open-meteo.com/v1/archive"

# Open-Meteo supports batch: multiple latitudes/longitudes in one call
# Format: latitudes, longitudes as comma-separated
def fetch_chunk(sites_chunk, start, end):
    lats = ",".join(sites_chunk['latitude'].astype(str))
    lons = ",".join(sites_chunk['longitude'].astype(str))
    params = {
        "latitude": lats,
        "longitude": lons,
        "start_date": start,
        "end_date": end,
        "hourly": ["temperature_2m", "relative_humidity_2m", "wet_bulb_temperature_2m",
                   "cloud_cover", "shortwave_radiation", "wind_speed_10m"],
        "timezone": "America/Los_Angeles"
    }
    return client.weather_api(url, params=params)

# Process in chunks (Open-Meteo limit: ~100 sites per call)
CHUNK = 50
all_dfs = []
t0 = time.time()
for chunk_start in range(0, len(sites), CHUNK):
    chunk = sites.iloc[chunk_start:chunk_start+CHUNK]
    try:
        responses = fetch_chunk(chunk, "2025-07-01", "2026-07-03")
        for idx, resp in enumerate(responses):
            site = chunk.iloc[idx]
            hourly = resp.Hourly()
            hourly_data = {"date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            )}
            for j, var in enumerate(["temperature_2m", "relative_humidity_2m",
                                      "wet_bulb_temperature_2m", "cloud_cover",
                                      "shortwave_radiation", "wind_speed_10m"]):
                hourly_data[var] = hourly.Variables(j).ValuesAsNumpy()
            df = pd.DataFrame(hourly_data)
            df['dc_id'] = site['dc_id']
            df['name'] = site['name']
            df['caiso_zone'] = site['caiso_zone']
            all_dfs.append(df)
    except Exception as e:
        print(f"  Chunk {chunk_start}-{chunk_start+CHUNK}: error {e}")
    elapsed = time.time() - t0
    done = min(chunk_start + CHUNK, len(sites))
    print(f"  {done}/{len(sites)} sites done ({elapsed:.0f}s)", flush=True)

if all_dfs:
    combined = pd.concat(all_dfs, ignore_index=True)
    out = os.path.join(OUT_DIR, 'openmeteo_ca_dc_1y.parquet')
    combined.to_parquet(out)
    sz = os.path.getsize(out) / 1024 / 1024
    print(f"\nSaved: {combined.shape[0]} rows × {combined.shape[1]} cols, {sz:.1f} MB")
    print(f"Unique sites: {combined['dc_id'].nunique()}")
    print(f"Date range: {combined['date'].min()} → {combined['date'].max()}")
