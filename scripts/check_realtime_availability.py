"""Check real-time data availability for 5-min refresh pipeline."""
import gridstatus
import pandas as pd
from datetime import datetime, timedelta

caiso = gridstatus.CAISO()
end = datetime.now()

print(f"Current time: {end}")
print()

# === Test real-time endpoints ===
endpoints = []

# 1. get_load (uses caiso.com/outlook, includes current)
print("=== 1. get_load (caiso.com/outlook, includes current) ===")
try:
    load = caiso.get_load(date=end)
    print(f"  Shape: {load.shape}")
    print(f"  Cols: {list(load.columns)}")
    print(f"  First: {load.iloc[0].to_dict()}")
    print(f"  Last:  {load.iloc[-1].to_dict()}")
    # Check freshness
    if 'Time' in load.columns:
        latest = load['Time'].max()
        if hasattr(latest, 'tz') and latest.tz is not None:
            lag_min = (pd.Timestamp.now(tz='US/Pacific') - latest).total_seconds() / 60
        else:
            lag_min = (pd.Timestamp.now(tz='US/Pacific').tz_localize(None) - latest).total_seconds() / 60
        print(f"  Lag: {lag_min:.1f} minutes")
    endpoints.append(('load', load))
except Exception as e:
    print(f"  Error: {e}")

# 2. get_fuel_mix (caiso.com/outlook, includes current)
print("\n=== 2. get_fuel_mix (caiso.com/outlook, includes current) ===")
try:
    fm = caiso.get_fuel_mix(date=end)
    print(f"  Shape: {fm.shape}")
    print(f"  Cols: {list(fm.columns)}")
    print(f"  First: {fm.iloc[0].to_dict()}")
    print(f"  Last:  {fm.iloc[-1].to_dict()}")
    if 'Time' in fm.columns:
        latest = fm['Time'].max()
        if hasattr(latest, 'tz') and latest.tz is not None:
            lag_min = (pd.Timestamp.now(tz='US/Pacific') - latest).total_seconds() / 60
        else:
            lag_min = (pd.Timestamp.now(tz='US/Pacific').tz_localize(None) - latest).total_seconds() / 60
        print(f"  Lag: {lag_min:.1f} minutes")
    endpoints.append(('fuel_mix', fm))
except Exception as e:
    print(f"  Error: {e}")

# 3. get_lmp with date=now (OASIS, lag ~1-2h)
print("\n=== 3. get_lmp(date=now) — OASIS, 5-min LMP ===")
try:
    lmp = caiso.get_lmp(date=end, market='REAL_TIME_5_MIN')
    print(f"  Shape: {lmp.shape}")
    print(f"  Cols: {list(lmp.columns)}")
    if len(lmp) > 0:
        print(f"  First: {lmp.iloc[0].to_dict()}")
        print(f"  Last:  {lmp.iloc[-1].to_dict()}")
    if 'Time' in lmp.columns and len(lmp) > 0:
        latest = lmp['Time'].max()
        if hasattr(latest, 'tz') and latest.tz is not None:
            lag_min = (pd.Timestamp.now(tz='US/Pacific') - latest).total_seconds() / 60
        else:
            lag_min = (pd.Timestamp.now(tz='US/Pacific').tz_localize(None) - latest).total_seconds() / 60
        print(f"  Lag: {lag_min:.1f} minutes")
    endpoints.append(('lmp', lmp))
except Exception as e:
    print(f"  Error: {e}")

# 4. get_lmp with date=yesterday (OASIS, 24h delay but reliable)
print("\n=== 4. get_lmp(date=yesterday) — OASIS, full 5-min LMP for prior day ===")
try:
    lmp_yest = caiso.get_lmp(date=end - timedelta(days=1), market='REAL_TIME_5_MIN')
    print(f"  Shape: {lmp_yest.shape}")
    if len(lmp_yest) > 0:
        print(f"  Date range: {lmp_yest['Time'].min()} → {lmp_yest['Time'].max()}")
    endpoints.append(('lmp_yesterday', lmp_yest))
except Exception as e:
    print(f"  Error: {e}")

# 5. Test direct caiso.com/outlook for real-time LMP
print("\n=== 5. Direct caiso.com/outlook current LMP? ===")
# Try a few known endpoints
import requests
for url in [
    'https://www.caiso.com/outlook/current/fuelsource.csv',
    'https://www.caiso.com/outlook/current/demand.csv',
    'https://www.caiso.com/outlook/current/slenergy.csv',  # system load
]:
    try:
        r = requests.get(url, timeout=5)
        print(f"  {url.split('/')[-1]}: status={r.status_code}, size={len(r.text)}")
        if r.status_code == 200 and len(r.text) > 100:
            # First line
            print(f"    Header: {r.text.split(chr(10))[0][:200]}")
    except Exception as e:
        print(f"  {url}: {e}")

# 6. NWS API for current weather
print("\n=== 6. NWS API (current weather) ===")
try:
    r = requests.get("https://api.weather.gov/points/37.39,-121.98", timeout=5,
                    headers={"User-Agent": "test"})
    print(f"  Status: {r.status_code}")
    if r.status_code == 200:
        j = r.json()
        print(f"  Office: {j.get('properties', {}).get('gridId')}")
        print(f"  Grid: {j.get('properties', {}).get('gridX')},{j.get('properties', {}).get('gridY')}")
except Exception as e:
    print(f"  Error: {e}")

# 7. Open-Meteo for current weather
print("\n=== 7. Open-Meteo current weather ===")
try:
    r = requests.get("https://api.open-meteo.com/v1/forecast?latitude=37.39&longitude=-121.98&current=temperature_2m,relative_humidity_2m,cloud_cover,wind_speed_10m&timezone=America/Los_Angeles", timeout=5)
    print(f"  Status: {r.status_code}")
    if r.status_code == 200:
        j = r.json()
        print(f"  Current: {j.get('current', {})}")
except Exception as e:
    print(f"  Error: {e}")

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
for name, ep in endpoints:
    if hasattr(ep, 'shape'):
        print(f"  {name}: {ep.shape}, lag varies")
