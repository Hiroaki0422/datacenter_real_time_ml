"""
Phase 1 Backfill: Pull 1 year of CAISO 5-min LMP + Fuel Mix
Writes to /root/project/dc_real_time/data/processed/
"""
import gridstatus
import pandas as pd
from datetime import datetime, timedelta
import time
import sys
import os

START_DATE = datetime(2025, 7, 1)   # ~1y ago from "now" (2026-07-04)
END_DATE = datetime(2026, 7, 3)     # yesterday
OUT_DIR = '/root/project/dc_real_time/data/processed'
os.makedirs(OUT_DIR, exist_ok=True)

caiso = gridstatus.CAISO()

def pull_lmp():
    """Pull daily LMP, one day at a time."""
    print(f"=== LMP backfill: {START_DATE.date()} to {END_DATE.date()} ===", flush=True)
    dfs = []
    days_total = (END_DATE - START_DATE).days
    t0 = time.time()
    failures = 0
    for i in range(days_total, 0, -1):
        d = END_DATE - timedelta(days=i-1)
        try:
            lmp = caiso.get_lmp(date=d, market='REAL_TIME_5_MIN')
            if len(lmp) > 0:
                dfs.append(lmp)
            else:
                failures += 1
        except Exception as e:
            failures += 1
            print(f"  {d.date()}: error {type(e).__name__}: {str(e)[:80]}", flush=True)
        if (days_total - i + 1) % 30 == 0:
            elapsed = time.time() - t0
            done = days_total - i + 1
            rate = elapsed / done
            eta = (days_total - done) * rate
            print(f"  {done}/{days_total} days done ({elapsed:.0f}s, ETA {eta/60:.1f}min)", flush=True)
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        out = os.path.join(OUT_DIR, 'caiso_lmp_1y.parquet')
        df.to_parquet(out)
        sz = os.path.getsize(out) / 1024 / 1024
        print(f"\nLMP saved: {df.shape[0]} rows, {sz:.1f} MB, {failures} failures")
    return dfs

def pull_fuel_mix():
    """Pull daily fuel mix, one day at a time."""
    print(f"\n=== Fuel Mix backfill: {START_DATE.date()} to {END_DATE.date()} ===", flush=True)
    dfs = []
    days_total = (END_DATE - START_DATE).days
    t0 = time.time()
    failures = 0
    for i in range(days_total, 0, -1):
        d = END_DATE - timedelta(days=i-1)
        try:
            fm = caiso.get_fuel_mix(date=d)
            if len(fm) > 0:
                dfs.append(fm)
        except Exception as e:
            failures += 1
            print(f"  {d.date()}: error {type(e).__name__}: {str(e)[:80]}", flush=True)
        if (days_total - i + 1) % 30 == 0:
            elapsed = time.time() - t0
            done = days_total - i + 1
            rate = elapsed / done
            eta = (days_total - done) * rate
            print(f"  {done}/{days_total} days done ({elapsed:.0f}s, ETA {eta/60:.1f}min)", flush=True)
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        out = os.path.join(OUT_DIR, 'caiso_fuel_mix_1y.parquet')
        df.to_parquet(out)
        sz = os.path.getsize(out) / 1024 / 1024
        print(f"\nFuel mix saved: {df.shape[0]} rows, {sz:.1f} MB, {failures} failures")

if __name__ == '__main__':
    pull_lmp()
    pull_fuel_mix()
    print("\n=== Backfill complete ===")
