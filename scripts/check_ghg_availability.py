"""Check GHG data availability by time window."""
import pandas as pd
from datetime import timedelta

lmp = pd.read_parquet('/root/project/dc_real_time/data/processed/caiso_lmp_1y.parquet')

print(f"Full data date range: {lmp['Time'].min()} -> {lmp['Time'].max()}")
print(f"Total rows: {len(lmp):,}")
print()

# Use the last date in the data as "now"
end = lmp['Time'].max()
start_90 = end - timedelta(days=90)
start_180 = end - timedelta(days=180)
start_365 = end - timedelta(days=365)

print(f"Last date in data: {end.date()}")
print(f"Last 90 days: {start_90.date()} -> {end.date()}")
print(f"Last 180 days: {start_180.date()} -> {end.date()}")
print(f"Last 365 days: {start_365.date()} -> {end.date()}")
print()

for label, start in [('Last 90 days', start_90),
                     ('Last 180 days', start_180),
                     ('Last 365 days', start_365)]:
    sub = lmp[lmp['Time'] >= start]
    nz = sub[sub['GHG'] > 0]
    print(f"{label}:")
    print(f"  Total rows: {len(sub):,}")
    print(f"  Non-zero GHG: {len(nz):,} ({len(nz)/len(sub)*100:.1f}%)")
    print(f"  Locations: {sub['Location'].nunique()}")
    print()

# May-Jul 2026 window specifically
print("=== May-Jul 2026 window (where GHG data exists) ===")
may_start = pd.Timestamp('2026-05-01', tz='US/Pacific')
sub = lmp[(lmp['Time'] >= may_start) & (lmp['Time'] <= end)]
nz = sub[sub['GHG'] > 0]
print(f"  Total rows: {len(sub):,}")
print(f"  Non-zero GHG: {len(nz):,} ({len(nz)/len(sub)*100:.1f}%)")
days_covered = (sub['Time'].max() - sub['Time'].min()).days
print(f"  Days covered: {days_covered}")
print(f"  Locations: {sub['Location'].nunique()}")
print(f"  Mean GHG (non-zero only): {nz['GHG'].mean():.2f}")
print(f"  Max GHG: {nz['GHG'].max():.2f}")

# For the carbon model: how much would train/val/test be?
print()
print("=== How much training data for a 60/20/20 split of May-Jul 2026? ===")
total_days = days_covered
train_days = int(total_days * 0.6)
val_days = int(total_days * 0.2)
test_days = total_days - train_days - val_days
print(f"  Total days: {total_days}")
print(f"  Train days: {train_days} (60%)")
print(f"  Val days: {val_days} (20%)")
print(f"  Test days: {test_days} (20%)")
print(f"  Train rows: ~{train_days * 288 * 3:,} (3 zones x 288 5-min/day)")
print(f"  Val rows: ~{val_days * 288 * 3:,}")
print(f"  Test rows: ~{test_days * 288 * 3:,}")
print(f"  Non-zero rows in train: ~{int(len(nz) * train_days / total_days):,}")
print(f"  Non-zero rows in val: ~{int(len(nz) * val_days / total_days):,}")
print(f"  Non-zero rows in test: ~{int(len(nz) * test_days / total_days):,}")
