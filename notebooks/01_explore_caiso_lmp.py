# %% [markdown]
# # Phase 1: CAISO 5-min LMP Exploration
# 
# **Goal**: Validate data quality, understand distributions, compute spike class frequencies, lock thresholds.
# 
# **Output**: Class frequency tables, threshold decisions, schema notes.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load 7-day sample
df = pd.read_parquet('/root/project/dc_real_time/data/processed/caiso_lmp_7d_sample.parquet')
print(f"Loaded {df.shape[0]} rows × {df.shape[1]} cols")
print(df.head(3))

# %%
# Per-zone LMP stats
print("\n=== LMP stats per zone (7 days) ===")
print(df.groupby('Location')['LMP'].describe())

# %%
# Multi-class spike label
def label_spike(lmp_series, window='4h', thresholds=(1.5, 3.0, 6.0)):
    baseline = lmp_series.rolling(window).mean()
    ratio = lmp_series / baseline
    labels = pd.cut(ratio, bins=[0, *thresholds, float('inf')], labels=[0,1,2,3], right=False)
    return labels.astype('Int64')

# Apply per zone
print("\n=== Spike class frequency (4h baseline, 1.5x/3x/6x) ===")
for loc, sub in df.groupby('Location'):
    sub = sub.sort_values('Time').set_index('Time')
    labels = label_spike(sub['LMP'])
    total = len(labels.dropna())
    print(f"\n{loc}:")
    for cls, name in {0:'Normal',1:'Moderate',2:'High',3:'Extreme'}.items():
        n = (labels == cls).sum()
        print(f"  Class {cls} ({name:8s}): {n:5d} ({n/total*100:5.2f}%)")

# %%
# Visualization: 24h LMP trace for one zone
fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for i, (loc, sub) in enumerate(df.groupby('Location')):
    sub = sub.sort_values('Time')
    ax[i].plot(sub['Time'], sub['LMP'], linewidth=0.5)
    ax[i].set_title(f'{loc} 5-min LMP (7d)')
    ax[i].set_ylabel('LMP ($/MWh)')
    ax[i].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/root/project/dc_real_time/artifacts/lmp_7d_trace.png', dpi=100)
plt.show()
