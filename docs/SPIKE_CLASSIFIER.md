# LMP Spike Classifier — Multi-Class Design

> **Decision (locked 2026-07-04)**: Replace binary spike classifier with **4-class multi-class classifier** based on LMP magnitude vs 4h-rolling baseline. Outputs a probability vector (one per class) for richer downstream visualization.

## 1. Motivation

Binary classifier (spike / not-spike) loses information that matters for downstream use:
- **Visualization**: a stack-probability bar per hour tells a much better story than a binary flag
- **Action tiers**: different spike magnitudes call for different advisory levels (notify vs pause-job)
- **Class imbalance handling**: hard binary threshold is brittle; soft probabilities are easier to tune downstream
- **Calibration**: 4-class softmax is more naturally calibrated than ad-hoc binary thresholds

## 2. Label Definition

### Baseline: 4h-rolling mean of LMP (per zone)

For each 5-min interval `t` in zone `z`:
```
baseline(z, t) = mean(LMP(z, t-4h : t))
ratio(z, t)    = LMP(z, t) / baseline(z, t)
```

### Class Boundaries (4 classes)

| Class | Label | Ratio Range | Visual Color | Advisory Implication |
|---|---|---|---|---|
| 0 | **Normal** | ratio < 1.5x | Green | No action |
| 1 | **Moderate** | 1.5x ≤ ratio < 3.0x | Yellow | Notify; consider deferring non-urgent |
| 2 | **High** | 3.0x ≤ ratio < 6.0x | Orange | Strong advisory; pause batch jobs |
| 3 | **Extreme** | ratio ≥ 6.0x | Red | Hard advisory; consider region shift |

### Rationale For Thresholds

- **1.5x**: roughly the 80th-90th percentile of normal LMP variance on a typical day; small spikes happen often
- **3.0x**: aligned with industry "spike" definition (LMP > 3× mean); the canonical threshold
- **6.0x**: extreme events (Feb 2021 Texas, Aug 2020 CAISO rolling blackouts); rare but exist

### Class Frequency Targets

Empirical (CAISO 2023 data, to verify in Phase 1):
- Class 0 (normal): ~85-90% of intervals
- Class 1 (moderate): ~7-10%
- Class 2 (high): ~2-4%
- Class 3 (extreme): <1%

If observed frequencies differ, **adjust thresholds** before locking (Phase 1 task).

## 3. Edge Cases & Filters

| Case | Handling |
|---|---|
| **Cold start** (first 4h of zone history) | Mark as "no label"; exclude from training |
| **Baseline = 0** (offline intervals) | Skip; mark as "no label" |
| **Missing LMP** | Forward-fill up to 15 min, then mark as "no label" |
| **Oscillation** (back-and-forth across boundaries within 30min) | Use mode of last 30min to reduce label noise |

## 4. Target Variable

For supervised training, each 5-min interval gets a label `y ∈ {0, 1, 2, 3}`.

For inference, model outputs `P(y=k) for k ∈ {0,1,2,3}` summing to 1.

## 5. Loss Function

**Multi-class log loss with class weights** (inverse frequency from training set):
```
L = -Σ_k w_k * y_k * log(p_k)
```
where `w_k ∝ 1 / freq(k)` to handle imbalance without focal loss complexity.

**Alternative considered**: focal loss for multi-class. Rejected because:
- Class weights simpler and equally effective for 4 classes
- Focal loss has more hyperparameters to tune
- We can revisit if calibration fails

## 6. Evaluation

| Metric | Why |
|---|---|
| **Multi-class log loss** | Primary; rewards calibrated probabilities |
| **One-vs-rest PR-AUC per class** | Per-class quality |
| **Confusion matrix** | Reveals which classes get confused |
| **Brier score** | Calibration check |
| **Reliability diagram** | Visual calibration |

**Acceptance** (Phase 6):
- Log loss < baseline (persistence) log loss
- Class 2+3 (high+extreme) PR-AUC > 0.5 (better than random)
- Reliability diagram: predicted probability ≈ empirical frequency within 0.05 for each class

## 7. Downstream Use

### Visualization
- **Stack-probability bar per zone per hour**: shows distribution of expected spike state
- **Time-series heatmap**: zones × hours, color = argmax class
- **Per-DC overlay**: stacked bar per DC, weighted by zone × DC mapping

### "Shift Advisory" Rule Engine (rule-based, not learned)
```
advisory = 
  "PAUSE"      if P(class=3) > 0.5 OR P(class=2) > 0.7
  "DEFERRABLE" if P(class=1) > 0.5
  "OK"         otherwise
```

### Carbon Overlay
- Combine `P(spike_class)` with Model B (carbon forecaster) for joint advisory
- High LMP + high carbon → strongest pause signal
- High LMP + low carbon (e.g., gas price spike but solar-heavy) → softer signal

## 8. Open Questions

| Question | Resolved in |
|---|---|
| Threshold values (1.5x, 3x, 6x) | Phase 1 (EDA) — adjust to match observed frequency |
| 4h window vs 6h or 8h | Phase 1 (EDA) — sensitivity analysis |
| Per-zone thresholds vs global | Phase 1 (EDA) — some zones are more volatile |
| Include forward-looking features? | Phase 2 (modeling) — if multi-step ahead prediction needed |

## 9. Implementation Sketch

```python
def label_spike_multiclass(lmp_series: pd.Series, window='4h', 
                           thresholds=(1.5, 3.0, 6.0)) -> pd.Series:
    """Multi-class spike label based on LMP vs rolling baseline."""
    baseline = lmp_series.rolling(window).mean()
    ratio = lmp_series / baseline
    labels = pd.cut(ratio, 
                    bins=[0, thresholds[0], thresholds[1], thresholds[2], float('inf')],
                    labels=[0, 1, 2, 3],
                    right=False)
    return labels.astype('Int64')  # nullable integer
```

`XGBoost` config:
```python
XGBClassifier(
    objective='multi:softprob',
    num_class=4,
    eval_metric='mlogloss',
    ...
)
```
