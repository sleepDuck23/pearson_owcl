# Pearson and MMD open-world change-point detection

Two online detectors compare adjacent reference and analysis windows, confirm
persistent changes, and recognize returning regimes from fresh post-alarm
representatives.

## Run the experiments

```bash
python3 main_mmd_pearson.py  # ICASSP §5.1: Gaussian reference example
python3 main_T1.py           # §5.2: distinct means, common variance
python3 main_T2.py           # §5.2: common mean, distinct variances
python3 main_T3.py           # §5.2: common mean and variance
python3 main_mnist.py        # §5.3: sequences A and B
```

Each main keeps its parameters as named constants at the top. `main_T_exp.py`
remains a compatibility entry point for T1. For execution without opening plots,
call the corresponding `main(show=False)` from Python. NumPy, Matplotlib, pandas,
and scikit-learn are required. The extra `main_wisdm.py` and `main_pamap2.py`
experiments retain their own configurations; WISDM uses synthetic surrogate data.

| Experiment | Segment size | n | m | MMD detection threshold | MMD novelty threshold |
|---|---:|---:|---:|---|---|
| Gaussian reference | 300 | 150 | 40 | 0.12 | 0.90 |
| T1 | 700 | 150 | 40 | Independent calibration | 0.63 |
| T2 | 700 | 300 | 100 | Independent calibration | 0.99 |
| T3 | 700 | 600 | 200 | Independent calibration | Independent calibration |
| MNIST A and B | 300 | 150 | 40 | 0.06 | 0.96 |

All experiments use six consecutive exceedances for confirmation. T1/T2/T3 each
run three independent streams (seeds 42, 43, 44), with eight true returns per
stream, and report aggregated re-identification out of 24. Their data generators
match population moments; they do not force finite batches to have identical
sample moments.

## Published settings and calibration

Reported numerical settings above follow `icassp_2027.pdf`, sections 5.1–5.3.
The PDF omits tier-specific detection thresholds, precise T1/T2 population
parameters, seeds, and the full precision of T3's printed novelty threshold
`1.00`. The Pearson manuscript supplies no experimental parameter table.
Consequently these scripts cannot promise exact reproduction of the paper's
counts. Implementation choices are explicitly marked in each main.

A threshold set to `None` is calibrated before evaluation using separate data
and a separate seed. This includes all Pearson thresholds and T3's unreported
full-precision novelty threshold. Pearson uses operator regularization
`REG_LAMBDA = 0.1`, an explicit implementation choice, with Gram ridge `n * lambda`.
Do not transfer thresholds between different regularizations or window sizes.

Calibration compares same-regime and different-regime confirmation runs and
recognition batches. It uses empirical quantile boundaries, selects the midpoint
when their tails separate, and otherwise retains the same-regime boundary and
reports overlap. This is a reproducible empirical calibration protocol, not the
unpublished original calibration or a statistical guarantee. Thresholds and
calibration seeds are printed with sufficient precision and are frozen before
all evaluation runs. Evaluation outcomes never feed back into calibration.

MNIST reserves disjoint per-digit pools for fitting PCA (500 images), calibrating
thresholds (1,500 images), and evaluation. PCA is fitted once on the preprocessing
pool; both models use the same 50-dimensional coordinates and Euclidean RBF
kernel, without extra per-image L2 normalization. No evaluation image is reused
within a sequence or across A and B. OpenML downloads are cached in `data/openml/`.

## Bandwidth and online procedure

Each initial or replacement representative is built in this order:

1. Collect its n observations.
2. Compute `gamma = 1 / (2 * median(pairwise squared distances))` using distinct
   pairs, with a small floor for degenerate distances.
3. Compute its kernel mean embedding and, for Pearson, its uncentered empirical
   second-moment operator through the regularized Gram inverse.
4. Store the representative with that bandwidth.

Every recognition candidate uses its own stored bandwidth for the entire
comparison. A fresh batch's bandwidth is computed before its embedding/operator
and stored when it becomes the assigned representative, including on recall.
The active bandwidth is then held fixed through sliding-window monitoring until
another representative is collected. Adaptive bandwidths are the chosen
experimental policy; the Pearson manuscript's fixed-kernel concentration bounds
do not directly establish guarantees for this adaptive selection.

The first score is evaluated at sample count `n + m`, and monitoring resumes at
`t_alarm + n + m`. The n observations immediately after an alarm form the
recognition batch. Post-alarm collection does not itself guarantee a homogeneous
batch, particularly with T3's long reference window and 700-sample segments.

## Evaluation

Change-point arrays and plots use zero-based indices of the first new-regime
sample. Models record alarms as one-based sample counts. Reported detection
delays equal `t_alarm - t0`, where `t0` is the count/index of the last old-regime
observation in the papers: an alarm on the first new sample has delay one.

Each paper experiment explicitly sets `MAX_DELAY = M + K_CONF`. This finite
horizon is an evaluation convention introduced here, not a value reported in the
PDF. Alarms outside it are false alarms; unmatched changes are misses. Matching
never crosses the next true change. Adjust the named constant to compare another
horizon, and report the horizon together with results.

Re-identification is counted against all true returning segments, including
those with missed detections. Success requires a timely matched alarm, a
homogeneous recognition batch completed before the next transition, and recall
of the original learned identity. Creating another ID for a known regime is a
failure. Novel discoveries, false alarms, misses, mixed representative batches,
per-run results, mean detection count, sample standard deviation, and aggregated
recall are also reported. Plots show both score traces and estimated regime IDs
without relabeling model errors using future ground truth.

## Checks

```bash
MPLBACKEND=Agg python3 -m unittest discover -s tests -v
```

Tests cover monitoring/restart timing, bandwidth/statistic consistency, the
recursive Pearson score against an independent feature-space operator solve,
detection deadlines, recurrence failures, and disjoint MNIST image allocation.
