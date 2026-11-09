# Shared MMD / Pearson benchmark evaluation

Both models receive the same observations. Ground truth is available only to the evaluator. Each run uses its own chronological, one-to-one regime mapping. The initial episode is excluded from assignment scores; later appearances of its regime are returns.

Change times and alarms are zero-based sample indices. Delay = confirmed alarm index − first new-episode sample index. Tolerance is inclusive and matching cannot cross an episode end. Assignment times are scored separately and must precede the episode end. All unmatched monitoring alarms count as false alarms.

Aggregates sum numerators and denominators and pool all matched detection delays. No thresholds were tuned for this evaluation. Several configured thresholds are arbitrary placeholders, so these results evaluate the checked-in settings rather than establish a tuned model comparison or an exact reproduction of ICASSP results.

WISDM is a synthetic activity surrogate, not WISDM recordings. PAMAP2 retains lambda=20 and Pearson detection threshold 0.80; its score is bounded by 0.05, so that configuration cannot confirm alarms.

Reproduce with `MPLBACKEND=Agg python3 -m experiments.benchmark`. Full settings, stream fingerprints, episode boundaries, decisions, and per-run scores are saved in `benchmark_results.json`.

## gaussian

1 paired stream(s); seeds: [42]. Alarm tolerance: **46 samples**. n=150, m=40, k=6; Pearson lambda=0.1.

MMD detection/novelty: 0.12/0.9; Pearson detection/novelty: 0.4/0.7.

| Metric | MMD | Pearson |
|---|---:|---:|
| Detected changes | 5/5 (100.0%) | 5/5 (100.0%) |
| False alarms | 0 | 0 |
| Mean detection delay (samples) | 17.00 | 18.20 |
| First-encounter discovery | 3/3 (100.0%) | 3/3 (100.0%) |
| Return-episode assignment | 2/2 (100.0%) | 2/2 (100.0%) |

## T1

3 paired stream(s); seeds: [42, 43, 44]. Alarm tolerance: **46 samples**. n=150, m=40, k=6; Pearson lambda=0.1.

MMD detection/novelty: 0.1/0.63; Pearson detection/novelty: 0.4/0.7.

| Metric | MMD | Pearson |
|---|---:|---:|
| Detected changes | 33/33 (100.0%) | 33/33 (100.0%) |
| False alarms | 0 | 0 |
| Mean detection delay (samples) | 15.91 | 18.64 |
| First-encounter discovery | 9/9 (100.0%) | 9/9 (100.0%) |
| Return-episode assignment | 24/24 (100.0%) | 24/24 (100.0%) |

## T2

3 paired stream(s); seeds: [42, 43, 44]. Alarm tolerance: **106 samples**. n=300, m=100, k=6; Pearson lambda=0.1.

MMD detection/novelty: 0.1/0.99; Pearson detection/novelty: 0.4/0.7.

| Metric | MMD | Pearson |
|---|---:|---:|
| Detected changes | 18/33 (54.5%) | 11/33 (33.3%) |
| False alarms | 0 | 0 |
| Mean detection delay (samples) | 70.33 | 88.36 |
| First-encounter discovery | 6/9 (66.7%) | 0/9 (0.0%) |
| Return-episode assignment | 12/24 (50.0%) | 1/24 (4.2%) |

## T3

3 paired stream(s); seeds: [42, 43, 44]. Alarm tolerance: **206 samples**. n=600, m=200, k=6; Pearson lambda=0.1.

MMD detection/novelty: 0.1/0.99; Pearson detection/novelty: 0.4/0.7.

| Metric | MMD | Pearson |
|---|---:|---:|
| Detected changes | 0/33 (0.0%) | 0/33 (0.0%) |
| False alarms | 0 | 0 |
| Mean detection delay (samples) | N/A | N/A |
| First-encounter discovery | 0/9 (0.0%) | 0/9 (0.0%) |
| Return-episode assignment | 0/24 (0.0%) | 0/24 (0.0%) |

## MNIST A

1 paired stream(s); seeds: [42]. Alarm tolerance: **46 samples**. n=150, m=40, k=6; Pearson lambda=0.1.

MMD detection/novelty: 0.06/0.96; Pearson detection/novelty: 0.4/0.7.

| Metric | MMD | Pearson |
|---|---:|---:|
| Detected changes | 10/10 (100.0%) | 9/10 (90.0%) |
| False alarms | 0 | 0 |
| Mean detection delay (samples) | 25.20 | 31.56 |
| First-encounter discovery | 6/6 (100.0%) | 3/6 (50.0%) |
| Return-episode assignment | 4/4 (100.0%) | 1/4 (25.0%) |

## MNIST B

1 paired stream(s); seeds: [42]. Alarm tolerance: **46 samples**. n=150, m=40, k=6; Pearson lambda=0.1.

MMD detection/novelty: 0.06/0.96; Pearson detection/novelty: 0.4/0.7.

| Metric | MMD | Pearson |
|---|---:|---:|
| Detected changes | 9/10 (90.0%) | 9/10 (90.0%) |
| False alarms | 0 | 0 |
| Mean detection delay (samples) | 27.33 | 34.11 |
| First-encounter discovery | 2/3 (66.7%) | 1/3 (33.3%) |
| Return-episode assignment | 7/7 (100.0%) | 4/7 (57.1%) |

## pamap2

1 paired stream(s); seeds: [None]. Alarm tolerance: **106 samples**. n=200, m=100, k=6; Pearson lambda=20.0.

MMD detection/novelty: 0.75/0.95; Pearson detection/novelty: 0.8/0.71.

| Metric | MMD | Pearson |
|---|---:|---:|
| Detected changes | 4/10 (40.0%) | 0/10 (0.0%) |
| False alarms | 7 | 0 |
| Mean detection delay (samples) | 64.75 | N/A |
| First-encounter discovery | 2/3 (66.7%) | 0/3 (0.0%) |
| Return-episode assignment | 1/7 (14.3%) | 0/7 (0.0%) |

## wisdm

1 paired stream(s); seeds: [42]. Alarm tolerance: **56 samples**. n=150, m=50, k=6; Pearson lambda=0.01.

MMD detection/novelty: 0.06/0.96; Pearson detection/novelty: 0.4/0.71.

| Metric | MMD | Pearson |
|---|---:|---:|
| Detected changes | 10/10 (100.0%) | 10/10 (100.0%) |
| False alarms | 0 | 0 |
| Mean detection delay (samples) | 26.70 | 21.20 |
| First-encounter discovery | 3/3 (100.0%) | 3/3 (100.0%) |
| Return-episode assignment | 7/7 (100.0%) | 7/7 (100.0%) |

## Execution environment

Python 3.9.6; NumPy 2.0.2; scikit-learn 1.6.1. BLAS thread limit: 1.
