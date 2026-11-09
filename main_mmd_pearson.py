"""ICASSP section 5.1: four Gaussian regimes with two returns."""

import numpy as np

# Re-export helpers used by the extra PAMAP2 and WISDM entry points.
from experiments.evaluation import (
    calculate_delays, detection_indices, evaluate_changepoints,
    get_regime_sequence, print_detection_summary,
)
from experiments.runner import labels_for_segments, run_comparison, validate_thresholds

N = 150
M = 40
K_CONF = 6
SEGMENT_LENGTH = 300
SEQUENCE = (1, 2, 3, 2, 1, 4)
MEANS = (0.0, 5.0, -4.0, 10.0)
STDS = (1.0, 1.0, 0.5, 2.0)
MMD_ETA = 0.12
MMD_NU = 0.90
SEED = 42
REG_LAMBDA = 0.1
PEARSON_TAU = 0.4  # Arbitrary placeholder; adjust for this experiment.
PEARSON_NU = 0.7  # Arbitrary placeholder; adjust for this experiment.
MAX_DELAY = M + K_CONF


def main(show=True):
    def sample(label, size, rng):
        return rng.normal(MEANS[label - 1], STDS[label - 1], (size, 1))

    thresholds = {'MMD': dict(detection=MMD_ETA, novelty=MMD_NU),
                  'Pearson': dict(detection=PEARSON_TAU, novelty=PEARSON_NU)}
    validate_thresholds(thresholds, REG_LAMBDA)
    rng = np.random.RandomState(SEED)
    data = np.concatenate([sample(label, SEGMENT_LENGTH, rng) for label in SEQUENCE]).ravel()
    cps = list(range(SEGMENT_LENGTH, len(data), SEGMENT_LENGTH))
    return run_comparison(
        data, cps, labels_for_segments(SEQUENCE, SEGMENT_LENGTH),
        title=f'Gaussian reference, seed={SEED}', n=N, m=M, k_conf=K_CONF,
        thresholds=thresholds, reg_lambda=REG_LAMBDA,
        max_delay=MAX_DELAY, show=show, seed=SEED)


if __name__ == '__main__':
    main()
