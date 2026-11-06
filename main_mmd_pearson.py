"""ICASSP section 5.1: four Gaussian regimes with two returns."""

import numpy as np

from experiments.calibration import calibrate_thresholds, print_calibration
# Re-export helpers used by the extra PAMAP2 and WISDM entry points.
from experiments.evaluation import (
    calculate_delays, detection_indices, evaluate_changepoints,
    get_regime_sequence, print_detection_summary,
)
from experiments.runner import labels_for_segments, run_comparison

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
PEARSON_TAU = None
PEARSON_NU = None
CALIBRATION_SEED = 10000
CALIBRATION_REPEATS = 12
MAX_DELAY = M + K_CONF


def main(show=True):
    def sample(label, size, rng):
        return rng.normal(MEANS[label - 1], STDS[label - 1], (size, 1))

    calibration = calibrate_thresholds(
        sample, (1, 2, 3, 4), n=N, m=M, k_conf=K_CONF,
        reg_lambda=REG_LAMBDA, seed=CALIBRATION_SEED,
        repeats=CALIBRATION_REPEATS, names=('Pearson',))
    overrides = {'Pearson': dict(detection=PEARSON_TAU, novelty=PEARSON_NU)}
    print(f'MMD paper settings: eta={MMD_ETA}, nu={MMD_NU}')
    print_calibration(calibration, overrides)
    thresholds = {'MMD': dict(detection=MMD_ETA, novelty=MMD_NU),
                  'Pearson': {key: calibration['Pearson'][key] if value is None else value
                              for key, value in overrides['Pearson'].items()}}
    rng = np.random.RandomState(SEED)
    data = np.concatenate([sample(label, SEGMENT_LENGTH, rng) for label in SEQUENCE]).ravel()
    cps = list(range(SEGMENT_LENGTH, len(data), SEGMENT_LENGTH))
    return run_comparison(
        data, cps, labels_for_segments(SEQUENCE, SEGMENT_LENGTH),
        title=f'Gaussian reference, seed={SEED}', n=N, m=M, k_conf=K_CONF,
        thresholds=thresholds, reg_lambda=REG_LAMBDA,
        max_delay=MAX_DELAY, show=show)


if __name__ == '__main__':
    main()
