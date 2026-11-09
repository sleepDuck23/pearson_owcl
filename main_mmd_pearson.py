"""ICASSP section 5.1: four Gaussian regimes with two returns."""

import numpy as np

from model.owcl_mmd import OWR_CPD
from model.owcl_pearson import PearsonOWRCPD
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


def build_models():
    """Edit every model parameter here; the shared runner uses these instances as-is."""
    return {
        'MMD': OWR_CPD(
            n=N,
            m=M,
            eta=MMD_ETA,
            nu=MMD_NU,
            k_conf=K_CONF,
            gamma=0.1,
            factor=1.0,
            dynamic_gamma=True,
        ),
        'Pearson': PearsonOWRCPD(
            n=N,
            m=M,
            tau=PEARSON_TAU,
            nu=PEARSON_NU,
            k_conf=K_CONF,
            reg_lambda=REG_LAMBDA,
            gamma=0.1,
            factor=1.0,
            dynamic_gamma=True,
            normalize_distance=False,
        ),
    }


def main(show=True):
    def sample(label, size, rng):
        return rng.normal(MEANS[label - 1], STDS[label - 1], (size, 1))

    models = build_models()
    validate_thresholds(
        {'MMD': dict(detection=models['MMD'].eta, novelty=models['MMD'].nu),
         'Pearson': dict(detection=models['Pearson'].tau, novelty=models['Pearson'].nu)},
        models['Pearson'].reg_lambda)
    rng = np.random.RandomState(SEED)
    data = np.concatenate([sample(label, SEGMENT_LENGTH, rng) for label in SEQUENCE]).ravel()
    cps = list(range(SEGMENT_LENGTH, len(data), SEGMENT_LENGTH))
    return run_comparison(
        data, cps, labels_for_segments(SEQUENCE, SEGMENT_LENGTH),
        title=f'Gaussian reference, seed={SEED}', models=models,
        max_delay=MAX_DELAY, show=show, seed=SEED)


if __name__ == '__main__':
    main()
