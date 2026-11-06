"""ICASSP section 5.2: population moment matching, shared by three mains."""

import numpy as np
from experiments.calibration import calibrate_thresholds, print_calibration
from experiments.runner import labels_for_segments, print_aggregate, run_comparison

SEQUENCE = (1, 2, 3, 4, 2, 1, 4, 3, 1, 4, 2, 3)


def sample_family(label, size, rng, means, stds):
    mean, std = means[label - 1], stds[label - 1]
    if label == 1:
        values = rng.normal(mean, std, size)
    elif label == 2:
        values = rng.laplace(mean, std / np.sqrt(2), size)
    elif label == 3:
        values = rng.exponential(std, size) - std + mean
    elif label == 4:
        values = rng.uniform(mean - std * np.sqrt(3), mean + std * np.sqrt(3), size)
    else:
        raise ValueError('Family label must be 1, 2, 3, or 4')
    return values[:, None]


def generate_moment_matched_stream(tier='T1', segment_length=700, seed=42,
                                   means=None, stds=None):
    defaults = {
        'T1': ((0, 3, -3, 6), (1, 1, 1, 1)),
        'T2': ((0, 0, 0, 0), (1, 2, 0.5, 3)),
        'T3': ((0, 0, 0, 0), (1, 1, 1, 1)),
    }
    if tier not in defaults:
        raise ValueError("Tier must be 'T1', 'T2', or 'T3'")
    if segment_length < 1:
        raise ValueError('Segment length must be positive')
    default_means, default_stds = defaults[tier]
    means = default_means if means is None else means
    stds = default_stds if stds is None else stds
    rng = np.random.RandomState(seed)
    data = np.concatenate([sample_family(label, segment_length, rng, means, stds)
                           for label in SEQUENCE]).ravel()
    return data, list(range(segment_length, len(data), segment_length)), list(SEQUENCE)


def run_tier(*, tier, n, m, k_conf, segment_length, means, stds, mmd_eta,
             mmd_nu, pearson_tau, pearson_nu, reg_lambda, calibration_seed,
             calibration_repeats, seeds, max_delay, show=True):
    if calibration_seed in seeds:
        raise ValueError('Calibration and evaluation seeds must be disjoint')

    def sample(label, size, rng):
        return sample_family(label, size, rng, means, stds)

    calibration = calibrate_thresholds(
        sample, (1, 2, 3, 4), n=n, m=m, k_conf=k_conf,
        reg_lambda=reg_lambda, seed=calibration_seed, repeats=calibration_repeats)
    overrides = {'MMD': dict(detection=mmd_eta, novelty=mmd_nu),
                 'Pearson': dict(detection=pearson_tau, novelty=pearson_nu)}
    print_calibration(calibration, overrides)
    thresholds = {
        name: {key: calibration[name][key] if value is None else value
               for key, value in parameters.items()}
        for name, parameters in overrides.items()
    }
    runs = []
    for seed in seeds:
        data, cps, sequence = generate_moment_matched_stream(
            tier, segment_length, seed, means, stds)
        runs.append(run_comparison(
            data, cps, labels_for_segments(sequence, segment_length),
            title=f'{tier}, seed={seed}', n=n, m=m, k_conf=k_conf,
            thresholds=thresholds, reg_lambda=reg_lambda,
            max_delay=max_delay, show=show))
    print_aggregate(runs)
    return dict(calibration=calibration, runs=runs)
