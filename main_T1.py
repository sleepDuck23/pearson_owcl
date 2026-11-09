"""ICASSP section 5.2, T1: distinct means and common variance."""

import matplotlib.pyplot as plt

from data.generation import generate_moment_matched_stream
from model.owcl_mmd import OWR_CPD
from model.owcl_pearson import PearsonOWRCPD
from experiments.runner import (
    labels_for_segments, print_aggregate, run_comparison, save_seed_report,
    validate_thresholds,
)

# Reported paper settings: 700 observations per segment, three streams.
N = 150
M = 40
K_CONF = 6
SEGMENT_LENGTH = 700
SEEDS = (42, 43, 44)
RANDOMIZE_FAMILIES = True  # Assign Gaussian/Laplace/Exponential/Uniform to IDs per seed.

# T1/T2 population values and all seeds are choices not specified in the PDF.
MEANS = (0.0, 3.0, -3.0, 6.0)
STDS = (1.0, 1.0, 1.0, 1.0)
MMD_ETA = 0.1  # Arbitrary placeholder; adjust for this experiment.
MMD_NU = 0.63
PEARSON_ETA = 0.2  # Arbitrary placeholder; adjust for this experiment.
PEARSON_NU = 0.7  # Arbitrary placeholder; adjust for this experiment.
REG_LAMBDA = 0.1
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
            eta=PEARSON_ETA,
            nu=PEARSON_NU,
            k_conf=K_CONF,
            reg_lambda=REG_LAMBDA,
            gamma=0.1,
            factor=1.0,
            dynamic_gamma=True,
            normalize_distance=False,
        ),
    }


def main(show=True, output_dir=None):
    """Print each seed separately; display all seed figures after the final run."""
    models = build_models()
    validate_thresholds(
        {'MMD': dict(detection=models['MMD'].eta, novelty=models['MMD'].nu),
         'Pearson': dict(detection=models['Pearson'].eta, novelty=models['Pearson'].nu)},
        models['Pearson'].reg_lambda)
    runs = []
    for index, seed in enumerate(SEEDS):
        if index:
            models = build_models()
        data, cps, sequence, stream_metadata = generate_moment_matched_stream(
            'T1', SEGMENT_LENGTH, seed, MEANS, STDS,
            randomize_families=RANDOMIZE_FAMILIES, return_metadata=True)
        runs.append(run_comparison(
            data, cps, labels_for_segments(sequence, SEGMENT_LENGTH),
            title=f'T1, seed={seed}', models=models,
            max_delay=MAX_DELAY, show=show, seed=seed,
            plot_dir=output_dir, block=False, stream_metadata=stream_metadata))
    aggregate = print_aggregate(runs)
    report = save_seed_report(runs, output_dir, 'T1') if output_dir is not None else None
    if show:
        # Display every seed's figures together after all results have printed.
        plt.show()
    return dict(runs=runs, aggregate=aggregate, report=report)


if __name__ == '__main__':
    main()
