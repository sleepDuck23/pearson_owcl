"""ICASSP section 5.2, T3: common mean and common variance."""

from data.generation import generate_moment_matched_stream
from model.owcl_mmd import OWR_CPD
from model.owcl_pearson import PearsonOWRCPD
from experiments.runner import labels_for_segments, print_aggregate, run_comparison, validate_thresholds

# Reported paper settings: 700 observations per segment, three streams.
N = 600
M = 200
K_CONF = 6
SEGMENT_LENGTH = 700
SEEDS = (42, 43, 44)

# T1/T2 population values and all seeds are choices not specified in the PDF.
MEANS = (0.0, 0.0, 0.0, 0.0)
STDS = (1.0, 1.0, 1.0, 1.0)
MMD_ETA = 0.1  # Arbitrary placeholder; adjust for this experiment.
MMD_NU = 0.99  # Arbitrary placeholder; adjust for this experiment.
PEARSON_TAU = 0.4  # Arbitrary placeholder; adjust for this experiment.
PEARSON_NU = 0.7  # Arbitrary placeholder; adjust for this experiment.
REG_LAMBDA = 0.1                # Pearson extension, not a reported ICASSP value.
MAX_DELAY = M + K_CONF          # Explicit evaluation horizon, not a paper setting.


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
    models = build_models()
    validate_thresholds(
        {'MMD': dict(detection=models['MMD'].eta, novelty=models['MMD'].nu),
         'Pearson': dict(detection=models['Pearson'].tau, novelty=models['Pearson'].nu)},
        models['Pearson'].reg_lambda)
    runs = []
    for index, seed in enumerate(SEEDS):
        if index:
            models = build_models()
        data, cps, sequence = generate_moment_matched_stream(
            'T3', SEGMENT_LENGTH, seed, MEANS, STDS)
        runs.append(run_comparison(
            data, cps, labels_for_segments(sequence, SEGMENT_LENGTH),
            title=f'T3, seed={seed}', models=models,
            max_delay=MAX_DELAY, show=show, seed=seed))
    print_aggregate(runs)
    return dict(runs=runs)


if __name__ == '__main__':
    main()
