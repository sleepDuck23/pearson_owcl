"""ICASSP section 5.2, T2: common mean and distinct variances."""

from data.generation import generate_moment_matched_stream
from experiments.runner import labels_for_segments, print_aggregate, run_comparison, validate_thresholds

# Reported paper settings: 700 observations per segment, three streams.
N = 300
M = 100
K_CONF = 6
SEGMENT_LENGTH = 700
SEEDS = (42, 43, 44)

# T1/T2 population values and all seeds are choices not specified in the PDF.
MEANS = (0.0, 0.0, 0.0, 0.0)
STDS = (1.0, 2.0, 0.5, 3.0)
MMD_ETA = 0.1  # Arbitrary placeholder; adjust for this experiment.
MMD_NU = 0.99                 # Table 1.
PEARSON_TAU = 0.4  # Arbitrary placeholder; adjust for this experiment.
PEARSON_NU = 0.7  # Arbitrary placeholder; adjust for this experiment.
REG_LAMBDA = 0.1                # Pearson extension, not a reported ICASSP value.
MAX_DELAY = M + K_CONF          # Explicit evaluation horizon, not a paper setting.


def main(show=True):
    thresholds = {'MMD': dict(detection=MMD_ETA, novelty=MMD_NU),
                  'Pearson': dict(detection=PEARSON_TAU, novelty=PEARSON_NU)}
    validate_thresholds(thresholds, REG_LAMBDA)
    runs = []
    for seed in SEEDS:
        data, cps, sequence = generate_moment_matched_stream(
            'T2', SEGMENT_LENGTH, seed, MEANS, STDS)
        runs.append(run_comparison(
            data, cps, labels_for_segments(sequence, SEGMENT_LENGTH),
            title=f'T2, seed={seed}', n=N, m=M, k_conf=K_CONF,
            thresholds=thresholds, reg_lambda=REG_LAMBDA,
            max_delay=MAX_DELAY, show=show, seed=seed))
    print_aggregate(runs)
    return dict(runs=runs)


if __name__ == '__main__':
    main()
