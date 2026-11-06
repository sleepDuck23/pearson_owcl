"""ICASSP section 5.2, T2: common mean and distinct variances."""

from experiments.moment_matched import run_tier

# Reported paper settings: 700 observations per segment, three streams.
N = 300
M = 100
K_CONF = 6
SEGMENT_LENGTH = 700
SEEDS = (42, 43, 44)

# T1/T2 population values and all seeds are choices not specified in the PDF.
MEANS = (0.0, 0.0, 0.0, 0.0)
STDS = (1.0, 2.0, 0.5, 3.0)
MMD_ETA = None                  # None: independent calibration.
MMD_NU = 0.99                 # Table 1.
PEARSON_TAU = None
PEARSON_NU = None
REG_LAMBDA = 0.1                # Pearson extension, not a reported ICASSP value.
CALIBRATION_SEED = 10002
CALIBRATION_REPEATS = 12
MAX_DELAY = M + K_CONF          # Explicit evaluation horizon, not a paper setting.


def main(show=True):
    return run_tier(
        tier='T2', n=N, m=M, k_conf=K_CONF, segment_length=SEGMENT_LENGTH,
        means=MEANS, stds=STDS, mmd_eta=MMD_ETA, mmd_nu=MMD_NU,
        pearson_tau=PEARSON_TAU, pearson_nu=PEARSON_NU,
        reg_lambda=REG_LAMBDA, calibration_seed=CALIBRATION_SEED,
        calibration_repeats=CALIBRATION_REPEATS, seeds=SEEDS,
        max_delay=MAX_DELAY, show=show)


if __name__ == '__main__':
    main()
