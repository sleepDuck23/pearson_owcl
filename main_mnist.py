"""ICASSP section 5.3: both MNIST sequences with held-out calibration."""

from pathlib import Path
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

from experiments.calibration import calibrate_thresholds, print_calibration
from experiments.runner import labels_for_segments, run_comparison

N = 150
M = 40
K_CONF = 6
SAMPLES_PER_DIGIT = 300
PCA_DIM = 50
MMD_ETA = 0.06
MMD_NU = 0.96
SEQUENCES = {
    'A': (3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5),
    'B': (4, 9, 4, 8, 3, 8, 9, 3, 4, 8, 3),
}

# Explicit choices for details not supplied by the paper. These partitions
# are disjoint; neither PCA nor thresholds use evaluation images.
PCA_IMAGES_PER_DIGIT = 500
CALIBRATION_IMAGES_PER_DIGIT = 1500
SPLIT_SEED = 2027
EVALUATION_SEED = 42
CALIBRATION_SEED = 10020
CALIBRATION_REPEATS = 12
REG_LAMBDA = 0.1
PEARSON_TAU = None
PEARSON_NU = None
MAX_DELAY = M + K_CONF
# fetch_openml appends its own 'openml' subdirectory to data_home.
DATA_HOME = Path(__file__).resolve().parent / 'data'


def load_mnist_partitions(pca_per_digit=PCA_IMAGES_PER_DIGIT,
                          calibration_per_digit=CALIBRATION_IMAGES_PER_DIGIT,
                          seed=SPLIT_SEED):
    print('Loading MNIST and reserving disjoint PCA, calibration, and evaluation images...')
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True,
                       as_frame=False, parser='auto', data_home=str(DATA_HOME))
    X, y = np.asarray(X, dtype=float) / 255.0, np.asarray(y).astype(int)
    rng = np.random.RandomState(seed)
    preprocessing, calibration, evaluation = [], {}, {}
    for digit in sorted(set(y)):
        indices = rng.permutation(np.flatnonzero(y == digit))
        split = pca_per_digit + calibration_per_digit
        if len(indices) <= split:
            raise ValueError(f'Digit {digit} has too few images for the requested disjoint splits')
        preprocessing.append(X[indices[:pca_per_digit]])
        calibration[int(digit)] = X[indices[pca_per_digit:split]]
        evaluation[int(digit)] = X[indices[split:]]
    return np.vstack(preprocessing), calibration, evaluation


def build_mnist_stream(sequence, evaluation_pools, samples_per_digit=300, offsets=None):
    """Consume fresh images; shared offsets also prevent reuse across A and B."""
    if not sequence or samples_per_digit < 1:
        raise ValueError('Provide a nonempty sequence and positive segment length')
    offsets = {} if offsets is None else offsets
    for digit in set(sequence):
        needed = offsets.get(digit, 0) + sequence.count(digit) * samples_per_digit
        if digit not in evaluation_pools or len(evaluation_pools[digit]) < needed:
            raise ValueError(f'Digit {digit} needs {needed} distinct evaluation images')
    segments, cps = [], []
    for i, digit in enumerate(sequence):
        start = offsets.get(digit, 0)
        segments.append(evaluation_pools[digit][start:start + samples_per_digit])
        offsets[digit] = start + samples_per_digit
        if i and digit != sequence[i - 1]:
            cps.append(i * samples_per_digit)
    return np.vstack(segments), cps


def generate_mnist_stream(sequence, samples_per_digit=300):
    """Compatibility helper; the main below loads/splits the dataset only once."""
    _, _, pools = load_mnist_partitions()
    return build_mnist_stream(sequence, pools, samples_per_digit)


def main(show=True):
    preprocessing, calibration_pools, evaluation_pools = load_mnist_partitions()
    print(f'Fitting PCA(d={PCA_DIM}) on separate preprocessing images...')
    pca = PCA(n_components=PCA_DIM, random_state=SPLIT_SEED)
    pca.fit(preprocessing)
    calibration_pools = {digit: pca.transform(images)
                         for digit, images in calibration_pools.items()}
    # Identical PCA coordinates and Euclidean distances for both models.
    rng = np.random.RandomState(EVALUATION_SEED)
    evaluation_pools = {digit: images[rng.permutation(len(images))]
                        for digit, images in evaluation_pools.items()}

    def sample(label, size, calibration_rng):
        pool = calibration_pools[label]
        if size > len(pool):
            raise ValueError('Increase CALIBRATION_IMAGES_PER_DIGIT')
        return pool[calibration_rng.choice(len(pool), size=size, replace=False)]

    digits = sorted({digit for sequence in SEQUENCES.values() for digit in sequence})
    calibration = calibrate_thresholds(
        sample, digits, n=N, m=M, k_conf=K_CONF, reg_lambda=REG_LAMBDA,
        seed=CALIBRATION_SEED, repeats=CALIBRATION_REPEATS, names=('Pearson',))
    overrides = {'Pearson': dict(detection=PEARSON_TAU, novelty=PEARSON_NU)}
    print(f'MMD paper settings: eta={MMD_ETA}, nu={MMD_NU}')
    print_calibration(calibration, overrides)
    thresholds = {'MMD': dict(detection=MMD_ETA, novelty=MMD_NU),
                  'Pearson': {key: calibration['Pearson'][key] if value is None else value
                              for key, value in overrides['Pearson'].items()}}
    offsets, results = {}, {}
    for name, sequence in SEQUENCES.items():
        raw, cps = build_mnist_stream(sequence, evaluation_pools, SAMPLES_PER_DIGIT, offsets)
        data = pca.transform(raw)
        results[name] = run_comparison(
            data, cps, labels_for_segments(sequence, SAMPLES_PER_DIGIT),
            title=f'MNIST sequence {name}', n=N, m=M, k_conf=K_CONF,
            thresholds=thresholds, reg_lambda=REG_LAMBDA,
            max_delay=MAX_DELAY, show=show)
    return dict(calibration=calibration, sequences=results)


if __name__ == '__main__':
    main()
