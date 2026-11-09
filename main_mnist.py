"""ICASSP section 5.3: both MNIST sequences with manually chosen thresholds."""

from pathlib import Path
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

from model.owcl_mmd import OWR_CPD
from model.owcl_pearson import PearsonOWRCPD
from experiments.runner import labels_for_segments, run_comparison, validate_thresholds

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

# Fit PCA on images separate from the evaluation streams.
PCA_IMAGES_PER_DIGIT = 500
SPLIT_SEED = 2027
EVALUATION_SEED = 42
REG_LAMBDA = 0.1
PEARSON_ETA = 0.4  # Arbitrary placeholder; adjust for this experiment.
PEARSON_NU = 0.7  # Arbitrary placeholder; adjust for this experiment.
MAX_DELAY = M + K_CONF
# fetch_openml appends its own 'openml' subdirectory to data_home.
DATA_HOME = Path(__file__).resolve().parent / 'data'


def load_mnist_partitions(pca_per_digit=PCA_IMAGES_PER_DIGIT, seed=SPLIT_SEED):
    print('Loading MNIST and reserving disjoint PCA and evaluation images...')
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True,
                       as_frame=False, parser='auto', data_home=str(DATA_HOME))
    X, y = np.asarray(X, dtype=float) / 255.0, np.asarray(y).astype(int)
    rng = np.random.RandomState(seed)
    preprocessing, evaluation = [], {}
    for digit in sorted(set(y)):
        indices = rng.permutation(np.flatnonzero(y == digit))
        split = pca_per_digit
        if len(indices) <= split:
            raise ValueError(f'Digit {digit} has too few images for the requested disjoint splits')
        preprocessing.append(X[indices[:pca_per_digit]])
        evaluation[int(digit)] = X[indices[split:]]
    return np.vstack(preprocessing), evaluation


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
    _, pools = load_mnist_partitions()
    return build_mnist_stream(sequence, pools, samples_per_digit)


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


def main(show=True):
    models = build_models()
    validate_thresholds(
        {'MMD': dict(detection=models['MMD'].eta, novelty=models['MMD'].nu),
         'Pearson': dict(detection=models['Pearson'].eta, novelty=models['Pearson'].nu)},
        models['Pearson'].reg_lambda)
    preprocessing, evaluation_pools = load_mnist_partitions()
    print(f'Fitting PCA(d={PCA_DIM}) on separate preprocessing images...')
    pca = PCA(n_components=PCA_DIM, random_state=SPLIT_SEED)
    pca.fit(preprocessing)
    # Identical PCA coordinates and Euclidean distances for both models.
    rng = np.random.RandomState(EVALUATION_SEED)
    evaluation_pools = {digit: images[rng.permutation(len(images))]
                        for digit, images in evaluation_pools.items()}

    offsets, results = {}, {}
    for index, (name, sequence) in enumerate(SEQUENCES.items()):
        if index:
            models = build_models()
        raw, cps = build_mnist_stream(sequence, evaluation_pools, SAMPLES_PER_DIGIT, offsets)
        data = pca.transform(raw)
        results[name] = run_comparison(
            data, cps, labels_for_segments(sequence, SAMPLES_PER_DIGIT),
            title=f'MNIST sequence {name}', models=models,
            max_delay=MAX_DELAY, show=show, seed=EVALUATION_SEED)
    return dict(sequences=results)


if __name__ == '__main__':
    main()
