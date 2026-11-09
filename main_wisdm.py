"""Synthetic activity surrogate (not real WISDM recordings)."""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from experiments.runner import labels_for_segments, run_comparison

N = 150
M = 50
K_CONF = 6
MAX_DELAY = M + K_CONF  # Inclusive tolerance in downsampled model observations.
SEED = 42
DOWNSAMPLE_RATE = 5
USE_PCA = False
PCA_D = 2
SEQUENCE = (0, 1, 0, 2, 3, 2, 1, 3, 0, 2, 3)
MMD_ETA = 0.06
MMD_NU = 0.96
PEARSON_TAU = 0.40
PEARSON_NU = 0.71
REG_LAMBDA = 1e-2


def generate_synthetic_activity_data(blocks_per_activity=None, seed=42):
    """Synthetic Gaussian activity surrogate; this does not load WISDM recordings."""
    print("Generating synthetic activity data (WISDM placeholder)...")
    block_size = 2500
    rng = np.random.RandomState(seed)
    specifications = {
        0: (0.5, [1.0, 0.0, 9.8]),
        1: (1.5, [0.0, 2.0, 9.8]),
        2: (0.8, [0.5, 0.5, 9.8]),
        3: (0.7, [-0.5, -0.5, 9.8]),
        4: (0.1, [0.0, 0.0, 9.8]),
        5: (0.1, [0.0, 9.8, 0.0]),
        6: (0.1, [9.8, 0.0, 0.0]),
    }
    counts = blocks_per_activity if blocks_per_activity is not None else dict.fromkeys(specifications, 1)
    activities = {}
    for act, count in sorted(counts.items()):
        if act not in specifications or not isinstance(count, (int, np.integer)) or count < 1:
            raise ValueError("Activity IDs and block counts must be valid")
        std, mean = specifications[act]
        activities[act] = rng.normal(0, std, (count * block_size, 3)) + mean
    return activities, block_size


def build_stream(sequence, activities, block_size):
    """Consume disjoint blocks per activity and derive boundaries from their lengths."""
    if not sequence or block_size < 1:
        raise ValueError("Provide a nonempty sequence and positive block size")
    stream, true_cps = [], []
    offsets = {act: 0 for act in sequence}
    current_step = 0
    for i, act_id in enumerate(sequence):
        if act_id not in activities:
            raise ValueError(f"No observations available for activity {act_id}")
        start = offsets[act_id]
        block = np.asarray(activities[act_id][start:start + block_size])
        if len(block) != block_size:
            raise ValueError(f"Not enough distinct observations for activity {act_id}")
        if block.ndim != 2 or not np.isfinite(block).all():
            raise ValueError(f"Activity {act_id} must contain finite observation rows")
        if i and act_id != sequence[i - 1]:
            true_cps.append(current_step)
        stream.append(block)
        offsets[act_id] += len(block)
        current_step += len(block)
    return np.vstack(stream), true_cps

def run_experiment(show=True, seed=SEED):
    counts = {act: SEQUENCE.count(act) for act in set(SEQUENCE)}
    activities, block_size = generate_synthetic_activity_data(counts, seed=seed)
    stream, _ = build_stream(SEQUENCE, activities, block_size)
    labels = labels_for_segments(SEQUENCE, block_size)[::DOWNSAMPLE_RATE]
    stream = stream[::DOWNSAMPLE_RATE]
    true_cps = (np.flatnonzero(labels[1:] != labels[:-1]) + 1).tolist()
    # Freeze observation-only preprocessing after the initial reference window.
    processed = StandardScaler().fit(stream[:N]).transform(stream)
    if USE_PCA:
        processed = PCA(n_components=PCA_D, random_state=seed).fit(processed[:N]).transform(processed)
    return run_comparison(
        processed, true_cps, labels,
        title=f'Synthetic activity (WISDM placeholder), seed={seed}',
        n=N, m=M, k_conf=K_CONF,
        thresholds={'MMD': dict(detection=MMD_ETA, novelty=MMD_NU),
                    'Pearson': dict(detection=PEARSON_TAU, novelty=PEARSON_NU)},
        reg_lambda=REG_LAMBDA, max_delay=MAX_DELAY, seed=seed, show=show)


if __name__ == '__main__':
    run_experiment()
