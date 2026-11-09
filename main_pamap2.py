"""Real PAMAP2 recordings, scored by the common benchmark evaluator."""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from data.manage_pamap2 import load_pamap2_full_blocks
from experiments.runner import labels_for_segments, run_comparison

N = 200
M = 100
K_CONF = 6
MAX_DELAY = M + K_CONF  # Inclusive tolerance in model observations at 100 Hz.
USE_PCA = False
PCA_D = 5
SEQUENCE = (0, 1, 0, 2, 3, 2, 1, 3, 0, 2, 3)
MMD_ETA = 0.75
MMD_NU = 0.95
PEARSON_TAU = 0.80
PEARSON_NU = 0.71
REG_LAMBDA = 20.0  # Preserve original settings; score <= 1/lambda = 0.05.


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

def run_experiment(show=True):
    counts = {act: SEQUENCE.count(act) for act in set(SEQUENCE)}
    activities, block_size = load_pamap2_full_blocks(
        block_size=1500, blocks_per_activity=counts)
    stream, true_cps = build_stream(SEQUENCE, activities, block_size)
    labels = labels_for_segments(SEQUENCE, block_size)
    processed = StandardScaler().fit(stream[:N]).transform(stream)
    if USE_PCA:
        processed = PCA(n_components=PCA_D, random_state=42).fit(processed[:N]).transform(processed)
    # This benchmark deliberately retains its pre-existing unreachable Pearson
    # threshold. Do not silently retune parameters using evaluation truth.
    return run_comparison(
        processed, true_cps, labels, title='PAMAP2 subject101',
        n=N, m=M, k_conf=K_CONF,
        thresholds={'MMD': dict(detection=MMD_ETA, novelty=MMD_NU),
                    'Pearson': dict(detection=PEARSON_TAU, novelty=PEARSON_NU)},
        reg_lambda=REG_LAMBDA, max_delay=MAX_DELAY, show=show,
        check_score_bounds=False)


if __name__ == '__main__':
    run_experiment()
