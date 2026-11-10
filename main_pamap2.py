"""Real PAMAP2 recordings, scored by the common benchmark evaluator."""

import numpy as np
from scipy.signal import decimate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from data.manage_pamap2 import load_pamap2_full_blocks
from model.owcl_mmd import OWR_CPD
from model.owcl_pearson import PearsonOWRCPD
from experiments.runner import labels_for_segments, run_comparison

RAW_SAMPLE_RATE_HZ = 50
DOWNSAMPLE_RATE = 2  # 1 = 100 Hz, 2 = 50 Hz, 4 = 25 Hz, 5 = 20 Hz.
N = 600  # Reference/recognition window in model observations (after downsampling).
M = 200  # Analysis window in model observations (after downsampling).
K_CONF = 6
MAX_DELAY = M + K_CONF  # Inclusive tolerance in downsampled model observations.
USE_PCA = True
PCA_D = 3
# Edit the activity order and observations per occurrence here.
SEQUENCE = (0, 1, 2, 3, 1, 0, 3, 2, 0, 3, 1, 2)
SEGMENT_LENGTH = 2000  # Raw 100 Hz samples per occurrence; duration stays unchanged.
# Dataset activity ID -> ground-truth regime ID used in SEQUENCE.
# Activity names follow the PAMAP2 dataset documentation.
ACTIVITY_MAPPING = {
    1: 0,   # Lying
    2: 1,   # Sitting
    3: 2,   # Standing
    4: 3,   # Walking
    5: 4,   # Running
    12: 5,  # Ascending stairs
    13: 6,  # Descending stairs
}
MMD_ETA = 0.95
MMD_NU = 0.2
PEARSON_ETA = 0.35
PEARSON_NU = 0.25
REG_LAMBDA = 1.0  # Pearson operator regularization.


def downsample_stream(stream, factor):
    """Causally low-pass filter sensor channels, then retain every factor-th row.

    Filter the assembled stream without activity labels or boundary resets.
    zero_phase=False prevents filtering from using future observations. The FIR
    filter's latency remains part of the measured online detection delay.
    """
    if (not isinstance(factor, (int, np.integer))
            or isinstance(factor, (bool, np.bool_)) or factor < 1):
        raise ValueError('DOWNSAMPLE_RATE must be a positive integer')
    stream = np.asarray(stream, dtype=float)
    if stream.ndim != 2 or not len(stream) or not np.isfinite(stream).all():
        raise ValueError('Provide a nonempty stream of finite sensor rows')
    if factor == 1:
        return stream
    return decimate(stream, factor, ftype='fir', axis=0, zero_phase=False)


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


def run_experiment(show=True):
    models = build_models()
    counts = {act: SEQUENCE.count(act) for act in set(SEQUENCE)}
    activities, block_size = load_pamap2_full_blocks(
        block_size=SEGMENT_LENGTH, blocks_per_activity=counts,
        activity_mapping=ACTIVITY_MAPPING)
    stream, _ = build_stream(SEQUENCE, activities, block_size)
    stream = downsample_stream(stream, DOWNSAMPLE_RATE)
    labels = labels_for_segments(SEQUENCE, block_size)[::DOWNSAMPLE_RATE]
    true_cps = (np.flatnonzero(labels[1:] != labels[:-1]) + 1).tolist()
    sample_rate_hz = RAW_SAMPLE_RATE_HZ / DOWNSAMPLE_RATE
    first_episode_end = true_cps[0] if true_cps else len(labels)
    for name, model in models.items():
        if model.n > first_episode_end:
            raise ValueError(
                f'{name}: reference window n={model.n} exceeds the first activity\'s '
                f'{first_episode_end} downsampled observations. Reduce N or '
                'DOWNSAMPLE_RATE, or increase SEGMENT_LENGTH.')
    filtering = 'no filtering' if DOWNSAMPLE_RATE == 1 else 'causal FIR anti-alias filtering'
    print(f'PAMAP2 sampling: {RAW_SAMPLE_RATE_HZ:g} Hz -> {sample_rate_hz:g} Hz '
          f'({filtering})')
    for name, model in models.items():
        print(f'{name} windows: reference/recognition {model.n / sample_rate_hz:g} s; '
              f'analysis {model.m / sample_rate_hz:g} s')
    processed = StandardScaler().fit(stream[:N]).transform(stream)
    if USE_PCA:
        processed = PCA(n_components=PCA_D, random_state=42).fit(processed[:N]).transform(processed)
    # Evaluate the manually configured thresholds without automatic retuning.
    result = run_comparison(
        processed, true_cps, labels, title=f'PAMAP2 subject101, {sample_rate_hz:g} Hz',
        models=models, max_delay=MAX_DELAY, show=show,
        check_score_bounds=False)
    result['config'].update(raw_sample_rate_hz=RAW_SAMPLE_RATE_HZ,
                            sample_rate_hz=sample_rate_hz,
                            downsample_rate=DOWNSAMPLE_RATE,
                            raw_segment_length=SEGMENT_LENGTH,
                            sampling_filter='none' if DOWNSAMPLE_RATE == 1 else 'causal FIR')
    return result


if __name__ == '__main__':
    run_experiment()
