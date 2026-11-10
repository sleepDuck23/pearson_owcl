"""Real WISDM Activity Recognition v1.1 smartphone accelerometer benchmark."""

import numpy as np
from scipy.signal import decimate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from data.manage_wisdm import DEFAULT_DATA_DIR, load_wisdm_blocks
from model.owcl_mmd import OWR_CPD
from model.owcl_pearson import PearsonOWRCPD
from experiments.runner import labels_for_segments, run_comparison

N = 150  # Reference/recognition window in downsampled observations (all regimes).
M = 50   # Analysis window in downsampled observations (all regimes).
K_CONF = 6
MAX_DELAY = M + K_CONF  # Inclusive tolerance in downsampled model observations.
SEED = 42
SUBJECT_ID = 1
DATA_DIR = DEFAULT_DATA_DIR
NOMINAL_SAMPLE_RATE_HZ = 20
DOWNSAMPLE_RATE = 1  # Keep the measured rows at their original nominal 20 Hz.
MAX_GAP_SECONDS = 0.5  # Split recordings at longer timestamp gaps or backward time.
USE_PCA = False
PCA_D = 2
# Edit the regime order and raw observations per occurrence here.
SEQUENCE = (0, 1, 2, 3, 1, 0, 3, 2, 0, 3, 1, 2)
SEGMENT_LENGTH = 700  # Real raw observations per activity occurrence.
# Original WISDM activity name -> ground-truth regime ID used in SEQUENCE.
ACTIVITY_MAPPING = {
    'Walking': 0,
    'Jogging': 1,
    'Upstairs': 2,
    'Downstairs': 3,
    'Sitting': 4,
    'Standing': 5,
}
MMD_ETA = 0.06
MMD_NU = 0.96
PEARSON_ETA = 0.15
PEARSON_NU = 0.25
REG_LAMBDA = 0.1  # Pearson operator regularization.


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


def run_experiment(show=True, seed=SEED):
    if (not isinstance(DOWNSAMPLE_RATE, (int, np.integer))
            or isinstance(DOWNSAMPLE_RATE, (bool, np.bool_)) or DOWNSAMPLE_RATE < 1):
        raise ValueError('DOWNSAMPLE_RATE must be a positive integer')
    models = build_models()
    counts = {act: SEQUENCE.count(act) for act in set(SEQUENCE)}
    activities, block_size, metadata = load_wisdm_blocks(
        subject_id=SUBJECT_ID, block_size=SEGMENT_LENGTH, blocks_per_activity=counts,
        activity_mapping=ACTIVITY_MAPPING, data_dir=DATA_DIR,
        max_gap_seconds=MAX_GAP_SECONDS, return_metadata=True)
    stream, _ = build_stream(SEQUENCE, activities, block_size)
    labels = labels_for_segments(SEQUENCE, block_size)[::DOWNSAMPLE_RATE]
    if DOWNSAMPLE_RATE > 1:
        stream = decimate(stream, DOWNSAMPLE_RATE, ftype='fir', axis=0, zero_phase=False)
    true_cps = (np.flatnonzero(labels[1:] != labels[:-1]) + 1).tolist()
    first_episode_end = true_cps[0] if true_cps else len(labels)
    for name, model in models.items():
        if model.n > first_episode_end:
            raise ValueError(f'{name}: initialization needs {model.n} observations, but '
                             f'the first episode has {first_episode_end} after downsampling. '
                             'Reduce N or DOWNSAMPLE_RATE, or increase SEGMENT_LENGTH.')
    print(f"Real {metadata['dataset']}, subject {SUBJECT_ID}; "
          f'nominal model rate {NOMINAL_SAMPLE_RATE_HZ / DOWNSAMPLE_RATE:g} Hz')
    for activity, regime in ACTIVITY_MAPPING.items():
        if regime in counts:
            print(f'  Regime {regime} = {activity}: {counts[regime]} distinct blocks '
                  f'of {block_size} measured rows')
    # Freeze observation-only preprocessing after the initial reference window.
    processed = StandardScaler().fit(stream[:N]).transform(stream)
    if USE_PCA:
        processed = PCA(n_components=PCA_D, random_state=seed).fit(processed[:N]).transform(processed)
    result = run_comparison(
        processed, true_cps, labels,
        title=f'WISDM AR v1.1, subject={SUBJECT_ID}',
        models=models, max_delay=MAX_DELAY, seed=seed, show=show)
    # Preserve file provenance and the exact real blocks behind each episode.
    used = dict.fromkeys(counts, 0)
    metadata['episode_sources'] = []
    for regime in SEQUENCE:
        metadata['episode_sources'].append(metadata['blocks'][regime][used[regime]])
        used[regime] += 1
    result['config'].update(dataset=metadata, downsample_rate=DOWNSAMPLE_RATE,
                            nominal_sample_rate_hz=NOMINAL_SAMPLE_RATE_HZ / DOWNSAMPLE_RATE,
                            raw_segment_length=SEGMENT_LENGTH,
                            sampling_filter='none' if DOWNSAMPLE_RATE == 1 else 'causal FIR')
    return result


if __name__ == '__main__':
    run_experiment()
