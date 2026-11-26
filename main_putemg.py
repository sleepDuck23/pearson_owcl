"""Recurring putEMG hand gestures, scored by the shared evaluator."""

from sklearn.preprocessing import StandardScaler

from experiments.evaluation import episode_boundaries
from experiments.runner import labels_for_segments, run_comparison
from model.owcl_mmd import OWR_CPD
from model.owcl_pearson import PearsonOWRCPD
from putemg.data import (DEFAULT_DATA_DIR, DEFAULT_RECORDING, FEATURE_NAMES,
                         GESTURE_NAMES, RAW_SAMPLE_RATE_HZ, load_putemg_stream)


DATA_DIR = DEFAULT_DATA_DIR
RECORDING = DEFAULT_RECORDING  # Participant 03, first repeats_long recording.
SEQUENCE = (0, 1, 2, 3, 1, 0, 3, 2, 0, 3, 1, 2)
WINDOW_SAMPLES = 256  # 50 ms of raw EMG per feature row.
SEGMENT_LENGTH = 36  # Feature rows per gesture episode.
N = 18  # Reference and recognition window in feature rows.
M = 8   # Analysis window in feature rows.
K_CONF = 6
MAX_DELAY = M + K_CONF  # Inclusive tolerance in feature rows.

# Starting values for manual tuning.
MMD_ETA = 0.15
MMD_NU = 0.90
PEARSON_ETA = 0.08
PEARSON_NU = 0.05
REG_LAMBDA = 1.0


def build_models():
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
    stream, metadata = load_putemg_stream(SEQUENCE, segment_length=SEGMENT_LENGTH,
                                          window_samples=WINDOW_SAMPLES,
                                          recording=RECORDING, data_dir=DATA_DIR)
    labels = labels_for_segments(SEQUENCE, SEGMENT_LENGTH)
    boundaries = episode_boundaries(labels)
    reference_lengths = [model.n for model in models.values()]
    if max(reference_lengths) > boundaries[1]:
        raise ValueError(f'The initial putEMG episode has {boundaries[1]} feature rows, but '
                         f'the largest reference window needs {max(reference_lengths)}. '
                         'Reduce N or increase SEGMENT_LENGTH.')

    # Share a scaler fitted within both models' initialization windows.
    processed = StandardScaler().fit(stream[:min(reference_lengths)]).transform(stream)
    sample_rate_hz = RAW_SAMPLE_RATE_HZ / WINDOW_SAMPLES
    print(f'putEMG sampling: {RAW_SAMPLE_RATE_HZ:g} Hz, {sample_rate_hz:g} feature rows/s '
          f'({WINDOW_SAMPLES} raw samples per row)')
    for gesture in dict.fromkeys(SEQUENCE):
        print(f'  Regime {gesture} = {GESTURE_NAMES[gesture]}')
    result = run_comparison(
        processed, boundaries[1:-1].tolist(), labels,
        title=f"putEMG subject {metadata['subject']}, {metadata['trajectory']}",
        models=models, max_delay=MAX_DELAY, show=show)
    result['config'].update(metadata, dataset='putEMG', raw_sample_rate_hz=RAW_SAMPLE_RATE_HZ,
                            sample_rate_hz=sample_rate_hz, window_samples=WINDOW_SAMPLES,
                            segment_length=SEGMENT_LENGTH,
                            raw_segment_length=SEGMENT_LENGTH * WINDOW_SAMPLES,
                            feature_names=FEATURE_NAMES)
    return result


if __name__ == '__main__':
    run_experiment()
