"""Recurring bearing conditions from CWRU, scored by the shared evaluator."""

from sklearn.preprocessing import StandardScaler

from cwru.data import (DEFAULT_DATA_DIR, FEATURE_NAMES, RAW_SAMPLE_RATE_HZ,
                       load_cwru_stream)
from experiments.evaluation import episode_boundaries
from experiments.runner import labels_for_segments, run_comparison
from model.owcl_mmd import OWR_CPD
from model.owcl_pearson import PearsonOWRCPD


DATA_DIR = DEFAULT_DATA_DIR
MOTOR_LOAD_HP = 0  # Use the same load for every condition.
SEQUENCE = (0, 1, 2, 3, 1, 0, 3, 2, 0, 3, 1, 2)
WINDOW_SAMPLES = 2048  # Raw vibration samples per feature row.
SEGMENT_LENGTH = 36  # Feature rows per occurrence.
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
    """Build the two detectors using the parameters selected for this experiment."""
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
    stream, sources = load_cwru_stream(SEQUENCE, segment_length=SEGMENT_LENGTH,
                                      window_samples=WINDOW_SAMPLES,
                                      motor_load=MOTOR_LOAD_HP, data_dir=DATA_DIR)
    labels = labels_for_segments(SEQUENCE, SEGMENT_LENGTH)
    boundaries = episode_boundaries(labels)
    true_cps = boundaries[1:-1].tolist()
    reference_lengths = [model.n for model in models.values()]
    reference_length = max(reference_lengths)
    if reference_length > boundaries[1]:
        raise ValueError(f'The initial CWRU episode has {boundaries[1]} feature rows, but the '
                         f'largest reference window needs {reference_length}. Reduce N or '
                         'increase SEGMENT_LENGTH.')

    # Share a scaler fitted within both models' initialization windows.
    processed = StandardScaler().fit(stream[:min(reference_lengths)]).transform(stream)
    sample_rate_hz = RAW_SAMPLE_RATE_HZ / WINDOW_SAMPLES
    print(f'CWRU sampling: {RAW_SAMPLE_RATE_HZ:g} Hz, {sample_rate_hz:g} feature rows/s '
          f'({WINDOW_SAMPLES} raw samples per row)')
    recording_by_regime = {source['regime']: source for source in sources}
    for regime, source in recording_by_regime.items():
        print(f"  Regime {regime} = {source['condition']} ({source['record_id']}.mat)")
    result = run_comparison(
        processed, true_cps, labels, title=f'CWRU drive-end vibration, {MOTOR_LOAD_HP} HP',
        models=models, max_delay=MAX_DELAY, show=show)
    result['config'].update(dataset='CWRU Bearing Data Center', motor_load_hp=MOTOR_LOAD_HP,
                            raw_sample_rate_hz=RAW_SAMPLE_RATE_HZ, sample_rate_hz=sample_rate_hz,
                            window_samples=WINDOW_SAMPLES, segment_length=SEGMENT_LENGTH,
                            raw_segment_length=SEGMENT_LENGTH * WINDOW_SAMPLES,
                            feature_names=FEATURE_NAMES, episode_sources=sources)
    return result


if __name__ == '__main__':
    run_experiment()
