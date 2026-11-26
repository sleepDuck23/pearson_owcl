"""Recurring TAU urban acoustic scenes, scored by the shared evaluator."""

from sklearn.preprocessing import StandardScaler

from experiments.evaluation import episode_boundaries
from experiments.runner import labels_for_segments, run_comparison
from model.owcl_mmd import OWR_CPD
from model.owcl_pearson import PearsonOWRCPD
from tau.data import DEFAULT_DATA_DIR, MEL_BANDS, RAW_SAMPLE_RATE_HZ, load_tau_stream


DATA_DIR = DEFAULT_DATA_DIR
CITY = 'barcelona'
DEVICE = 'a'
SEQUENCE = ('airport', 'metro', 'park', 'street_traffic', 'metro', 'airport',
            'street_traffic', 'park', 'airport', 'street_traffic', 'metro', 'park')
WINDOW_SAMPLES = 4410  # 100 ms of audio per feature row.
SEGMENT_LENGTH = 100  # 10 seconds of audio per episode.
N = 40  # Reference and recognition window in feature rows.
M = 15  # Analysis window in feature rows.
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
    stream, sources = load_tau_stream(SEQUENCE, segment_length=SEGMENT_LENGTH,
                                       window_samples=WINDOW_SAMPLES, city=CITY,
                                       device=DEVICE, data_dir=DATA_DIR)
    labels = labels_for_segments(SEQUENCE, SEGMENT_LENGTH)
    boundaries = episode_boundaries(labels)
    reference_lengths = [model.n for model in models.values()]
    if max(reference_lengths) > boundaries[1]:
        raise ValueError(f'The initial TAU episode has {boundaries[1]} feature rows, but '
                         f'the largest reference window needs {max(reference_lengths)}. '
                         'Reduce N or increase SEGMENT_LENGTH.')

    processed = StandardScaler().fit(stream[:min(reference_lengths)]).transform(stream)
    sample_rate_hz = RAW_SAMPLE_RATE_HZ / WINDOW_SAMPLES
    print(f'TAU sampling: {RAW_SAMPLE_RATE_HZ:g} Hz, {sample_rate_hz:g} feature rows/s, '
          f'{MEL_BANDS} log-mel energies per row')
    result = run_comparison(
        processed, boundaries[1:-1].tolist(), labels,
        title=f'TAU Urban Acoustic Scenes 2020 Mobile, {CITY}, device {DEVICE}',
        models=models, max_delay=MAX_DELAY, show=show)
    result['config'].update(dataset='TAU Urban Acoustic Scenes 2020 Mobile Development v2',
                            dataset_doi='10.5281/zenodo.3819968', city=CITY, device=DEVICE,
                            raw_sample_rate_hz=RAW_SAMPLE_RATE_HZ, sample_rate_hz=sample_rate_hz,
                            window_samples=WINDOW_SAMPLES, segment_length=SEGMENT_LENGTH,
                            feature_count=MEL_BANDS, episode_sources=sources)
    return result


if __name__ == '__main__':
    run_experiment()
