"""Recurring CORe50 objects, scored by the shared change-point evaluator."""

from core50.data import (DEFAULT_DATA_DIR, FEATURE_SIZE, FRAME_RATE_HZ, OBJECT_CATEGORIES,
                         load_core50_stream)
from experiments.evaluation import episode_boundaries
from experiments.runner import labels_for_segments, run_comparison
from model.owcl_mmd import OWR_CPD
from model.owcl_pearson import PearsonOWRCPD


DATA_DIR = DEFAULT_DATA_DIR
SESSION = 1
SEQUENCE = (1, 11, 21, 41, 11, 1, 41, 21, 1, 41, 11, 21)  # Official object IDs.
SEGMENT_LENGTH = 60  # Consecutive RGB frames per episode.
N = 24  # Reference and recognition window in frames.
M = 12  # Analysis window in frames.
K_CONF = 6
MAX_DELAY = M + K_CONF  # Inclusive tolerance in frames.

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
    stream, sources = load_core50_stream(SEQUENCE, segment_length=SEGMENT_LENGTH,
                                          session=SESSION, data_dir=DATA_DIR)
    labels = labels_for_segments(SEQUENCE, SEGMENT_LENGTH)
    boundaries = episode_boundaries(labels)
    reference_length = max(model.n for model in models.values())
    if reference_length > boundaries[1]:
        raise ValueError(f'The initial CORe50 episode has {boundaries[1]} frames, but the largest '
                         f'reference window needs {reference_length}. Reduce N or increase SEGMENT_LENGTH.')

    print(f'CORe50 session s{SESSION}, {FRAME_RATE_HZ} frames/s, '
          f'{FEATURE_SIZE[0]}×{FEATURE_SIZE[1]} RGB features')
    for object_id in dict.fromkeys(SEQUENCE):
        print(f'  Regime {object_id} = {OBJECT_CATEGORIES[(object_id - 1) // 5]} (o{object_id})')
    result = run_comparison(
        stream, boundaries[1:-1].tolist(), labels, title=f'CORe50 objects, session s{SESSION}',
        models=models, max_delay=MAX_DELAY, show=show)
    result['config'].update(dataset='CORe50', session=SESSION, sample_rate_hz=FRAME_RATE_HZ,
                            feature_image_size=FEATURE_SIZE, feature_count=stream.shape[1],
                            segment_length=SEGMENT_LENGTH, episode_sources=sources)
    return result


if __name__ == '__main__':
    run_experiment()
