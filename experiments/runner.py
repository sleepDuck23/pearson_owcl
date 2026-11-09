"""Run identical observations through both models and report their results."""

import contextlib
import io
import hashlib
import numpy as np
import matplotlib.pyplot as plt

from model.owcl_mmd import OWR_CPD
from model.owcl_pearson import PearsonOWRCPD
from experiments.evaluation import (
    aggregate_metrics, comparison_table, detection_indices, episode_boundaries,
    evaluate_model, validate_max_delay,
)


def make_model(name, n, m, k_conf, detection, novelty, reg_lambda=0.1):
    common = dict(n=n, m=m, k_conf=k_conf, gamma=0.1, factor=1.0,
                  dynamic_gamma=True)
    if name == "MMD":
        return OWR_CPD(eta=detection, nu=novelty, **common)
    if name == "Pearson":
        return PearsonOWRCPD(tau=detection, nu=novelty,
                             reg_lambda=reg_lambda, normalize_distance=False,
                             **common)
    raise ValueError(f"Unknown model: {name}")


def labels_for_segments(sequence, segment_length):
    return np.repeat(sequence, segment_length)


def validate_thresholds(thresholds, reg_lambda, check_score_bounds=True):
    """Check manually chosen values; never estimate or replace thresholds."""
    if not np.isfinite(reg_lambda) or reg_lambda <= 0:
        raise ValueError('REG_LAMBDA must be finite and positive')
    for name in ("MMD", "Pearson"):
        parameters = thresholds[name]
        for key in ('detection', 'novelty'):
            value = parameters[key]
            if value is None:
                constant = {('MMD', 'detection'): 'MMD_ETA',
                            ('MMD', 'novelty'): 'MMD_NU',
                            ('Pearson', 'detection'): 'PEARSON_TAU',
                            ('Pearson', 'novelty'): 'PEARSON_NU'}[(name, key)]
                raise ValueError(f'Set {constant} in the experiment main file before running; '
                                 'thresholds are chosen manually.')
            if not np.isfinite(value):
                raise ValueError(f'{name} {key} threshold must be finite')
        bound = 2.0 if name == 'MMD' else 1.0 / reg_lambda
        if parameters['detection'] < 0 or (check_score_bounds and parameters['detection'] >= bound):
            raise ValueError(f'{name} detection threshold must be below its score bound {bound}')
        novelty_bound = 1.0 if name == 'MMD' else bound
        if parameters['novelty'] < 0 or (check_score_bounds and parameters['novelty'] > novelty_bound):
            raise ValueError(f'{name} novelty threshold must be between 0 and {novelty_bound}')


def run_comparison(data, true_cps, true_labels, *, title, n, m, k_conf,
                   thresholds, reg_lambda, max_delay, show=True, seed=None,
                   check_score_bounds=True):
    if len(data) != len(true_labels) or not np.isfinite(data).all():
        raise ValueError('The stream must contain finite observations and one truth label per sample')
    validate_thresholds(thresholds, reg_lambda, check_score_bounds)
    validate_max_delay(max_delay)
    boundaries = episode_boundaries(true_labels)
    if list(true_cps) != boundaries[1:-1].tolist():
        raise ValueError('True changes must equal the episode-label boundaries')
    if boundaries[1] < n:
        raise ValueError('Initialization must finish within the initial episode')
    models = {}
    for name in ("MMD", "Pearson"):
        if seed is not None:
            np.random.seed(seed)
        parameters = thresholds[name]
        model = make_model(name, n, m, k_conf, parameters['detection'],
                           parameters['novelty'], reg_lambda)
        with contextlib.redirect_stdout(io.StringIO()):
            for x in data:
                model.update(x)
        models[name] = model
    print(f"\n{title}\nTrue change points: {list(true_cps)}")
    print(f'Alarm tolerance: {max_delay} samples (inclusive); delay = alarm index - change index')
    results = {name: evaluate_model(true_cps, true_labels, model, max_delay)
               for name, model in models.items()}
    print(comparison_table(results))
    if show:
        plot_comparison(data, true_cps, true_labels, models, title)
    return dict(models=models, metrics=results, thresholds=thresholds,
                config=dict(title=title, seed=seed, n=n, m=m, k_conf=k_conf,
                            reg_lambda=reg_lambda, max_delay=max_delay,
                            gamma=0.1, factor=1.0, dynamic_gamma=True,
                            normalize_distance=False, stream_length=len(data),
                            stream_sha256=hashlib.sha256(
                                np.ascontiguousarray(data, dtype='<f8').tobytes()).hexdigest(),
                            true_cps=list(true_cps),
                            episode_labels=np.asarray(true_labels)[boundaries[:-1]].tolist()))


def print_aggregate(runs):
    metrics = {name: aggregate_metrics([run['metrics'][name] for run in runs])
               for name in ('MMD', 'Pearson')}
    print(f"\nAGGREGATE OVER {len(runs)} PAIRED STREAMS")
    print(f"Alarm tolerance: {metrics['MMD']['max_delay']} samples (inclusive)")
    print(comparison_table(metrics))
    return metrics


def plot_comparison(data, true_cps, true_labels, models, title):
    scalar = np.asarray(data).ndim == 1 or np.asarray(data).shape[1] == 1
    fig, axes = plt.subplots(3 if scalar else 2, 1,
                             figsize=(12, 10 if scalar else 7), sharex=True)
    fig.suptitle(title)
    if scalar:
        axes[0].plot(np.asarray(data).reshape(-1), color='black', linewidth=0.5)
        axes[0].set_ylabel('Signal')
        score_axes = axes[1:]
    else:
        score_axes = axes
    for ax, (name, model), color in zip(score_axes, models.items(), ('teal', 'purple')):
        scores = model.mmd_scores if name == 'MMD' else model.pearson_div_scores
        threshold = model.eta if name == 'MMD' else model.tau
        ax.plot(scores, color=color, label=f'{name} score', linewidth=1)
        ax.axhline(threshold, color='orange', linestyle='--', label='Threshold')
        for i, cp in enumerate(detection_indices(model)):
            ax.axvline(cp, color='red', alpha=0.7, label='Alarm' if i == 0 else None)
        ax.legend(loc='upper right')
        ax.set_ylabel(name)
    for ax in axes:
        for cp in true_cps:
            ax.axvline(cp, color='black', linestyle=':', alpha=0.5)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel('Stream index (zero-based)')
    fig.tight_layout()
    # Preserve the model's IDs; do not relabel mistakes using future truth.
    lookup, truth = {}, []
    for label in true_labels:
        lookup.setdefault(label, len(lookup))
        truth.append(lookup[label])
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(f'{title}: regime IDs (truth ordered by first appearance)')
    for ax, (name, model) in zip(axes, models.items()):
        prediction = np.asarray(model.regime_log, dtype=float)
        prediction[prediction < 0] = np.nan
        ax.step(range(len(truth)), truth, where='post', color='grey', label='Truth')
        ax.step(range(len(prediction)), prediction, where='post', color='red', label=name)
        ax.set_ylabel('Regime ID')
        ax.legend()
    axes[-1].set_xlabel('Stream index (zero-based)')
    fig.tight_layout()
    plt.show()
