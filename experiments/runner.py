"""Run identical observations through both models and report their results."""

import contextlib
import io
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from experiments.evaluation import (
    aggregate_metrics, comparison_table, detection_indices, episode_boundaries,
    evaluate_model, validate_max_delay,
)


def model_parameters(model, name):
    """Snapshot the supplied model's actual settings before streaming changes gamma."""
    keys = ['n', 'm', 'k_conf', 'gamma', 'factor', 'dynamic_gamma', 'eta', 'nu']
    if name == 'Pearson':
        keys += ['reg_lambda', 'normalize_distance']
    return {key: getattr(model, key) for key in keys}


def labels_for_segments(sequence, segment_length):
    return np.repeat(sequence, segment_length)


def stream_description(metadata):
    """Display generator metadata without exposing it to either model."""
    lines = ['| True regime | Distribution | Population mean | Population std |',
             '|---|---|---:|---:|']
    for regime in metadata['regimes']:
        lines.append(f"| {regime['regime']} | {regime['distribution']} | "
                     f"{regime['mean']:g} | {regime['std']:g} |")
    return '\n'.join(lines)


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
                            ('Pearson', 'detection'): 'PEARSON_ETA',
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


def run_comparison(data, true_cps, true_labels, *, title, models,
                   max_delay, show=True, seed=None, check_score_bounds=True,
                   plot_dir=None, block=True, stream_metadata=None):
    """Stream through models constructed in the experiment main, without overrides.

    Every model receives only observations. Truth is used for evaluation/plots.
    Models must be fresh instances; each repeated run needs a new pair.
    """
    if len(data) != len(true_labels) or not np.isfinite(data).all():
        raise ValueError('The stream must contain finite observations and one truth label per sample')
    if set(models) != {'MMD', 'Pearson'}:
        raise ValueError('Supply fresh MMD and Pearson models from the experiment main')
    parameters = {name: model_parameters(models[name], name) for name in ('MMD', 'Pearson')}
    thresholds = {'MMD': dict(detection=models['MMD'].eta, novelty=models['MMD'].nu),
                  'Pearson': dict(detection=models['Pearson'].eta, novelty=models['Pearson'].nu)}
    validate_thresholds(thresholds, models['Pearson'].reg_lambda, check_score_bounds)
    validate_max_delay(max_delay)
    boundaries = episode_boundaries(true_labels)
    if list(true_cps) != boundaries[1:-1].tolist():
        raise ValueError('True changes must equal the episode-label boundaries')
    for name in ('MMD', 'Pearson'):
        model = models[name]
        if model.t != 0:
            raise ValueError(f'{name}: construct a fresh model for each run')
        if boundaries[1] < model.n:
            raise ValueError(f'{name}: initialization must finish within the initial episode')
    for name in ('MMD', 'Pearson'):
        if seed is not None:
            np.random.seed(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            for x in data:
                models[name].update(x)
    print(f"\n{title}\nTrue change points: {list(true_cps)}")
    if stream_metadata is not None:
        print(stream_description(stream_metadata))
    print(f'Alarm tolerance: {max_delay} samples (inclusive); delay = alarm index - change index')
    results = {name: evaluate_model(true_cps, true_labels, models[name], max_delay)
               for name in ('MMD', 'Pearson')}
    print(comparison_table(results))
    plots = {}
    if show or plot_dir is not None:
        prefix = Path(plot_dir) / f'seed_{seed}' if plot_dir is not None else None
        plots = plot_comparison(data, true_cps, true_labels, models, title,
                                show=show, block=block, output_prefix=prefix)
    return dict(models=models, metrics=results, thresholds=thresholds, plots=plots,
                config=dict(title=title, seed=seed, max_delay=max_delay,
                            stream_metadata=stream_metadata,
                            model_parameters=parameters, stream_length=len(data),
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


def save_seed_report(runs, output_dir, title):
    """Save individual seed tables/plots and a pooled table for one benchmark."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = {name: aggregate_metrics([run['metrics'][name] for run in runs])
                 for name in ('MMD', 'Pearson')}
    lines = [f'# {title}: individual seed results', '',
             'Both models use the same stream for each seed and the shared evaluator. '
             'Delay is measured at confirmed alarms; assignment is scored when the decision becomes available.', '']
    for run in runs:
        config = run['config']
        lines += [f"## Seed {config['seed']}", '',
                  f"Alarm tolerance: **{config['max_delay']} samples** (inclusive).", '']
        if config.get('stream_metadata') is not None:
            lines += [stream_description(config['stream_metadata']), '']
        for name, parameters in config['model_parameters'].items():
            lines += [f'{name}: ' + ', '.join(f'{key}={value}' for key, value in parameters.items()) + '.', '']
        lines += [comparison_table(run['metrics']), '']
        for kind, path in run['plots'].items():
            lines += [f"![Seed {config['seed']} {kind}]({Path(path).name})", '']
    lines += ['## Aggregate over all seeds', '', comparison_table(aggregate), '']
    report_path = output_dir / 'results.md'
    report_path.write_text('\n'.join(lines))
    serializable = [dict(config=run['config'], thresholds=run['thresholds'],
                         metrics=run['metrics'], plots=run['plots']) for run in runs]
    (output_dir / 'results.json').write_text(json.dumps(
        dict(runs=serializable, aggregate=aggregate), indent=2, allow_nan=False,
        default=lambda value: value.item() if isinstance(value, np.generic) else str(value)) + '\n')
    print(f'Individual seed results and plots saved to {report_path}')
    return str(report_path)


def plot_comparison(data, true_cps, true_labels, models, title, *, show=True,
                    block=True, output_prefix=None):
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
        threshold = model.eta
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
    score_figure = fig
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
    plots = {}
    if output_prefix is not None:
        output_prefix = Path(output_prefix)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        for kind, figure in (('scores', score_figure), ('regimes', fig)):
            path = output_prefix.with_name(f'{output_prefix.name}_{kind}.png')
            figure.savefig(path, dpi=150)
            plots[kind] = str(path)
    if show:
        plt.show(block=block)
    else:
        plt.close(score_figure)
        plt.close(fig)
    return plots
