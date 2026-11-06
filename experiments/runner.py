"""Run identical observations through both models and report their results."""

import contextlib
import io
import numpy as np
import matplotlib.pyplot as plt

from experiments.calibration import make_model
from experiments.evaluation import detection_indices, print_detection_summary


def labels_for_segments(sequence, segment_length):
    return np.repeat(sequence, segment_length)


def run_comparison(data, true_cps, true_labels, *, title, n, m, k_conf,
                   thresholds, reg_lambda, max_delay, show=True):
    if len(data) != len(true_labels) or not np.isfinite(data).all():
        raise ValueError('The stream must contain finite observations and one truth label per sample')
    models = {}
    for name in ("MMD", "Pearson"):
        parameters = thresholds[name]
        bound = 2.0 if name == 'MMD' else 1.0 / reg_lambda
        if not 0 <= parameters['detection'] < bound:
            raise ValueError(f'{name} detection threshold must be below its score bound {bound}')
        novelty_bound = 1.0 if name == 'MMD' else bound
        if not 0 <= parameters['novelty'] < novelty_bound:
            raise ValueError(f'{name} novelty threshold must be below {novelty_bound}')
        model = make_model(name, n, m, k_conf, parameters['detection'],
                           parameters['novelty'], reg_lambda)
        with contextlib.redirect_stdout(io.StringIO()):
            for x in data:
                model.update(x)
        models[name] = model
    print(f"\n{title}\nTrue change points: {list(true_cps)}")
    results = {}
    for name, model in models.items():
        print(f"\n{name} OWR MODEL")
        results[name] = print_detection_summary(
            true_cps, model, max_delay=max_delay, true_labels=true_labels)
    if show:
        plot_comparison(data, true_cps, true_labels, models, title)
    return dict(models=models, metrics=results, thresholds=thresholds)


def print_aggregate(runs):
    print(f"\nAGGREGATE OVER {len(runs)} INDEPENDENT STREAMS")
    for name in ("MMD", "Pearson"):
        metrics = [run['metrics'][name] for run in runs]
        counts = np.array([len(value['matches']) for value in metrics])
        target = len(metrics[0]['matches']) + len(metrics[0]['missed_changes'])
        deviation = np.std(counts, ddof=1) if len(counts) > 1 else 0.0
        print(f"{name}: detection mean {np.mean(counts):.2f}/{target}, std {deviation:.2f}; "
              f"false alarms {sum(len(value['false_alarms']) for value in metrics)}")
        for label, key in (("Re-identification", "recall"), ("Novel discovery", "novel")):
            correct = sum(value[key + '_correct'] for value in metrics)
            total = sum(value[key + '_total'] for value in metrics)
            print(f"  {label}: {correct}/{total}")


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
