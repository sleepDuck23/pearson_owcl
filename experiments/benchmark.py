"""Run configured benchmarks and save auditable, paired model comparisons.

Usage: MPLBACKEND=Agg python3 -m experiments.benchmark
"""

import argparse
import importlib
import json
import platform
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.evaluation import aggregate_metrics, comparison_table

BENCHMARKS = {
    'gaussian': 'main_mmd_pearson',
    'T1': 'main_T1',
    'T2': 'main_T2',
    'T3': 'main_T3',
    'mnist': 'main_mnist',
    'pamap2': 'main_pamap2',
    'wisdm': 'main_wisdm',
}


def audit_run(run):
    return dict(config=run['config'], thresholds=run['thresholds'], metrics=run['metrics'],
                # Keep native observation-count timestamps to audit conversion.
                model_events={name: dict(alarm_counts=model.global_changepoints,
                                         assignments=model.recognition_log)
                              for name, model in run['models'].items()})


def save_reports(report, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'benchmark_results.json').write_text(
        json.dumps(report, indent=2, allow_nan=False,
                   default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + '\n')
    lines = ['# Shared MMD / Pearson benchmark evaluation', '',
             'Both models receive the same observations. Ground truth is available only to the evaluator. '
             'Each run uses its own chronological, one-to-one regime mapping. The initial episode is '
             'excluded from assignment scores; later appearances of its regime are returns.', '',
             'Change times and alarms are zero-based sample indices. Delay = confirmed alarm index − '
             'first new-episode sample index. Tolerance is inclusive and matching cannot cross an episode '
             'end. Assignment times are scored separately and must precede the episode end. '
             'All unmatched monitoring alarms count as false alarms.', '',
             'Aggregates sum numerators and denominators and pool all matched detection delays. '
             'No thresholds were tuned for this evaluation. Several configured thresholds are arbitrary '
             'placeholders, so these results evaluate the checked-in settings rather than establish '
             'a tuned model comparison or an exact reproduction of ICASSP results.', '',
             'WISDM is a synthetic activity surrogate, not WISDM recordings. PAMAP2 retains '
             'lambda=20 and Pearson detection threshold 0.80; its score is bounded by 0.05, '
             'so that configuration cannot confirm alarms.', '',
             'Reproduce with `MPLBACKEND=Agg python3 -m experiments.benchmark`. '
             'Full settings, stream fingerprints, episode boundaries, decisions, and per-run scores '
             'are saved in `benchmark_results.json`.', '']
    for title, benchmark in report['benchmarks'].items():
        configs = [run['config'] for run in benchmark['runs']]
        seeds = [c['seed'] for c in configs]
        settings = configs[0]
        thresholds = benchmark['runs'][0]['thresholds']
        lines += [f'## {title}', '',
                  f"{len(configs)} paired stream(s); seeds: {seeds}. "
                  f"Alarm tolerance: **{settings['max_delay']} samples**. "
                  f"n={settings['n']}, m={settings['m']}, k={settings['k_conf']}; "
                  f"Pearson lambda={settings['reg_lambda']}.", '',
                  f"MMD detection/novelty: {thresholds['MMD']['detection']}/"
                  f"{thresholds['MMD']['novelty']}; Pearson detection/novelty: "
                  f"{thresholds['Pearson']['detection']}/{thresholds['Pearson']['novelty']}.", '',
                  comparison_table(benchmark['aggregate']), '']
    lines += ['## Execution environment', '',
              f"Python {report['environment']['python']}; NumPy {report['environment']['numpy']}; "
              f"scikit-learn {report['environment']['sklearn']}. BLAS thread limit: 1.", '']
    (output_dir / 'benchmark_report.md').write_text('\n'.join(lines))


def main(argv=None):
    import sklearn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmarks', nargs='+', choices=BENCHMARKS, default=list(BENCHMARKS))
    parser.add_argument('--output-dir', type=Path, default=Path('results'))
    args = parser.parse_args(argv)
    report = dict(environment=dict(python=platform.python_version(), numpy=np.__version__,
                                   sklearn=sklearn.__version__), benchmarks={})
    with threadpool_limits(limits=1):
        for benchmark in args.benchmarks:
            module = importlib.import_module(BENCHMARKS[benchmark])
            execute = module.run_experiment if benchmark in ('wisdm', 'pamap2') else module.main
            output = execute(show=False)
            if 'sequences' in output:
                groups = {f'MNIST {name}': [run] for name, run in output['sequences'].items()}
            elif 'runs' in output:
                groups = {benchmark: output['runs']}
            else:
                groups = {benchmark: [output]}
            for title, runs in groups.items():
                report['benchmarks'][title] = dict(
                    aggregate={name: aggregate_metrics([run['metrics'][name] for run in runs])
                               for name in ('MMD', 'Pearson')},
                    runs=[audit_run(run) for run in runs])
            save_reports(report, args.output_dir)
    print(f"\nSaved {args.output_dir / 'benchmark_report.md'} and benchmark_results.json")
    return report


if __name__ == '__main__':
    main()
