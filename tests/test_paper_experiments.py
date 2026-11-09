import contextlib
import copy
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.evaluation import evaluate_changepoints, evaluate_recognition
from data.generation import generate_moment_matched_stream
from model.owcl_mmd import OWR_CPD
from model.owcl_pearson import PearsonOWRCPD
from main_mnist import build_mnist_stream, load_mnist_partitions


def make_model(name, n, m, k_conf, detection, novelty, reg_lambda=0.1):
    """Small-model convenience for algorithm tests, not experiment configuration."""
    common = dict(n=n, m=m, k_conf=k_conf)
    if name == 'MMD':
        return OWR_CPD(eta=detection, nu=novelty, **common)
    return PearsonOWRCPD(tau=detection, nu=novelty, reg_lambda=reg_lambda, **common)


class ModelTests(unittest.TestCase):
    def test_confirmation_counter_resets_on_a_non_exceedance(self):
        for name in ('MMD', 'Pearson'):
            model = make_model(name, 4, 2, 3, 0.5, 0.5)
            target = ('model.owcl_mmd.OWR_CPD._compute_mmd_squared' if name == 'MMD'
                      else 'model.owcl_pearson.pearson_score')
            with self.subTest(name=name), patch(target, side_effect=[1, 1, 0, 1, 1, 1]):
                with contextlib.redirect_stdout(io.StringIO()):
                    for x in range(11):
                        model.update([float(x)])
                self.assertEqual(model.global_changepoints, [11])

    def test_scores_start_at_full_windows_and_resume_without_extra_sample(self):
        for name in ('MMD', 'Pearson'):
            with self.subTest(name=name), contextlib.redirect_stdout(io.StringIO()):
                model = make_model(name, 4, 2, 1, 0, 0.5)
                scores = [model.update([float(x)]) for x in range(18)]
                self.assertEqual(model.global_changepoints, [6, 12, 18])
                self.assertEqual(np.flatnonzero(np.isfinite(scores)).tolist(), [5, 11, 17])
                self.assertEqual([e['time'] for e in model.recognition_log], [4, 10, 16])

    def test_bandwidth_precedes_new_statistics_and_candidates_keep_their_gamma(self):
        observations = np.array([0, 1, 2, 3, 10, 12, 30, 50, 80, 120], dtype=float)[:, None]
        for name in ('MMD', 'Pearson'):
            with self.subTest(name=name), contextlib.redirect_stdout(io.StringIO()):
                model = make_model(name, 4, 2, 1, 0, 0.5)
                for x in observations[:4]:
                    model.update(x)
                old = copy.deepcopy(model.regimes[0])
                method = '_compute_coherence' if name == 'MMD' else '_compute_pearson_divergence'
                compare = getattr(model, method)
                comparisons = []

                def record(batch, representative):
                    comparisons.append(representative['gamma'])
                    return compare(batch, representative)

                with patch.object(model, method, side_effect=record):
                    for x in observations[4:]:
                        model.update(x)
                self.assertEqual(comparisons, [old['gamma']])
                new = model.regimes[model.active_regime_id]
                batch = observations[6:10]
                distances = (batch - batch.T) ** 2
                expected_gamma = 1 / (2 * np.median(distances[np.triu_indices(4, 1)]))
                self.assertAlmostEqual(new['gamma'], expected_gamma)
                self.assertNotEqual(old['gamma'], new['gamma'])
                K = np.exp(-expected_gamma * distances)
                np.testing.assert_allclose(new['mu_K'], K.mean(axis=1))
                if name == 'Pearson':
                    np.testing.assert_allclose(new['W'], np.linalg.inv(K + 4 * model.reg_lambda * np.eye(4)))

    def test_recursive_pearson_matches_independent_feature_space_operator(self):
        rng = np.random.RandomState(713)
        model = PearsonOWRCPD(8, 3, 1e9, 0.5, 2, reg_lambda=0.1)
        for x in rng.normal(size=(65, 3)):
            score = model.update(x)
            if not np.isfinite(score):
                continue
            joint = np.vstack((model.X, model.Y))
            K = model._rbf_kernel(joint, joint)
            values, vectors = np.linalg.eigh(K)
            features = vectors * np.sqrt(np.maximum(values, 0))[None, :]
            delta = features[8:].mean(axis=0) - features[:8].mean(axis=0)
            T = features[:8].T @ features[:8] / 8
            expected = 0.5 * delta @ np.linalg.solve(T + 0.1 * np.eye(11), delta)
            self.assertAlmostEqual(score, expected, places=10)


class EvaluationTests(unittest.TestCase):
    @staticmethod
    def event(start, end, regime, is_new, alarm):
        # Model timestamps are observation counts; batch ends are exclusive.
        return dict(batch_start=start, batch_end=end, time=end, regime=regime,
                    is_new=is_new, alarm_time=alarm)

    def score(self, sequence, decisions, alarms=None, max_delay=4):
        from experiments.evaluation import evaluate_model
        labels = np.repeat(sequence, 10)
        cps = list(range(10, len(labels), 10))
        if alarms is None:
            alarms = [e['alarm_time'] for e in decisions if e['alarm_time'] is not None]
        model = SimpleNamespace(n=3, t=len(labels), recognition_log=decisions,
                                global_changepoints=alarms)
        return evaluate_model(cps, labels, model, max_delay)

    def test_causal_matching_inclusive_deadline_and_zero_delay(self):
        result = evaluate_changepoints([10, 20], [2, 10, 12, 24, 25, 29], 4,
                                      stream_length=30, warmup_end=3)
        self.assertEqual(result['matches'], [(10, 10), (20, 24)])
        self.assertEqual(result['delays'], [0, 4])
        self.assertEqual(result['false_alarms'], [12, 25, 29])
        self.assertEqual(result['ignored_warmup_alarms'], [2])
        result = evaluate_changepoints([10, 20], [9, 15, 20, 20], 4)
        self.assertEqual(result['matches'], [(20, 20)])
        self.assertEqual(result['false_alarms'], [9, 15, 20])
        self.assertEqual(result['missed_changes'], [10])

    def test_matching_cannot_cross_next_change_or_stream_end(self):
        result = evaluate_changepoints([10, 20], [20, 29], 100, stream_length=30)
        self.assertEqual(result['matches'], [(20, 20)])
        self.assertEqual(result['false_alarms'], [29])
        for delay in (None, -1, float('inf'), 2.5):
            with self.assertRaises(ValueError):
                evaluate_changepoints([10], [11], delay)
        with self.assertRaises(ValueError):
            evaluate_changepoints([10], [30], 4, stream_length=30)
        self.assertEqual(evaluate_changepoints([10], [10], 0)['delays'], [0])

    def test_user_sanity_check_and_late_discovery(self):
        sequence = [1, 2, 1, 3, 2, 1]
        ids = [1, 1, 1, 2, 3, 1]
        events = [self.event(0, 3, ids[0], True, None)]
        events += [self.event(i * 10 + 1, i * 10 + 4, ids[i], i in (3, 4), i * 10 + 1)
                   for i in range(1, len(sequence))]
        result = self.score(sequence, events)
        self.assertEqual((result['novel_correct'], result['novel_total']), (1, 2))
        self.assertEqual((result['recall_correct'], result['recall_total']), (3, 3))
        self.assertEqual(result['mapping'], {1: 1, 3: 2, 2: 3})
        self.assertFalse(result['episodes'][1]['correct'])

    def test_missing_alarms_assignments_and_duplicates_remain_failures(self):
        events = [self.event(0, 3, 0, True, None),
                  self.event(11, 14, 1, True, 11),
                  self.event(21, 24, 2, True, 21)]
        result = self.score([1, 2, 1, 3, 2], events, [11, 21, 41])
        self.assertEqual((result['novel_correct'], result['novel_total']), (1, 2))
        self.assertEqual((result['recall_correct'], result['recall_total']), (0, 2))
        self.assertEqual(result['mapping'], {1: 0, 2: 1})
        self.assertEqual(result['detected_count'], 3)

    def test_assignment_deadline_uses_decision_not_alarm_time(self):
        init = self.event(0, 3, 0, True, None)
        # Count 20 means index 19: available on the final sample of episode 2.
        events = [init, self.event(11, 20, 1, True, 11)]
        result = self.score([1, 2, 1], events, [11, 21])
        self.assertEqual(result['novel_correct'], 1)
        self.assertEqual(result['delays'], [0, 0])
        events[1]['time'] = 21  # index 20 is already the next episode.
        result = self.score([1, 2, 1], events, [11, 21])
        self.assertEqual(result['novel_correct'], 0)
        self.assertEqual(result['mapping'], {1: 0})
        self.assertEqual(result['mean_delay'], 0)

    def test_mixed_batch_is_not_an_extra_scoring_rule(self):
        events = [self.event(0, 3, 0, True, None),
                  self.event(11, 14, 1, True, 11),
                  self.event(19, 22, 0, False, 21)]
        self.assertEqual(self.score([1, 2, 1], events)['recall_correct'], 1)

    def test_extra_alarm_cannot_rescue_failed_matched_assignment(self):
        events = [self.event(0, 3, 0, True, None),
                  self.event(11, 12, 0, False, 11),
                  self.event(13, 16, 1, True, 13)]
        result = self.score([1, 2], events)
        self.assertEqual(result['novel_correct'], 0)
        self.assertEqual(result['false_alarms'], [12])
        self.assertEqual(result['mapping'], {1: 0})

    def test_ids_created_by_unmatched_alarms_are_not_fresh_later(self):
        events = [self.event(0, 3, 0, True, None),
                  self.event(5, 8, 1, True, 5),
                  self.event(11, 14, 1, False, 11),
                  self.event(21, 24, 0, False, 21),
                  self.event(31, 34, 2, True, 31)]
        result = self.score([1, 2, 1, 2], events)
        self.assertEqual((result['novel_correct'], result['recall_correct']), (0, 2))
        self.assertEqual(result['mapping'], {1: 0, 2: 2})

    def test_unavailable_assignment_does_not_replace_alarm_delay(self):
        events = [self.event(0, 3, 0, True, None), self.event(15, 19, 1, True, 15)]
        result = self.score([1, 2], events)
        self.assertEqual(result['delays'], [4])
        self.assertEqual(result['episodes'][1]['assignment_time'], 18)

    def test_aggregate_pools_counts_and_detection_weighted_delays(self):
        from experiments.evaluation import aggregate_metrics, comparison_table
        first = self.score([1, 2], [self.event(0, 3, 0, True, None)], [15])
        second = self.score([1, 2, 1], [self.event(0, 3, 0, True, None)], [11, 21])
        pooled = aggregate_metrics([first, second])
        self.assertEqual((pooled['detected_count'], pooled['change_total']), (3, 3))
        self.assertAlmostEqual(pooled['mean_delay'], 4 / 3)
        self.assertEqual(pooled['novel_total'], 2)
        empty = self.score([1], [self.event(0, 3, 0, True, None)])
        self.assertIsNone(empty['mean_delay'])
        table = comparison_table({'MMD': empty, 'Pearson': empty})
        self.assertIn('0/0 (N/A)', table)


class ExperimentTests(unittest.TestCase):
    def test_runner_pairs_observations_and_model_rng_without_passing_truth(self):
        from experiments.runner import run_comparison
        observations, draws, models = {}, {}, {}

        def prepare(name):
            observations[name], draws[name] = [], []
            model = make_model(name, 3, 2, 1, 0.5, 0.5)
            original_update = model.update

            def update(x):
                observations[name].append(float(x))
                draws[name].append(np.random.random())
                return original_update(x)

            model.update = update
            return model

        models = {name: prepare(name) for name in ('MMD', 'Pearson')}
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_comparison(
                np.arange(20), [10], np.repeat([901, 402], 10), title='paired',
                models=models, max_delay=4, seed=73, show=False)
        self.assertEqual(observations['MMD'], observations['Pearson'])
        self.assertEqual(observations['MMD'], list(range(20)))
        self.assertEqual(draws['MMD'], draws['Pearson'])
        self.assertEqual(result['metrics']['MMD']['mapping'][901], 0)
        self.assertEqual(result['config']['seed'], 73)
        for name, model in models.items():
            self.assertIs(result['models'][name], model)

    def test_main_uses_supplied_constructor_settings_and_records_them(self):
        import main_mmd_pearson
        models = {
            'MMD': OWR_CPD(n=8, m=3, eta=0.11, nu=0.82, k_conf=2,
                           gamma=0.25, factor=1.75, dynamic_gamma=False),
            'Pearson': PearsonOWRCPD(n=6, m=2, tau=0.42, nu=0.72, k_conf=1,
                                     reg_lambda=0.2, gamma=0.7, factor=2.25,
                                     dynamic_gamma=False, normalize_distance=True),
        }
        with patch.object(main_mmd_pearson, 'build_models', return_value=models):
            with patch.object(main_mmd_pearson, 'SEGMENT_LENGTH', 20):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = main_mmd_pearson.main(show=False)
        for name, model in models.items():
            self.assertIs(result['models'][name], model)
            self.assertEqual(model.t, 120)
        settings = result['config']['model_parameters']
        self.assertEqual(settings['MMD'], dict(n=8, m=3, eta=0.11, nu=0.82,
                                              k_conf=2, gamma=0.25, factor=1.75,
                                              dynamic_gamma=False))
        self.assertEqual(settings['Pearson'], dict(n=6, m=2, tau=0.42, nu=0.72,
                                                  k_conf=1, reg_lambda=0.2, gamma=0.7,
                                                  factor=2.25, dynamic_gamma=False,
                                                  normalize_distance=True))
        self.assertEqual(models['MMD'].gamma, 0.25)
        self.assertEqual(models['Pearson'].gamma, 0.7)

    def test_runner_rejects_reused_models_before_streaming_either(self):
        from experiments.runner import run_comparison
        models = {name: make_model(name, 3, 2, 1, 0.5, 0.5)
                  for name in ('MMD', 'Pearson')}
        models['Pearson'].update(0)
        with self.assertRaisesRegex(ValueError, 'fresh model'):
            run_comparison(np.arange(20), [], np.zeros(20), title='reused',
                           models=models, max_delay=4, show=False)
        self.assertEqual(models['MMD'].t, 0)

    def test_activity_entrypoints_use_shared_assignment_scores(self):
        import main_wisdm, main_pamap2
        rng = np.random.RandomState(3)
        for module in (main_wisdm, main_pamap2):
            source = ('generate_synthetic_activity_data' if module is main_wisdm
                      else 'load_pamap2_full_blocks')
            activities = {0: rng.normal(size=(22, 3)), 1: rng.normal(size=(11, 3))}
            settings = dict(N=2, M=1, K_CONF=1, MAX_DELAY=4, SEQUENCE=(0, 1, 0))
            if module is main_wisdm:
                settings['DOWNSAMPLE_RATE'] = 4
            with self.subTest(module=module.__name__), patch.multiple(module, **settings):
                with patch.object(module, source, return_value=(activities, 11)):
                    with contextlib.redirect_stdout(io.StringIO()):
                        result = module.run_experiment(show=False)
            for metrics in result['metrics'].values():
                self.assertEqual((metrics['change_total'], metrics['novel_total'],
                                  metrics['recall_total']), (2, 1, 1))
            self.assertEqual(result['config']['true_cps'],
                             [3, 6] if module is main_wisdm else [11, 22])
            if module is main_pamap2:
                self.assertEqual(result['models']['Pearson'].reg_lambda, 20)

    def test_both_mnist_sequences_use_common_pca_and_manual_thresholds(self):
        import main_mnist
        rng = np.random.RandomState(15)
        digits = sorted({d for seq in main_mnist.SEQUENCES.values() for d in seq})
        centers = {d: rng.normal(size=10) for d in digits}
        preprocessing = np.vstack([rng.normal(centers[d], 0.2, (20, 10)) for d in digits])
        evaluation = {d: rng.normal(centers[d], 0.2, (100, 10)) for d in digits}
        with patch.multiple(main_mnist, N=8, M=3, K_CONF=2, SAMPLES_PER_DIGIT=20,
                            PCA_DIM=4, PEARSON_TAU=0.4, PEARSON_NU=0.7, MAX_DELAY=8):
            with patch.object(main_mnist, 'load_mnist_partitions',
                              return_value=(preprocessing, evaluation)):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = main_mnist.main(show=False)
        self.assertEqual(set(result['sequences']), {'A', 'B'})
        for name, run in result['sequences'].items():
            for model in run['models'].values():
                self.assertEqual(model.t, 220)
                self.assertEqual(len(model.X[0]), 4)
            self.assertEqual(run['models']['Pearson'].tau, 0.4)
            self.assertEqual(run['models']['Pearson'].nu, 0.7)
            self.assertFalse(run['models']['Pearson'].normalize_distance)
            metrics = run['metrics']['Pearson']
            self.assertEqual(metrics['recall_total'], 4 if name == 'A' else 7)

    def test_impossible_pearson_threshold_is_rejected(self):
        from experiments.runner import run_comparison
        models = {'MMD': make_model('MMD', 4, 2, 2, 0.1, 0.9),
                  'Pearson': make_model('Pearson', 4, 2, 2, 0.46, 0.71, reg_lambda=10)}
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'score bound'):
            run_comparison(np.arange(20), [], np.zeros(20), title='invalid',
                           models=models, max_delay=4, show=False)

    def test_paper_lengths_and_windows(self):
        import main_T1, main_T2, main_T3, main_mmd_pearson
        for module, windows in ((main_T1, (150, 40)), (main_T2, (300, 100)),
                                (main_T3, (600, 200))):
            self.assertEqual((module.N, module.M), windows)
            self.assertEqual(module.SEGMENT_LENGTH, 700)
            self.assertEqual(len(set(module.SEEDS)), 3)
            data, cps, sequence = generate_moment_matched_stream(module.__name__[-2:])
            self.assertEqual(len(data), 8400)
            self.assertEqual(cps, list(range(700, 8400, 700)))
            self.assertEqual(len(sequence) - len(set(sequence)), 8)
        self.assertEqual((main_mmd_pearson.SEGMENT_LENGTH, main_mmd_pearson.M,
                          main_mmd_pearson.MMD_ETA), (300, 40, 0.12))

    def test_missing_mnist_threshold_fails_before_downloading(self):
        import main_mnist
        with patch.object(main_mnist, 'PEARSON_TAU', None):
            with patch.object(main_mnist, 'fetch_openml') as download:
                with self.assertRaisesRegex(ValueError, 'Set PEARSON_TAU'):
                    main_mnist.main(show=False)
        download.assert_not_called()

    def test_tiers_use_manual_thresholds_without_replacing_them(self):
        import main_T1, main_T2, main_T3
        for module in (main_T1, main_T2, main_T3):
            with self.subTest(tier=module.__name__), patch.multiple(
                    module, N=8, M=3, K_CONF=2, SEGMENT_LENGTH=20,
                    SEEDS=(42,), MAX_DELAY=8, MMD_ETA=0.12, MMD_NU=0.9,
                    PEARSON_TAU=0.4, PEARSON_NU=0.7):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = module.main(show=False)
                run = result['runs'][0]
                self.assertEqual(run['models']['MMD'].eta, 0.12)
                self.assertEqual(run['models']['MMD'].nu, 0.9)
                self.assertEqual(run['models']['Pearson'].tau, 0.4)
                self.assertEqual(run['models']['Pearson'].nu, 0.7)
                self.assertEqual(run['models']['Pearson'].t, 240)

    def test_mnist_partitions_and_repeated_digits_never_reuse_images(self):
        X = np.arange(120, dtype=float)[:, None]
        y = np.repeat([0, 1], 60)
        with patch('main_mnist.fetch_openml', return_value=(X, y)):
            preprocessing, evaluation = load_mnist_partitions(5, 42)
        groups = [set(preprocessing.ravel()),
                  set(np.concatenate(list(evaluation.values())).ravel())]
        for i in range(len(groups)):
            for j in range(i):
                self.assertFalse(groups[i] & groups[j])
        offsets = {}
        first, cps = build_mnist_stream([0, 1, 0], evaluation, 5, offsets)
        second, _ = build_mnist_stream([1, 0, 1], evaluation, 5, offsets)
        self.assertEqual(cps, [5, 10])
        self.assertEqual(len(set(np.r_[first.ravel(), second.ravel()])), 30)
        with self.assertRaises(ValueError):
            build_mnist_stream([0] * 20, evaluation, 5, offsets)


if __name__ == '__main__':
    unittest.main()
