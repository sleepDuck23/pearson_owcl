import contextlib
import copy
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.calibration import calibrate_thresholds, make_model
from experiments.evaluation import evaluate_changepoints, evaluate_recognition
from experiments.moment_matched import generate_moment_matched_stream
from model.owcl_mmd import OWR_CPD
from model.owcl_pearson import PearsonOWRCPD
from main_mnist import build_mnist_stream, load_mnist_partitions


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
    def test_deadline_matching_duplicates_and_paper_delay_convention(self):
        result = evaluate_changepoints([10, 20], [2, 10, 12, 27], max_delay=4)
        self.assertEqual(result['matches'], [(10, 10)])
        self.assertEqual(result['delays'], [1])
        self.assertEqual(result['false_alarms'], [2, 12, 27])
        self.assertEqual(result['missed_changes'], [20])

    @staticmethod
    def event(start, end, regime, is_new, alarm):
        return dict(batch_start=start, batch_end=end, time=end, regime=regime,
                    is_new=is_new, alarm_time=alarm)

    def test_recall_requires_original_id_and_homogeneous_timely_batch(self):
        labels = np.repeat([1, 2, 1], 10)
        events = [self.event(0, 3, 0, True, None),
                  self.event(11, 14, 1, True, 11),
                  self.event(21, 24, 0, False, 21)]
        model = SimpleNamespace(n=3, recognition_log=events)
        matched = evaluate_changepoints([10, 20], [10, 20], max_delay=4)
        result = evaluate_recognition(labels, model, matched)
        self.assertEqual((result['recall_correct'], result['recall_total']), (1, 1))
        self.assertEqual((result['novel_correct'], result['novel_total']), (2, 2))
        events[-1]['is_new'], events[-1]['regime'] = True, 2
        self.assertEqual(evaluate_recognition(labels, model, matched)['recall_correct'], 0)
        events[-1] = self.event(19, 22, 0, False, 21)
        result = evaluate_recognition(labels, model, matched)
        self.assertEqual(result['contaminated_batches'], 1)
        self.assertEqual(result['recall_correct'], 0)
        model.recognition_log = events[:2]
        self.assertEqual(evaluate_recognition(labels, model, matched)['recall_total'], 1)


class ExperimentTests(unittest.TestCase):
    def test_both_mnist_sequences_run_through_common_pca_and_calibration(self):
        import main_mnist
        rng = np.random.RandomState(15)
        digits = sorted({d for seq in main_mnist.SEQUENCES.values() for d in seq})
        centers = {d: rng.normal(size=10) for d in digits}
        preprocessing = np.vstack([rng.normal(centers[d], 0.2, (20, 10)) for d in digits])
        calibration = {d: rng.normal(centers[d], 0.2, (100, 10)) for d in digits}
        evaluation = {d: rng.normal(centers[d], 0.2, (100, 10)) for d in digits}
        with patch.multiple(main_mnist, N=8, M=3, K_CONF=2, SAMPLES_PER_DIGIT=20,
                            PCA_DIM=4, CALIBRATION_REPEATS=2, MAX_DELAY=8):
            with patch.object(main_mnist, 'load_mnist_partitions',
                              return_value=(preprocessing, calibration, evaluation)):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = main_mnist.main(show=False)
        self.assertEqual(set(result['sequences']), {'A', 'B'})
        for name, run in result['sequences'].items():
            for model in run['models'].values():
                self.assertEqual(model.t, 220)
                self.assertEqual(len(model.X[0]), 4)
            self.assertFalse(run['models']['Pearson'].normalize_distance)
            metrics = run['metrics']['Pearson']
            self.assertEqual(metrics['recall_total'], 4 if name == 'A' else 7)

    def test_impossible_pearson_threshold_is_rejected(self):
        from experiments.runner import run_comparison
        thresholds = {'MMD': dict(detection=0.1, novelty=0.9),
                      'Pearson': dict(detection=0.46, novelty=0.71)}
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'score bound'):
            run_comparison(np.arange(20), [], np.zeros(20), title='invalid',
                           n=4, m=2, k_conf=2, thresholds=thresholds,
                           reg_lambda=10, max_delay=4, show=False)

    def test_paper_lengths_windows_and_disjoint_seeds(self):
        import main_T1, main_T2, main_T3, main_mmd_pearson
        for module, windows in ((main_T1, (150, 40)), (main_T2, (300, 100)),
                                (main_T3, (600, 200))):
            self.assertEqual((module.N, module.M), windows)
            self.assertEqual(module.SEGMENT_LENGTH, 700)
            self.assertEqual(len(set(module.SEEDS)), 3)
            self.assertNotIn(module.CALIBRATION_SEED, module.SEEDS)
            data, cps, sequence = generate_moment_matched_stream(module.__name__[-2:])
            self.assertEqual(len(data), 8400)
            self.assertEqual(cps, list(range(700, 8400, 700)))
            self.assertEqual(len(sequence) - len(set(sequence)), 8)
        self.assertIsNone(main_T3.MMD_NU)
        self.assertEqual((main_mmd_pearson.SEGMENT_LENGTH, main_mmd_pearson.M,
                          main_mmd_pearson.MMD_ETA), (300, 40, 0.12))

    def test_calibration_is_deterministic_without_touching_global_rng(self):
        def sample(label, size, rng):
            return rng.normal(label * 5, 1, (size, 2))
        np.random.seed(17)
        state = np.random.get_state()
        parameters = dict(n=8, m=3, k_conf=2, repeats=3, seed=123)
        first = calibrate_thresholds(sample, (0, 1), **parameters)
        self.assertEqual(first, calibrate_thresholds(sample, (0, 1), **parameters))
        np.testing.assert_array_equal(np.random.get_state()[1], state[1])
        self.assertLess(first['MMD']['novelty'], 1.0)
        self.assertLess(first['Pearson']['detection'], 10.0)

    def test_mnist_partitions_and_repeated_digits_never_reuse_images(self):
        X = np.arange(120, dtype=float)[:, None]
        y = np.repeat([0, 1], 60)
        with patch('main_mnist.fetch_openml', return_value=(X, y)):
            preprocessing, calibration, evaluation = load_mnist_partitions(5, 10, 42)
        groups = [set(preprocessing.ravel()),
                  set(np.concatenate(list(calibration.values())).ravel()),
                  set(np.concatenate(list(evaluation.values())).ravel())]
        for i in range(3):
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
