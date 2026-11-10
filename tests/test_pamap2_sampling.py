"""Check timing, causality and anti-aliasing of PAMAP2 frequency reduction."""

import unittest
import contextlib
import io
from unittest.mock import patch

import numpy as np

from main_pamap2 import downsample_stream
import main_pamap2 as pamap2


class Pamap2SamplingTests(unittest.TestCase):
    def test_original_rate_keeps_observations_unchanged(self):
        data = np.random.RandomState(7).normal(size=(101, 3))
        np.testing.assert_array_equal(downsample_stream(data, 1), data)

    def test_future_observations_do_not_change_past_output(self):
        data = np.random.RandomState(7).normal(size=(401, 3))
        for factor in (2, 4, 5):
            prefix = downsample_stream(data[:203], factor)
            complete = downsample_stream(data, factor)
            np.testing.assert_allclose(prefix, complete[:len(prefix)], rtol=0, atol=1e-14)
            self.assertEqual(len(complete), len(data[::factor]))

    def test_filter_preserves_slow_signal_and_suppresses_aliasing(self):
        time = np.arange(4000) / 100
        data = np.column_stack((np.sin(2 * np.pi * 2 * time),
                                np.sin(2 * np.pi * 40 * time)))
        result = downsample_stream(data, 2)[100:]
        rms = np.sqrt(np.mean(result ** 2, axis=0))
        self.assertGreater(rms[0], 0.65)
        self.assertLess(rms[1], 0.02)
        # Simple row skipping would preserve the unwanted 40 Hz component
        # as a 10 Hz alias at the new 50 Hz rate.
        self.assertGreater(np.sqrt(np.mean(data[::2, 1] ** 2)), 0.65)

    def test_invalid_downsampling_factors_fail_clearly(self):
        for factor in (0, -1, 2.5, True):
            with self.subTest(factor=factor), self.assertRaisesRegex(ValueError, 'positive integer'):
                downsample_stream(np.ones((20, 3)), factor)

    def run_fixture(self, factor, n=3):
        def load_blocks(*, block_size, blocks_per_activity, activity_mapping):
            activities = {act: np.arange(count * block_size * 3, dtype=float).reshape(-1, 3)
                          for act, count in blocks_per_activity.items()}
            return activities, block_size

        with patch.multiple(pamap2, N=n, M=2, USE_PCA=False, SEQUENCE=(0, 1, 0),
                            SEGMENT_LENGTH=11, DOWNSAMPLE_RATE=factor, RAW_SAMPLE_RATE_HZ=100), \
                patch.object(pamap2, 'load_pamap2_full_blocks', side_effect=load_blocks), \
                patch.object(pamap2, 'run_comparison', return_value={'config': {}}) as runner, \
                contextlib.redirect_stdout(io.StringIO()):
            result = pamap2.run_experiment(show=False)
        return result, runner.call_args

    def test_stream_labels_and_boundaries_follow_the_reduced_rate(self):
        for factor in (1, 4):
            with self.subTest(factor=factor):
                result, call = self.run_fixture(factor)
                data, changes, labels = call.args
                expected_labels = np.repeat([0, 1, 0], 11)[::factor]
                np.testing.assert_array_equal(labels, expected_labels)
                self.assertEqual(len(data), len(labels))
                self.assertEqual(changes, [int(np.ceil(11 / factor)), int(np.ceil(22 / factor))])
                self.assertEqual(result['config']['sample_rate_hz'], 100 / factor)
                self.assertEqual(result['config']['raw_segment_length'], 11)

    def test_reference_window_must_fit_after_downsampling(self):
        with self.assertRaisesRegex(ValueError, '3 downsampled observations'):
            self.run_fixture(factor=4, n=4)


if __name__ == '__main__':
    unittest.main()
