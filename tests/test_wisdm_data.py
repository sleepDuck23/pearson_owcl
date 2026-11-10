"""Verify real-record parsing, participant isolation, and non-replayed blocks."""

import io
import tarfile
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import numpy as np

from data.manage_wisdm import RAW_FILENAME, download_wisdm, load_wisdm_blocks


def row(person, activity, sample, value, *, trailing_comma=False):
    ending = ',;' if trailing_comma else ';'
    return f'{person},{activity},{1_000_000_000 + sample * 50_000_000},{value},2,3{ending}'


class WisdmDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        (self.root / 'readme.txt').write_text('Fixture documentation')

    def tearDown(self):
        self.directory.cleanup()

    def load(self, records, *, count=1, size=3, mapping=None):
        (self.root / RAW_FILENAME).write_text('\n'.join(records))
        return load_wisdm_blocks(
            subject_id=1, block_size=size, blocks_per_activity={0: count},
            activity_mapping={'Walking': 0} if mapping is None else mapping,
            data_dir=self.root, return_metadata=True)

    def test_only_requested_person_and_fresh_source_rows_are_used(self):
        records = [row(2, 'Walking', i, -100 - i) for i in range(8)]
        records += [row(1, 'Walking', i, i) for i in range(8)]
        with patch('urllib.request.urlopen', side_effect=AssertionError('Unexpected download')):
            activities, size, metadata = self.load(records, count=2)
        self.assertEqual(size, 3)
        np.testing.assert_array_equal(activities[0][:, 0], np.arange(6))
        sources = metadata['blocks'][0]
        self.assertEqual([block['source_record_first'] for block in sources], [8, 11])
        self.assertEqual([block['source_record_last'] for block in sources], [10, 13])
        self.assertTrue(all(block['subject_id'] == 1 for block in sources))
        self.assertEqual(len(metadata['raw_sha256']), 64)

    def test_delimiters_and_equal_timestamps_preserve_real_values(self):
        records = [row(1, 'Walking', 0, 10, trailing_comma=True)
                   + row(1, 'Walking', 0, 20), row(1, 'Walking', 1, 30)]
        activities, _, _ = self.load(records)
        np.testing.assert_array_equal(activities[0][:, 0], [10, 20, 30])

    def test_invalid_rows_break_runs_instead_of_stitching_them(self):
        for invalid in ('malformed', '1,Walking,0,1,2,3;',
                        '1,Walking,1100000000,nan,2,3;'):
            with self.subTest(invalid=invalid):
                records = [row(1, 'Walking', i, i) for i in range(2)]
                records += [invalid]
                records += [row(1, 'Walking', i, i) for i in range(3, 5)]
                with self.assertRaisesRegex(ValueError, 'longest run: 2'):
                    self.load(records)

    def test_participant_activity_gaps_and_backward_time_break_runs(self):
        prefixes = [row(1, 'Walking', i, i) for i in range(2)]
        for interruption in (row(2, 'Walking', 2, 99), row(1, 'Jogging', 2, 99)):
            with self.subTest(interruption=interruption), self.assertRaisesRegex(ValueError, 'found 0'):
                self.load(prefixes + [interruption] + [row(1, 'Walking', i, i) for i in (3, 4)])
        for samples in ((0, 1, 30, 31), (10, 11, 0, 1)):
            with self.subTest(samples=samples), self.assertRaisesRegex(ValueError, 'found 0'):
                self.load([row(1, 'Walking', sample, i) for i, sample in enumerate(samples)])

    def test_missing_activity_raises_without_switching_people_or_filling(self):
        records = [row(2, 'Standing', i, i) for i in range(20)]
        with self.assertRaisesRegex(ValueError, 'subject 1, Standing.*found 0'):
            self.load(records, mapping={'Standing': 0})

    def test_download_keeps_official_docs_and_does_not_extract_archive_paths(self):
        payload = io.BytesIO()
        contents = {RAW_FILENAME: row(1, 'Walking', 0, 1).encode(),
                    'readme.txt': b'Dataset notice',
                    'WISDM_ar_v1.1_raw_about.txt': b'Source description',
                    'unused.txt': b'Do not extract'}
        with tarfile.open(fileobj=payload, mode='w:gz') as archive:
            for name, data in contents.items():
                member = tarfile.TarInfo('../../' + name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        with patch('urllib.request.urlopen', return_value=io.BytesIO(payload.getvalue())) as request:
            path = download_wisdm(self.root)
        self.assertEqual(request.call_count, 1)
        self.assertEqual(path.read_bytes(), contents[RAW_FILENAME])
        self.assertEqual((self.root / 'readme.txt').read_bytes(), contents['readme.txt'])
        self.assertFalse((self.root / 'unused.txt').exists())
        with patch('urllib.request.urlopen', side_effect=AssertionError('Unexpected download')):
            self.assertEqual(download_wisdm(self.root), path)

    def test_download_failure_does_not_create_data_or_a_synthetic_fallback(self):
        with patch('urllib.request.urlopen', side_effect=urllib.error.URLError('Offline')):
            with self.assertRaises(urllib.error.URLError):
                download_wisdm(self.root)
        self.assertFalse((self.root / RAW_FILENAME).exists())
        self.assertEqual(list(self.root.glob('*.part')), [])


if __name__ == '__main__':
    unittest.main()
