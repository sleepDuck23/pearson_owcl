"""Load real WISDM Activity Recognition v1.1 smartphone accelerometer records."""

import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np


DATASET_URL = ('https://www.cis.fordham.edu/wisdm/includes/datasets/latest/'
               'WISDM_ar_latest.tar.gz')
RAW_FILENAME = 'WISDM_ar_v1.1_raw.txt'
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / 'wisdm'
ACTIVITY_NAMES = ('Walking', 'Jogging', 'Upstairs', 'Downstairs', 'Sitting', 'Standing')


def download_wisdm(data_dir=DEFAULT_DATA_DIR):
    """Cache the official archive and copy only its raw data and documentation.

    No archive paths are extracted, and no synthetic fallback is provided.
    Keeping readme.txt alongside the records preserves the dataset's notice.
    """
    data_dir = Path(data_dir)
    raw_path = data_dir / RAW_FILENAME
    readme_path = data_dir / 'readme.txt'
    if raw_path.is_file() and raw_path.stat().st_size and readme_path.is_file():
        return raw_path
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / 'WISDM_ar_v1.1.tar.gz'
    if not archive_path.is_file():
        partial = archive_path.with_suffix('.gz.part')
        print(f'Downloading official WISDM Activity Recognition data from {DATASET_URL}')
        request = urllib.request.Request(DATASET_URL, headers={'User-Agent': 'pearson_x2/1.0'})
        try:
            with urllib.request.urlopen(request, timeout=60) as response, partial.open('wb') as target:
                shutil.copyfileobj(response, target)
            # Reject an error page before committing it as the cached archive.
            with tarfile.open(partial, 'r:gz'):
                pass
            partial.replace(archive_path)
        finally:
            partial.unlink(missing_ok=True)

    required = (RAW_FILENAME, 'readme.txt', 'WISDM_ar_v1.1_raw_about.txt')
    with tarfile.open(archive_path, 'r:gz') as archive:
        members = archive.getmembers()
        for filename in required:
            matching = [member for member in members
                        if member.isfile() and Path(member.name).name == filename]
            if len(matching) != 1:
                raise ValueError(f'The WISDM archive must contain exactly one {filename}')
            destination = data_dir / filename
            temporary = destination.with_suffix(destination.suffix + '.part')
            try:
                with archive.extractfile(matching[0]) as source, temporary.open('wb') as target:
                    shutil.copyfileobj(source, target)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
    return raw_path


def _subject_runs(raw_path, subject_id, selected_activities, max_gap_ns):
    """Yield uninterrupted, valid runs in source order for the selected person.

    Equal timestamps are retained: Android can assign the same timestamp to
    different measured rows. Backward timestamps, long gaps, invalid rows, and
    subject/activity changes break runs. No sorting, filling or interpolation.
    """
    rows, stamps, records = [], [], []
    activity, previous_time = None, None
    record_index = -1
    with Path(raw_path).open(encoding='utf-8') as source:
        for line in source:
            for text in line.split(';'):
                text = text.strip()
                if not text:
                    continue
                record_index += 1
                # Some genuine source records have an empty trailing CSV field.
                fields = [field.strip() for field in text.rstrip(',').split(',')]
                try:
                    if len(fields) != 6:
                        raise ValueError()
                    person, label, timestamp = int(fields[0]), fields[1], int(fields[2])
                    values = [float(value) for value in fields[3:]]
                    valid = (person == subject_id and label in selected_activities
                             and timestamp > 0 and np.isfinite(values).all())
                except ValueError:
                    valid = False

                continuous = (valid and activity == label and previous_time is not None
                              and 0 <= timestamp - previous_time <= max_gap_ns)
                if not continuous and rows:
                    yield activity, np.asarray(rows), np.asarray(stamps, dtype=np.int64), records
                    rows, stamps, records = [], [], []
                if valid:
                    activity, previous_time = label, timestamp
                    rows.append(values)
                    stamps.append(timestamp)
                    records.append(record_index)
                else:
                    activity, previous_time = None, None
    if rows:
        yield activity, np.asarray(rows), np.asarray(stamps, dtype=np.int64), records


def load_wisdm_blocks(*, subject_id, block_size, blocks_per_activity, activity_mapping,
                      data_dir=DEFAULT_DATA_DIR, max_gap_seconds=0.5, return_metadata=False):
    """Take disjoint measured blocks, never mixing people or replaying samples.

    activity_mapping maps original WISDM activity names to truth regime IDs.
    Insufficient data raises an error rather than changing the person, block
    length, requested activities, or measurements.
    """
    if (not isinstance(subject_id, (int, np.integer)) or isinstance(subject_id, (bool, np.bool_))
            or not 1 <= subject_id <= 36):
        raise ValueError('WISDM subject_id must be an integer from 1 to 36')
    if (not isinstance(block_size, (int, np.integer)) or isinstance(block_size, (bool, np.bool_))
            or block_size < 1):
        raise ValueError('block_size must be a positive integer')
    if not np.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
        raise ValueError('max_gap_seconds must be finite and positive')
    mapping, counts = dict(activity_mapping), dict(blocks_per_activity)
    if (not mapping or not set(mapping).issubset(ACTIVITY_NAMES)
            or len(set(mapping.values())) != len(mapping)):
        raise ValueError('Map valid WISDM activity names to distinct truth regime IDs')
    if not counts:
        raise ValueError('Request at least one activity block')
    for regime, count in counts.items():
        if (regime not in mapping.values() or not isinstance(count, (int, np.integer))
                or isinstance(count, (bool, np.bool_)) or count < 1):
            raise ValueError('Requested regime IDs and block counts must be valid')

    raw_path = download_wisdm(data_dir)
    selected = {name for name, regime in mapping.items() if regime in counts}
    chunks = {regime: [] for regime in counts}
    provenance = {regime: [] for regime in counts}
    available, largest_run = dict.fromkeys(counts, 0), dict.fromkeys(counts, 0)
    for activity, values, timestamps, records in _subject_runs(
            raw_path, subject_id, selected, int(max_gap_seconds * 1e9)):
        regime = mapping[activity]
        available[regime] += len(values)
        largest_run[regime] = max(largest_run[regime], len(values))
        for start in range(0, len(values) - block_size + 1, block_size):
            if len(chunks[regime]) == counts[regime]:
                break
            end = start + block_size
            chunks[regime].append(values[start:end])
            provenance[regime].append(dict(
                activity=activity, regime=int(regime), subject_id=int(subject_id),
                source_record_first=records[start], source_record_last=records[end - 1],
                timestamp_first_ns=int(timestamps[start]), timestamp_last_ns=int(timestamps[end - 1]),
                samples=int(block_size)))

    names = {regime: activity for activity, regime in mapping.items()}
    for regime, blocks in chunks.items():
        if len(blocks) != counts[regime]:
            raise ValueError(
                f'WISDM subject {subject_id}, {names[regime]} (regime {regime}): '
                f'need {counts[regime]} distinct contiguous blocks of {block_size} samples; '
                f'found {len(blocks)}. Valid samples: {available[regime]}; longest run: '
                f'{largest_run[regime]}. Change SUBJECT_ID, SEQUENCE or SEGMENT_LENGTH '
                'in main_wisdm.py; samples are never replayed or generated.')
    result = {regime: np.vstack(blocks) for regime, blocks in chunks.items()}, int(block_size)
    if not return_metadata:
        return result
    digest = hashlib.sha256()
    with raw_path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    metadata = dict(dataset='WISDM Activity Recognition v1.1', source_url=DATASET_URL,
                    raw_file=str(raw_path), raw_sha256=digest.hexdigest(),
                    subject_id=int(subject_id), nominal_sample_rate_hz=20,
                    channels=['x_acceleration', 'y_acceleration', 'z_acceleration'],
                    max_gap_seconds=float(max_gap_seconds),
                    activity_mapping=mapping, blocks=provenance)
    return (*result, metadata)
