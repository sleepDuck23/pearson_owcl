"""Load official putEMG recordings and assemble labelled gesture episodes."""

from collections import Counter, deque
from http.client import HTTPException
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen
from zipfile import BadZipFile

import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / 'recordings'
DEFAULT_RECORDING = 'emg_gestures-03-repeats_long-2018-05-11-11-05-00-695'
DOWNLOAD_URL = 'https://chmura.put.poznan.pl/s/45NY5snj0U4tgQz/download'
RECORDING_PATTERN = re.compile(
    r'emg_gestures-(?P<subject>\d{2})-(?P<trajectory>repeats_long|sequential|repeats_short)'
    r'-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{3}')
RAW_SAMPLE_RATE_HZ = 5120
LABEL_COLUMN = 'TRAJ_GT'
TIMESTAMP_COLUMNS = ('Timestamp', 'Unnamed: 0')
EMG_COLUMNS = [f'EMG_{channel}' for channel in range(1, 25)]
FEATURE_NAMES = tuple(f'log_rms_{channel}' for channel in EMG_COLUMNS)
GESTURE_NAMES = {
    0: 'Idle', 1: 'Fist', 2: 'Flexion', 3: 'Extension',
    6: 'Index pinch', 7: 'Middle pinch', 8: 'Ring pinch', 9: 'Little-finger pinch',
}


def _read_recording(path):
    columns = set(EMG_COLUMNS) | {LABEL_COLUMN, *TIMESTAMP_COLUMNS}
    dtypes = dict.fromkeys(EMG_COLUMNS, 'float32')
    dtypes.update(dict.fromkeys((LABEL_COLUMN, *TIMESTAMP_COLUMNS), 'float64'))
    try:
        frame = pd.read_csv(path, usecols=lambda column: column in columns, dtype=dtypes)
    except (OSError, ValueError, BadZipFile) as exc:
        raise ValueError(f'Cannot read putEMG recording {path}') from exc
    missing = (set(EMG_COLUMNS) | {LABEL_COLUMN}) - set(frame)
    timestamp_column = next((name for name in TIMESTAMP_COLUMNS if name in frame), None)
    if timestamp_column is None:
        missing.add('Timestamp')
    if missing or frame.empty:
        detail = f'missing columns {sorted(missing)}' if missing else 'no samples'
        raise ValueError(f'Invalid putEMG recording {path} ({detail})')
    return frame, timestamp_column


def _load_recording(recording, data_dir):
    csv_path = data_dir / f'{recording}.csv'
    archive_path = data_dir / f'{recording}.zip'
    for path in (csv_path, archive_path):
        if path.is_file():
            frame, timestamp_column = _read_recording(path)
            return frame, timestamp_column, path

    query = urlencode({'path': '/Data-CSV', 'files': archive_path.name})
    url = f'{DOWNLOAD_URL}?{query}'
    data_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=data_dir) as temporary_dir:
        staged = Path(temporary_dir) / archive_path.name
        print(f'Downloading putEMG recording {archive_path.name}...')
        try:
            with urlopen(url, timeout=60) as response, staged.open('wb') as destination:
                shutil.copyfileobj(response, destination)
        except (URLError, TimeoutError, HTTPException) as exc:
            raise RuntimeError(f'Could not download putEMG. Download {url} to {archive_path}.') from exc
        try:
            frame, timestamp_column = _read_recording(staged)
        except ValueError as exc:
            raise ValueError(f'The download at {url} did not contain a valid putEMG recording. '
                             f'Place the official ZIP or CSV in {data_dir}.') from exc
        staged.replace(archive_path)
    return frame, timestamp_column, archive_path


def load_putemg_stream(sequence, *, segment_length, window_samples,
                        recording=DEFAULT_RECORDING, data_dir=DEFAULT_DATA_DIR):
    """Build a feature stream using one centered segment per labelled occurrence."""
    sequence = tuple(sequence)
    if not sequence or any(gesture not in GESTURE_NAMES for gesture in sequence):
        raise ValueError(f'Use a nonempty sequence of putEMG gesture IDs {tuple(GESTURE_NAMES)}')
    for name, value, minimum in (('segment_length', segment_length, 1),
                                 ('window_samples', window_samples, 2)):
        if (not isinstance(value, (int, np.integer))
                or isinstance(value, (bool, np.bool_)) or value < minimum):
            raise ValueError(f'{name} must be an integer of at least {minimum}')
    match = RECORDING_PATTERN.fullmatch(recording)
    if match is None:
        raise ValueError('Use an official putEMG gesture recording filename without its extension')

    frame, timestamp_column, path = _load_recording(recording, Path(data_dir))
    labels = frame[LABEL_COLUMN].to_numpy()
    timestamps = frame[timestamp_column].to_numpy(dtype=float)
    boundaries = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    counts = Counter(sequence)
    pools = {gesture: deque() for gesture in counts}
    raw_length = int(segment_length * window_samples)
    channel_indices = [frame.columns.get_loc(channel) for channel in EMG_COLUMNS]
    sample_period = 1 / RAW_SAMPLE_RATE_HZ
    for start, stop in zip(boundaries[:-1].tolist(), boundaries[1:].tolist()):
        gesture = labels[start]
        if gesture not in pools or len(pools[gesture]) == counts[gesture] or stop - start < raw_length:
            continue
        raw_start = start + (stop - start - raw_length) // 2
        raw_stop = raw_start + raw_length
        time = timestamps[raw_start:raw_stop]
        if not np.isfinite(time).all() or not np.allclose(
                np.diff(time), sample_period, rtol=0, atol=sample_period / 2):
            continue
        raw = frame.iloc[raw_start:raw_stop, channel_indices].to_numpy(dtype=float)
        if not np.isfinite(raw).all():
            continue
        windows = raw.reshape(segment_length, window_samples, len(EMG_COLUMNS))
        # Center each channel within its window to remove the local DC offset.
        rms = windows.std(axis=1)
        features = np.log(np.maximum(rms, 1e-12))
        source = dict(regime=int(gesture), gesture=GESTURE_NAMES[gesture],
                      raw_start=raw_start, raw_stop=raw_stop,
                      gesture_start=start, gesture_stop=stop,
                      start_time_s=float(time[0]), last_sample_time_s=float(time[-1]))
        pools[gesture].append((features, source))
        if all(len(pools[gesture]) == count for gesture, count in counts.items()):
            break

    for gesture, count in counts.items():
        if len(pools[gesture]) < count:
            raise ValueError(f'{path.name} has {len(pools[gesture])} valid occurrences of '
                             f'{GESTURE_NAMES[gesture]} with {raw_length} consecutive samples. '
                             f'The sequence needs {count}. Reduce SEGMENT_LENGTH or '
                             'WINDOW_SAMPLES, or select a different recording.')
    episodes, sources = zip(*(pools[gesture].popleft() for gesture in sequence))
    metadata = dict(file=str(path), recording=recording, subject=match['subject'],
                    trajectory=match['trajectory'], label_column=LABEL_COLUMN,
                    episode_sources=list(sources))
    return np.vstack(episodes), metadata
