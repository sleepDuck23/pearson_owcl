"""Select real TAU 2020 Mobile clips and extract causal acoustic features."""

from collections import Counter, defaultdict, deque
import csv
from http.client import HTTPException
from io import BufferedReader, TextIOWrapper
from itertools import zip_longest
from pathlib import Path, PurePosixPath
import shutil
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

import numpy as np
from scipy.io import wavfile

from data.remote_zip import HTTPRangeReader


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / 'recordings'
DATASET_NAME = 'TAU-urban-acoustic-scenes-2020-mobile-development'
RECORD_URL = 'https://zenodo.org/records/3819968/files'
META_ARCHIVE = f'{DATASET_NAME}.meta.zip'
AUDIO_ARCHIVES = tuple(f'{DATASET_NAME}.audio.{i}.zip' for i in range(1, 17))
RAW_SAMPLE_RATE_HZ = 44_100
MEL_BANDS = 24


def _read_metadata(path):
    def read(file):
        reader = csv.DictReader(file, delimiter='\t')
        columns = {'filename', 'scene_label', 'identifier', 'source_label'}
        if not columns.issubset(reader.fieldnames or ()):
            raise ValueError(f'{path} is missing the TAU metadata columns')
        return list(reader)

    if path.suffix == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as file:
            return read(file)
    with ZipFile(path) as archive:
        with archive.open(f'{DATASET_NAME}/meta.csv') as file:
            return read(TextIOWrapper(file, encoding='utf-8-sig', newline=''))


def _load_metadata(data_dir, root):
    csv_path = root / 'meta.csv'
    if csv_path.is_file():
        return _read_metadata(csv_path)
    archive = data_dir / META_ARCHIVE
    if archive.is_file():
        return _read_metadata(archive)

    data_dir.mkdir(parents=True, exist_ok=True)
    print('Downloading TAU 2020 Mobile metadata...')
    with TemporaryDirectory(dir=data_dir) as temporary_dir:
        staged = Path(temporary_dir) / META_ARCHIVE
        with urlopen(f'{RECORD_URL}/{META_ARCHIVE}', timeout=60) as response, staged.open('wb') as file:
            shutil.copyfileobj(response, file)
        records = _read_metadata(staged)
        staged.replace(archive)
    return records


def _cache_file(path, contents):
    """Publish a complete cached file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temporary_dir:
        staged = Path(temporary_dir) / path.name
        staged.write_bytes(contents)
        staged.replace(path)


def _select_clips(sequence, records, city, device):
    counts = Counter(sequence)
    by_scene = {scene: defaultdict(list) for scene in counts}
    seen_files = set()
    for row in records:
        scene, location = row['scene_label'], row['identifier']
        if (scene not in by_scene or row['source_label'] != device
                or location.rpartition('-')[0] != city):
            continue
        path = PurePosixPath(row['filename'])
        if (len(path.parts) != 2 or path.parts[0] != 'audio' or path.suffix != '.wav'
                or '\\' in path.name):
            raise ValueError(f'Invalid TAU audio path in metadata: {row["filename"]}')
        if path in seen_files:
            raise ValueError(f'Duplicate TAU audio path in metadata: {path}')
        seen_files.add(path)
        by_scene[scene][location].append(row)

    clips = {}
    for scene, count in counts.items():
        locations = by_scene[scene].values()
        candidates = [row for group in zip_longest(*locations) for row in group if row is not None]
        if len(candidates) < count:
            raise ValueError(f'TAU {city}, device {device}, supplies {len(candidates)} clips for '
                             f'{scene}, but the sequence needs {count}.')
        clips[scene] = deque(candidates[:count])
    return [clips[scene].popleft() for scene in sequence]


def _cache_audio(filenames, data_dir, root):
    missing = {f'{DATASET_NAME}/{name}': root / name for name in filenames if not (root / name).is_file()}
    if not missing:
        return

    local_archives = [data_dir / name for name in AUDIO_ARCHIVES if (data_dir / name).is_file()]

    def save_members(archive):
        for name in archive.namelist():
            if name in missing:
                _cache_file(missing[name], archive.read(name))
                del missing[name]

    for path in local_archives:
        with ZipFile(path) as archive:
            save_members(archive)
        if not missing:
            return
    for name in AUDIO_ARCHIVES:
        if data_dir / name in local_archives:
            continue
        print(f'Checking TAU audio archive {name}...')
        with BufferedReader(HTTPRangeReader(f'{RECORD_URL}/{name}'), buffer_size=512 * 1024) as remote:
            with ZipFile(remote) as archive:
                save_members(archive)
        if not missing:
            return
    raise ValueError(f'Could not find {len(missing)} selected recordings in the TAU audio archives')


def _mel_filters(window_samples):
    frequencies = np.fft.rfftfreq(window_samples, d=1 / RAW_SAMPLE_RATE_HZ)
    mel_frequencies = np.log1p(frequencies / 700)
    edges = np.linspace(np.log1p(50 / 700), np.log1p(16_000 / 700), MEL_BANDS + 2)
    left = (mel_frequencies - edges[:-2, None]) / np.diff(edges)[:-1, None]
    right = (edges[2:, None] - mel_frequencies) / np.diff(edges)[1:, None]
    filters = np.maximum(0, np.minimum(left, right))
    weights = filters.sum(axis=1, keepdims=True)
    if np.any(weights == 0):
        raise ValueError('WINDOW_SAMPLES is too small to resolve the TAU mel filters')
    return filters / weights


def _clip_features(path, segment_length, window, filters):
    sample_rate, audio = wavfile.read(path)
    needed = segment_length * len(window)
    if sample_rate != RAW_SAMPLE_RATE_HZ or audio.ndim != 1:
        raise ValueError(f'{path} must contain mono {RAW_SAMPLE_RATE_HZ:g} Hz TAU audio')
    if audio.size < needed:
        raise ValueError(f'{path} has {audio.size} samples, but the episode needs {needed}. '
                         'Reduce SEGMENT_LENGTH or WINDOW_SAMPLES.')
    rows = audio[:needed].astype(float).reshape(segment_length, len(window))
    if np.issubdtype(audio.dtype, np.signedinteger):
        rows /= -np.iinfo(audio.dtype).min
    if not np.isfinite(rows).all():
        raise ValueError(f'{path} contains non-finite audio samples')
    spectrum = np.fft.rfft(rows * window, axis=1)
    power = np.abs(spectrum) ** 2 / np.sum(window ** 2)
    band_power = np.einsum('ij,kj->ik', power, filters)
    return np.log(np.maximum(band_power, 1e-12))


def load_tau_stream(sequence, *, segment_length, window_samples, city='barcelona', device='a',
                    data_dir=DEFAULT_DATA_DIR):
    """Build a stream using one contiguous portion of a distinct WAV per episode."""
    sequence = tuple(sequence)
    if not sequence or any(not isinstance(scene, str) or not scene for scene in sequence):
        raise ValueError('Provide a nonempty sequence of TAU scene labels')
    for name, value in (('segment_length', segment_length), ('window_samples', window_samples)):
        if (not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
                or value < 1):
            raise ValueError(f'{name} must be a positive integer')
    if not isinstance(city, str) or not city or not isinstance(device, str) or not device:
        raise ValueError('Provide a TAU city and recording device')

    window = np.hanning(window_samples)
    filters = _mel_filters(window_samples)
    data_dir = Path(data_dir)
    root = (data_dir if (data_dir / 'meta.csv').is_file() or (data_dir / 'audio').is_dir()
            else data_dir / DATASET_NAME)
    try:
        records = _load_metadata(data_dir, root)
        clips = _select_clips(sequence, records, city, device)
        _cache_audio([row['filename'] for row in clips], data_dir, root)
    except (OSError, HTTPException, BadZipFile, KeyError) as exc:
        raise RuntimeError(f'Could not load the official TAU metadata or audio. Place the TAU '
                           f'2020 Mobile v2 ZIPs or extracted dataset in {data_dir}.') from exc

    episodes, sources = [], []
    for scene, row in zip(sequence, clips):
        path = root / row['filename']
        episodes.append(_clip_features(path, segment_length, window, filters))
        sources.append(dict(scene=scene, file=row['filename'], location=row['identifier'],
                            city=city, device=device, raw_start=0,
                            raw_stop=int(segment_length * window_samples)))
    return np.vstack(episodes), sources
