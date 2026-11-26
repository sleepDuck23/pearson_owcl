"""Download CWRU recordings and assemble recurring drive-end vibration episodes."""

from collections import Counter
from http.client import HTTPException
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from urllib.error import URLError
from urllib.request import urlopen
from zlib import error as ZlibError

import numpy as np
from scipy.io import loadmat


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / 'recordings'
RAW_SAMPLE_RATE_HZ = 48_000
DOWNLOAD_BASE_URL = 'https://engineering.case.edu/sites/default/files'
REGIME_NAMES = {
    0: 'Healthy',
    1: 'Inner-race fault',
    2: 'Ball fault',
    3: 'Outer-race fault at 6 o’clock',
}
# Columns are motor loads 0–3 HP. Fault recordings are 0.007-inch drive-end defects.
RECORD_IDS = {
    0: (97, 98, 99, 100),
    1: (109, 110, 111, 112),
    2: (122, 123, 124, 125),
    3: (135, 136, 137, 138),
}
FEATURE_NAMES = ('log_rms', 'crest_factor', 'spectral_centroid_hz')


def _read_drive_end(path, variable):
    try:
        values = loadmat(path, variable_names=[variable], squeeze_me=True)
    except (OSError, ValueError, IndexError, NotImplementedError, ZlibError) as exc:
        raise ValueError(f'Cannot read CWRU MATLAB recording {path}') from exc
    if variable not in values:
        raise ValueError(f'{path} is missing the expected drive-end channel {variable}')
    signal = np.asarray(values[variable])
    if (signal.ndim != 1 or not np.isrealobj(signal)
            or not np.issubdtype(signal.dtype, np.number)
            or not signal.size or not np.isfinite(signal).all()):
        raise ValueError(f'{variable} in {path} must be a finite real vibration vector')
    return signal.astype(float, copy=False)


def _load_recording(record_id, data_dir):
    path = data_dir / f'{record_id}.mat'
    variable = f'X{record_id:03d}_DE_time'
    url = f'{DOWNLOAD_BASE_URL}/{path.name}'
    if path.is_file():
        return _read_drive_end(path, variable)

    data_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=data_dir) as temporary_dir:
        staged_path = Path(temporary_dir) / path.name
        print(f'Downloading CWRU recording {path.name}...')
        try:
            with urlopen(url, timeout=60) as response, staged_path.open('wb') as destination:
                shutil.copyfileobj(response, destination)
        except (URLError, TimeoutError, HTTPException) as exc:
            raise RuntimeError(f'Could not download {url}. Download it manually to {path}.') from exc
        try:
            signal = _read_drive_end(staged_path, variable)
        except ValueError as exc:
            raise ValueError(f'{url} did not contain a valid CWRU recording with {variable}. '
                             f'Download the MATLAB recording manually to {path}.') from exc
        staged_path.replace(path)
    return signal


def _window_features(windows):
    """Compute one feature row after each complete vibration window arrives."""
    centered = windows - windows.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(centered ** 2, axis=1))
    rms_floor = np.maximum(rms, 1e-12)
    crest_factor = np.max(np.abs(centered), axis=1) / rms_floor

    window_samples = windows.shape[1]
    spectrum = np.fft.rfft(centered * np.hanning(window_samples), axis=1)
    frequencies = np.fft.rfftfreq(window_samples, d=1 / RAW_SAMPLE_RATE_HZ)[1:]
    power = np.abs(spectrum[:, 1:]) ** 2
    if window_samples % 2 == 0:
        power[:, -1] *= 0.5  # Nyquist has no negative-frequency counterpart.
    total_power = power.sum(axis=1)
    centroid = np.divide(np.sum(power * frequencies, axis=1), total_power,
                         out=np.zeros_like(total_power), where=total_power > 0)
    return np.column_stack((np.log(rms_floor), crest_factor, centroid))


def load_cwru_stream(sequence, *, segment_length, window_samples, motor_load=0,
                     data_dir=DEFAULT_DATA_DIR):
    """Return feature rows and source intervals for an ordered series of episodes.

    Each episode takes the next contiguous block from its condition's recording.
    Source intervals use zero-based raw sample offsets with an exclusive stop.
    """
    sequence = tuple(sequence)
    if not sequence or any(regime not in RECORD_IDS for regime in sequence):
        raise ValueError(f'Provide a nonempty sequence using CWRU regime IDs {tuple(RECORD_IDS)}')
    if (not isinstance(motor_load, (int, np.integer))
            or isinstance(motor_load, (bool, np.bool_)) or not 0 <= motor_load <= 3):
        raise ValueError('motor_load must be an integer from 0 to 3 HP')
    for name, value, minimum in (('segment_length', segment_length, 1),
                                 ('window_samples', window_samples, 4)):
        if (not isinstance(value, (int, np.integer))
                or isinstance(value, (bool, np.bool_)) or value < minimum):
            raise ValueError(f'{name} must be an integer of at least {minimum}')

    data_dir = Path(data_dir)
    raw_segment_length = segment_length * window_samples
    features_by_regime = {}
    counts = Counter(sequence)
    for regime, count in counts.items():
        record_id = RECORD_IDS[regime][motor_load]
        signal = _load_recording(record_id, data_dir)
        needed = count * raw_segment_length
        if signal.size < needed:
            raise ValueError(f'CWRU record {record_id} has {signal.size} samples, but regime '
                             f'{regime} needs {needed} for {count} distinct episodes. Reduce '
                             'SEGMENT_LENGTH, WINDOW_SAMPLES, or the number of occurrences.')
        windows = signal[:needed].reshape(-1, window_samples)
        features_by_regime[regime] = _window_features(windows)

    offsets = dict.fromkeys(counts, 0)
    episodes, sources = [], []
    for regime in sequence:
        start = offsets[regime]
        stop = start + segment_length
        episodes.append(features_by_regime[regime][start:stop])
        record_id = RECORD_IDS[regime][motor_load]
        sources.append(dict(regime=regime, condition=REGIME_NAMES[regime],
                            record_id=record_id, file=str(data_dir / f'{record_id}.mat'),
                            raw_start=start * window_samples, raw_stop=stop * window_samples))
        offsets[regime] = stop
    return np.vstack(episodes), sources
