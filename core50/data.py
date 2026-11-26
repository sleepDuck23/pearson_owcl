"""Read chronological RGB clips of recurring object instances from CORe50."""

from collections import Counter, deque
from contextlib import ExitStack
from http.client import HTTPException
from io import BufferedReader
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile

import numpy as np
from PIL import Image

from data.remote_zip import HTTPRangeReader


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / 'recordings'
ARCHIVE_NAME = 'core50_128x128.zip'
DOWNLOAD_URL = f'http://bias.csr.unibo.it/maltoni/download/core50/{ARCHIVE_NAME}'
FRAME_RATE_HZ = 20
FEATURE_SIZE = (8, 8)
OBJECT_CATEGORIES = ('Plug adapter', 'Mobile phone', 'Scissors', 'Light bulb', 'Can',
                     'Glasses', 'Ball', 'Marker', 'Cup', 'Remote control')
FRAME_NAME = re.compile(r'C_(\d+)_(\d+)_(\d+)\.png')


def _archive_members(archive, session, objects):
    by_directory = {f'o{object_id}': object_id for object_id in objects}
    members = {object_id: [] for object_id in objects}
    for name in archive.namelist():
        path = name.rsplit('/', 3)[-3:]
        if len(path) == 3 and path[0] == f's{session}' and path[1] in by_directory:
            members[by_directory[path[1]]].append(name)
    return members


def _indexed_frames(names, session, object_id):
    frames = {}
    for name in names:
        match = FRAME_NAME.fullmatch(name.rsplit('/', 1)[-1])
        if match is None:
            continue
        video_session, video_object, frame = map(int, match.groups())
        if (video_session, video_object) != (session, object_id):
            continue
        if frame in frames:
            raise ValueError(f'Duplicate CORe50 frame {frame} in s{session}/o{object_id}')
        frames[frame] = name
    return frames


def _cache_track(source_archive, frames, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temporary_dir:
        staged = Path(temporary_dir) / path.name
        with ZipFile(staged, 'w') as output:
            for name in frames.values():
                output.writestr(name, source_archive.read(name))
        staged.replace(path)


def _clip_features(frames, open_frame, *, session, object_id, count, segment_length, source):
    frame_ids = np.array(sorted(frames), dtype=int)
    clips = deque()
    runs = np.split(frame_ids, np.flatnonzero(np.diff(frame_ids) != 1) + 1)
    for run in runs:
        for offset in range(0, len(run) - segment_length + 1, segment_length):
            clip_ids = run[offset:offset + segment_length]
            rows = []
            for frame in clip_ids:
                name = frames[frame]
                try:
                    with open_frame(name) as file, Image.open(file, formats=['PNG']) as image:
                        thumbnail = image.convert('RGB').resize(FEATURE_SIZE, Image.Resampling.BOX)
                        rows.append(np.asarray(thumbnail, dtype=float).ravel() / 255.0)
                except (OSError, BadZipFile) as exc:
                    raise ValueError(f'Cannot read CORe50 frame {name} in {source}') from exc
            clips.append((np.vstack(rows), dict(
                object_id=object_id, category=OBJECT_CATEGORIES[(object_id - 1) // 5],
                session=session, source=str(source), first_frame=int(clip_ids[0]),
                last_frame=int(clip_ids[-1]), first_file=frames[clip_ids[0]],
                last_file=frames[clip_ids[-1]])))
            if len(clips) == count:
                return clips
    raise ValueError(f'CORe50 s{session}/o{object_id} supplies {len(clips)} of {count} requested '
                     f'clips of {segment_length} consecutive frames in {source}. '
                     'Check the recordings or reduce SEGMENT_LENGTH or the number of repeats.')


def load_core50_stream(sequence, *, segment_length, session=1, data_dir=DEFAULT_DATA_DIR):
    """Return one feature row per frame and the source of each episode in sequence order."""
    sequence = tuple(sequence)
    if not sequence or any(not isinstance(obj, (int, np.integer)) or isinstance(obj, (bool, np.bool_))
                           or not 1 <= obj <= 50 for obj in sequence):
        raise ValueError('Use a nonempty sequence of CORe50 object IDs from 1 to 50')
    for name, value, maximum in (('session', session, 11), ('segment_length', segment_length, None)):
        if (not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
                or value < 1 or (maximum is not None and value > maximum)):
            raise ValueError(f'{name} must be an integer from 1 to 11' if maximum
                             else f'{name} must be a positive integer')

    counts = Counter(sequence)
    data_dir = Path(data_dir)
    extracted = data_dir / 'core50_128x128'
    root = extracted if extracted.is_dir() else data_dir
    local_archive = data_dir / ARCHIVE_NAME
    has_local_archive = local_archive.is_file()
    clips = {}
    with ExitStack() as stack:
        source_archive = None
        for object_id, count in counts.items():
            track = root / f's{session}' / f'o{object_id}'
            cached = data_dir / f's{session}-o{object_id}.zip'
            if track.is_dir():
                frames = _indexed_frames((path.name for path in track.iterdir() if path.is_file()),
                                         session, object_id)
                source = track

                def open_frame(name):
                    return (track / name).open('rb')

            else:
                try:
                    if cached.is_file():
                        archive = stack.enter_context(ZipFile(cached))
                        frames = _indexed_frames(archive.namelist(), session, object_id)
                        source = cached
                    else:
                        if source_archive is None:
                            if has_local_archive:
                                archive_file = local_archive
                            else:
                                print('Opening the official CORe50 ZIP via HTTP range requests...')
                                archive_file = stack.enter_context(BufferedReader(
                                    HTTPRangeReader(DOWNLOAD_URL), buffer_size=2 ** 20))
                            source_archive = stack.enter_context(ZipFile(archive_file))
                            members = _archive_members(source_archive, session, counts)
                        frames = _indexed_frames(members[object_id], session, object_id)
                        archive, source = source_archive, local_archive
                        if not has_local_archive and frames:
                            print(f'Caching CORe50 s{session}/o{object_id} ({len(frames)} frames)...')
                            _cache_track(source_archive, frames, cached)
                            archive, source = stack.enter_context(ZipFile(cached)), cached
                    open_frame = archive.open
                except (OSError, HTTPException, BadZipFile) as exc:
                    raise RuntimeError(f'Could not read CORe50 s{session}/o{object_id}. Check the '
                                       f'local cache, or place {ARCHIVE_NAME} or the extracted '
                                       f'core50_128x128 directory in {data_dir}.') from exc
            clips[object_id] = _clip_features(
                frames, open_frame, session=session, object_id=object_id, count=count,
                segment_length=segment_length, source=source)

    episodes, sources = zip(*(clips[obj].popleft() for obj in sequence))
    return np.vstack(episodes), list(sources)
