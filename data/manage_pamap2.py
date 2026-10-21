import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd

def download_and_extract_pamap2(data_dir="data/PAMAP2_Dataset"):
    """Downloads and extracts the PAMAP2 dataset if it doesn't already exist."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip"
    zip_path = "data/PAMAP2_Dataset.zip"
    
    if not os.path.exists(data_dir):
        print("Downloading PAMAP2 Dataset (this may take a moment)...")
        os.makedirs("data", exist_ok=True)
        urllib.request.urlretrieve(url, zip_path)
        
        print("Extracting PAMAP2 Dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("data/")
        os.remove(zip_path)
        print("PAMAP2 ready.")
    else:
        print("PAMAP2 Dataset already exists locally. Skipping download.")

def _load_activity_blocks(feature_columns, block_size, blocks_per_activity):
    """Collect non-overlapping blocks from contiguous runs of subject101."""
    if not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 12: 5, 13: 6}
    counts = (dict.fromkeys(mapping.values(), 1) if blocks_per_activity is None
              else dict(blocks_per_activity))
    for act, count in counts.items():
        if act not in mapping.values() or not isinstance(count, (int, np.integer)) or count < 1:
            raise ValueError("Activity IDs and block counts must be valid")
    download_and_extract_pamap2()
    file_path = "data/PAMAP2_Dataset/Protocol/subject101.dat"
    print(f"Parsing {file_path}...")
    df = pd.read_csv(file_path, sep=r"\s+", header=None,
                     usecols=[0, 1] + list(feature_columns))

    # Preserve labels and timestamps until runs have been identified. Do not
    # splice across activity transitions or missing timestamps (> 1.5 samples).
    dt = df[0].diff()
    new_run = df[1].ne(df[1].shift()) | dt.gt(0.015) | dt.le(0)
    pools = {act: [] for act in counts}
    for _, run in df.groupby(new_run.cumsum(), sort=False):
        act = mapping.get(int(run[1].iloc[0]))
        if act not in pools or len(pools[act]) >= counts[act]:
            continue
        # Causal sample-and-hold within this run only. Leading missing values
        # are discarded, never filled using future observations or other labels.
        features = run[list(feature_columns)].ffill().dropna()
        values = features.to_numpy(dtype=float)
        for start in range(0, len(values) - block_size + 1, block_size):
            block = values[start:start + block_size]
            if not np.isfinite(block).all():
                raise ValueError(f"Non-finite sensor values in activity {act}")
            pools[act].append(block)
            if len(pools[act]) == counts[act]:
                break

    activities = {}
    for act, chunks in pools.items():
        if len(chunks) != counts[act]:
            raise ValueError(
                f"Activity {act} needs {counts[act]} distinct contiguous blocks "
                f"of {block_size} samples; found {len(chunks)}. "
                "Reduce the sequence/block size or use additional recordings.")
        activities[act] = np.vstack(chunks)
    return activities, block_size


def load_pamap2_blocks(block_size=300, blocks_per_activity=None):
    """Hand acceleration (zero-based columns 4, 5, 6); no replay or padding."""
    return _load_activity_blocks([4, 5, 6], block_size, blocks_per_activity)


def load_pamap2_full_blocks(block_size=1500, blocks_per_activity=None):
    """Heart rate and valid IMU channels, with causal within-run filling.

    Exclude the 12 orientation channels marked invalid by the dataset authors.
    The resulting observations have 40 features, excluding timestamp and label.
    """
    invalid_orientation = set(range(16, 20)) | set(range(33, 37)) | set(range(50, 54))
    columns = [column for column in range(2, 54) if column not in invalid_orientation]
    return _load_activity_blocks(columns, block_size, blocks_per_activity)
