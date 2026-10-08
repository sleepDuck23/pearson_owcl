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

def load_pamap2_blocks(block_size=300):
    """
    Reads a PAMAP2 subject file, cleans NaNs, extracts the Hand 3D Accelerometer,
    and maps 7 distinct activities to dictionary keys 0-6 to match the sequence builder.
    """
    download_and_extract_pamap2()
    
    # We will use subject101 as our continuous stream source
    file_path = "data/PAMAP2_Dataset/Protocol/subject101.dat"
    
    # PAMAP2 Columns: 1: timestamp, 2: activity_id, 3: heart_rate, 4-6: Hand Accel 3D
    # We extract columns 1 (activity) and 4, 5, 6 (Hand X, Y, Z acceleration)
    print(f"Parsing {file_path}...")
    df = pd.read_csv(file_path, sep='\s+', header=None, usecols=[1, 4, 5, 6])
    df.columns = ['activity_id', 'x', 'y', 'z']
    
    # Drop transitional periods (activity 0) and any dropped packet NaNs
    df = df[df['activity_id'] != 0].dropna()
    
    # Map PAMAP2 specific IDs to our 0-6 keys
    # 1: lying, 2: sitting, 3: standing, 4: walking, 5: running, 12: asc_stairs, 13: desc_stairs
    activity_mapping = {
        1: 0,  
        2: 1,  
        3: 2,  
        4: 3,  
        5: 4,  
        12: 5, 
        13: 6  
    }
    
    activities = {}
    for pamap_id, target_key in activity_mapping.items():
        # Extract the continuous block for this specific activity
        block = df[df['activity_id'] == pamap_id][['x', 'y', 'z']].values
        
        if len(block) >= block_size:
            activities[target_key] = block[:block_size]
        else:
            # Fallback/padding if a subject didn't perform an activity long enough
            print(f"Warning: Activity {pamap_id} too short. Padding with local mean.")
            pad_length = block_size - len(block)
            padding = np.tile(np.mean(block, axis=0), (pad_length, 1)) if len(block) > 0 else np.zeros((pad_length, 3))
            activities[target_key] = np.vstack([block, padding])
            
    return activities, block_size

def load_pamap2_full_blocks(block_size=1500):
    """
    Reads all 54 columns of a PAMAP2 subject file, interpolates mismatched 
    sampling rates, and extracts the full body sensor network.
    """
    download_and_extract_pamap2()
    file_path = "data/PAMAP2_Dataset/Protocol/subject101.dat"
    
    print(f"Parsing all columns from {file_path}...")
    # Load all 54 columns. Col 0 is timestamp, Col 1 is activity_id.
    df = pd.read_csv(file_path, sep='\s+', header=None)
    
    # Rename activity column for easy filtering
    df.rename(columns={1: 'activity_id'}, inplace=True)
    
    # Drop transitional periods (activity 0)
    df = df[df['activity_id'] != 0]
    
    # THE CRITICAL STEP: Interpolate the 9Hz heart rate and dropped packets 
    # to match the 100Hz IMU stream.
    df.fillna(method='ffill', inplace=True) # Forward fill previous valid readings
    df.fillna(method='bfill', inplace=True) # Backward fill any remaining gaps at the start
    
    activity_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 12: 5, 13: 6}
    activities = {}
    
    for pamap_id, target_key in activity_mapping.items():
        # Extract all 52 feature columns (skipping timestamp and activity_id)
        block = df[df['activity_id'] == pamap_id].iloc[:, 2:].values
        
        if len(block) >= block_size:
            activities[target_key] = block[:block_size]
        else:
            print(f"Warning: Activity {pamap_id} too short. Padding with local mean.")
            pad_length = block_size - len(block)
            padding = np.tile(np.mean(block, axis=0), (pad_length, 1)) if len(block) > 0 else np.zeros((pad_length, block.shape[1]))
            activities[target_key] = np.vstack([block, padding])
            
    return activities, block_size