"""
Configuration file for SEN12MS preprocessing pipeline
Edit the paths and parameters here according to your setup
"""

import os


# Path to SEN12MS dataset
DATA_DIR = r"path/CropLEM/data/sen12ms/"

# Output directory for processed data (created automatically)
OUTPUT_DIR = "processed_data/"


# SENSOR CONFIGURATION

# Which sensors to use: ['s1', 's2'] or ['s1'] or ['s2']
SENSORS = ['s1', 's2']  # Using both Sentinel-1 and Sentinel-2

# Sentinel-1 has 2 bands (VV, VH polarizations)
# Sentinel-2 has 13 bands (different spectral bands)
# Total channels when using both: 2 + 13 = 15 channels


# PREPROCESSING PARAMETERS

IMAGE_SIZE = 256

# Normalization method: 'standardize' (mean=0, std=1) or 'minmax' (0-1 range)
NORM_METHOD = 'standardize'

# Number of samples to use for calculating normalization statistics
# Use smaller number for faster computation (e.g., 100)
# Use larger number for more accurate statistics (e.g., 500)
NORM_SAMPLE_SIZE = 100


# TRAIN/VAL SPLIT

# Validation split ratio (0.2 = 20% validation, 80% training)
VAL_SPLIT = 0.2

# Random seed for reproducibility
RANDOM_SEED = 42


# DATALOADER PARAMETERS

BATCH_SIZE = 32

# Number of worker processes for data loading
# Set to 0 on Windows to avoid multiprocessing issues
# Set to 2-4 on Linux/Mac for faster loading
NUM_WORKERS = 0

# Use pinned memory for faster GPU transfer (True if using GPU)
PIN_MEMORY = True

# Drop last incomplete batch
DROP_LAST = True


# FILE PATTERNS

# File extensions to search for
FILE_EXTENSIONS = ['*.tif', '*.tiff']

# Sensor folder patterns (SEN12MS structure)
SENSOR_PATTERNS = {
    's1': 's1_*',  # Sentinel-1 folders
    's2': 's2_*'   # Sentinel-2 folders
}


# VALIDATION

def validate_config():
    """Validate configuration parameters"""
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        raise ValueError(f"DATA_DIR does not exist: {DATA_DIR}")
    
    # Check sensors
    valid_sensors = ['s1', 's2']
    for sensor in SENSORS:
        if sensor not in valid_sensors:
            raise ValueError(f"Invalid sensor: {sensor}. Must be one of {valid_sensors}")
    
    # Check normalization method
    if NORM_METHOD not in ['standardize', 'minmax']:
        raise ValueError(f"Invalid NORM_METHOD: {NORM_METHOD}")
    
    # Check split ratio
    if not 0 < VAL_SPLIT < 1:
        raise ValueError(f"VAL_SPLIT must be between 0 and 1, got {VAL_SPLIT}")
    
    # Check batch size
    if BATCH_SIZE < 1:
        raise ValueError(f"BATCH_SIZE must be positive, got {BATCH_SIZE}")
    
    print("Configuration validated successfully")
    return True

if __name__ == "__main__":
    # Quick test of configuration
    print("Current Configuration:")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Sensors: {SENSORS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Validation split: {VAL_SPLIT}")
    print(f"  Num workers: {NUM_WORKERS}")
    
    try:
        validate_config()
    except ValueError as e:
        print(f"\nConfiguration error: {e}")