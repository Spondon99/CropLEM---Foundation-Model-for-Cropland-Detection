"""
Preprocessing methods for SEN12MS satellite imagery
Contains all utility functions for data loading, normalization, and dataset creation
"""

import numpy as np
import rasterio
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict
from tqdm import tqdm


# FILE DISCOVERY

def find_sensor_files(data_dir: str, sensor: str, extensions: List[str]) -> List[Path]:
    """
    Find all files for a specific sensor in the dataset
    
    Args:
        data_dir: Base directory containing SEN12MS data
        sensor: Sensor type ('s1' or 's2')
        extensions: List of file extensions to search (e.g., ['*.tif', '*.tiff'])
    
    Returns:
        List of Path objects pointing to found files
    """
    data_path = Path(data_dir)
    all_files = []
    
    # Search pattern for this specific SEN12MS structure
    # Pattern: ROIs*_summer_s1/ROIs*_summer/s1_*/*.tif
    sensor_pattern = f"ROIs*_summer_{sensor}"

    # Find all sensor-specific root folders (e.g. ROIs1868_summer_s1)
    sensor_folders = list(data_path.glob(sensor_pattern))

    print(f" Found {len(sensor_folders)} {sensor.upper()} region folders")

    if len(sensor_folders) == 0:
        print(f" Warning: No folders matching pattern '{sensor_pattern}' found in {data_dir}")
        return []
    
    # For each sensor folder, go into ROIs*_summer/s*_N/ to find .tif files
    for sensor_folder in sensor_folders:
        # Get the inner folder (e.g. ROIs1868_summer)
        inner_folders = [f for f in sensor_folder.iterdir() if f.is_dir()]

        for inner_folder in inner_folders:
            # Find all s1_* or s2_* folders
            tile_folders = list(inner_folder.glob(f"{sensor}_*"))

            for tile_folder in tile_folders:
                # Find all .tif/.tiff files in this tile folder
                for ext in extensions:
                    files = list(tile_folder.glob(ext))
                    all_files.extend(files)
    
    # Remove duplicates and sort
    all_files = sorted(list(set(all_files)))

    return all_files



def match_sensor_files(s1_files: List[Path], s2_files: List[Path]) -> List[Tuple[Path, Path]]:
    """
    Match Sentinel-1 and Sentinel-2 files that correspond to the same location
    
    SEN12MS naming convention: ROIs{id}_{season}_{sensor}_{region}_{patch}.tif
    Example: ROIs1970_summer_s1_1_p123.tif matches ROIs1970_summer_s2_1_p123.tif
    
    Args:
        s1_files: List of Sentinel-1 file paths
        s2_files: List of Sentinel-2 file paths
    
    Returns:
        List of tuples (s1_path, s2_path) for matched pairs
    """
    # Create lookup dictionary for S2 files based on their identifier
    s2_dict = {}
    for s2_file in s2_files:
        # Extract identifier (everything except sensor type)
        # Example: ROIs1970_summer_s2_1_p123.tif -> ROIs1970_summer_1_p123
        name = s2_file.stem  # Filename without extension
        identifier = name.replace('_s2_', '_').replace('_s2', '')
        s2_dict[identifier] = s2_file
    
    # Match S1 files to S2 files
    matched_pairs = []
    for s1_file in s1_files:
        name = s1_file.stem
        identifier = name.replace('_s1_', '_').replace('_s1', '')
        
        if identifier in s2_dict:
            matched_pairs.append((s1_file, s2_dict[identifier]))
    
    return matched_pairs



# DATA LOADING

def load_tiff(file_path: Path) -> Tuple[np.ndarray, dict]:
    """
    Load a GeoTIFF file and return as numpy array
    
    Args:
        file_path: Path to .tif/.tiff file
    
    Returns:
        Tuple of (image array, metadata)
        Image shape: (channels, height, width)
    """
    with rasterio.open(file_path) as src:
        img = src.read()  # Shape: (channels, height, width)
        metadata = src.meta
    return img, metadata


def load_matched_pair(s1_path: Path, s2_path: Path) -> np.ndarray:
    """
    Load and concatenate S1 and S2 images
    
    Args:
        s1_path: Path to Sentinel-1 image
        s2_path: Path to Sentinel-2 image
    
    Returns:
        Combined array with shape (s1_channels + s2_channels, height, width)
        Typically (2 + 13, 256, 256) = (15, 256, 256)
    """
    s1_img, _ = load_tiff(s1_path)
    s2_img, _ = load_tiff(s2_path)
    
    # Concatenate along channel dimension
    combined = np.concatenate([s1_img, s2_img], axis=0)
    
    return combined



# NORMALIZATION STATISTICS

def calculate_normalization_stats(
    file_list: List[Path], 
    sample_size: int = 300,
    is_paired: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate per-channel mean and standard deviation for normalization
    
    Args:
        file_list: List of file paths (or tuples of paired paths)
        sample_size: Number of samples to use for statistics
        is_paired: Whether file_list contains paired (s1, s2) tuples
    
    Returns:
        Tuple of (mean, std) arrays with shape (num_channels,)
    """
    # Sample files
    sample_size = min(sample_size, len(file_list))
    sampled_files = np.random.choice(len(file_list), sample_size, replace=False)
    
    print(f"Calculating normalization statistics from {sample_size} samples...")
    
    all_means = []
    all_stds = []
    
    for idx in tqdm(sampled_files, desc="Processing samples"):
        try:
            if is_paired:
                # Load paired S1 and S2
                s1_path, s2_path = file_list[idx]
                img = load_matched_pair(s1_path, s2_path)
            else:
                # Load single sensor
                img, _ = load_tiff(file_list[idx])
            
            # Handle NaN/Inf values
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Calculate per-channel statistics
            means = img.reshape(img.shape[0], -1).mean(axis=1)
            stds = img.reshape(img.shape[0], -1).std(axis=1)
            
            all_means.append(means)
            all_stds.append(stds)
            
        except Exception as e:
            print(f"\nWarning: Could not process file at index {idx}: {e}")
            continue
    
    if len(all_means) == 0:
        raise ValueError("Could not process any files successfully")
    
    # Average across all sampled images
    dataset_mean = np.array(all_means).mean(axis=0)
    dataset_std = np.array(all_stds).mean(axis=0)
    
    # Avoid division by zero
    dataset_std = np.where(dataset_std == 0, 1.0, dataset_std)
    
    print(f"Statistics calculated successfully!")
    print(f"  Number of channels: {len(dataset_mean)}")
    print(f"  Mean range: [{dataset_mean.min():.2f}, {dataset_mean.max():.2f}]")
    print(f"  Std range: [{dataset_std.min():.2f}, {dataset_std.max():.2f}]")
    
    return dataset_mean, dataset_std



# PYTORCH DATASET CLASSES

class SingleSensorDataset(Dataset):
    """Dataset for single sensor (S1 or S2)"""
    
    def __init__(
        self, 
        file_list: List[Path], 
        mean: np.ndarray, 
        std: np.ndarray,
        transform=None
    ):
        """
        Args:
            file_list: List of file paths
            mean: Per-channel mean for normalization
            std: Per-channel std for normalization
            transform: Optional transforms to apply
        """
        self.file_list = file_list
        self.mean = mean
        self.std = std
        self.transform = transform
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # Load image
        img, _ = load_tiff(self.file_list[idx])
        
        # Normalize
        img_normalized = (img - self.mean[:, None, None]) / (self.std[:, None, None] + 1e-8)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_normalized).float()
        
        # Apply transforms if any
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        return img_tensor


class DualSensorDataset(Dataset):
    """Dataset for combined S1 + S2"""
    
    def __init__(
        self, 
        paired_files: List[Tuple[Path, Path]], 
        mean: np.ndarray, 
        std: np.ndarray,
        transform=None
    ):
        """
        Args:
            paired_files: List of (s1_path, s2_path) tuples
            mean: Per-channel mean for normalization (length = s1_channels + s2_channels)
            std: Per-channel std for normalization
            transform: Optional transforms to apply
        """
        self.paired_files = paired_files
        self.mean = mean
        self.std = std
        self.transform = transform
    
    def __len__(self):
        return len(self.paired_files)
    
    def __getitem__(self, idx):
        # Load and combine S1 + S2
        s1_path, s2_path = self.paired_files[idx]
        img = load_matched_pair(s1_path, s2_path)
        
        # Normalize
        img_normalized = (img - self.mean[:, None, None]) / (self.std[:, None, None] + 1e-8)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_normalized).float()
        
        # Apply transforms if any
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        return img_tensor



# TRAIN/VAL SPLIT

def split_train_val(
    file_list: List, 
    val_split: float = 0.2, 
    random_seed: int = 42
) -> Tuple[List, List]:
    """
    Split file list into train and validation sets
    
    Args:
        file_list: List of files (or tuples of paired files)
        val_split: Fraction of data for validation (0.0 to 1.0)
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_files, val_files)
    """
    train_files, val_files = train_test_split(
        file_list,
        test_size=val_split,
        random_state=random_seed
    )
    
    return train_files, val_files



# VALIDATION UTILITIES

def validate_batch(batch: torch.Tensor, expected_channels: int) -> bool:
    """
    Validate a batch tensor
    
    Args:
        batch: Batch tensor to validate
        expected_channels: Expected number of channels
    
    Returns:
        True if valid, raises error otherwise
    """
    # Check shape
    if len(batch.shape) != 4:
        raise ValueError(f"Expected 4D tensor (B, C, H, W), got shape {batch.shape}")
    
    # Check channels
    if batch.shape[1] != expected_channels:
        raise ValueError(f"Expected {expected_channels} channels, got {batch.shape[1]}")
    
    # Check for NaN or Inf
    if torch.isnan(batch).any():
        raise ValueError("Batch contains NaN values")
    
    if torch.isinf(batch).any():
        raise ValueError("Batch contains Inf values")
    
    return True


def print_dataset_info(dataset, name: str = "Dataset"):
    """Print information about a dataset"""
    print(f"\n{name} Information:")
    print(f"  Number of samples: {len(dataset)}")
    
    # Get one sample
    sample = dataset[0]
    print(f"  Sample shape: {sample.shape}")
    print(f"  Sample dtype: {sample.dtype}")
    print(f"  Value range: [{sample.min():.3f}, {sample.max():.3f}]")
    print(f"  Memory per sample: {sample.element_size() * sample.nelement() / 1024**2:.2f} MB")