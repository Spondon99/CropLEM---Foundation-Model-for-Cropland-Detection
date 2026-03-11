"""
Main preprocessing pipeline execution script for SEN12MS dataset
Run this script to prepare data for MAE self-supervised learning

Usage:
    python pipeline_execute.py
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# Import configuration and preprocessing methods
import config
from preprocessing_methods import (
    find_sensor_files,
    match_sensor_files,
    calculate_normalization_stats,
    SingleSensorDataset,
    DualSensorDataset,
    split_train_val,
    validate_batch,
    print_dataset_info
)


def main():
    """Main preprocessing pipeline execution"""
    
    print("="*60)
    print("SEN12MS PREPROCESSING PIPELINE")
    print("="*60)
    
    
    # STEP 1: VALIDATE CONFIGURATION
    print("\n[Step 1/6] Validating configuration...")
    try:
        config.validate_config()
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("\nPlease edit config.py to fix the issue.")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    
    
    # STEP 2: DISCOVER FILES
    print("\n[Step 2/6] Discovering sensor files...")
    
    sensors_to_use = config.SENSORS
    file_data = {}
    
    for sensor in sensors_to_use:
        print(f"\n  Searching for {sensor.upper()} files...")
        files = find_sensor_files(
            config.DATA_DIR, 
            sensor, 
            config.FILE_EXTENSIONS
        )
        file_data[sensor] = files
        print(f"  Found {len(files)} {sensor.upper()} files")
        
        if len(files) == 0:
            print(f"  No files found for sensor {sensor}")
            print(f"  Check your DATA_DIR and sensor folder structure")
            sys.exit(1)
    
    
    # STEP 3: MATCH SENSOR FILES (if using both)
    if len(sensors_to_use) == 2 and 's1' in sensors_to_use and 's2' in sensors_to_use:
        print("\n[Step 3/6] Matching S1 and S2 files...")
        
        matched_pairs = match_sensor_files(
            file_data['s1'], 
            file_data['s2']
        )
        
        print(f"  Matched {len(matched_pairs)} pairs")
        
        if len(matched_pairs) == 0:
            print("  No matched pairs found")
            print("  S1 and S2 files may have different naming patterns")
            sys.exit(1)
        
        # Use matched pairs
        all_files = matched_pairs
        is_dual_sensor = True
        expected_channels = 15  # S1 (2 channels) + S2 (13 channels)
        
    else:
        # Single sensor mode
        print("\n[Step 3/6] Using single sensor mode...")
        sensor = sensors_to_use[0]
        all_files = file_data[sensor]
        is_dual_sensor = False
        
        # Determine expected channels
        if sensor == 's1':
            expected_channels = 2
        else:  # s2
            expected_channels = 13
        
        print(f"  Using {len(all_files)} {sensor.upper()} files")
    
    
    # STEP 4: CALCULATE NORMALIZATION STATISTICS
    print("\n[Step 4/6] Calculating normalization statistics...")
    
    mean, std = calculate_normalization_stats(
        all_files,
        sample_size=config.NORM_SAMPLE_SIZE,
        is_paired=is_dual_sensor
    )
    
    print(f"\n  Per-channel mean: {mean}")
    print(f"  Per-channel std: {std}")
    
    
    # STEP 5: SPLIT TRAIN/VAL
    print("\n[Step 5/6] Splitting data into train/validation sets...")
    
    train_files, val_files = split_train_val(
        all_files,
        val_split=config.VAL_SPLIT,
        random_seed=config.RANDOM_SEED
    )
    
    print(f"  Training samples: {len(train_files)}")
    print(f"  Validation samples: {len(val_files)}")
    print(f"  Split ratio: {1-config.VAL_SPLIT:.0%} train / {config.VAL_SPLIT:.0%} val")
    
    
    # STEP 6: CREATE DATASETS AND DATALOADERS
    print("\n[Step 6/6] Creating PyTorch datasets and dataloaders...")
    
    # Choose appropriate dataset class
    if is_dual_sensor:
        DatasetClass = DualSensorDataset
    else:
        DatasetClass = SingleSensorDataset
    
    # Create datasets (no augmentation)
    train_dataset = DatasetClass(train_files, mean, std, transform=None)
    val_dataset = DatasetClass(val_files, mean, std, transform=None)
    
    # Print dataset info
    print_dataset_info(train_dataset, "Training Dataset")
    print_dataset_info(val_dataset, "Validation Dataset")
    
    # Create dataloaders
    print("\n  Creating DataLoaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=config.DROP_LAST
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=False
    )
    
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    
    # VALIDATION: TEST DATALOADER
    print("\n[Validation] Testing DataLoader...")
    
    try:
        # Get first batch
        batch = next(iter(train_loader))
        
        # Validate batch
        validate_batch(batch, expected_channels)
        
        print(f"  Batch shape: {batch.shape}")
        print(f"  Batch dtype: {batch.dtype}")
        print(f"  Value range: [{batch.min():.3f}, {batch.max():.3f}]")
        print(f"  Memory per batch: {batch.element_size() * batch.nelement() / 1024**2:.2f} MB")
        
    except Exception as e:
        print(f"  DataLoader test failed: {e}")
        sys.exit(1)
    
    
    # SUMMARY
    print("\n" + "="*60)
    print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nDataset Summary:")
    print(f"  • Total samples: {len(all_files)}")
    print(f"  • Training samples: {len(train_files)}")
    print(f"  • Validation samples: {len(val_files)}")
    print(f"  • Image size: {config.IMAGE_SIZE}×{config.IMAGE_SIZE}")
    print(f"  • Number of channels: {expected_channels}")
    print(f"  • Batch size: {config.BATCH_SIZE}")
    
    print(f"\nConfiguration:")
    print(f"  • Sensors: {', '.join([s.upper() for s in config.SENSORS])}")
    print(f"  • Normalization: {config.NORM_METHOD}")
    print(f"  • Val split: {config.VAL_SPLIT:.0%}")
    
    print(f"\nOutputs:")
    print(f"  • train_loader: Ready for MAE training")
    print(f"  • val_loader: Ready for validation")
    print(f"  • Expected input shape: (batch_size, {expected_channels}, {config.IMAGE_SIZE}, {config.IMAGE_SIZE})")
    
    print(f"\nNext Steps:")
    print(f"  1. Use train_loader for MAE training")
    print(f"  2. Use val_loader for validation during training")
    print(f"  3. Save normalization stats (mean, std) for downstream tasks")
    
    print("\n" + "="*60)
    
    # Return dataloaders for use in other scripts
    return train_loader, val_loader, mean, std


if __name__ == "__main__":
    # Execute pipeline
    train_loader, val_loader, mean, std = main()
    
    # Optional: Save normalization stats for later use
    print("\nSaving normalization statistics...")
    output_dir = Path(config.OUTPUT_DIR)
    np.save(output_dir / "normalization_mean.npy", mean)
    np.save(output_dir / "normalization_std.npy", std)
    print(f"  Saved to {output_dir / 'normalization_mean.npy'}")
    print(f"  Saved to {output_dir / 'normalization_std.npy'}")
    
    print("\nAll done! DataLoaders are ready for MAE training.")