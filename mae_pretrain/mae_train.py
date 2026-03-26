"""
Main training script for MAE pretraining
Run this file to train the Masked Autoencoder on SEN12MS data

Usage:
    python train.py
"""

import sys
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add parent directory to path to import preprocessing modules
sys.path.append('path_to_project/CropLEM/')

# Import configuration
import mae_config as config

# Import MAE model
from mae_model import build_mae

# Import utilities
from mae_utils import (
    setup_directories,
    save_checkpoint,
    save_encoder_only,
    visualize_reconstruction,
    TrainingLogger,
    get_cosine_schedule_with_warmup,
    get_model_size,
    count_parameters,
    AverageMeter,
    print_training_header,
    print_epoch_summary,
    load_checkpoint
)

# Import preprocessing pipeline
from data_preprocessing.preprocessing_methods import (
    find_sensor_files,
    match_sensor_files,
    calculate_normalization_stats,
    SingleSensorDataset,
    DualSensorDataset,
    split_train_val
)

from torch.utils.data import DataLoader



# DATA PREPARATION

def prepare_data():
    """
    Prepare training and validation data loaders
    Loads data using the preprocessing pipeline
    """
    print("\nPREPARING DATA\n")
    
    # Load normalization stats if they exist
    mean_path = Path(config.NORM_MEAN_PATH)
    std_path = Path(config.NORM_STD_PATH)
    
    if mean_path.exists() and std_path.exists():
        print("\nLoading saved normalization statistics...")
        mean = np.load(mean_path)
        std = np.load(std_path)
        print(f"  Mean shape: {mean.shape}")
        print(f"  Std shape: {std.shape}")
    else:
        print("\nNormalization stats not found!")
        print("   Please run the preprocessing pipeline first:")
        print("   python ../pipeline_execute.py")
        sys.exit(1)
    
    # Find sensor files
    print("\nDiscovering sensor files...")
    file_data = {}
    
    for sensor in config.SENSORS:
        files = find_sensor_files(
            config.DATA_ROOT,
            sensor,
            ['*.tif', '*.tiff']
        )
        file_data[sensor] = files
        print(f"  Found {len(files)} {sensor.upper()} files")
        
        if len(files) == 0:
            print(f"  No files found for sensor {sensor}")
            sys.exit(1)
    
    # Match sensor files if using both S1 and S2
    if len(config.SENSORS) == 2 and 's1' in config.SENSORS and 's2' in config.SENSORS:
        print("\nMatching S1 and S2 files...")
        matched_pairs = match_sensor_files(file_data['s1'], file_data['s2'])
        print(f"  Matched {len(matched_pairs)} pairs")
        
        all_files = matched_pairs
        is_dual_sensor = True
        DatasetClass = DualSensorDataset
    else:
        sensor = config.SENSORS[0]
        all_files = file_data[sensor]
        is_dual_sensor = False
        DatasetClass = SingleSensorDataset
    
    # Split train/val
    print(f"\nSplitting data (train/val = {1-config.VAL_SPLIT:.0%}/{config.VAL_SPLIT:.0%})...")
    train_files, val_files = split_train_val(
        all_files,
        val_split=config.VAL_SPLIT,
        random_seed=config.RANDOM_SEED
    )
    print(f"  Training samples: {len(train_files)}")
    print(f"  Validation samples: {len(val_files)}")
    
    # Create datasets
    print("\nCreating PyTorch datasets...")
    train_dataset = DatasetClass(train_files, mean, std, transform=None)
    val_dataset = DatasetClass(val_files, mean, std, transform=None)
    
    # Create dataloaders
    print("Creating DataLoaders...")
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
    
    return train_loader, val_loader



# ======TRAINING FUNCTIONS=======

def train_one_epoch(model, train_loader, optimizer, scheduler, scaler, epoch, config):
    model.train()
    loss_meter = AverageMeter()
    
    for batch_idx, imgs in enumerate(train_loader):
        imgs = imgs.to(config.DEVICE)
        
        # ===== DIAGNOSTIC CHECKS =====
        # Check 1: Input data
        if torch.isnan(imgs).any() or torch.isinf(imgs).any():
            print(f"!!!Batch {batch_idx}: Bad input data!")
            continue
        
        # Check 2: Input range
        if batch_idx % 100 == 0:
            print(f"  Input range: [{imgs.min():.2f}, {imgs.max():.2f}]")
        
        # Forward pass
        if config.USE_AMP:
            with torch.amp.autocast('cuda'):
                loss, pred, mask = model(imgs)
        else:
            loss, pred, mask = model(imgs)
        
        # Check 3: Loss value
        if torch.isnan(loss):
            print(f"!!!Batch {batch_idx}: NaN loss detected!")
            print(f"  Predictions range: [{pred.min():.2f}, {pred.max():.2f}]")
            print(f"  Saving debug info...")
            torch.save({
                'imgs': imgs.cpu(),
                'pred': pred.cpu(),
                'mask': mask.cpu(),
                'batch_idx': batch_idx
            }, f"debug_batch_{batch_idx}.pth")
            break  # Stop training to investigate
        
        # Check 4: Gradient norms (before clipping)
        if batch_idx % 100 == 0:
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"  Gradient norm: {total_norm:.2f}")
        
        # Backward pass
        optimizer.zero_grad()
        
        if config.USE_AMP:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        scheduler.step()
        loss_meter.update(loss.item(), imgs.size(0))
        
        # Print progress
        if batch_idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch [{epoch}/{config.NUM_EPOCHS}] "
                  f"Batch [{batch_idx}/{len(train_loader)}] "
                  f"Loss: {loss_meter.avg:.4f} "
                  f"LR: {lr:.6e}")
    
    return loss_meter.avg


def validate(model, val_loader, config):
    """Validate the model"""
    model.eval()
    loss_meter = AverageMeter()
    
    with torch.no_grad():
        for imgs in val_loader:
            imgs = imgs.to(config.DEVICE)
            
            if config.USE_AMP:
                with torch.amp.autocast('cuda'):
                    loss, _, _ = model(imgs)
            else:
                loss, _, _ = model(imgs)
            
            loss_meter.update(loss.item(), imgs.size(0))
    
    return loss_meter.avg


def get_reconstruction_samples(model, val_loader, config):
    """Get samples for visualization"""
    model.eval()
    
    with torch.no_grad():
        # Get first batch
        imgs = next(iter(val_loader))
        imgs = imgs.to(config.DEVICE)
        
        # Forward pass
        if config.USE_AMP:
            with torch.amp.autocast('cuda'):
                _, pred_imgs, mask = model(imgs)
        else:
            _, pred_imgs, mask = model(imgs)
    
    return imgs, pred_imgs, mask



# ======MAIN TRAINING LOOP=======

def main():
    """Main training function"""
    
    # Print configuration
    print("\n" + "-"*70)
    print("MAE PRETRAINING - STARTING")
    print("-"*70)
    
    # Validate configuration
    config.validate_config()
    config.print_config()
    
    # Setup directories
    setup_directories(config)
    
    # Prepare data
    train_loader, val_loader = prepare_data()
    
    # Build model
    print("\n" + "-"*70)
    print("BUILDING MODEL")
    print("-"*70)
    
    model = build_mae(config)
    model = model.to(config.DEVICE)
    
    # Print model info
    num_params = count_parameters(model)
    model_size = get_model_size(model)
    print(f"\nModel built successfully")
    print(f"  Parameters: {num_params:,}")
    print(f"  Model size: {model_size:.2f} MB")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.BASE_LR,
        betas=config.BETAS,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Setup learning rate scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_epochs=config.WARMUP_EPOCHS,
        num_training_epochs=config.NUM_EPOCHS,
        base_lr=config.BASE_LR,
        min_lr=config.MIN_LR,
        steps_per_epoch=len(train_loader)
    )
    
    # Setup mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if config.USE_AMP else None
    
    # Setup logger
    log_path = Path(config.LOGS_DIR) / "training_log.csv"
    logger = TrainingLogger(log_path)
    print(f"\nLogging to {log_path}")

    # Resume from checkpoint if exists
    start_epoch = 1
    best_val_loss = float('inf')
    resume_checkpoint = Path(config.CHECKPOINT_DIR) / "best_checkpoint.pth"

    if resume_checkpoint.exists():
        print(f"\n**Found checkpoint: {resume_checkpoint}")
        user_input = input("Resume from checkpoint? (y/n): ")
        if user_input.lower() == 'y':
            epoch_loaded, loaded_loss = load_checkpoint(resume_checkpoint, model, optimizer)
            start_epoch = epoch_loaded + 1
            best_val_loss = loaded_loss
            print(f"   Resuming from epoch {start_epoch}")
            print(f"   Best val loss so far {best_val_loss:.6f}")
        else:
            print("   Starting fresh training")
    else:
        print("\n**No checkpoint found - starting fresh")
    
    # Training loop
    print("\n" + "-"*70)
    print("TRAINING")
    print("-"*70)
    
    print_training_header()
    
    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, epoch, config
        )
        
        # Validate
        val_loss = validate(model, val_loader, config)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print_epoch_summary(epoch, train_loss, val_loss, current_lr, epoch_time)
        
        # Log metrics
        logger.log(epoch, train_loss, val_loss, current_lr, epoch_time)
        
        # Save checkpoint
        if epoch % config.SAVE_FREQ == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss, config,
                filename=f"checkpoint_epoch_{epoch:04d}.pth"
            )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss, config,
                filename="best_checkpoint.pth"
            )
            print(f"  New best model! Val loss: {val_loss:.6f}")
        
        # Save reconstruction samples
        if epoch % config.SAMPLE_FREQ == 0 or epoch == 1:
            imgs, pred_imgs, mask = get_reconstruction_samples(model, val_loader, config)
            visualize_reconstruction(imgs, pred_imgs, mask, config, epoch)
        
        # Save final checkpoint and encoder
        if epoch == config.NUM_EPOCHS:
            save_checkpoint(
                model, optimizer, epoch, val_loss, config,
                filename="final_checkpoint.pth"
            )
            save_encoder_only(model, config, filename="mae_encoder.pth")
    
    # Training complete
    print("\n" + "-"*70)
    print("TRAINING COMPLETED.")
    print("-"*70)
    
    print(f"\nFinal Results:")
    print(f"  Best validation loss: {best_val_loss:.6f}")
    print(f"  Total epochs: {config.NUM_EPOCHS}")
    
    print(f"\nSaved Files:")
    print(f"  Checkpoints: {config.CHECKPOINT_DIR}/")
    print(f"  Encoder: {config.CHECKPOINT_DIR}/mae_encoder.pth")
    print(f"  Samples: {config.SAMPLES_DIR}/")
    print(f"  Logs: {log_path}")
    
    print(f"\nNext Steps:")
    print(f"  1. Check reconstruction samples in {config.SAMPLES_DIR}/")
    print(f"  2. Review training log in {log_path}")
    print(f"  3. Use encoder ({config.CHECKPOINT_DIR}/mae_encoder.pth) for downstream tasks")
    
    print("\n=======MAE pretraining complete!========\n")



# ======ENTRY POINT=======

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Progress has been saved in checkpoints")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
