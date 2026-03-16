"""
Utility functions for MAE training
Includes checkpointing, visualization, and logging
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import csv


# ======DIRECTORY SETUP=======

def setup_directories(config):
    """Create necessary directories for outputs"""
    directories = [
        config.OUTPUT_DIR,
        config.CHECKPOINT_DIR,
        config.SAMPLES_DIR,
        config.LOGS_DIR
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print(f"Created output directories in {config.OUTPUT_DIR}/")



# ======CHECKPOINTING=======

def save_checkpoint(model, optimizer, epoch, loss, config, filename="checkpoint.pth"):
    """
    Save model checkpoint
    
    Args:
        model: MAE model
        optimizer: optimizer
        epoch: current epoch
        loss: current loss
        config: configuration
        filename: checkpoint filename
    """
    checkpoint_path = Path(config.CHECKPOINT_DIR) / filename
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': {
            'img_size': config.IMAGE_SIZE,
            'patch_size': config.PATCH_SIZE,
            'in_channels': config.IN_CHANNELS,
            'embed_dim': config.EMBED_DIM,
            'encoder_depth': config.ENCODER_DEPTH,
            'encoder_num_heads': config.ENCODER_NUM_HEADS,
            'decoder_embed_dim': config.DECODER_EMBED_DIM,
            'decoder_depth': config.DECODER_DEPTH,
            'decoder_num_heads': config.DECODER_NUM_HEADS,
            'mask_ratio': config.MASK_RATIO
        }
    }
    
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load model checkpoint
    
    Args:
        checkpoint_path: path to checkpoint file
        model: MAE model
        optimizer: optimizer (optional)
    
    Returns:
        epoch: epoch number from checkpoint
        loss: loss from checkpoint
    """
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Loaded checkpoint from epoch {epoch}")
    
    return epoch, loss


def save_encoder_only(model, config, filename="mae_encoder.pth"):
    """
    Save only the encoder (foundation model for downstream tasks)
    
    Args:
        model: MAE model
        config: configuration
        filename: encoder checkpoint filename
    """
    encoder_path = Path(config.CHECKPOINT_DIR) / filename
    
    encoder_checkpoint = {
        'encoder_state_dict': model.encoder.state_dict(),
        'config': {
            'img_size': config.IMAGE_SIZE,
            'patch_size': config.PATCH_SIZE,
            'in_channels': config.IN_CHANNELS,
            'embed_dim': config.EMBED_DIM,
            'depth': config.ENCODER_DEPTH,
            'num_heads': config.ENCODER_NUM_HEADS,
        }
    }
    
    torch.save(encoder_checkpoint, encoder_path)
    print(f"Saved encoder to {encoder_path}")
    
    return encoder_path



# ======VISUALIZATION=======

def visualize_reconstruction(
    original, 
    reconstructed, 
    mask, 
    config,
    epoch,
    num_samples=4
):
    """
    Visualize original images, masked images, and reconstructions
    """
    # Check for NaN values
    if torch.isnan(reconstructed).any():
        print(f"  !!!Skipping visualization - predictions contain NaN")
        return
    
    # Move to CPU and convert to numpy with explicit float32
    original = original.cpu().numpy().astype(np.float32)
    reconstructed = reconstructed.cpu().numpy().astype(np.float32)
    mask = mask.cpu().numpy()
    
    # Limit number of samples
    num_samples = min(num_samples, original.shape[0])
    
    # Create masked version
    masked = create_masked_image(original, mask, config)
    
    # For visualization, use first 3 channels as RGB
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 4))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Normalize for display (per-image min-max) - ensure float32
        orig_rgb = normalize_for_display(original[i, :3]).astype(np.float32)
        masked_rgb = normalize_for_display(masked[i, :3]).astype(np.float32)
        recon_rgb = normalize_for_display(reconstructed[i, :3]).astype(np.float32)
        
        # Clip to valid range [0, 1]
        orig_rgb = np.clip(orig_rgb, 0, 1)
        masked_rgb = np.clip(masked_rgb, 0, 1)
        recon_rgb = np.clip(recon_rgb, 0, 1)
        
        # Plot
        axes[i, 0].imshow(orig_rgb.transpose(1, 2, 0))
        axes[i, 0].set_title("Original")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(masked_rgb.transpose(1, 2, 0))
        axes[i, 1].set_title("Masked (75%)")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(recon_rgb.transpose(1, 2, 0))
        axes[i, 2].set_title("Reconstructed")
        axes[i, 2].axis('off')
    
    plt.suptitle(f"MAE Reconstruction - Epoch {epoch}", fontsize=16, y=0.995)
    plt.tight_layout()
    
    # Save figure
    save_path = Path(config.SAMPLES_DIR) / f"reconstruction_epoch_{epoch:04d}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  **Saved reconstruction sample to {save_path}")



def create_masked_image(images, mask, config):
    """
    Create masked version of images for visualization
    
    Args:
        images: (B, C, H, W)
        mask: (B, num_patches) - 1 is masked, 0 is visible
        config: configuration
    
    Returns:
        masked_images: (B, C, H, W) with masked patches set to 0
    """
    B, C, H, W = images.shape
    p = config.PATCH_SIZE
    h = w = H // p
    
    masked_images = images.copy()
    
    for b in range(B):
        mask_2d = mask[b].reshape(h, w)
        for i in range(h):
            for j in range(w):
                if mask_2d[i, j] == 1:  # Masked patch
                    masked_images[b, :, i*p:(i+1)*p, j*p:(j+1)*p] = 0
    
    return masked_images


def normalize_for_display(img):
    """
    Normalize image for display (per-channel min-max)
    
    Args:
        img: (C, H, W)
    
    Returns:
        normalized: (C, H, W) in range [0, 1] as float32
    """
    img = img.astype(np.float32)  # Ensure float32
    img_normalized = np.zeros_like(img, dtype=np.float32)
    
    for c in range(img.shape[0]):
        channel = img[c]
        min_val = channel.min()
        max_val = channel.max()
        if max_val > min_val:
            img_normalized[c] = (channel - min_val) / (max_val - min_val)
        else:
            img_normalized[c] = 0
    
    return img_normalized



# ======LOGGING=======

class TrainingLogger:
    """Simple CSV logger for training metrics"""
    
    def __init__(self, log_path):
        self.log_path = log_path
        self.fieldnames = ['epoch', 'train_loss', 'val_loss', 'learning_rate', 'time']
        
        # Create CSV file with headers
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
    
    def log(self, epoch, train_loss, val_loss, lr, elapsed_time):
        """Log training metrics"""
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({
                'epoch': epoch,
                'train_loss': f'{train_loss:.6f}',
                'val_loss': f'{val_loss:.6f}',
                'learning_rate': f'{lr:.6e}',
                'time': f'{elapsed_time:.2f}'
            })



# ======LEARNING RATE SCHEDULING=======

def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_epochs,
    num_training_epochs,
    base_lr,
    min_lr,
    steps_per_epoch
):
    """
    Create learning rate scheduler with warmup and cosine decay
    
    Args:
        optimizer: optimizer
        num_warmup_epochs: number of warmup epochs
        num_training_epochs: total number of training epochs
        base_lr: maximum learning rate
        min_lr: minimum learning rate
        steps_per_epoch: number of steps (batches) per epoch
    
    Returns:
        scheduler: learning rate scheduler
    """
    def lr_lambda(current_step):
        current_epoch = current_step / steps_per_epoch
        
        # Warmup phase
        if current_epoch < num_warmup_epochs:
            return current_epoch / num_warmup_epochs
        
        # Cosine decay phase
        progress = (current_epoch - num_warmup_epochs) / (num_training_epochs - num_warmup_epochs)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        return min_lr / base_lr + (1 - min_lr / base_lr) * cosine_decay
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return scheduler



# ======TRAINING UTILITIES=======

def get_model_size(model):
    """Calculate model size in MB"""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds):
    """Format seconds to human-readable time"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}hr"


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



# ======PROGRESS DISPLAY=======

def print_training_header():
    """Print header for training progress"""
    print("\n" + "-"*80)
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'LR':<12} {'Time':<10}")
    print("-"*80)


def print_epoch_summary(epoch, train_loss, val_loss, lr, elapsed_time):
    """Print summary for one epoch"""
    print(f"{epoch:<8} {train_loss:<12.6f} {val_loss:<12.6f} {lr:<12.6e} {format_time(elapsed_time):<10}")



# ======TESTING=======

if __name__ == "__main__":
    print("Testing utility functions...")
    
    # Test AverageMeter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"AverageMeter: {meter.avg:.2f}")
    
    # Test time formatting
    print(f"Time format: {format_time(3725)}")
    
    print("\nUtilities are working correctly.")