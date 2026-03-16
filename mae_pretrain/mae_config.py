"""
Configuration file for MAE pretraining
Edit parameters here to customize your training
"""

import torch


# =======PATHS========

# Path to the SEN12MS data
DATA_ROOT = r"path/data/sen12ms"

# Path to saved normalization stats
NORM_MEAN_PATH = "path/processed_data/normalization_mean.npy"
NORM_STD_PATH = "path/processed_data/normalization_std.npy"

# Output directory for checkpoints and logs
OUTPUT_DIR = "path/mae_pretrain/mae_outputs"
CHECKPOINT_DIR = "path/mae_pretrain/mae_outputs/checkpoints"
SAMPLES_DIR = "path/mae_pretrain/mae_outputs/samples"
LOGS_DIR = "path/mae_pretrain/mae_outputs/logs"


# =======DATA PARAMETERS========

# Sensors being used
SENSORS = ['s1', 's2']

# Input channels (S1: 2 channels, S2: 13 channels)
IN_CHANNELS = 15

IMAGE_SIZE = 256

# Train/Val split
VAL_SPLIT = 0.2
RANDOM_SEED = 42


# ======MODEL ARCHITECTURE=======

# Patch size (image is divided into patches of this size)
PATCH_SIZE = 16  # 256 / 16 = 16 patches per side = 256 total patches

# Embedding dimension
EMBED_DIM = 768  # Standard ViT-Base size

# Encoder configuration
ENCODER_DEPTH = 12  # Number of transformer layers in encoder
ENCODER_NUM_HEADS = 12  # Number of attention heads
ENCODER_MLP_RATIO = 4.0  # MLP hidden dim = EMBED_DIM * MLP_RATIO

# Decoder configuration (lighter than encoder)
DECODER_EMBED_DIM = 512  # Decoder embedding dimension
DECODER_DEPTH = 8  # Number of transformer layers in decoder
DECODER_NUM_HEADS = 16  # Number of attention heads in decoder
DECODER_MLP_RATIO = 4.0

# Masking
MASK_RATIO = 0.75


# ======TRAINING PARAMETERS=======

# Number of epochs
NUM_EPOCHS = 30

# Batch size (adjust based on GPU memory)
BATCH_SIZE = 32

# Learning rate
BASE_LR = 5e-6
MIN_LR = 1e-6  

# Optimizer
WEIGHT_DECAY = 0.05
BETAS = (0.9, 0.95)

# Learning rate warmup
WARMUP_EPOCHS = 10  # Gradually increase LR for first N epochs


# ======DATALOADER PARAMETERS=======

NUM_WORKERS = 0  # Set to 0 for Windows, 4-8 for Linux/Mac
PIN_MEMORY = True  # Faster GPU transfer
DROP_LAST = True  # Drop incomplete batches


# ======CHECKPOINTING=======

# Save checkpoint every N epochs
SAVE_FREQ = 10

# Keep only the best N checkpoints (saves disk space)
KEEP_BEST_N = 3

# Save reconstruction samples every N epochs
SAMPLE_FREQ = 10


# ======DEVICE=======

# Automatically use GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Automatic mixed precision training
USE_AMP = True


# ======LOGGING=======

# Print training info every N batches
PRINT_FREQ = 10


# ======VALIDATION=======

def validate_config():
    """Validate configuration"""
    
    # Check image size is divisible by patch size
    if IMAGE_SIZE % PATCH_SIZE != 0:
        raise ValueError(f"IMAGE_SIZE ({IMAGE_SIZE}) must be divisible by PATCH_SIZE ({PATCH_SIZE})")
    
    # Check mask ratio
    if not 0 < MASK_RATIO < 1:
        raise ValueError(f"MASK_RATIO must be between 0 and 1, got {MASK_RATIO}")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, training will be VERY slow on CPU")
        print("   Consider using Google Colab or a cloud GPU")
    else:
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    return True


# ======SUMMARY=======

def print_config():
    """Print current configuration"""
    print("\nMAE TRAINING CONFIGURATION\n")
    
    print(f"\nData:")
    print(f"  Input channels: {IN_CHANNELS}")
    print(f"  Image size: {IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"  Patch size: {PATCH_SIZE}×{PATCH_SIZE}")
    print(f"  Patches per image: {(IMAGE_SIZE // PATCH_SIZE) ** 2}")
    
    print(f"\nModel:")
    print(f"  Encoder depth: {ENCODER_DEPTH} layers")
    print(f"  Encoder embedding dim: {EMBED_DIM}")
    print(f"  Decoder depth: {DECODER_DEPTH} layers")
    print(f"  Mask ratio: {MASK_RATIO:.0%}")
    
    print(f"\nTraining:")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Base learning rate: {BASE_LR}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    
    print(f"\nCheckpoints:")
    print(f"  Save every: {SAVE_FREQ} epochs")
    print(f"  Output dir: {OUTPUT_DIR}")
    
    print(f"\nDevice:")
    print(f"  Device: {DEVICE}")
    print(f"  Mixed precision: {USE_AMP}")
    
    print("-"*70 + "\n")


if __name__ == "__main__":
    validate_config()
    print_config()