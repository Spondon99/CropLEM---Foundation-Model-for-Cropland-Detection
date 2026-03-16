# MAE Pretraining for Satellite Imagery

Train a Masked Autoencoder (MAE) on unlabeled SEN12MS satellite imagery to create a foundation model for cropland detection.

## Files

```
mae_pretraining/
├── mae_config.py      # Configuration and hyperparameters
├── mae_model.py       # MAE architecture
├── mae_utils.py       # Helper functions
├── mae_train.py       # Main training script (RUN THIS)
└── README.md          
```

---

## Quick Start

### Step 1: Check GPU

```python
import torch
print(torch.cuda.is_available())  # Should print True if you have GPU
print(torch.cuda.get_device_name(0))  
```

If False, install PyTorch with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Edit Configuration

Open `mae_config.py` and check these settings:

```python
# Paths (verify these match your setup)
DATA_ROOT = r"path/data/sen12ms"
NORM_MEAN_PATH = "path/processed_data/normalization_mean.npy"
NORM_STD_PATH = "path/processed_data/normalization_std.npy"

# Training
NUM_EPOCHS = 30  # Can take a very long time even with decent GPU
BATCH_SIZE = 32   # Reduce to 16 if out of memory
```

### Step 3: Run Training

```bash
python mae_train.py
```

That's it! Training will start automatically.

---

## What Happens During Training

### The MAE Process

1. **Input**: 256×256 satellite image with 15 channels (S1 + S2)
2. **Split into patches**: 16×16 patches -> 256 total patches
3. **Random masking**: Hide 75% of patches (192 masked, 64 visible)
4. **Encoder**: Process only visible 25% -> Create embeddings
5. **Decoder**: Try to reconstruct masked 75%
6. **Loss**: Compare reconstruction to original -> Update weights

### Example Output

```
----------------------------------------------------------------------
MAE PRETRAINING - STARTING
----------------------------------------------------------------------

  GPU detected: NVIDIA GeForce RTX 3060
  Memory: 12.0 GB

----------------------------------------------------------------------
PREPARING DATA
----------------------------------------------------------------------

Loading saved normalization statistics...
Discovering sensor files...
  Found 600 S1 files
  Found 600 S2 files
Matching S1 and S2 files...
  Matched 600 pairs
Splitting data (train/val = 80%/20%)...
  Training samples: 480
  Validation samples: 120

----------------------------------------------------------------------
BUILDING MODEL
----------------------------------------------------------------------

Model built successfully
  Parameters: 86,573,568
  Model size: 330.14 MB

----------------------------------------------------------------------
TRAINING
----------------------------------------------------------------------

Epoch    Train Loss   Val Loss     LR           Time      
----------------------------------------------------------------------
1        0.125463     0.118234     1.500000e-05  2.5min
  New best model! Val loss: 0.118234
  Saved reconstruction sample
10       0.089123     0.084567     1.485000e-04  2.3min
  Saved checkpoint
...
100      0.045678     0.043210     1.000000e-06  2.2min
  Saved final checkpoint
  Saved encoder

----------------------------------------------------------------------
TRAINING COMPLETED!
----------------------------------------------------------------------

Final Results:
  Best validation loss: 0.043210
  Total epochs: 100

Saved Files:
  Checkpoints: mae_outputs/checkpoints/
  Encoder: mae_outputs/checkpoints/mae_encoder.pth
  Samples: mae_outputs/samples/
  Logs: mae_outputs/logs/training_log.csv
```

---

## Output Files

### Checkpoints (mae_outputs/checkpoints/)
- **`mae_encoder.pth`** **Main output!** - Use this for downstream tasks
- `best_checkpoint.pth` - Best performing model during training
- `final_checkpoint.pth` - Last epoch checkpoint
- `checkpoint_epoch_XXXX.pth` - Periodic checkpoints (every 10 epochs)

### Reconstruction Samples (mae_outputs/samples/)
- `reconstruction_epoch_0001.png` - Shows how well model learned (epoch 1)
- `reconstruction_epoch_0010.png` - Progress at epoch 10
- `reconstruction_epoch_030.png` - Final results

These images show:
1. **Original** - Ground truth image
2. **Masked** - 75% of patches hidden
3. **Reconstructed** - Model's attempt to fill in masked areas

### Logs (mae_outputs/logs/)
- `training_log.csv` - Training metrics per epoch (loss, learning rate, time)

---

## Configuration Options

### Training Duration

```python
NUM_EPOCHS = 2   # Test training
NUM_EPOCHS = 20   # Try for decent results
NUM_EPOCHS = 40   # Best results, but much more time consuming
```

### Batch Size (Memory)

```python
BATCH_SIZE = 32   # Default (needs ~10GB GPU memory)
BATCH_SIZE = 16   # If out of memory
BATCH_SIZE = 8    # If still out of memory
```

### Model Size

**Current (ViT-Base):**
```python
EMBED_DIM = 768
ENCODER_DEPTH = 12
```

**Smaller (ViT-Small) - faster, less accurate:**
```python
EMBED_DIM = 384
ENCODER_DEPTH = 12
```

**Larger (ViT-Large) - slower, more accurate:**
```python
EMBED_DIM = 1024
ENCODER_DEPTH = 24
```

---

## Monitoring Training

### Check GPU Usage

```bash
# In another terminal while training
nvidia-smi
```

Should show ~90-100% GPU utilization and memory usage.

### Check Reconstruction Quality

Open images in `mae_outputs/samples/`:
- Early epochs: Reconstructions will be blurry/wrong
- Middle epochs: Starting to see structure
- Late epochs: Should look very similar to originals

### Check Loss Curve

Open `mae_outputs/logs/training_log.csv`:
- Training loss should decrease smoothly
- Validation loss should follow training loss
- If val_loss >> train_loss -> Overfitting

---

## Troubleshooting

### "CUDA out of memory"

**Solution 1: Reduce batch size**
```python
BATCH_SIZE = 16  # or 8
```

**Solution 2: Reduce model size**
```python
EMBED_DIM = 384
DECODER_EMBED_DIM = 256
```

### "Normalization stats not found"

Run preprocessing first:
```bash
cd ..
cd data_preprocessing
python pipeline_execute.py
cd ..
cd mae_pretraining
python mae_train.py
```

### Training is very slow

**Check GPU usage:**
```bash
nvidia-smi
```

If GPU utilization is low:
- Set `NUM_WORKERS = 4` (instead of 0) if on Linux
- Enable `USE_AMP = True` in config (should already be on)

### Loss is not decreasing

- Check reconstruction samples - are they improving?
- Try training for more epochs
- Check learning rate - might be too high or too low

---

## Expected Training Time

**On RTX 3060 (12GB):**
- 1 epoch: ~40 minutes
- 10 epochs: ~8 hours
- 30 epochs: ~22 hours

**Depends on:**
- Number of training samples
- Batch size
- Model size

---

## How to Know Training is Working

**Good signs:**
1. GPU utilization ~90-100%
2. Loss decreasing steadily
3. Reconstructions improving over epochs
4. Val loss following train loss closely

**Bad signs:**
1. !GPU utilization <50%
2. !Loss stuck or increasing
3. !Reconstructions stay blurry
4. !Val loss >> train loss (overfitting)

---

## Next Steps After Training

Once training completes, you have a **foundation model** (`mae_encoder.pth`). You can use it for downstream cropland classification.

---

## Understanding the Code

### mae_config.py
- All hyperparameters in one place
- Easy to experiment with different settings

### mae_model.py
- `PatchEmbed`: Splits image into patches
- `TransformerBlock`: Self-attention + MLP
- `ViTEncoder`: Vision Transformer encoder
- `MAEDecoder`: Reconstructs masked patches
- `MaskedAutoencoder`: Complete MAE (encoder + decoder + masking)

### mae_utils.py
- Checkpointing: Save/load models
- Visualization: Reconstruction samples
- Logging: CSV logs for metrics

### mae_train.py
- Main training loop
- Data loading
- Optimizer + scheduler setup
- Training + validation

---

## Tips

1. **Start small**: Train for 2-3 epochs first to verify everything works
2. **Monitor GPU**: Use `nvidia-smi` to check utilization
3. **Check samples**: Look at reconstructions every 10 epochs (when training fully)
4. **Save checkpoints**: Don't delete them until downstream task is done
5. **Document settings**: If you change hyperparameters, note them down

---

## What is MAE Learning?

The encoder learns to:
- Recognize patterns in satellite imagery
- Understand vegetation signatures
- Detect field boundaries
- Identify seasonal changes
- Capture texture and structure

These learned representations will make cropland classification **much easier** than training from scratch!

---
