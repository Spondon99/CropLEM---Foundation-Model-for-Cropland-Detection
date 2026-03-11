# SEN12MS Preprocessing Pipeline

A clean, organized preprocessing pipeline for SEN12MS satellite imagery dataset, preparing data for Masked Autoencoder (MAE) self-supervised learning.

## File Structure

```
CropLEM project/data preprocessing/
├── config.py                    # Configuration file
├── preprocessing_methods.py     # Utility functions
├── pipeline_execute.py          # Main execution script
├── processed_data/              # Output directory (auto-created)
│   ├── normalization_mean.npy
│   └── normalization_std.npy
└── README.md                    # This file
```

## Quick Start

### Step 1: Install Dependencies

```bash
pip install numpy rasterio scikit-learn torch torchvision tqdm
```

### Step 2: Edit Configuration

Open `config.py` and update:

```python
# Path to SEN12MS data
DATA_DIR = "path/SEN12MS/ROIs1970_summer/"

# Which sensors to use
SENSORS = ['s1', 's2']  # Both S1 and S2

# Batch size (can be adjusted based on GPU)
BATCH_SIZE = 32  # Reduce if out of memory
```

### Step 3: Run Pipeline

```bash
python pipeline_execute.py
```

That's it! The script will:
- Find all S1 and S2 files
- Match corresponding pairs
- Calculate normalization statistics
- Create train/validation split
- Prepare PyTorch DataLoaders
- Save normalization stats

## What We Get

After running the pipeline:

1. **DataLoaders ready for training**
   - `train_loader`: For MAE training
   - `val_loader`: For validation
   - Shape: `(batch_size, 15, 256, 256)` for S1+S2

2. **Normalization stats saved**
   - `processed_data/normalization_mean.npy`
   - `processed_data/normalization_std.npy`
   - **Keep these files for**:
     - MAE training
     - Downstream cropland classifier
     - Inference on new images

## Configuration Options

### Sensors

```python
SENSORS = ['s1', 's2']  # Both sensors (15 channels)
SENSORS = ['s2']        # Only Sentinel-2 (13 channels)
SENSORS = ['s1']        # Only Sentinel-1 (2 channels)
```

### Batch Size

Adjust based on GPU memory:
- 32: Good for 8-12GB GPU
- 16: Good for 6-8GB GPU
- 8: Good for 4GB GPU

### Number of Workers

```python
NUM_WORKERS = 0  # Windows (recommended)
NUM_WORKERS = 4  # Linux/Mac (faster)
```

### Validation Split

```python
VAL_SPLIT = 0.2  # 20% validation, 80% training
VAL_SPLIT = 0.1  # 10% validation, 90% training
```

## Using the DataLoaders

### In the "MAE Training" Script

```python
from pipeline_execute import main

# Get DataLoaders
train_loader, val_loader, mean, std = main()

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        # batch.shape = (32, 15, 256, 256)
        # Feed to the MAE model
        pass
```

### Loading Saved Stats Later

```python
import numpy as np

mean = np.load("processed_data/normalization_mean.npy")
std = np.load("processed_data/normalization_std.npy")
```

## Troubleshooting

### Problem: "No files found"

**Solution:** Check your `DATA_DIR` in `config.py`

```python
# Make sure this path exists and contains your data
DATA_DIR = "path/to/SEN12MS/ROIs1970_summer/"
```

### Problem: DataLoader hangs on Windows

**Solution:** Set `NUM_WORKERS = 0` in `config.py`

```python
NUM_WORKERS = 0  # Change from 4 to 0
```

### Problem: Out of GPU memory

**Solution:** Reduce batch size in `config.py`

```python
BATCH_SIZE = 16  # Or even 8
```

### Problem: "No matched pairs found"

**Solution:** Your S1 and S2 files might have different naming. Try single sensor:

```python
SENSORS = ['s2']  # Use only Sentinel-2
```

## Expected Output

When we run `pipeline_execute.py`, we should see:

```
============================================================
SEN12MS PREPROCESSING PIPELINE
============================================================

[Step 1/6] Validating configuration...
 Configuration validated successfully
 Output directory: /path/to/processed_data

[Step 2/6] Discovering sensor files...
  Searching for S1 files...
   Found 600 S1 files
  Searching for S2 files...
   Found 600 S2 files

[Step 3/6] Matching S1 and S2 files...
  Matched 600 pairs

[Step 4/6] Calculating normalization statistics...
Processing samples: 100% (100/100)
  Statistics calculated successfully
  Number of channels: 15

[Step 5/6] Splitting data into train/validation sets...
  Training samples: 480
  Validation samples: 120

[Step 6/6] Creating PyTorch datasets and dataloaders...
  Training batches: 15
  Validation batches: 4

[Validation] Testing DataLoader...
  Batch shape: torch.Size([32, 15, 256, 256])
  Batch dtype: torch.float32

============================================================
PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!
============================================================
```

## Next Steps

1. **MAE Training**: Use the DataLoaders to train our Masked Autoencoder
2. **Save the Model**: Keep the trained MAE encoder weights
3. **Downstream Task**: Use MAE embeddings for cropland detection

## Technical Details

### Channel Information

**Sentinel-1 (2 channels):**
- VV polarization
- VH polarization

**Sentinel-2 (13 channels):**
- Band 1: Coastal aerosol (443 nm)
- Band 2: Blue (490 nm)
- Band 3: Green (560 nm)
- Band 4: Red (665 nm)
- Band 5-7: Red edge
- Band 8: NIR (842 nm)
- Band 8A: Narrow NIR
- Band 9: Water vapor
- Band 10: SWIR - Cirrus
- Band 11-12: SWIR

### Data Flow

```
Raw .tif files
    ↓
File discovery & matching
    ↓
Normalization (mean=0, std=1)
    ↓
PyTorch Dataset
    ↓
DataLoader (batching + parallel loading)
    ↓
Ready for MAE training!
```

## Questions?

Common questions are answered in the troubleshooting section above. For more help, check the comments in the code files.
