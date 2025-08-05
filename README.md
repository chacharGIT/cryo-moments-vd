# Cryo-EM Volume Distribution Analysis Framework

This repository provides a comprehensive framework for analyzing 3D molecular structures from cryo-electron microscopy data using Volume Distribution Models (VDM), statistical moments, and neural network architectures. The framework supports multiple distribution types including von Mises-Fisher (vMF) mixtures and S2 delta mixtures for modeling orientation distributions on SO(3).

## Overview

The framework addresses the cryo-EM inverse problem by representing molecular orientations as distributions over the rotation group SO(3) and using analytical moments as features for machine learning. It includes tools for real data acquisition from EMDB, synthetic data generation, spectral analysis, neural network training, and comprehensive data storage solutions.

## Key Features

### Volume Distribution Model (VDM)
- **Core Architecture**: Object-oriented representation of 3D volumes with associated SO(3) distributions
- **Distribution Support**: von Mises-Fisher mixtures and S2 delta mixtures with analytical moment computation
- **ASPIRE Integration**: Direct compatibility with ASPIRE library for volume handling and projection operations
- **Metadata Management**: Tracking of distribution parameters, shapes, and generation settings

### Data Management
- **EMDB Integration**: Automated downloading and filtering of electron microscopy maps from EMDB database
- **Real and Synthetic Data**: Tools for working with both real cryo-EM data and synthetic datasets

### Neural Networks
- **VNN Architecture**: coVariance Neural Networks based on second moment data
- **Multiple Loss Functions**: Support for Sinkhorn divergence, L2 loss, and distribution-specific losses

### Spectral Analysis
- **Eigendecomposition**: Complete framework for second moment spectral analysis with GPU acceleration
- **Visualization Tools**: Plotting capabilities for eigenvalues, eigenvectors, and distribution comparisons

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

All parameters are managed through YAML configuration files in `config/parameters.yaml`


### 1. Generate VDM from EMDB Data
```python
from src.data.vdm_generator import generate_vdm_from_volume
from src.data.emdb_downloader import download_emdb_volume

# Download and create VDM with delta mixture distribution
volume = download_emdb_volume("EMD-1234")
vdm = generate_vdm_from_volume(volume, distribution_type="s2_delta_mixture")
```

### 2. Perform Spectral Analysis (Optional)
```python
from src.utils.spectral_analysis import perform_complete_spectral_analysis

# Analyze second moment eigendecomposition
eigenvalues, eigenvectors = perform_complete_spectral_analysis(
    vdm, output_dir="./outputs/spectral_analysis"
)
```

### 3. Train Neural Network
```python
# Single data training
python src/training/train_moments_vnn_single.py

# Batch training
python src/training/train_moments_vnn_batch.py
```

## Project Structure
- `src/`: Main source code
  - `networks/`: Neural network architectures and loss functions
  - `data/`: Data generation and preparation scripts
  - `training/`: Training scripts and routines
  - `utils/`: Helper functions for moments, batching, etc.
  - `volume_distribution_model.py`: VDM object definition
- `config/`: Configuration files (YAML, Python)
- `outputs/`: Model checkpoints, logs, and results
- `requirements.txt`: Python dependencies
---
