# Cryo-EM Orientation Distribution Modeling (VDM + DPF)

This repository contains a framework for learning and analyzing **orientation distributions** in cryo-EM using:
- **Analytical moment / subspace features** derived from volumes and spherical harmonics–style representations, and
- A **diffusion-based model on S²** (DPF) that learns a score field for functions sampled on the sphere.

The codebase supports both **real EMDB volumes** and **synthetic distributions** (e.g., vMF mixtures), and provides end-to-end workflows for:
1. downloading/loading volumes,
2. generating moment/subspace datasets,
3. training score-based models (including conditional variants), and
4. sampling / inference with diffusion solvers.

> Requirements: Python 3.12, PyTorch (see `requirements.txt`).  
> Configuration lives in `config/parameters.yaml` (and `config/config.py`).

---

## Repository Layout

- `src/networks/dpf/`  
  Diffusion Probabilistic Field (DPF) components:
  - **Score network** wrapping a Perceiver-style backbone
  - **PerceiverIO** with optional **conditional cross-attention**
  - Forward diffusion utilities (noise schedules, `q_sample`, etc.)
  - Losses for score matching
  - Conditional encoders (e.g., “moment encoders” producing `cond_feat` tokens)

- `src/data/`  
  Dataset construction and IO:
  - EMDB download/load helpers (ASPIRE integration)
  - Dataset/dataloader builders for moment/subspace representations
  - Volume splits / caching in `outputs/` (paths configurable)

- `src/utils/`  
  Math + geometry utilities used across the project:
  - S² point generation (e.g., Fibonacci sphere)
  - Rotation / interpolation utilities for spherical functions
  - Spectral / moment processing helpers
  - Visualization helpers

- `src/` (core)
  - Core objects for volumes/distributions (VDM)
  - Common glue for configs, pipeline wiring, and shared primitives

---


## Installation

**Requirements**: Python 3.12 and PyTorch 2.5.1 are specifically required for this framework.

```bash
pip install -r requirements.txt
```

Note: Ensure you have Python 3.12 installed before running the installation command.

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

## Training

### DPF training
Main entrypoint:
```bash
python src/training/train_dpf.py
```

Notes:
- Distributed training uses `torchrun` (if enabled in the script/config).
- Conditional training can freeze most of the score model and train only:
  - conditional cross-attention layers,
  - conditional scales,
  - optional time-modulation (gamma/beta) modules,
  - plus the conditional encoder.

(Exact behavior depends on `settings`.)

---

## Inference / Sampling

Main entrypoint:
```bash
python src/inference/sample_from_dpf.py
```

Supported modes typically include:
- `single`: evaluate score prediction and reconstruct `x0` from `x_t`
- `diffusion`: run an iterative sampler/solver (e.g., DPM-Solver style)

Outputs:
- comparison plots of `x0`, `x_t`, predicted score, and reconstructed `x0_est`
- optional conditional-vs-unconditional comparisons

---
