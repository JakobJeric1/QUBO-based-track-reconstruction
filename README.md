# QUBO-based Track Reconstruction

This project implements a full pipeline for reconstructing particle tracks using Quadratic Unconstrained Binary Optimization (QUBO) models.
It’s largely based on the original [hepqpr-qallse](https://github.com/derlin/hepqpr-qallse) repository, with only minimal modifications for Python 3.12 compatibility.
Additional scripts demonstrate solving QUBOs using the OpenJij simulated quantum annealer.

> Note: The original `qbsolv` backend is not included, as it is currently incompatible with Python 3.12 and no longer maintained.

## Installation

> ⚠️ Requires Python 3.12 — newer versions (e.g. 3.13) are not supported due to dependency issues.

```bash
# create and activate a virtual environment with Python 3.12 (on Linux/macOS)
python3.12 -m venv .venv
source .venv/bin/activate        # (Windows: .venv\Scripts\activate)

# upgrade pip inside the venv
python -m pip install --upgrade pip setuptools wheel

# install directly from GitHub
pip install "git+https://github.com/JakobJeric1/QUBO-based_track_reconstruction@main"
```

## Data

The repository includes a small sample of 10 events under `data/train_10_events` for quick testing and development.  
Full TrackML datasets with thousands of events are publicly available at:  
https://www.kaggle.com/c/trackml-particle-identification
