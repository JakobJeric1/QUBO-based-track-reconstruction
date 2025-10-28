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

## Pipeline Overview

The project follows an end-to-end workflow designed for high-energy physics track reconstruction:



### 1. Dataset Preparation
Raw TrackML hit data are parsed into *doublets* and from them into *triplets* that represent possible particle trajectory segments.

### 2. QUBO Construction
Each event is formulated as a **Quadratic Unconstrained Binary Optimization (QUBO)** problem, where each binary variable represents the inclusion or exclusion of a triplet.  
The cost function enforces smoothness and mutual exclusivity constraints, favoring physically consistent tracks.

### 3. Solver Execution
QUBOs are minimized using either:

- **Classical simulated annealing** (`dwave-neal`)  
- **Quantum-inspired simulated annealing** (`OpenJij SQA`)

Each solver explores the energy landscape to find combinations of segments forming valid tracks.

### 4. Evaluation and Scoring
The reconstructed tracks are compared against the ground truth to compute **precision** and **recall** of the doublets.
