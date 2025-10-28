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

The algorithm performs end-to-end reconstruction of particle tracks from raw detector hits using a QUBO-based optimization approach.  
Below you can see a quick overview of the main algorithmic steps, which are briefly described in the sections that follow.  
Steps **1–4** correspond directly to the main scripts included in the [`scripts/`](scripts) folder, each implementing one stage of the pipeline.

<p align="center">
  <img src="https://github.com/user-attachments/assets/b206146d-8844-4071-9d16-e8d430e37eb7" width="700" alt="QUBO pipeline diagram">
</p>

### 1. Dataset Preparation – [`make_datasets.py`](scripts/make_datasets.py)
Generates reduced TrackML datasets at multiple hit densities.  
For each event, the script builds *doublets* and *triplets* representing possible particle segments and saves them in `scripts/DATASETS/`.  
It also records key statistics (number of hits, tracks, and noise) in a CSV file for later analysis.

### 2. QUBO Construction – [`build_qubos.py`](scripts/build_qubos.py)
Transforms each prepared dataset into a **Quadratic Unconstrained Binary Optimization (QUBO)** model.  
Every triplet becomes a binary variable, and the objective function encodes geometric smoothness and mutual exclusion between incompatible triplets.  
The generated QUBOs are stored as `.pickle` files under `scripts/QUBOs/` along with metadata describing the problem size.

### 3. Solver Execution – [`solve_qubos_neal.py`](scripts/solve_qubos_neal.py) and [`solve_qubos_sqa.py`](scripts/solve_qubos_sqa.py)
Solves the QUBO instances using two different approaches:
- **Classical simulated annealing** via `dwave-neal`  
- **Quantum-inspired simulated annealing** via `OpenJij SQA`

Each solver searches the energy landscape for optimal combinations of triplets forming physically consistent tracks.  
Results (precision, recall, TrackML score, and runtime) are saved to `RESULTS_NEAL/` or `RESULTS_SQA/`.

### 4. Evaluation and Scoring – [`plot_scores.py`](scripts/plot_scores.py)
Aggregates solver outputs and computes average precision and recall as functions of the number of reconstructed particles per event.  
Generates publication-quality plots (e.g., *Precision & Recall vs # particles*) stored as `.svg` figures in the results folders.


<p align="center">
  <img src="https://github.com/user-attachments/assets/f91f7b5a-aa84-43b8-8137-384cea7d42a9" width="48%" alt="Track reconstruction result">
  <img src="https://github.com/user-attachments/assets/362fdaec-2285-41bc-aa31-6ffb91f53043" width="48%" alt="Scoring plot">
</p>

