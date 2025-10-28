# QUBO-based Track Reconstruction

This project implements a full pipeline for reconstructing particle tracks using Quadratic Unconstrained Binary Optimization (QUBO) models.
It’s largely based on the original [hepqpr-qallse](https://github.com/derlin/hepqpr-qallse) repository, with only minimal modifications for Python 3.12 compatibility.
Additional scripts demonstrate solving QUBOs using the OpenJij simulated quantum annealer.

> Note: The original `qbsolv` backend is not included, as it is currently incompatible with Python 3.12 and no longer maintained.

## Installation

> ⚠️ Requires Python 3.12 — newer versions (e.g. 3.13) are not supported due to dependency issues.

```bash
# create and activate a virtual environment with Python 3.12 (Linux/macOS)
python3.12 -m venv .venv
source .venv/bin/activate        # (Windows: .venv\Scripts\activate)

# upgrade pip inside the venv
python -m pip install --upgrade pip setuptools wheel

# install directly from GitHub
pip install "git+https://github.com/JakobJeric1/QUBO-based_track_reconstruction@main"
```

## Data

The repository includes a small sample of 10 events under `data/train_10_events` for quick testing and development. Full TrackML datasets with thousands of events are publicly available at: https://www.kaggle.com/c/trackml-particle-identification

## Pipeline Overview

The algorithm performs end-to-end reconstruction of particle tracks from raw (TrackML) detector hits using a QUBO-based optimization approach.
Below you can see a quick overview of the main algorithmic steps, which are briefly described in the sections that follow. Steps **1–4** correspond
directly to the core scripts in the [`scripts/`](scripts) folder, each representing one conceptual stage of the workflow.

<p align="center">
  <img src="https://github.com/user-attachments/assets/b206146d-8844-4071-9d16-e8d430e37eb7" width="700" alt="QUBO pipeline diagram">
</p>

### 1. Dataset Preparation – [`make_datasets.py`](scripts/1_make_datasets.py)
The process begins with transforming raw detector hits into meaningful geometric relationships. In this step, nearby hits are connected into
*doublets* and then combined into *triplets*, which represent short, locally consistent fragments of potential particle trajectories.
This forms the building blocks from which complete tracks will later be reconstructed.

### 2. QUBO Construction – [`build_qubos.py`](scripts/2_build_qubos.py)
Each event is then expressed as a **QUBO problem** — a mathematical formulation where every triplet becomes a binary decision variable.
The optimization goal balances two competing ideas:
it rewards smooth, physically plausible track continuations, and penalizes conflicting or overlapping segments.
The result is a compact representation of the entire tracking problem as an energy landscape that can be explored by solvers.

### 3. Solver Execution – [`solve_qubos_neal.py`](scripts/3a_solve_qubos_neal.py) and [`solve_qubos_sqa.py`](scripts/3b_solve_qubos_sqa.py)
At this stage, specialized annealing algorithms are used to search the energy landscape for the lowest-energy configurations.
These correspond to the most consistent combinations of triplets — in other words, the most likely particle tracks.
Two solvers are compared: a classical simulated annealer (`dwave-neal`) and a quantum-inspired one (`OpenJij SQA`), allowing us to study how both approaches perform on the same problem.

### 4. Evaluation and Scoring – [`plot_scores.py`](scripts/4_plot_scores.py)

Finally, the reconstructed tracks are compared against the known ground truth from the simulated detector data.
Precision and recall metrics quantify how accurately the algorithm identifies true tracks while avoiding false ones.
To understand the algorithm’s behavior across different conditions, results are evaluated for **10 separate events** at multiple **dataset densities**.
The plots below summarize how reconstruction quality changes with event complexity and solver type.

<p align="center">
  <img src="https://github.com/user-attachments/assets/f91f7b5a-aa84-43b8-8137-384cea7d42a9" width="48%" alt="Scoring plot SA">
  <img src="https://github.com/user-attachments/assets/362fdaec-2285-41bc-aa31-6ffb91f53043" width="48%" alt="Scoring plot SQA">
</p>

<p align="center"><em>Figure 2 — Precision and recall for ten TrackML events at varying dataset densities, comparing classical simulated annealing (SA, left) and quantum-inspired simulated annealing (SQA, right).</em></p>
