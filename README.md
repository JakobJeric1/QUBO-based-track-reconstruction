# QUBO-based Track Reconstruction


This repository contains an implementation of a particle track reconstruction pipeline
based on Quadratic Unconstrained Binary Optimization (QUBO) models.

All core ideas, reconstruction logic, and pipeline structure originate from the original
[hepqpr-qallse](https://github.com/derlin/hepqpr-qallse) project, and full credit for the
algorithmic design goes to its original author.
This repository does not introduce a new tracking method, but provides a modified version
of the original source code that is compatible with Python 3.12.

> Note: The original `qbsolv` backend is not included, as it is currently incompatible with Python 3.12 and no longer maintained.

The modified source code introduces a clear separation between two interchangeable
implementations: a *reference* backend and a *fast* backend.
The reference backend closely follows the behaviour and structure of the original
implementation, while the fast backend provides a functionally equivalent alternative
with improved runtime characteristics.

The differences between the two backends are most pronounced in the QUBO construction
phase, which remains the dominant computational bottleneck of the overall pipeline.
While the underlying model and constraints are unchanged, the fast backend reorganizes
parts of the [source](src/hepqpr/qallse/fast) code to reduce overhead and improve execution time.


A comparison of wall-clock runtimes for the QUBO construction step shows that the fast
backend achieves approximately a fourfold speedup compared to the reference
implementation, while producing equivalent results.

<p align="center">
  <img src="https://github.com/user-attachments/assets/8ff312bf-fc33-49f8-a5a4-34c2ff53967d" width="70%" alt="QUBO build time comparison">
</p>

<p align="center"><em>
Wall-clock time comparison of the QUBO construction step for the reference and fast backends, red dots represent full events.
</em></p>

## Installation

> ⚠️ **Requirement:** Python 3.12 is mandatory. Versions 3.13+ are currently unsupported due to dependency constraints.

### Full Project & Benchmarking
This setup clones the entire repository—including the research scripts and sample data—and installs the package in editable mode within a dedicated virtual environment.

```bash
# Clone and enter repo
git clone https://github.com/JakobJeric1/QUBO-based_track_reconstruction.git && cd QUBO-based_track_reconstruction

# Create and activate venv
python3.12 -m venv venv
source venv/bin/activate              # Windows: .\venv\Scripts\activate

# Install package in editable mode
pip install --upgrade pip setuptools wheel && pip install -e .
```

### Standard Library Install

Installs the reconstruction engine directly as a dependency. This is intended for users who want to import the tracking logic into their own external projects.

```bash
pip install "git+https://github.com/JakobJeric1/QUBO-based_track_reconstruction@main"
```

## Configuration & Data

### Backend Control

The system features two interchangeable backends. The fast backend (default) is optimized for runtime efficiency, while the reference backend provides strict parity with the original codebase. Switch between them using environment variables:

```bash
# Set to reference logic
export QALLSE_BACKEND=reference       # Windows: $env:QALLSE_BACKEND="reference"

# Revert to optimized logic (default)
export QALLSE_BACKEND=fast            # Windows: $env:QALLSE_BACKEND="fast"
```

### TrackML Datasets

### TrackML Datasets
The pipeline scripts expect detector hits located in the `data/` directory.

* **Fetch Sample Data (10 events):** If you installed the library only but need the included samples for testing, use this one-liner to pull the data folder:
```bash
git clone --depth 1 https://github.com/JakobJeric1/QUBO-based_track_reconstruction.git tmp && cp -r tmp/data ./data && rm -rf tmp
```
* **Full Datasets**: For large-scale benchmarks, download the official TrackML dataset from Kaggle [TrackML Particle Tracking Challange](https://www.kaggle.com/competitions/trackml-particle-identification/data)

## Pipeline Overview

The algorithm performs end-to-end reconstruction of particle tracks from raw detector hits using a QUBO-based optimization approach. The process is divided into four main stages, each handled by the scripts in the [`scripts/`](scripts) folder.

<p align="center">
  <img src="https://github.com/user-attachments/assets/5bd2068b-5ea7-4e5e-89f0-1a4461e93ef8" width="700" alt="QUBO pipeline diagram">
</p>



### 1. Dataset Preparation – [`1_make_datasets.py`](scripts/1_make_datasets.py)
This stage transforms raw TrackML data into sub-sampled events at varying densities. It applies a transverse momentum cut ($P_T \geq 1.0$ GeV) to focus on high-momentum particles and identifies the initial geometric connections (doublets) between hits. This creates a manageable set of candidates for the reconstruction pipeline.

### 2. QUBO Construction – [`2_build_QUBO.py`](scripts/2_build_QUBO.py)
The tracking problem is formulated as a mathematical optimization model. Potential track segments (triplets) are represented as binary variables. The script defines an energy landscape where physically smooth trajectories are assigned rewards (lower energy) and conflicting or unphysical overlaps are penalized.

### 3. Solver Execution – [`3a_solve_neal.py`](scripts/3a_solve_neal.py) & [`3b_solve_sqa.py`](scripts/3b_solve_sqa.py)
The energy landscape is explored using two different annealing techniques to find the global minimum—representing the most likely set of real particle tracks. 
* **Neal** uses classical Simulated Annealing (SA), relying on thermal fluctuations to reach the ground state. 
* **OpenJij (SQA)** uses Simulated Quantum Annealing, employing simulated quantum tunneling to traverse energy barriers. 
Both scripts output detailed tracking metrics and energy statistics for comparison.

### 4. Evaluation and Plotting – [`4_plot.py`](scripts/4_plot.py)
The final step aggregates the solver results and evaluates them against the ground truth. It calculates Precision (purity) and Recall (efficiency) and generates visualizations using a 3rd-degree polynomial fit. This allows for a clear comparison of how each solver handles increasing event complexity and particle density.

<p align="center">
  <img src="https://github.com/user-attachments/assets/f91f7b5a-aa84-43b8-8137-384cea7d42a9" width="48%" alt="Scoring plot SA">
  <img src="https://github.com/user-attachments/assets/362fdaec-2285-41bc-aa31-6ffb91f53043" width="48%" alt="Scoring plot SQA">
</p>

<p align="center"><em>Figure 2 — Comparison of Precision and Recall for ten TrackML events at varying densities using Simulated Annealing (left) and Simulated Quantum Annealing (right). Red dots represent full events. </em></p>
