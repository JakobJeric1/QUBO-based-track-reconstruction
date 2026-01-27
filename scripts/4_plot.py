#!/usr/bin/env python3
import os, os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = op.dirname(op.abspath(__file__))

# -------- CONFIG --------
PT = 1.00
# NEAL (Primary)
CSV_RESULTS = op.join(HERE, "RESULTS_NEAL", "solve_qubos_neal.csv")
OUT_SVG      = op.join(HERE, "RESULTS_NEAL", "scores_by_particles.svg")
TITLE_PFX    = "Neal"

# SQA (Toggle)
# CSV_RESULTS = op.join(HERE, "RESULTS_SQA", "solve_qubos_sqa.csv")
# OUT_SVG      = op.join(HERE, "RESULTS_SQA", "scores_by_particles.svg")
# TITLE_PFX    = "SQA"

CSV_PARTS = op.join(HERE, "Stats", "Datasets", "datasets1.csv")
# ------------------------

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "svg.fonttype": "none"
})

def fit_band(x_obs, y_obs, deg, xs):
    X = np.vander(x_obs, N=deg + 1, increasing=True)
    beta, *_ = np.linalg.lstsq(X, y_obs, rcond=None)
    r = y_obs - X @ beta
    s2 = float((r @ r) / max(len(x_obs) - (deg + 1), 1))
    Xinv = np.linalg.inv(X.T @ X)
    Xs = np.vander(xs, N=deg + 1, increasing=True)
    ys = Xs @ beta
    se = np.sqrt(np.sum((Xs @ Xinv) * Xs, axis=1) * s2)
    return ys, se

df = pd.read_csv(CSV_RESULTS, usecols=["event","ds","precision","recall"]).dropna()
parts = pd.read_csv(CSV_PARTS, usecols=["event","ds","num_important_tracks"]).rename(columns={"num_important_tracks":"n_particles"})
df = df.merge(parts, on=["event","ds"], how="inner").query("n_particles > 0")

g = df.groupby(["event","ds"], as_index=False)[["precision","recall","n_particles"]].mean().sort_values("n_particles")
x = g["n_particles"].to_numpy(float)

plt.figure(figsize=(11.5, 7.2))
colors = {"precision":"C0", "recall":"C1"}

if np.unique(x).size >= 4:
    xs = np.linspace(x.min(), x.max(), 300)
    for metric in ["precision", "recall"]:
        ys, se = fit_band(x, g[metric], 3, xs)
        plt.plot(xs, ys, lw=2.5, color=colors[metric])
        plt.fill_between(xs, ys - se, ys + se, alpha=0.18, color=colors[metric])

plt.scatter(x, g["precision"], marker="o", s=55, color=colors["precision"], alpha=0.9, zorder=5)
plt.scatter(x, g["recall"], marker="o", s=55, color=colors["recall"], alpha=0.9, zorder=5)

handles = [Line2D([0], [0], color=colors["precision"], lw=2.5, label="Precision"),
           Line2D([0], [0], color=colors["recall"], lw=2.5, label="Recall")]
plt.legend(handles=handles, loc="lower right", frameon=True)

plt.xlabel(rf"# particles $P_T \geq {PT:.2f}\,\mathrm{{GeV}}$ / event")
plt.ylabel("Score")
plt.ylim(0.7, 1.02)
plt.grid(alpha=0.25)
plt.title(f"{TITLE_PFX}: Precision & Recall vs Density")
plt.tight_layout()

os.makedirs(op.dirname(OUT_SVG), exist_ok=True)
plt.savefig(OUT_SVG, format="svg", bbox_inches="tight")
print(f"[OK] saved {OUT_SVG}")