#!/usr/bin/env python3
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.lines import Line2D

HERE = op.dirname(op.abspath(__file__))

PT = 1.00

CSV_RESULTS = op.join(HERE, "RESULTS_SQA", "solve_qubos_sqa.csv")               # op.join(HERE, "RESULTS_NEAL", "solve_qubos_neal.csv") if plotting for neal
CSV_PARTS   = op.join(HERE, "Stats", "Datasets", "datasets1.csv")
OUT_SVG     = op.join(HERE, "RESULTS_SQA", "scores_by_particles_pt1p00GeV.svg") # op.join(HERE, "RESULTS_NEAL", "scores_by_particles_pt1p00GeV.svg") if plotting for neal

df = pd.read_csv(CSV_RESULTS, usecols=["event","ds","precision","recall"]).dropna()
parts = pd.read_csv(CSV_PARTS, usecols=["event","ds","num_important_tracks"]).rename(columns={"num_important_tracks":"n_particles"})
df = df.merge(parts, on=["event","ds"], how="inner").query("n_particles > 0")

g = (df.groupby(["event","ds"], as_index=False)[["precision","recall","n_particles"]]
        .mean().sort_values("n_particles"))
x = g["n_particles"].to_numpy(float)

def fit_band(x_obs, y_obs, deg, xs):
    p = deg + 1
    X = np.vander(x_obs, N=p, increasing=True)
    beta, *_ = np.linalg.lstsq(X, y_obs, rcond=None)
    r = y_obs - X @ beta
    s2 = float((r @ r) / max(len(x_obs) - p, 1))
    Xinv = np.linalg.inv(X.T @ X)
    Xs = np.vander(xs, N=p, increasing=True)
    ys = Xs @ beta
    se = np.sqrt(np.sum((Xs @ Xinv) * Xs, axis=1) * s2)
    return ys, se

colors = {"precision":"C0", "recall":"C1"}
plt.figure(figsize=(11.5, 7.2))
lw = 2.5

if np.unique(x).size >= 4:
    xs = np.linspace(x.min(), x.max(), 300)
    yP = g["precision"].to_numpy(float)
    ysP, seP = fit_band(x, yP, 3, xs)
    plt.plot(xs, ysP, lw=lw, ls="-", color=colors["precision"], label="Precision")
    plt.fill_between(xs, ysP - seP, ysP + seP, alpha=0.18, color=colors["precision"])
    yR = g["recall"].to_numpy(float)
    ysR, seR = fit_band(x, yR, 3, xs)
    plt.plot(xs, ysR, lw=lw, ls="-", color=colors["recall"], label="Recall")
    plt.fill_between(xs, ysR - seR, ysR + seR, alpha=0.18, color=colors["recall"])

plt.scatter(x, g["precision"], marker="o", s=55, color=colors["precision"], alpha=0.9, zorder=5)
plt.scatter(x, g["recall"],    marker="o", s=55, color=colors["recall"],    alpha=0.9, zorder=5)

right_handles = [
    Line2D([0],[0], color=colors["precision"], lw=lw, ls="-", label="Precision"),
    Line2D([0],[0], color=colors["recall"],    lw=lw, ls="-", label="Recall"),
]
plt.legend(handles=[*right_handles], labels=["Precision","Recall"], loc="lower right", frameon=True, framealpha=0.95)

plt.xlabel(rf"# particles $P_T \geq {PT:.2f}\,\mathrm{{GeV}}$ / event")
plt.ylabel("Score")
plt.ylim(0.7, 1.0)
plt.grid(alpha=0.25)
plt.title(rf"SQA: Precision & Recall vs # particles with $P_T \geq {PT:.2f}\,\mathrm{{GeV}}$")
plt.tight_layout()
plt.rcParams["svg.fonttype"] = "none"
plt.savefig(OUT_SVG, format="svg", bbox_inches="tight")
print(f"[OK] saved {OUT_SVG}")