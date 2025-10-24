#!/usr/bin/env python3
import os, os.path as op
from itertools import product
from tqdm import tqdm
from hepqpr.qallse.cli.func import DataWrapper, QallseD0, dumper, pd, time_this

HERE = op.dirname(op.abspath(__file__))

# -------- PATHS --------
DATASETS   = op.join(HERE, "DATASETS")
OUT_QUBO   = op.join(HERE, "QUBOs")
STATS_FILE = op.join(HERE, "Stats", "QUBO", "build_qubo.csv")
# -----------------------

EVENTS = [1000,1001,1002,1003,1004,1005,1006,1007,1008,1009]
DSS    = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]

os.makedirs(OUT_QUBO, exist_ok=True)
os.makedirs(op.dirname(STATS_FILE), exist_ok=True)

if not os.path.exists(STATS_FILE):
    with open(STATS_FILE, "w") as f:
        f.write("event,percent,n_doublets,n_triplets,n_qplets,q,cpu_time,wall_time\n")

for event, ds in tqdm(list(product(EVENTS, DSS)), desc="QUBOs", unit="job"):
    hits = op.join(DATASETS, f"ds{ds}", f"event00000{event}-hits.csv")
    dbls = hits.replace("-hits.csv", "-doublets.csv")

    if not (os.path.exists(hits) and os.path.exists(dbls)):
        tqdm.write(f"skip evt {event} ds{ds}: dataset missing")
        continue

    dw = DataWrapper.from_path(hits)
    doublets = pd.read_csv(dbls)

    with time_this() as tinfo:
        model = QallseD0(dw)
        model.build_model(doublets)

    Q = dumper.dump_model(
        model,
        output_path=OUT_QUBO,
        prefix=f"evt{event}-ds{ds}-",
        xplets_kwargs=dict(),
        qubo_kwargs=dict(w_marker=None, c_marker=None),
    )

    cpu, wall = tinfo
    with open(STATS_FILE, "a") as f:
        f.write(f"{event},{ds},{len(model.qubo_doublets)},{len(model.qubo_triplets)},{len(model.quadruplets)},{len(Q)},{cpu:.2f},{wall:.2f}\n")