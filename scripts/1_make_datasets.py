#!/usr/bin/env python3
import os, os.path as op
from itertools import product
from tqdm import tqdm
import logging
from hepqpr.qallse.dsmaker import create_dataset
from hepqpr.qallse.cli.func import time_this


HERE = op.dirname(op.abspath(__file__))
ROOT = op.abspath(op.join(HERE, ".."))


# -------- PATHS --------
TRACKML_TRAIN = op.join(ROOT, "data", "train_10_events")
DATASETS_OUT  = op.join(HERE, "DATASETS")
STATS_CSV     = op.join(HERE, "Stats", "Datasets", "datasets1.csv")
# -----------------------

EVENTS = [1000,1001,1002,1003,1004,1005,1006,1007,1008,1009]
DSS    = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]

os.makedirs(DATASETS_OUT, exist_ok=True)
os.makedirs(op.dirname(STATS_CSV), exist_ok=True)
if not op.exists(STATS_CSV):
    with open(STATS_CSV, "w") as f:
        f.write("event,ds,seed,num_hits,num_tracks,num_important_tracks,num_noise,cpu_time,wall_time\n")

logging.getLogger("hepqpr").setLevel(logging.WARNING)
logging.getLogger("hepqpr.qallse").setLevel(logging.WARNING)

jobs = list(product(EVENTS, DSS))
pbar = tqdm(total=len(jobs), desc="Datasets", unit="job")

for event, ds in jobs:
    seed = event*100 + ds
    hits_csv = op.join(TRACKML_TRAIN, f"event00000{event}-hits.csv")

    if not op.exists(hits_csv):
        pbar.write(f"skip evt {event} ds{ds}: missing hits")
        pbar.update(1)
        continue

    try:
        with time_this() as t:
            metas, _ = create_dataset(
                density=ds/100.0,
                input_path=hits_csv,
                output_path=DATASETS_OUT,
                prefix=f"ds{ds}",
                min_hits_per_track=5,
                high_pt_cut=1.0,
                random_seed=seed,
                double_hits_ok=False,
                gen_doublets=True,
            )
        cpu, wall = t
        with open(STATS_CSV, "a") as f:
            f.write(f"{event},{ds},{seed},"
                    f"{metas.get('num_hits',0)},{metas.get('num_tracks',0)},"
                    f"{metas.get('num_important_tracks',0)},{metas.get('num_noise',0)},"
                    f"{cpu:.2f},{wall:.2f}\n")

        pbar.set_postfix_str(f"evt={event} ds={ds} hits={metas.get('num_hits',0)}")
    except Exception as e:
        pbar.write(f"error evt {event} ds{ds}: {e}")

    pbar.update(1)

pbar.close()