#!/usr/bin/env python3
import os, os.path as op, pickle
from itertools import product
from tqdm import tqdm
from hepqpr.qallse.cli.func import (
    DataWrapper, process_response, diff_rows,
    solve_neal, time_this
)


HERE = op.dirname(op.abspath(__file__))


# -------- PATHS --------
DATASETS = op.join(HERE, "DATASETS")
QUBO_DIR = op.join(HERE, "QUBOs")
OUT_DIR  = op.join(HERE, "RESULTS_NEAL")
BASE_NAME = "solve_qubos_neal.csv"
# -----------------------

EVENTS = [1000,1001,1002,1003,1004,1005,1006,1007,1008,1009]
DSS    = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]

os.makedirs(OUT_DIR, exist_ok=True)

OUT_FILE = op.join(OUT_DIR, BASE_NAME)
if op.exists(OUT_FILE):
    i = 1
    while True:
        new_file = op.join(OUT_DIR, f"solve_qubos_neal({i}).csv")
        if not op.exists(new_file):
            OUT_FILE = new_file
            break
        i += 1

with open(OUT_FILE, "w") as f:
    f.write("event,ds,n_true,n_gen,n_real,n_real_all,n_fakes,"
            "precision,recall,trackml,missings,en,en0,dE,cpu_time,wall_time\n")

with tqdm(total=len(EVENTS)*len(DSS), desc="Solving QUBOs (Neal)", unit="job") as pbar:
    for event, ds in product(EVENTS, DSS):
        hits = op.join(DATASETS, f"ds{ds}", f"event00000{event}-hits.csv")
        dbls = hits.replace("-hits.csv", "-doublets.csv")
        qubo_path = op.join(QUBO_DIR, f"evt{event}-ds{ds}-qubo.pickle")

        if not all(map(op.exists, [hits, dbls, qubo_path])):
            pbar.write(f"Missing data for evt {event} ds{ds}")
            pbar.update(1)
            continue

        dw = DataWrapper.from_path(hits)
        with open(qubo_path, "rb") as f:
            Q = pickle.load(f)
        en0 = dw.compute_energy(Q)

        with time_this() as tinfo:
            response = solve_neal(Q)
            final_doublets, final_tracks = process_response(response)

        p, r, miss = dw.compute_score(final_doublets)
        trackml = dw.compute_trackml_score(final_tracks)
        en = response.record.energy[0]

        _, _, d_real = diff_rows(dw.get_real_doublets(), final_doublets)
        _, d_fakes, d_real_all = diff_rows(
            dw.get_real_doublets(with_unfocused=True), final_doublets
        )

        cpu, wall = tinfo
        with open(OUT_FILE, "a") as f:
            f.write(f"{event},{ds},{len(dw.get_real_doublets())},{len(final_doublets)},"
                    f"{len(d_real)},{len(d_real_all)},{len(d_fakes)},"
                    f"{p:.3f},{r:.3f},{trackml:.3f},{len(miss)},"
                    f"{en:.3f},{en0:.3f},{en-en0:.3f},{cpu:.2f},{wall:.2f}\n")

        pbar.set_postfix_str(f"evt {event} ds{ds} pur={p:.2f} rec={r:.2f} E={en:.2f}")
        pbar.update(1)