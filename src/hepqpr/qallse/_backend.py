# src/hepqpr/qallse/_backend.py
import os
from .fast.qallse import Qallse as QallseFast
from .reference.qallse import Qallse as QallseReference
from .fastest.qallse import Qallse as QallseFastest

def backend():
    return os.getenv("QALLSE_BACKEND", "fast").strip().lower()

def get_qallse_backend():
    b = backend()

    if b == "fast":
        return QallseFast
    if b == "reference":
        return QallseReference
    if b == "fastest":
        return QallseFastest

    raise ValueError(f"Invalid QALLSE_BACKEND={b!r}. Use 'fast', 'fastest', or 'reference'.")
