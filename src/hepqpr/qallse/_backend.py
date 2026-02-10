# src/hepqpr/qallse/_backend.py
import os
from .fast.qallse import Qallse as QallseFast
from .reference.qallse import Qallse as QallseReference
# Add this line, aliasing Qallse from your new fastest module
from .fastest.qallse import Qallse as QallseFastest

def get_qallse_backend():
    # This line keeps "fast" as the default, as you wanted
    b = os.getenv("QALLSE_BACKEND", "fast").strip().lower()

    if b == "fast":
        return QallseFast
    if b == "reference":
        return QallseReference
    # Add this block for the new option
    if b == "fastest":
        return QallseFastest

    # Update the error message
    raise ValueError(f"Invalid QALLSE_BACKEND={b!r}. Use 'fast', 'fastest', or 'reference'.")
