import os

def backend() -> str:
    # default: fast
    b = os.getenv("QALLSE_BACKEND", "fast").strip().lower()
    if b in ("fast", "reference"):
        return b
    raise ValueError(f"Invalid QALLSE_BACKEND={b!r}. Use 'fast' or 'reference'.")
