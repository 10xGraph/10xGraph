# ruff: noqa: INP001
import sys
from pathlib import Path


# Make `graph` and `ingest` importable when pytest runs from backend/ or from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
