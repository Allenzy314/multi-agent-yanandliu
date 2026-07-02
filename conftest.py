"""Pytest bootstrap: put the repo root on sys.path so the top-level packages
(`schemas`, `oracle`, `postprocess`, `agent`, `eval`) import cleanly."""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
