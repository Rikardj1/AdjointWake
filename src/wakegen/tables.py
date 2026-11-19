"""CSV exchange with the flow solver.

STAR-CCM+ writes table headers such as ``"X (m)"`` or ``"Velocity[i] (m/s)"``.
Columns are therefore matched by pattern on a normalised header (quotes and
units removed, lower case) instead of by exact name.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np


def _normalise(name: str) -> str:
    name = name.strip().strip('"').strip()
    name = re.sub(r"\s*\([^)]*\)\s*$", "", name)  # trailing unit "(m/s)"
    return name.lower()


def read_table(path: str | Path) -> tuple[list[str], np.ndarray]:
    """Return (normalised header names, float array of shape (rows, cols))."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = [row for row in reader if row and any(cell.strip() for cell in row)]
    names = [_normalise(h) for h in header]
    data = np.array(rows, dtype=float) if rows else np.zeros((0, len(names)))
    if data.ndim != 2 or data.shape[1] != len(names):
        raise ValueError(f"{path}: ragged table")
    return names, data


def find_column(names: list[str], patterns: list[str], what: str, path: object = "") -> int:
    """Index of the single column whose name matches one of the regex patterns."""
    for pattern in patterns:
        hits = [i for i, n in enumerate(names) if re.fullmatch(pattern, n)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise ValueError(f"{path}: several columns match {what}: {[names[i] for i in hits]}")
    raise ValueError(f"{path}: no column for {what}; header is {names}")


_COMPONENT = {
    0: [r"\[i\]", r"\[0\]", r"[ _]?x"],
    1: [r"\[j\]", r"\[1\]", r"[ _]?y"],
    2: [r"\[k\]", r"\[2\]", r"[ _]?z"],
}


def vector_patterns(stem: str, component: int) -> list[str]:
    """Patterns for one component of a vector field, e.g. stem=r'velocity'."""
    return [stem + suffix for suffix in _COMPONENT[component]]


def columns(names: list[str], data: np.ndarray, spec: dict[str, list[str]], path: object = "") -> dict[str, np.ndarray]:
    return {key: data[:, find_column(names, pats, key, path)] for key, pats in spec.items()}


POSITION = {"x": [r"x"], "y": [r"y"], "z": [r"z"]}
VOLUME = {"volume": [r"volume", r"vol", r"cell volume"]}


def read_cells(path: str | Path) -> dict[str, np.ndarray]:
    """Cell centroids and volumes: keys xyz (n,3) and volume (n,)."""
    names, data = read_table(path)
    col = columns(names, data, POSITION | VOLUME, path)
    return {"xyz": np.column_stack([col["x"], col["y"], col["z"]]), "volume": col["volume"]}


def read_vector_cells(path: str | Path, stem: str) -> dict[str, np.ndarray]:
    """Cells plus one vector field whose column names start with regex ``stem``."""
    names, data = read_table(path)
    spec = dict(POSITION | VOLUME)
    for i, key in enumerate(("vx", "vy", "vz")):
        spec[key] = vector_patterns(stem, i)
    col = columns(names, data, spec, path)
    return {
        "xyz": np.column_stack([col["x"], col["y"], col["z"]]),
        "volume": col["volume"],
        "vector": np.column_stack([col["vx"], col["vy"], col["vz"]]),
    }


def write_table(path: str | Path, header: list[str], data: np.ndarray) -> None:
    """Plain CSV with a one-line header, readable as a STAR-CCM+ file table."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.savetxt(tmp, data, delimiter=",", header=",".join(header), comments="", fmt="%.9e")
    tmp.replace(path)  # the solver never sees a half-written table
