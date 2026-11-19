"""Reader for the target wake.

Format (as in ``examples/WakeEdStar.csv``): one header line ``X,Y,Z,Vx,Vy,Vz``,
then one row per point in the propeller plane. X is the shaft axis pointing
downstream, the points lie on concentric rings in the Y-Z plane around the
shaft axis, and the velocities are Cartesian components in the same axes,
normally divided by the ship speed. The file's axes are used as the
simulation's laboratory axes, so no angle or sign convention is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .tables import columns, read_table

TWO_PI = 2.0 * np.pi


@dataclass
class WakeRing:
    radius: float  # [m]
    theta: np.ndarray  # (n,) sorted, in [0, 2 pi); theta = atan2(y, z)
    velocity: np.ndarray  # (n, 3) Cartesian [m/s]


@dataclass
class WakeTable:
    rings: list[WakeRing]  # sorted by radius

    @property
    def radii(self) -> np.ndarray:
        return np.array([ring.radius for ring in self.rings])

    @property
    def r_min(self) -> float:
        return self.rings[0].radius

    @property
    def r_max(self) -> float:
        return self.rings[-1].radius

    def points(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """All table points as (y, z, velocity)."""
        y = np.concatenate([r.radius * np.sin(r.theta) for r in self.rings])
        z = np.concatenate([r.radius * np.cos(r.theta) for r in self.rings])
        u = np.concatenate([r.velocity for r in self.rings])
        return y, z, u


_WAKE_COLUMNS = {
    "x": [r"x"],
    "y": [r"y"],
    "z": [r"z"],
    "vx": [r"vx", r"u", r"velocity\[i\]"],
    "vy": [r"vy", r"v", r"velocity\[j\]"],
    "vz": [r"vz", r"w", r"velocity\[k\]"],
}


def read_wake(
    path: str | Path,
    coord_scale: float = 1.0,
    velocity_scale: float = 1.0,
    ring_tol: float = 1.0e-3,
) -> WakeTable:
    """Read the wake point cloud and group it into rings.

    ``ring_tol`` is the radius gap, relative to the largest radius, that
    separates two rings.
    """
    names, data = read_table(path)
    col = columns(names, data, _WAKE_COLUMNS, path)
    if len(data) < 3:
        raise ValueError(f"{path}: too few wake points")

    y = col["y"] * coord_scale
    z = col["z"] * coord_scale
    x = col["x"] * coord_scale
    u = np.column_stack([col["vx"], col["vy"], col["vz"]]) * velocity_scale
    r = np.hypot(y, z)
    if np.ptp(x) > 1.0e-3 * r.max():
        raise ValueError(f"{path}: wake points are not in one plane X = const")
    if r.min() <= 0.0:
        raise ValueError(f"{path}: a wake point lies on the shaft axis")

    order = np.argsort(r, kind="stable")
    breaks = np.nonzero(np.diff(r[order]) > ring_tol * r.max())[0] + 1
    rings: list[WakeRing] = []
    for members in np.split(order, breaks):
        theta = np.mod(np.arctan2(y[members], z[members]), TWO_PI)
        theta[theta > TWO_PI - 1.0e-9] = 0.0
        by_angle = np.argsort(theta, kind="stable")
        theta = theta[by_angle]
        vel = u[members][by_angle]
        theta, vel = _merge_duplicates(theta, vel)
        if len(theta) < 3:
            raise ValueError(f"{path}: ring at r = {r[members].mean():.4g} has fewer than 3 angles")
        rings.append(WakeRing(float(r[members].mean()), theta, vel))
    return WakeTable(rings)


def _merge_duplicates(theta: np.ndarray, vel: np.ndarray, tol: float = 1.0e-6):
    """Average points given twice at the same angle (e.g. 0 and 360 degrees)."""
    group = np.concatenate([[0], np.cumsum(np.diff(theta) > tol)])
    if group[-1] > 0 and theta[-1] - theta[0] > TWO_PI - tol:
        group[group == group[-1]] = 0
    count = np.bincount(group)
    theta_out = np.bincount(group, weights=theta) / count
    vel_out = np.column_stack([np.bincount(group, weights=vel[:, i]) / count for i in range(3)])
    return theta_out, vel_out
