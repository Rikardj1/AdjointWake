"""Continuous target velocity field built from the wake table.

Interpolation is done on the Cartesian velocity components: periodic in the
angle on each ring, then across the rings in radius. PCHIP (shape-preserving
cubic) keeps measured peaks without overshoot; "linear" is the plain fallback.

Inside the innermost ring the innermost values are held. Outside the outermost
ring the outermost values are held up to ``r_hold`` and then blend smoothly to
the free stream at ``r_blend``. Only the part inside the cost region is ever
compared with the flow; the rest just gives the starting forces a smooth edge.
"""

from __future__ import annotations

import numpy as np

from .wake_io import TWO_PI, WakeTable


def _pchip_slopes(x: np.ndarray, y: np.ndarray, periodic: bool) -> np.ndarray:
    """Fritsch-Carlson slopes along axis 0. ``x`` is (n,), ``y`` is (n, ...)."""
    n = len(x)
    shape = (n,) + (1,) * (y.ndim - 1)
    if periodic:
        h = np.mod(np.roll(x, -1) - x, TWO_PI)  # h[i] = x[i+1] - x[i], wrapped
        h[h == 0.0] = TWO_PI  # single-interval ring
        delta = (np.roll(y, -1, axis=0) - y) / h.reshape(shape)
        h_prev, d_prev = np.roll(h, 1), np.roll(delta, 1, axis=0)
        return _harmonic(h_prev.reshape(shape), h.reshape(shape), d_prev, delta)
    h = np.diff(x)
    delta = np.diff(y, axis=0) / h.reshape((n - 1,) + shape[1:])
    slopes = np.empty_like(y)
    slopes[0], slopes[-1] = delta[0], delta[-1]
    if n > 2:
        hs = h.reshape((n - 1,) + shape[1:])
        slopes[1:-1] = _harmonic(hs[:-1], hs[1:], delta[:-1], delta[1:])
    return slopes


def _harmonic(h_prev, h_next, d_prev, d_next):
    w1 = 2.0 * h_next + h_prev
    w2 = h_next + 2.0 * h_prev
    same_sign = d_prev * d_next > 0.0
    safe_prev = np.where(same_sign, d_prev, 1.0)
    safe_next = np.where(same_sign, d_next, 1.0)
    return np.where(same_sign, (w1 + w2) / (w1 / safe_prev + w2 / safe_next), 0.0)


def _hermite(t, h, y0, y1, d0, d1):
    t2, t3 = t * t, t * t * t
    return (
        (2 * t3 - 3 * t2 + 1) * y0
        + (t3 - 2 * t2 + t) * h * d0
        + (-2 * t3 + 3 * t2) * y1
        + (t3 - t2) * h * d1
    )


def _interp_ring(theta_q: np.ndarray, theta: np.ndarray, values: np.ndarray, kind: str) -> np.ndarray:
    """Periodic interpolation of ring values (n, 3) at angles theta_q (N,)."""
    n = len(theta)
    i0 = (np.searchsorted(theta, theta_q, side="right") - 1) % n
    i1 = (i0 + 1) % n
    h = np.mod(theta[i1] - theta[i0], TWO_PI)
    h[h == 0.0] = TWO_PI
    t = (np.mod(theta_q - theta[i0], TWO_PI) / h)[:, None]
    if kind == "linear":
        return (1.0 - t) * values[i0] + t * values[i1]
    slopes = _pchip_slopes(theta, values, periodic=True)
    return _hermite(t, h[:, None], values[i0], values[i1], slopes[i0], slopes[i1])


def _smoothstep(s: np.ndarray) -> np.ndarray:
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


class WakeField:
    """Target velocity u(y, z) in the propeller plane, Cartesian [m/s]."""

    def __init__(self, table: WakeTable, freestream, r_blend: float, kind: str = "pchip", r_hold: float | None = None):
        self.table = table
        self.freestream = np.asarray(freestream, dtype=float)
        self.r_hold = table.r_max if r_hold is None else max(float(r_hold), table.r_max)
        self.r_blend = float(r_blend)
        if self.r_blend <= self.r_hold:
            raise ValueError("r_blend must be larger than the outermost wake radius and r_hold")
        self.kind = kind

    def evaluate(self, y, z) -> np.ndarray:
        y = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
        z = np.atleast_1d(np.asarray(z, dtype=float)).ravel()
        table = self.table
        r = np.hypot(y, z)
        theta = np.mod(np.arctan2(y, z), TWO_PI)

        per_ring = np.stack([_interp_ring(theta, ring.theta, ring.velocity, self.kind) for ring in table.rings])
        radii = table.radii
        if len(radii) == 1:
            value = per_ring[0]
        else:
            rc = np.clip(r, radii[0], radii[-1])
            i0 = np.clip(np.searchsorted(radii, rc, side="right") - 1, 0, len(radii) - 2)
            h = (radii[i0 + 1] - radii[i0])[:, None]
            t = (rc - radii[i0])[:, None] / h
            pick = np.arange(len(r))
            v0, v1 = per_ring[i0, pick], per_ring[i0 + 1, pick]
            if self.kind == "linear":
                value = (1.0 - t) * v0 + t * v1
            else:
                slopes = _pchip_slopes(radii, per_ring, periodic=False)
                value = _hermite(t, h, v0, v1, slopes[i0, pick], slopes[i0 + 1, pick])

        blend = _smoothstep((r - self.r_hold) / (self.r_blend - self.r_hold))[:, None]
        return (1.0 - blend) * value + blend * self.freestream

    def disk_mean(self, r_min: float | None = None, r_max: float | None = None, n_r: int = 40, n_theta: int = 180):
        """Area-weighted mean velocity over the annulus [r_min, r_max]."""
        r_min = self.table.r_min if r_min is None else r_min
        r_max = self.table.r_max if r_max is None else r_max
        edges = np.linspace(r_min, r_max, n_r + 1)
        r_mid = 0.5 * (edges[:-1] + edges[1:])
        theta = (np.arange(n_theta) + 0.5) * TWO_PI / n_theta
        rr, tt = np.meshgrid(r_mid, theta, indexing="ij")
        u = self.evaluate(rr * np.sin(tt), rr * np.cos(tt)).reshape(n_r, n_theta, 3)
        weight = (r_mid * np.diff(edges))[:, None, None]
        return (u * weight).sum(axis=(0, 1)) / (weight.sum() * n_theta)


def polar_unit_vectors(y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radial and tangential unit vectors (n, 3); tangential is along increasing theta."""
    r = np.maximum(np.hypot(y, z), 1.0e-30)
    s, c = y / r, z / r
    zero = np.zeros_like(r)
    return np.column_stack([zero, s, c]), np.column_stack([zero, c, -s])
