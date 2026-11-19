"""Force parametrisation.

The design variables are force per unit area ``F[i, j, :]`` (three Cartesian
components) on a polar grid in the force disk. The force density applied to a
cell at (x, y, z) is

    f = g(x) * window(r) * bilinear(F)(r, theta)

where ``g`` is a Gaussian in x with unit integral and ``window`` tapers the
force to zero on the disk axis and at the outer edge. r and theta are measured
from the disk centre, which is offset from the shaft axis against the inflow's
cross-flow so that the disturbance arrives centred on the propeller disk. The map F -> f is
linear; ``apply_T`` is its exact transpose and turns a cell-wise sensitivity
into the gradient with respect to ``F``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Case
from .wake_io import TWO_PI


def _smoothstep(s: np.ndarray) -> np.ndarray:
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


@dataclass
class ControlGrid:
    r_nodes: np.ndarray  # (nr,) node radii [m], increasing, r_nodes[0] > 0
    n_theta: int
    x0: float  # centre of the force disk [m]
    sigma: float  # std of the axial Gaussian [m]
    edge_taper: float  # width of the outer taper [m]
    centre: tuple[float, float] = (0.0, 0.0)  # (y, z) of the disk axis [m]

    @classmethod
    def from_case(cls, case: Case, freestream=None) -> "ControlGrid":
        radius = case.radius
        r_nodes = np.linspace(case.ctrl_r_min_R * radius, case.ctrl_r_max_R * radius, case.ctrl_n_r)
        centre = (0.0, 0.0)
        if freestream is not None:
            drift = (case.prop_plane_x - case.force_plane_x) / freestream[0]
            centre = (-freestream[1] * drift, -freestream[2] * drift)
        return cls(r_nodes, case.ctrl_n_theta, case.force_plane_x, case.ctrl_sigma, case.ctrl_edge_taper_R * radius, centre)

    def _polar(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dy, dz = xyz[:, 1] - self.centre[0], xyz[:, 2] - self.centre[1]
        return np.hypot(dy, dz), np.mod(np.arctan2(dy, dz), TWO_PI)

    # ------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.r_nodes), self.n_theta, 3)

    @property
    def size(self) -> int:
        return len(self.r_nodes) * self.n_theta * 3

    @property
    def theta_nodes(self) -> np.ndarray:
        return np.arange(self.n_theta) * TWO_PI / self.n_theta

    def node_yz(self) -> tuple[np.ndarray, np.ndarray]:
        """Node positions (y, z) in the simulation's axes."""
        rr, tt = np.meshgrid(self.r_nodes, self.theta_nodes, indexing="ij")
        return self.centre[0] + rr * np.sin(tt), self.centre[1] + rr * np.cos(tt)

    def node_area(self) -> np.ndarray:
        """Disk area represented by each node, shape (nr, ntheta)."""
        r = self.r_nodes
        edges = np.concatenate([[0.0], 0.5 * (r[:-1] + r[1:]), [r[-1]]])
        ring = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        return np.repeat(ring[:, None] / self.n_theta, self.n_theta, axis=1)

    # ------------------------------------------------------------------
    def axial_profile(self, x: np.ndarray) -> np.ndarray:
        """Gaussian g(x) [1/m], cut off beyond three standard deviations."""
        s = (x - self.x0) / self.sigma
        g = np.exp(-0.5 * s * s) / (self.sigma * np.sqrt(TWO_PI))
        return np.where(np.abs(s) <= 3.0, g, 0.0)

    def window(self, r: np.ndarray) -> np.ndarray:
        inner = np.minimum(1.0, r / self.r_nodes[0])
        outer = 1.0 - _smoothstep((r - (self.r_nodes[-1] - self.edge_taper)) / self.edge_taper)
        return inner * outer

    def in_zone(self, xyz: np.ndarray) -> np.ndarray:
        """Cells that can receive a force."""
        r, _ = self._polar(xyz)
        return (np.abs(xyz[:, 0] - self.x0) <= 3.0 * self.sigma) & (r < self.r_nodes[-1])

    def _stencil(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Flat node indices (N, 4) and weights (N, 4) of the map F -> f."""
        nr, nt = len(self.r_nodes), self.n_theta
        r, theta = self._polar(xyz)

        rc = np.clip(r, self.r_nodes[0], self.r_nodes[-1])
        i0 = np.clip(np.searchsorted(self.r_nodes, rc, side="right") - 1, 0, nr - 2)
        tr = (rc - self.r_nodes[i0]) / (self.r_nodes[i0 + 1] - self.r_nodes[i0])

        s = theta / (TWO_PI / nt)
        j0 = np.floor(s).astype(int) % nt
        tj = s - np.floor(s)
        j1 = (j0 + 1) % nt

        scale = self.axial_profile(xyz[:, 0]) * self.window(r)
        idx = np.stack([i0 * nt + j0, i0 * nt + j1, (i0 + 1) * nt + j0, (i0 + 1) * nt + j1], axis=1)
        w = np.stack([(1 - tr) * (1 - tj), (1 - tr) * tj, tr * (1 - tj), tr * tj], axis=1)
        return idx, w * scale[:, None]

    def apply(self, force: np.ndarray, xyz: np.ndarray) -> np.ndarray:
        """Force density f (N, 3) [N/m3] at cell centroids from F (nr, nt, 3) [N/m2]."""
        idx, w = self._stencil(xyz)
        flat = force.reshape(-1, 3)
        return np.einsum("nk,nkc->nc", w, flat[idx])

    def apply_T(self, sens: np.ndarray, xyz: np.ndarray) -> np.ndarray:
        """Transpose of ``apply``: cell sensitivities (N, 3) -> gradient (nr, nt, 3)."""
        idx, w = self._stencil(xyz)
        n_nodes = len(self.r_nodes) * self.n_theta
        out = np.empty((n_nodes, 3))
        for c in range(3):
            out[:, c] = np.bincount(idx.ravel(), weights=(w * sens[:, c : c + 1]).ravel(), minlength=n_nodes)
        return out.reshape(self.shape)


# ----------------------------------------------------------------------
# Regularisation on the dimensionless force Fhat = F / (rho Vs^2)
# ----------------------------------------------------------------------
def regularisation(fhat: np.ndarray, alpha: float, beta: float) -> tuple[float, np.ndarray]:
    """R = alpha/2 mean|F|^2 + beta/2 (mean|dF/dr step|^2 + mean|dF/dtheta step|^2), and dR/dF."""
    n = fhat[..., 0].size
    value = 0.5 * alpha * np.sum(fhat**2) / n
    grad = alpha * fhat / n

    d_theta = np.roll(fhat, -1, axis=1) - fhat
    value += 0.5 * beta * np.sum(d_theta**2) / n
    grad += beta * (np.roll(d_theta, 1, axis=1) - d_theta) / n

    d_r = fhat[1:] - fhat[:-1]
    value += 0.5 * beta * np.sum(d_r**2) / n
    grad[1:] += beta * d_r / n
    grad[:-1] -= beta * d_r / n
    return float(value), grad
