"""Cost function in the propeller plane.

J = sum_cells V m [ w_a (du.e_x)^2 + w_t (du.e_theta)^2 + w_r (du.e_r)^2 ] / (Vs^2 sum V m)

with du = u - u_target and m the mask of cells inside the cost annulus. J is
the weighted mean-square velocity error in units of Vs^2, so sqrt(J) reads as
an RMS error in fractions of the ship speed. The flow solver integrates the
same integrand (without the normalisation) as its adjoint cost function.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Case
from .wake_field import WakeField, polar_unit_vectors
from .wake_io import WakeTable


@dataclass
class CostRegion:
    x: float  # propeller plane [m]
    half_thickness: float  # [m]
    r_min: float
    r_max: float

    @classmethod
    def from_case(cls, case: Case, table: WakeTable) -> "CostRegion":
        margin = case.cost_margin_R * case.radius
        r_min = max(table.r_min - margin, 0.0) if case.cost_r_min_R < 0 else case.cost_r_min_R * case.radius
        r_max = table.r_max + margin if case.cost_r_max_R < 0 else case.cost_r_max_R * case.radius
        return cls(case.prop_plane_x, 0.5 * case.cost_slab_thickness_D * case.diameter, r_min, r_max)

    def mask(self, xyz: np.ndarray) -> np.ndarray:
        r = np.hypot(xyz[:, 1], xyz[:, 2])
        inside = (np.abs(xyz[:, 0] - self.x) <= self.half_thickness) & (r >= self.r_min) & (r <= self.r_max)
        return inside.astype(float)


@dataclass
class CostValue:
    total: float  # weighted, normalised J
    axial: float  # unweighted mean-square error per component, in Vs^2
    tangential: float
    radial: float

    def rms(self) -> tuple[float, float, float]:
        return (np.sqrt(self.axial), np.sqrt(self.tangential), np.sqrt(self.radial))


def target_at_cells(field: WakeField, xyz: np.ndarray) -> np.ndarray:
    return field.evaluate(xyz[:, 1], xyz[:, 2])


def cost(case: Case, region: CostRegion, field: WakeField, xyz: np.ndarray, volume: np.ndarray, velocity: np.ndarray) -> CostValue:
    weight = volume * region.mask(xyz)
    norm = weight.sum() * case.ship_speed**2
    if norm <= 0.0:
        raise ValueError("no cells inside the cost region; check prop_plane_x and cost_slab_thickness_D")
    du = velocity - target_at_cells(field, xyz)
    e_r, e_t = polar_unit_vectors(xyz[:, 1], xyz[:, 2])
    axial = float(np.sum(weight * du[:, 0] ** 2) / norm)
    tangential = float(np.sum(weight * np.einsum("nc,nc->n", du, e_t) ** 2) / norm)
    radial = float(np.sum(weight * np.einsum("nc,nc->n", du, e_r) ** 2) / norm)
    total = case.weight_axial * axial + case.weight_tangential * tangential + case.weight_radial * radial
    return CostValue(total, axial, tangential, radial)
