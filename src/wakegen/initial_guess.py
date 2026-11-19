"""Analytic starting point for the forces, from thin actuator-disk theory.

Axial:       F_x = rho/2 (u_x^2 - U_x^2)          (total-pressure jump along a streamline)
Transverse:  F_t = rho u_disk (u_t - U_t)         (transverse momentum added to the mass flux)

with u the target velocity, U the free stream and u_disk = (U_x + u_x)/2 the
axial velocity at the disk. With an inclined free stream the disturbance made
at a node arrives in the propeller plane displaced sideways, so the target is
sampled at that displaced position (the force disk is offset by the same
amount, see ControlGrid.centre).
"""

from __future__ import annotations

import numpy as np

from .control import ControlGrid
from .wake_field import WakeField


def initial_force(grid: ControlGrid, field: WakeField, density: float, prop_plane_x: float) -> np.ndarray:
    """Force per unit area F (nr, ntheta, 3) [N/m2]."""
    free = field.freestream
    distance = prop_plane_x - grid.x0
    y, z = grid.node_yz()
    y_plane = y + free[1] / free[0] * distance
    z_plane = z + free[2] / free[0] * distance
    u = field.evaluate(y_plane, z_plane).reshape(grid.shape)

    force = np.empty(grid.shape)
    force[..., 0] = 0.5 * density * (u[..., 0] ** 2 - free[0] ** 2)
    u_disk = 0.5 * (free[0] + u[..., 0])
    force[..., 1] = density * u_disk * (u[..., 1] - free[1])
    force[..., 2] = density * u_disk * (u[..., 2] - free[2])
    return force
