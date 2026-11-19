"""A stand-in for the flow solver, for tests and dry runs only.

It speaks the same file protocol as the STAR-CCM+ macro (reads
``forces_table.csv`` and ``target_table.csv``, writes ``achieved_cells.csv``
and ``adjoint_cells.csv``) but replaces the flow by a simple model: the force
column above each point is smeared by a Gaussian (mixing between the force
disk and the propeller plane), carried sideways by the inflow angle, and turned
into velocity by actuator-disk relations. It is nonlinear and has an exact
adjoint, which is what the optimisation loop needs to be tested against. It is
not a prediction of what the real flow will do.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import Case, read_properties
from .tables import read_table, write_table

_CELL_HEADER = ['"X (m)"', '"Y (m)"', '"Z (m)"', '"Volume (m^3)"']
_VELOCITY = ['"Velocity[i] (m/s)"', '"Velocity[j] (m/s)"', '"Velocity[k] (m/s)"']
# Placeholder name; the real STAR-CCM+ field-function name is fixed in phase 0.
_ADJOINT = ['"Adjoint of Cost w.r.t. Momentum[i]"', '"Adjoint of Cost w.r.t. Momentum[j]"', '"Adjoint of Cost w.r.t. Momentum[k]"']


def _disk_points(r_max: float, spacing: float) -> np.ndarray:
    n = int(np.ceil(r_max / spacing))
    axis = (np.arange(-n, n) + 0.5) * spacing
    yy, zz = np.meshgrid(axis, axis, indexing="ij")
    keep = np.hypot(yy, zz) < r_max
    return np.column_stack([yy[keep], zz[keep]])


def write_cells(case: Case, case_dir: str | Path, r_cost_max: float, h_control: float, h_target: float, layers: int = 5, centre=(0.0, 0.0)) -> None:
    """Write control_cells.csv and target_cells.csv on simple Cartesian meshes."""
    case_dir = Path(case_dir)
    sigma = case.ctrl_sigma
    dx = 6.0 * sigma / layers
    yz = _disk_points(case.ctrl_r_max_R * case.radius + np.hypot(*centre) + h_control, h_control)
    x = case.force_plane_x + (np.arange(layers) + 0.5 - 0.5 * layers) * dx
    xyz = np.column_stack([np.repeat(x, len(yz)), np.tile(yz, (layers, 1))])
    write_table(case_dir / "control_cells.csv", _CELL_HEADER, np.column_stack([xyz, np.full(len(xyz), h_control**2 * dx)]))

    yz = _disk_points(r_cost_max + 2.0 * h_target, h_target)
    thickness = case.cost_slab_thickness_D * case.diameter
    xyz = np.column_stack([np.full(len(yz), case.prop_plane_x), yz])
    write_table(case_dir / "target_cells.csv", _CELL_HEADER, np.column_stack([xyz, np.full(len(yz), h_target**2 * thickness)]))


def solve(case: Case, case_dir: str | Path, blur: float, adjoint: bool = True) -> float:
    """Emulate one flow (+ adjoint) solve. Returns the un-normalised cost J."""
    case_dir = Path(case_dir)
    inlet = read_properties(case_dir / "inlet.properties")
    free = np.array([float(inlet[k]) for k in ("inlet_vx", "inlet_vy", "inlet_vz")])
    rho = case.density

    _, forces = read_table(case_dir / "forces_table.csv")
    names, cells = read_table(case_dir / "control_cells.csv")
    if not np.allclose(forces[:, :3], cells[:, :3], atol=1e-8 * case.diameter):
        raise ValueError("forces_table.csv is not on the control cells")
    volume = cells[:, 3]
    # column-integrated force per unit area: sum over x of f V, per (y, z) column
    key = np.round(cells[:, 1:3] / (1e-6 * case.diameter)).astype(np.int64)
    columns, inverse = np.unique(key, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    col_yz = np.column_stack([np.bincount(inverse, weights=cells[:, 1 + i]) for i in range(2)]) / np.bincount(inverse)[:, None]
    col_force = np.column_stack([np.bincount(inverse, weights=forces[:, 3 + i] * volume) for i in range(3)])

    _, target = read_table(case_dir / "target_table.csv")
    _, tcells = read_table(case_dir / "target_cells.csv")
    p_yz, u_target, w = target[:, 1:3], target[:, 3:6], target[:, 6:10]
    p_vol = tcells[:, 3]

    shift = free[1:] / free[0] * (case.prop_plane_x - case.force_plane_x)
    d2 = ((p_yz[:, None, :] - (col_yz + shift)[None, :, :]) ** 2).sum(axis=2)
    kernel = np.exp(-0.5 * d2 / blur**2) / (2.0 * np.pi * blur**2)  # (targets, columns) [1/m2]
    area_force = kernel @ col_force  # [N/m2]

    floor = (0.05 * free[0]) ** 2
    head = free[0] ** 2 + 2.0 * area_force[:, 0] / rho
    clipped = head < floor
    ux = np.sqrt(np.where(clipped, floor, head))
    u_disk = 0.5 * (free[0] + ux)
    u = np.column_stack([ux, free[1] + area_force[:, 1] / (rho * u_disk), free[2] + area_force[:, 2] / (rho * u_disk)])

    du = u - u_target
    integrand = w[:, 0] * du[:, 0] ** 2 + w[:, 1] * du[:, 1] ** 2 + w[:, 2] * du[:, 2] ** 2 + 2.0 * w[:, 3] * du[:, 1] * du[:, 2]
    cost = float(np.sum(p_vol * integrand))
    write_table(case_dir / "achieved_cells.csv", _CELL_HEADER + _VELOCITY, np.column_stack([tcells[:, :4], u]))

    if adjoint:
        dj_du = p_vol[:, None] * 2.0 * np.column_stack([
            w[:, 0] * du[:, 0],
            w[:, 1] * du[:, 1] + w[:, 3] * du[:, 2],
            w[:, 2] * du[:, 2] + w[:, 3] * du[:, 1],
        ])
        dux_dfx = np.where(clipped, 0.0, 1.0 / (rho * ux))
        dudisk_dfx = 0.5 * dux_dfx
        dj_df = np.column_stack([
            dj_du[:, 0] * dux_dfx
            - (dj_du[:, 1] * area_force[:, 1] + dj_du[:, 2] * area_force[:, 2]) / (rho * u_disk**2) * dudisk_dfx,
            dj_du[:, 1] / (rho * u_disk),
            dj_du[:, 2] / (rho * u_disk),
        ])
        col_adjoint = kernel.T @ dj_df  # dJ / d(sum f V) per column
        write_table(case_dir / "adjoint_cells.csv", _CELL_HEADER + _ADJOINT, np.column_stack([cells[:, :4], col_adjoint[inverse]]))
    return cost
