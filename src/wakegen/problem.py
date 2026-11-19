"""One calibration case: glue between the wake, the control grid, the cost and the files.

Files in the case directory (see README for the full protocol):

    from the solver   control_cells.csv   X, Y, Z, Volume of cells around the force disk
                      target_cells.csv    X, Y, Z, Volume of cells around the propeller plane
                      achieved_cells.csv  X, Y, Z, Volume, Velocity[i,j,k] of the target cells
                      adjoint_cells.csv   X, Y, Z, Volume, adjoint momentum[i,j,k] of the control cells
    to the solver     forces_table.csv    X, Y, Z, fx, fy, fz           [N/m3]
                      target_table.csv    X, Y, Z, utx, uty, utz, wxx, wyy, wzz, wyz
                      inlet.properties    free-stream velocity vector
                      action.properties   what to solve next
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import Case
from .control import ControlGrid, regularisation
from .initial_guess import initial_force
from .objective import CostRegion, CostValue, cost, target_at_cells
from .optimizer import Options
from .tables import read_cells, read_vector_cells, write_table
from .wake_field import WakeField, polar_unit_vectors
from .wake_io import read_wake

FORCES_JSON = "wake_forces.json"
STATE = "state.npz"


class Problem:
    def __init__(self, case: Case, case_dir: str | Path = "."):
        self.case = case
        self.dir = Path(case_dir)
        wake_path = Path(case.wake_file)
        if not wake_path.is_absolute():
            wake_path = self.dir / wake_path
        self.table = read_wake(wake_path, case.wake_coord_scale, case.wake_velocity_scale)
        self.region = CostRegion.from_case(case, self.table)
        r_outer = case.ctrl_r_max_R * case.radius
        if self.region.r_max >= r_outer:
            raise ValueError(
                f"the cost region reaches r = {self.region.r_max:.4g} m but the force disk only "
                f"{r_outer:.4g} m; increase ctrl_r_max_R or check diameter / wake_coord_scale"
            )
        axial = np.array([case.ship_speed, 0.0, 0.0])
        mean = WakeField(self.table, axial, r_outer, case.wake_interp).disk_mean()
        self.wake_mean = mean
        self.freestream = np.array([case.ship_speed, mean[1], mean[2]]) if case.crossflow == "inlet" else axial
        self.field = WakeField(self.table, self.freestream, r_outer, case.wake_interp, r_hold=self.region.r_max)
        self.grid = ControlGrid.from_case(case, self.freestream)
        # optimisation variable x = Fhat * sqrt(node area / mean area), Fhat = F / (rho Vs^2)
        area = self.grid.node_area()
        self.sqrt_area = np.sqrt(area / area.mean())[..., None] * np.ones(3)
        self.force_unit = case.density * case.ship_speed**2

    # --- variables -----------------------------------------------------
    def x_from_fhat(self, fhat: np.ndarray) -> np.ndarray:
        return (fhat * self.sqrt_area).ravel()

    def fhat_from_x(self, x: np.ndarray) -> np.ndarray:
        return x.reshape(self.grid.shape) / self.sqrt_area

    @property
    def step_scale(self) -> np.ndarray:
        return (1.0 / self.sqrt_area).ravel()

    def options(self) -> Options:
        c = self.case
        return Options(
            c.lbfgs_memory, c.armijo_c1, c.noise_tol, c.max_backtracks, c.max_iterations,
            c.initial_step, c.max_step, c.tol_cost, c.tol_rel_decrease,
        )

    def initial_fhat(self, start: str = "analytic") -> np.ndarray:
        if start == "zero":
            return np.zeros(self.grid.shape)
        return initial_force(self.grid, self.field, self.case.density, self.case.prop_plane_x) / self.force_unit

    # --- to the solver --------------------------------------------------
    def write_forces_table(self, fhat: np.ndarray, cells_file: str = "control_cells.csv", out: str = "forces_table.csv") -> None:
        xyz = read_cells(self.dir / cells_file)["xyz"]
        density = self.grid.apply(fhat * self.force_unit, xyz)
        write_table(self.dir / out, ["X", "Y", "Z", "fx", "fy", "fz"], np.column_stack([xyz, density]))

    def write_target_table(self, cells_file: str = "target_cells.csv", out: str = "target_table.csv") -> int:
        """Target velocity and the symmetric weight tensor of the cost integrand per cell.

        integrand = wxx dux^2 + wyy duy^2 + wzz duz^2 + 2 wyz duy duz,  du = u - ut
        """
        c = self.case
        xyz = read_cells(self.dir / cells_file)["xyz"]
        mask = self.region.mask(xyz)
        e_r, _ = polar_unit_vectors(xyz[:, 1], xyz[:, 2])
        s, co = e_r[:, 1], e_r[:, 2]
        wt, wr = c.weight_tangential, c.weight_radial
        weights = np.column_stack([
            mask * c.weight_axial,
            mask * (wt * co**2 + wr * s**2),
            mask * (wt * s**2 + wr * co**2),
            mask * (wr - wt) * s * co,
        ])
        header = ["X", "Y", "Z", "utx", "uty", "utz", "wxx", "wyy", "wzz", "wyz"]
        write_table(self.dir / out, header, np.column_stack([xyz, target_at_cells(self.field, xyz), weights]))
        return int(mask.sum())

    def write_forces_json(self, fhat: np.ndarray, summary: dict | None = None) -> None:
        c, g = self.case, self.grid
        payload = {
            "format": "wakegen-forces-1",
            "note": "fhat = F / (density * ship_speed^2), F is force per unit disk area [N/m2], axes as in the wake file",
            "diameter": c.diameter,
            "ship_speed": c.ship_speed,
            "density": c.density,
            "prop_plane_x": c.prop_plane_x,
            "force_plane_x": g.x0,
            "sigma": g.sigma,
            "edge_taper": g.edge_taper,
            "centre": list(g.centre),
            "r_nodes": g.r_nodes.tolist(),
            "n_theta": g.n_theta,
            "freestream": self.freestream.tolist(),
            "summary": summary or {},
            "fhat": np.round(fhat, 9).tolist(),
        }
        path = self.dir / FORCES_JSON
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)

    # --- from the solver ------------------------------------------------
    def read_cost(self, achieved_file: str = "achieved_cells.csv") -> tuple[CostValue, float]:
        """Cost of the current flow solution and the volume of the cost region."""
        cells = read_vector_cells(self.dir / achieved_file, r"velocity")
        value = cost(self.case, self.region, self.field, cells["xyz"], cells["volume"], cells["vector"])
        return value, float(np.sum(cells["volume"] * self.region.mask(cells["xyz"])))

    def read_gradient(self, cost_volume: float, adjoint_file: str = "adjoint_cells.csv") -> np.ndarray:
        """d(normalised J)/d(Fhat) from the solver's adjoint momentum field."""
        c = self.case
        cells = read_vector_cells(self.dir / adjoint_file, c.adjoint_column)
        sens = c.adjoint_scale * cells["vector"]
        if c.adjoint_times_volume:
            sens = sens * cells["volume"][:, None]
        d_force = self.grid.apply_T(sens, cells["xyz"])  # d(J_solver)/dF
        return d_force * self.force_unit / (c.ship_speed**2 * cost_volume)

    def total(self, flow_cost: float, fhat: np.ndarray) -> tuple[float, float, np.ndarray]:
        """(J + R, R, dR/dFhat)."""
        reg, reg_grad = regularisation(fhat, self.case.reg_alpha, self.case.reg_beta)
        return flow_cost + reg, reg, reg_grad


def load_forces_json(path: str | Path) -> tuple[ControlGrid, np.ndarray, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format") != "wakegen-forces-1":
        raise ValueError(f"{path}: not a wakegen force file")
    grid = ControlGrid(np.array(data["r_nodes"]), int(data["n_theta"]), data["force_plane_x"], data["sigma"], data["edge_taper"], tuple(data["centre"]))
    return grid, np.array(data["fhat"]), data
