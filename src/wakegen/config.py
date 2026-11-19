"""Case configuration, read from a Java-style ``case.properties`` file.

The same file is read by the STAR-CCM+ macro (java.util.Properties) and by
Python, so there is one source of truth for a case.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path


def read_properties(path: str | Path) -> dict[str, str]:
    """Parse ``key = value`` lines. ``#`` and ``!`` start a comment line."""
    out: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line[0] in "#!":
            continue
        if "=" not in line:
            raise ValueError(f"{path}: line without '=': {raw!r}")
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def write_properties(path: str | Path, values: dict[str, object]) -> None:
    lines = [f"{k} = {v}" for k, v in values.items()]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass
class Case:
    # --- flow and geometry (SI units) ---
    diameter: float = 1.0  # propeller diameter D [m]
    ship_speed: float = 1.0  # Vs [m/s]; the wake file is V/Vs
    density: float = 998.0  # [kg/m3]
    hub_ratio: float = 0.2  # r_hub / R
    prop_plane_x: float = 0.0  # x of the propeller plane in the simulation [m]
    force_plane_x_D: float = -1.0  # force disk centre, in D relative to the propeller plane

    # --- target wake file (X,Y,Z,Vx,Vy,Vz point cloud in the propeller plane) ---
    wake_file: str = "wake.csv"
    wake_coord_scale: float = 1.0  # file coordinates * scale = metres
    wake_velocity_normalised: bool = True  # True: file holds V/Vs
    wake_interp: str = "pchip"  # "pchip" or "linear"

    # --- control (force) parametrisation ---
    ctrl_n_r: int = 24
    ctrl_n_theta: int = 72
    ctrl_r_min_R: float = 0.10  # innermost node radius / R; forces taper to 0 on the axis
    ctrl_r_max_R: float = 1.30  # outer edge of the force disk / R
    ctrl_edge_taper_R: float = 0.15  # width of the outer taper to zero / R
    ctrl_sigma_D: float = 0.03  # std of the axial Gaussian profile / D

    # --- cost function ---
    cost_r_min_R: float = -1.0  # < 0: innermost wake-table radius minus cost_margin_R
    cost_r_max_R: float = -1.0  # < 0: outermost wake-table radius plus cost_margin_R
    cost_margin_R: float = 0.03  # the cost annulus extends this far beyond the table's end rings / R
    cost_slab_thickness_D: float = 0.02  # thickness of the cell slab at the propeller plane / D
    weight_axial: float = 1.0
    weight_tangential: float = 1.0
    weight_radial: float = 1.0

    # --- uniform cross-flow part of the target ---
    crossflow: str = "inlet"  # "inlet": set by the inflow direction; "forces": forces only

    # --- optimiser ---
    reg_alpha: float = 1.0e-5  # Tikhonov weight on |F|^2
    reg_beta: float = 1.0e-4  # weight on node-to-node differences of F
    initial_step: float = 0.02  # first steepest-descent step, max change of F/(rho Vs^2)
    max_step: float = 0.20  # cap on the change of F/(rho Vs^2) per iteration
    lbfgs_memory: int = 10
    armijo_c1: float = 1.0e-4
    noise_tol: float = 0.0  # relative slack on J in the line search; 0 demands a strict decrease
    max_backtracks: int = 6
    max_iterations: int = 40
    tol_cost: float = 0.0  # stop when J <= tol_cost
    tol_rel_decrease: float = 1.0e-3  # stop when J drops by less than this fraction, twice in a row

    # --- how the solver's adjoint field maps to dJ/d(force density) ---
    # dJ/df_cell = adjoint_scale * adjoint_momentum_cell * (cell volume if adjoint_times_volume)
    # Both are fixed by the finite-difference check in phase 0.
    adjoint_scale: float = 1.0
    adjoint_times_volume: bool = True
    adjoint_column: str = ".*adjoint.*"  # regex for the adjoint momentum column names, before [i]/[j]/[k]

    # ------------------------------------------------------------------
    @property
    def radius(self) -> float:
        return 0.5 * self.diameter

    @property
    def force_plane_x(self) -> float:
        return self.prop_plane_x + self.force_plane_x_D * self.diameter

    @property
    def ctrl_sigma(self) -> float:
        return self.ctrl_sigma_D * self.diameter

    @property
    def wake_velocity_scale(self) -> float:
        return self.ship_speed if self.wake_velocity_normalised else 1.0

    @classmethod
    def from_properties(cls, path: str | Path) -> "Case":
        raw = read_properties(path)
        known = {f.name: f for f in fields(cls)}
        kwargs: dict[str, object] = {}
        for key, text in raw.items():
            if key not in known:
                continue  # keys used only by the Java macro (mesh sizes, iterations, ...)
            kind = known[key].type
            if kind in ("float", float):
                kwargs[key] = float(text)
            elif kind in ("int", int):
                kwargs[key] = int(float(text))
            elif kind in ("bool", bool):
                kwargs[key] = text.lower() in ("1", "true", "yes", "on")
            else:
                kwargs[key] = text
        case = cls(**kwargs)
        case.validate()
        return case

    def validate(self) -> None:
        if self.diameter <= 0 or self.ship_speed <= 0 or self.density <= 0:
            raise ValueError("diameter, ship_speed and density must be positive")
        if self.crossflow not in ("inlet", "forces"):
            raise ValueError("crossflow must be 'inlet' or 'forces'")
        if self.wake_interp not in ("pchip", "linear"):
            raise ValueError("wake_interp must be 'pchip' or 'linear'")
        if not 0 < self.ctrl_r_min_R < self.ctrl_r_max_R:
            raise ValueError("need 0 < ctrl_r_min_R < ctrl_r_max_R")
        if self.force_plane_x_D >= 0:
            raise ValueError("force_plane_x_D must be negative (upstream of the propeller)")
