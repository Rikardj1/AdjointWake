"""Compare the achieved wake with the target at the wake-table points."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .problem import Problem
from .tables import read_vector_cells
from .wake_field import polar_unit_vectors


def sample_at_points(xyz: np.ndarray, volume: np.ndarray, values: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Cell values at (y, z) points by a local linear least-squares fit.

    Cells are weighted by volume and a Gaussian of 0.6 local cell sizes. The
    linear fit removes the flattening a plain weighted mean gives in gradients.
    """
    size = np.cbrt(volume)
    out = np.empty((len(y), values.shape[1]))
    for start in range(0, len(y), 256):
        sl = slice(start, start + 256)
        dy = xyz[None, :, 1] - y[sl, None]
        dz = xyz[None, :, 2] - z[sl, None]
        d2 = dy**2 + dz**2
        h = 0.6 * size[np.argmin(d2, axis=1)][:, None]
        w = np.exp(-0.5 * d2 / h**2) * volume[None, :]
        basis = np.stack([np.ones_like(dy), dy / h, dz / h], axis=2)  # (points, cells, 3)
        normal = np.einsum("pc,pci,pcj->pij", w, basis, basis)
        rhs = np.einsum("pc,pci,cv->piv", w, basis, values)
        normal += 1.0e-9 * np.trace(normal, axis1=1, axis2=2)[:, None, None] * np.eye(3)
        out[sl] = np.linalg.solve(normal, rhs)[:, 0, :]
    return out


def build_report(problem: Problem, achieved_file: str = "achieved_cells.csv") -> dict:
    case, table = problem.case, problem.table
    cells = read_vector_cells(problem.dir / achieved_file, r"velocity")
    slab = np.abs(cells["xyz"][:, 0] - case.prop_plane_x) <= problem.region.half_thickness
    if not slab.any():
        raise ValueError("no cells in the propeller-plane slab")
    xyz, vol, vel = cells["xyz"][slab], cells["volume"][slab], cells["vector"][slab]

    y, z, target = table.points()
    achieved = sample_at_points(xyz, vol, vel, y, z)
    # The same sampling applied to the target itself: what the mesh cannot resolve of the table.
    resolved = sample_at_points(xyz, vol, problem.field.evaluate(xyz[:, 1], xyz[:, 2]), y, z)
    scale = case.wake_velocity_scale
    e_r, e_t = polar_unit_vectors(y, z)

    def components(delta):
        delta = delta / case.ship_speed
        return np.column_stack([delta[:, 0], np.einsum("nc,nc->n", delta, e_t), np.einsum("nc,nc->n", delta, e_r)])

    comp = components(achieved - target)
    floor = components(resolved - target)

    radius = np.hypot(y, z)
    rings = []
    for ring in table.rings:
        sel = np.isclose(radius, ring.radius, rtol=1.0e-6)
        rings.append({
            "r_over_R": ring.radius / case.radius,
            "rms": np.sqrt(np.mean(comp[sel] ** 2, axis=0)).tolist(),
            "max": np.max(np.abs(comp[sel]), axis=0).tolist(),
            "floor": np.sqrt(np.mean(floor[sel] ** 2, axis=0)).tolist(),
        })
    return {
        "y": y, "z": z,
        "target": target / scale, "achieved": achieved / scale,
        "error": comp,  # axial, tangential, radial, in fractions of Vs
        "rms": np.sqrt(np.mean(comp**2, axis=0)).tolist(),
        "max": np.max(np.abs(comp), axis=0).tolist(),
        "floor": np.sqrt(np.mean(floor**2, axis=0)).tolist(),
        "rings": rings,
    }


def write_report(problem: Problem, rep: dict, stem: str = "wake_achieved") -> list[Path]:
    """Achieved wake in the input format, a text summary, and plots if matplotlib is present."""
    case = problem.case
    out_dir = problem.dir
    written = []

    coords = np.column_stack([np.full_like(rep["y"], 0.0), rep["y"], rep["z"]]) / case.wake_coord_scale
    csv_path = out_dir / f"{stem}.csv"
    np.savetxt(csv_path, np.column_stack([coords, rep["achieved"]]), delimiter=",", header="X,Y,Z,Vx,Vy,Vz", comments="", fmt="%.6g")
    written.append(csv_path)

    lines = [
        "Achieved wake minus target at the wake-table points, in fractions of the ship speed.",
        "rms/max: achieved - target.  floor: rms of (target sampled from the mesh - target),",
        "i.e. the part of the table the propeller-plane mesh cannot resolve. An rms close to",
        "the floor means the mesh, not the force field, limits the match.",
        "",
        f"{'r/R':>7} {'rms Va':>8} {'rms Vt':>8} {'rms Vr':>8} {'max Va':>8} {'max Vt':>8} {'max Vr':>8} {'floor Va':>9} {'floor Vt':>9} {'floor Vr':>9}",
    ]
    for ring in rep["rings"]:
        lines.append(f"{ring['r_over_R']:7.3f} " + " ".join(f"{v:8.4f}" for v in ring["rms"] + ring["max"]) + " " + " ".join(f"{v:9.4f}" for v in ring["floor"]))
    lines.append(f"{'all':>7} " + " ".join(f"{v:8.4f}" for v in rep["rms"] + rep["max"]) + " " + " ".join(f"{v:9.4f}" for v in rep["floor"]))
    txt_path = out_dir / f"{stem}_summary.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written.append(txt_path)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return written

    names = ["Va/Vs", "Vy/Vs", "Vz/Vs"]
    fig, axes = plt.subplots(len(problem.table.rings), 3, figsize=(13, 2.2 * len(problem.table.rings)), sharex=True, squeeze=False)
    radius = np.hypot(rep["y"], rep["z"])
    theta = np.degrees(np.mod(np.arctan2(rep["y"], rep["z"]), 2 * np.pi))
    for row, ring in enumerate(problem.table.rings):
        sel = np.isclose(radius, ring.radius, rtol=1.0e-6)
        order = np.argsort(theta[sel])
        for c in range(3):
            ax = axes[row, c]
            ax.plot(theta[sel][order], rep["target"][sel][order, c], "k-", label="target")
            ax.plot(theta[sel][order], rep["achieved"][sel][order, c], "C3--", label="achieved")
            ax.grid(alpha=0.3)
            if c == 0:
                ax.set_ylabel(f"r/R = {ring.radius / case.radius:.2f}")
            if row == 0:
                ax.set_title(names[c])
    for ax in axes[-1]:
        ax.set_xlabel("theta [deg], from +Z towards +Y")
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    png_path = out_dir / f"{stem}_cuts.png"
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    written.append(png_path)
    return written
