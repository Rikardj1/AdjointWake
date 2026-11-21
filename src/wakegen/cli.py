"""Command line: ``python -m wakegen <command> --case case.properties``.

    info     summary of the target wake and the inflow vector
    prepare  after meshing: first force table, target table, optimiser state
    step     after each solve: read the result, write the next force table and action
    report   achieved wake against the target at the wake-table points
    apply    evaluate a finished force field on the cells of another mesh

Dry run without a flow solver (see surrogate.py; not a flow prediction):

    surrogate-cells  write control_cells.csv and target_cells.csv on a simple mesh
    surrogate-solve  stand in for one flow (and adjoint) solve
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from .config import Case, write_properties
from .optimizer import Lbfgs
from .problem import STATE, Problem, load_forces_json
from .report import build_report, write_report
from .tables import read_cells, write_table

HISTORY = "history.csv"
HISTORY_HEADER = [
    "evaluation", "iteration", "solved", "cost_total", "cost_flow", "cost_reg",
    "rms_axial", "rms_tangential", "rms_radial", "step", "next_action",
]


def _problem(args) -> Problem:
    case_path = Path(args.case)
    return Problem(Case.from_properties(case_path), case_path.parent)


def _write_action(problem: Problem, opt: Lbfgs) -> None:
    write_properties(problem.dir / "action.properties", {
        "action": opt.action,
        "iteration": opt.iteration,
        "evaluations": opt.evaluations,
        "cost": f"{opt.cost:.9e}",
        "message": opt.message,
    })


def cmd_info(args) -> int:
    problem = _problem(args)
    case, table = problem.case, problem.table
    print(f"wake file      : {case.wake_file}")
    print(f"coordinates    : file * {case.wake_scale:.6g} = metres")
    print(f"rings          : {len(table.rings)}, points per ring {[len(r.theta) for r in table.rings]}")
    print("r/R            : " + ", ".join(f"{r / case.radius:.3f}" for r in table.radii))
    mean = problem.wake_mean / case.ship_speed
    print(f"disk mean V/Vs : axial {mean[0]:.4f}, Y {mean[1]:+.4f}, Z {mean[2]:+.4f}")
    free = problem.freestream
    print(f"inflow [m/s]   : {free[0]:.6g}, {free[1]:.6g}, {free[2]:.6g}  (crossflow = {case.crossflow})")
    g = problem.grid
    print(f"force disk     : x = {g.x0:.4g} m, centre y = {g.centre[0]:+.4g} m, z = {g.centre[1]:+.4g} m, "
          f"r <= {g.r_nodes[-1]:.4g} m, {g.size} variables")
    print(f"cost region    : x = {case.prop_plane_x:.4g} m, {problem.region.r_min:.4g} <= r <= {problem.region.r_max:.4g} m")
    ratio = table.r_max / case.radius
    if not 0.8 <= ratio <= 1.5:
        print(f"WARNING: outermost wake radius is {ratio:.2f} R; check diameter and wake_diameter")
    return 0


def cmd_prepare(args) -> int:
    problem = _problem(args)
    ratio = problem.table.r_max / problem.case.radius
    if not 0.8 <= ratio <= 1.5 and not args.allow_scale:
        print(f"ERROR: the outermost wake ring is at {ratio:.2f} R (expected about 1.0-1.2 R); check diameter "
              f"and wake_diameter, or pass --allow-scale", file=sys.stderr)
        return 2
    free = problem.freestream
    write_properties(problem.dir / "inlet.properties", {
        "inlet_vx": f"{free[0]:.9e}", "inlet_vy": f"{free[1]:.9e}", "inlet_vz": f"{free[2]:.9e}",
    })
    n_cost = problem.write_target_table()
    if n_cost == 0:
        print("ERROR: no target cells inside the cost region", file=sys.stderr)
        return 2
    fhat = problem.initial_fhat(args.start)
    opt = Lbfgs(problem.x_from_fhat(fhat), problem.options(), problem.step_scale)
    problem.write_forces_table(problem.fhat_from_x(opt.x_trial))
    opt.save(problem.dir / STATE)
    with (problem.dir / HISTORY).open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(HISTORY_HEADER)
    _write_action(problem, opt)
    print(f"WAKE_PREPARED variables={problem.grid.size} cost_cells={n_cost} start={args.start} action={opt.action}")
    return 0


def cmd_step(args) -> int:
    problem = _problem(args)
    opt = Lbfgs.load(problem.dir / STATE)
    solved = opt.action
    if solved == "stop":
        _write_action(problem, opt)
        print(f"WAKE_STOP {opt.message}")
        return 0

    if args.failed:
        return _step_failed(problem, opt)

    fhat_trial = problem.fhat_from_x(opt.x_trial)
    value, cost_volume = problem.read_cost()
    total, reg, reg_grad = problem.total(value.total, fhat_trial)
    grad = None
    if solved in ("primal_adjoint", "adjoint"):
        grad = ((problem.read_gradient(cost_volume) + reg_grad) / problem.sqrt_area).ravel()
    step_taken, accepted_before = opt.step, opt.iteration
    opt.tell(total if solved != "adjoint" else None, grad)

    problem.write_forces_table(problem.fhat_from_x(opt.x_trial))
    if solved in ("primal_adjoint", "adjoint"):  # a point was accepted
        rms = value.rms()
        problem.write_forces_json(problem.fhat_from_x(opt.x), {
            "iteration": opt.iteration, "cost": opt.cost,
            "rms_axial": rms[0], "rms_tangential": rms[1], "rms_radial": rms[2],
        })
    opt.save(problem.dir / STATE)
    _write_action(problem, opt)

    rms = value.rms()
    with (problem.dir / HISTORY).open("a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow([
            opt.evaluations, accepted_before, solved, f"{total:.9e}", f"{value.total:.9e}", f"{reg:.9e}",
            f"{rms[0]:.6f}", f"{rms[1]:.6f}", f"{rms[2]:.6f}", f"{step_taken:.4e}", opt.action,
        ])
    print(
        f"WAKE_ITER {opt.iteration} solved={solved} J={total:.6e} "
        f"rms_a={rms[0]:.5f} rms_t={rms[1]:.5f} rms_r={rms[2]:.5f} next={opt.action}"
        + (f" ({opt.message})" if opt.action == "stop" else "")
    )
    return 0


def _step_failed(problem: Problem, opt: Lbfgs) -> int:
    """The flow solve of a line-search trial failed (diverged): reject it and take a shorter step."""
    if opt.action != "primal":
        print(f"ERROR: --failed only applies to a line-search trial (action 'primal'), not '{opt.action}'", file=sys.stderr)
        return 2
    step_taken, accepted_before = opt.step, opt.iteration
    opt.tell(float("inf"))
    problem.write_forces_table(problem.fhat_from_x(opt.x_trial))
    opt.save(problem.dir / STATE)
    _write_action(problem, opt)
    with (problem.dir / HISTORY).open("a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow([
            opt.evaluations, accepted_before, "primal_failed", "inf", "inf", "", "", "", "", f"{step_taken:.4e}", opt.action,
        ])
    print(f"WAKE_ITER {opt.iteration} solved=primal_failed step={step_taken:.4e} next={opt.action} step_next={opt.step:.4e}"
          + (f" ({opt.message})" if opt.action == "stop" else ""))
    return 0


def cmd_report(args) -> int:
    problem = _problem(args)
    rep = build_report(problem)
    files = write_report(problem, rep)
    rms, worst = rep["rms"], rep["max"]
    print(f"WAKE_REPORT rms_a={rms[0]:.5f} rms_t={rms[1]:.5f} rms_r={rms[2]:.5f} "
          f"max_a={worst[0]:.5f} max_t={worst[1]:.5f} max_r={worst[2]:.5f}")
    for path in files:
        print(f"  wrote {path}")
    return 0


def cmd_apply(args) -> int:
    grid, fhat, data = load_forces_json(args.forces)
    density = args.density if args.density else data["density"]
    speed = args.ship_speed if args.ship_speed else data["ship_speed"]
    if args.prop_plane_x is not None:
        grid.x0 += args.prop_plane_x - data["prop_plane_x"]
    xyz = read_cells(args.cells)["xyz"]
    keep = grid.in_zone(xyz)
    forces = grid.apply(fhat * density * speed**2, xyz[keep])
    write_table(args.out, ["X", "Y", "Z", "fx", "fy", "fz"], np.column_stack([xyz[keep], forces]))
    print(f"WAKE_APPLIED cells={int(keep.sum())} of {len(xyz)} density={density:g} ship_speed={speed:g} -> {args.out}")
    return 0 if keep.any() else 2


def cmd_surrogate_cells(args) -> int:
    from . import surrogate

    problem = _problem(args)
    radius = problem.case.radius
    surrogate.write_cells(problem.case, problem.dir, problem.region.r_max, args.h_control_R * radius,
                          args.h_target_R * radius, args.layers, problem.grid.centre)
    print("WAKE_SURROGATE cells written")
    return 0


def cmd_surrogate_solve(args) -> int:
    from . import surrogate

    problem = _problem(args)
    cost = surrogate.solve(problem.case, problem.dir, args.blur_R * problem.case.radius, adjoint=args.adjoint)
    print(f"WAKE_SURROGATE solved adjoint={args.adjoint} J_solver={cost:.6e}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wakegen", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func in (("info", cmd_info), ("prepare", cmd_prepare), ("step", cmd_step), ("report", cmd_report)):
        p = sub.add_parser(name)
        p.add_argument("--case", default="case.properties")
        p.set_defaults(func=func)
        if name == "step":
            p.add_argument("--failed", action="store_true",
                           help="the flow solve of the current line-search trial diverged: reject it, shorter step")
        if name == "prepare":
            p.add_argument("--start", choices=("analytic", "zero"), default="analytic")
            p.add_argument("--allow-scale", action="store_true",
                           help="run although the wake rings do not reach about R (diameter / wake_diameter)")
    p = sub.add_parser("apply")
    p.add_argument("--forces", default="wake_forces.json")
    p.add_argument("--cells", required=True, help="CSV with X, Y, Z, Volume of the cells around the force disk")
    p.add_argument("--out", default="forces_table.csv")
    p.add_argument("--density", type=float, default=0.0, help="override the calibration density")
    p.add_argument("--ship-speed", type=float, default=0.0, help="override the calibration ship speed")
    p.add_argument("--prop-plane-x", type=float, default=None, help="x of the propeller plane in the target simulation")
    p.set_defaults(func=cmd_apply)
    p = sub.add_parser("surrogate-cells")
    p.add_argument("--case", default="case.properties")
    p.add_argument("--h-control-R", type=float, default=0.06)
    p.add_argument("--h-target-R", type=float, default=0.03)
    p.add_argument("--layers", type=int, default=5)
    p.set_defaults(func=cmd_surrogate_cells)
    p = sub.add_parser("surrogate-solve")
    p.add_argument("--case", default="case.properties")
    p.add_argument("--blur-R", type=float, default=0.06, help="mixing length between force disk and propeller plane / R")
    p.add_argument("--adjoint", action="store_true")
    p.set_defaults(func=cmd_surrogate_solve)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
