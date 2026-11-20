"""The whole file protocol, with the surrogate standing in for STAR-CCM+."""

import csv

import numpy as np
import pytest

from wakegen import cli, surrogate
from wakegen.config import Case, read_properties
from wakegen.problem import Problem, load_forces_json
from wakegen.tables import read_table


def make_case(case_dir, **overrides):
    text = (case_dir / "case.properties").read_text()
    text += "".join(f"\n{key} = {value}" for key, value in overrides.items()) + "\n"
    (case_dir / "case.properties").write_text(text)
    case = Case.from_properties(case_dir / "case.properties")
    problem = Problem(case, case_dir)
    radius = case.radius
    surrogate.write_cells(case, case_dir, problem.region.r_max, 0.10 * radius, 0.08 * radius, layers=3, centre=problem.grid.centre)
    return case, problem


def run(case_dir, *args):
    return cli.main([*args, "--case", str(case_dir / "case.properties")])


def loop(case, case_dir, blur):
    solves = []
    while (action := read_properties(case_dir / "action.properties")["action"]) != "stop":
        surrogate.solve(case, case_dir, blur, adjoint=action in ("primal_adjoint", "adjoint"))
        solves.append(action)
        assert run(case_dir, "step") == 0
    surrogate.solve(case, case_dir, blur, adjoint=False)  # final flow at the best point
    return solves


def test_gradient_from_adjoint_files_matches_finite_differences(case_dir):
    case, problem = make_case(case_dir)
    blur = 0.08 * case.radius
    assert run(case_dir, "prepare") == 0
    fhat = problem.initial_fhat()

    def flow_cost(f):
        problem.write_forces_table(f)
        surrogate.solve(case, case_dir, blur, adjoint=False)
        return problem.read_cost()[0].total

    problem.write_forces_table(fhat)
    surrogate.solve(case, case_dir, blur, adjoint=True)
    _, cost_volume = problem.read_cost()
    grad = problem.read_gradient(cost_volume)

    rng = np.random.default_rng(3)
    for _ in range(3):
        direction = rng.normal(size=fhat.shape)
        eps = 1e-5
        fd = (flow_cost(fhat + eps * direction) - flow_cost(fhat - eps * direction)) / (2 * eps)
        assert np.sum(grad * direction) == pytest.approx(fd, rel=2e-4)


def test_analytic_start_is_close_when_there_is_little_mixing(case_dir):
    case, problem = make_case(case_dir)
    assert run(case_dir, "prepare", "--start", "analytic") == 0
    surrogate.solve(case, case_dir, 0.06 * case.radius, adjoint=False)
    start = problem.read_cost()[0]
    problem.write_forces_table(np.zeros(problem.grid.shape))
    surrogate.solve(case, case_dir, 0.06 * case.radius, adjoint=False)
    nothing = problem.read_cost()[0]
    assert start.total < 0.05 * nothing.total
    assert max(start.rms()) < 0.03


def test_loop_from_zero_forces_reaches_the_target(case_dir):
    case, problem = make_case(case_dir, max_iterations=15)
    blur = 0.08 * case.radius
    assert run(case_dir, "prepare", "--start", "zero") == 0
    solves = loop(case, case_dir, blur)

    assert solves[0] == "primal_adjoint" and solves.count("adjoint") == 15
    with (case_dir / "history.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(solves)
    accepted = [float(r["cost_total"]) for r in rows if r["solved"] != "primal"]
    assert np.all(np.diff(accepted) <= 0)
    assert accepted[-1] < 0.05 * accepted[0]

    final = problem.read_cost()[0]
    assert final.total == pytest.approx(float(rows[-1]["cost_flow"]), rel=1e-9)  # solver state is the best point
    assert max(final.rms()) < 0.02

    # the stored force field reproduces the table the solver last received
    grid, fhat, data = load_forces_json(case_dir / "wake_forces.json")
    _, sent = read_table(case_dir / "forces_table.csv")
    np.testing.assert_allclose(grid.apply(fhat * case.density * case.ship_speed**2, sent[:, :3]), sent[:, 3:], atol=1e-4 * np.abs(sent[:, 3:]).max())
    assert data["summary"]["iteration"] == 15

    assert run(case_dir, "step") == 0  # a further call after stop is harmless
    assert read_properties(case_dir / "action.properties")["action"] == "stop"


def test_wrong_adjoint_sign_is_detected_and_forces_are_left_at_the_start(case_dir):
    case, problem = make_case(case_dir, adjoint_scale=-1.0)
    assert run(case_dir, "prepare") == 0
    _, start = read_table(case_dir / "forces_table.csv")
    solves = loop(case, case_dir, 0.08 * case.radius)
    result = read_properties(case_dir / "action.properties")
    assert "line search failed" in result["message"] and result["iteration"] == "0"
    assert solves.count("adjoint") == 0
    _, end = read_table(case_dir / "forces_table.csv")
    np.testing.assert_allclose(end, start, atol=1e-6 * np.abs(start[:, 3:]).max())


def test_report_writes_the_achieved_wake_in_the_input_format(case_dir):
    case, problem = make_case(case_dir, max_iterations=4)
    assert run(case_dir, "prepare") == 0
    loop(case, case_dir, 0.08 * case.radius)
    assert run(case_dir, "report") == 0
    names, achieved = read_table(case_dir / "wake_achieved.csv")
    _, wake = read_table(case_dir / "WakeEdStar.csv")
    assert names == ["x", "y", "z", "vx", "vy", "vz"] and achieved.shape == wake.shape
    np.testing.assert_allclose(np.sort(np.hypot(achieved[:, 1], achieved[:, 2])), np.sort(np.hypot(wake[:, 1], wake[:, 2])), atol=1e-5)
    assert np.sqrt(np.mean((np.sort(achieved[:, 3]) - np.sort(wake[:, 3])) ** 2)) < 0.05  # V/Vs, like the input
    assert 0.4 < achieved[:, 3].min() and achieved[:, 3].max() < 1.05
    summary = (case_dir / "wake_achieved_summary.txt").read_text()
    assert "floor" in summary and summary.count("\n") >= 12


def test_apply_puts_the_forces_on_another_mesh_and_scales_with_speed(case_dir, tmp_path):
    case, problem = make_case(case_dir, max_iterations=2)
    assert run(case_dir, "prepare") == 0
    loop(case, case_dir, 0.08 * case.radius)
    _, cells = read_table(case_dir / "control_cells.csv")
    other = cells[::3].copy()
    other[:, 0] += 2.0  # the other simulation has its propeller plane at x = 2
    far = other[:5].copy()
    far[:, 0] += 5.0  # cells outside the force zone are dropped
    other_file = tmp_path / "other_cells.csv"
    np.savetxt(other_file, np.vstack([other, far]), delimiter=",", header="X,Y,Z,Volume", comments="")
    out = tmp_path / "forces_other.csv"
    args = ["apply", "--forces", str(case_dir / "wake_forces.json"), "--cells", str(other_file), "--out", str(out), "--prop-plane-x", "2.0"]
    assert cli.main(args + ["--ship-speed", str(2 * case.ship_speed)]) == 0
    _, applied = read_table(out)
    _, original = read_table(case_dir / "forces_table.csv")
    keep = problem.grid.in_zone(cells[::3, :3])
    assert len(applied) == keep.sum()
    np.testing.assert_allclose(applied[:, 3:], 4.0 * original[::3][keep][:, 3:], atol=1e-4 * np.abs(original[:, 3:]).max())


def test_clear_errors_for_bad_input(case_dir, capsys):
    (case_dir / "case.properties").write_text((case_dir / "case.properties").read_text().replace("wake_diameter = 1.0", "wake_diameter = 0"))
    (case_dir / "case.properties").write_text((case_dir / "case.properties").read_text().replace("diameter = 0.944", "diameter = 0.5"))
    assert run(case_dir, "info") == 2
    assert "force disk" in capsys.readouterr().err
    assert cli.main(["info", "--case", str(case_dir / "missing.properties")]) == 2
