"""WakeLoop.java driving the Python side, with the surrogate as the solver."""

import csv
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from wakegen.config import read_properties

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(shutil.which("javac") is None, reason="no JDK")


def dry_run(case_dir, tmp_path, *extra):
    text = (case_dir / "case.properties").read_text().replace("max_iterations = 40", "max_iterations = 5")
    (case_dir / "case.properties").write_text(text)
    build = tmp_path / "build"
    subprocess.run(["javac", "-d", str(build), str(ROOT / "java/WakeLoop.java"), str(ROOT / "java/DryRunLoop.java")], check=True, capture_output=True)
    return subprocess.run(
        ["java", "-cp", str(build), "DryRunLoop", str(case_dir / "case.properties"), sys.executable, str(ROOT / "src"), "0.08", *extra],
        capture_output=True, text=True, cwd=tmp_path,
    )


def test_java_loop_runs_to_stop_and_reports(case_dir, tmp_path):
    done = dry_run(case_dir, tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "WAKE_REPORT" in done.stdout
    last = [line for line in done.stdout.splitlines() if line.startswith("WAKE_DONE")][-1]
    assert "adjoint=6 " in last  # the starting point plus one per accepted iteration
    assert True and "max_iterations reached" in last
    assert read_properties(case_dir / "action.properties")["action"] == "stop"
    assert (case_dir / "wake_achieved.csv").exists() and (case_dir / "wake_forces.json").exists()


def test_a_diverged_trial_is_rejected_and_the_loop_goes_on(case_dir, tmp_path):
    done = dry_run(case_dir, tmp_path, "3")  # primal 1 = starting point, 2-3 = line-search trials
    assert done.returncode == 0, done.stdout + done.stderr
    failed = [line for line in done.stdout.splitlines() if line.startswith("WAKE_PRIMAL_FAILED")]
    assert len(failed) == 1 and "non-finite residual" in failed[0] and "second line" not in failed[0]
    rows = list(csv.DictReader((case_dir / "history.csv").open()))
    bad = [i for i, row in enumerate(rows) if row["solved"] == "primal_failed"]
    assert len(bad) == 1 and rows[bad[0]]["next_action"] == "primal"
    assert float(rows[bad[0] + 1]["step"]) == pytest.approx(0.5 * float(rows[bad[0]]["step"]))
    assert "WAKE_REPORT" in done.stdout and read_properties(case_dir / "action.properties")["action"] == "stop"


def test_a_diverged_starting_point_stops_the_loop(case_dir, tmp_path):
    done = dry_run(case_dir, tmp_path, "1")
    assert done.returncode != 0 and "non-finite residual" in done.stderr
