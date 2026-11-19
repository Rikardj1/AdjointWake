"""WakeLoop.java driving the Python side, with the surrogate as the solver."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from wakegen.config import read_properties

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(shutil.which("javac") is None, reason="no JDK")


def test_java_loop_runs_to_stop_and_reports(case_dir, tmp_path):
    text = (case_dir / "case.properties").read_text().replace("max_iterations = 40", "max_iterations = 5")
    (case_dir / "case.properties").write_text(text)
    build = tmp_path / "build"
    subprocess.run(["javac", "-d", str(build), str(ROOT / "java/WakeLoop.java"), str(ROOT / "java/DryRunLoop.java")], check=True, capture_output=True)
    done = subprocess.run(
        ["java", "-cp", str(build), "DryRunLoop", str(case_dir / "case.properties"), sys.executable, str(ROOT / "src"), "0.08"],
        capture_output=True, text=True, cwd=tmp_path,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "WAKE_REPORT" in done.stdout
    last = [line for line in done.stdout.splitlines() if line.startswith("WAKE_DONE")][-1]
    assert "adjoint=6 " in last  # the starting point plus one per accepted iteration
    assert True and "max_iterations reached" in last
    assert read_properties(case_dir / "action.properties")["action"] == "stop"
    assert (case_dir / "wake_achieved.csv").exists() and (case_dir / "wake_forces.json").exists()
