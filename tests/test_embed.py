"""The STAR macros carry a copy of java/WakeLoop.java; it must not drift."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_wakeloop_copy_is_current():
    result = subprocess.run([sys.executable, str(ROOT / "tools" / "embed_wakeloop.py"), "--check"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_properties_keep_windows_paths(tmp_path):
    from wakegen.config import read_properties

    path = tmp_path / "case.properties"
    path.write_text('\ufeff# comment\npython = "C:\\Users\\me\\python.exe"\nwakegen_path = C:\\Repo\\src\n', encoding="utf-8")
    values = read_properties(path)
    assert values["python"] == "C:\\Users\\me\\python.exe"
    assert values["wakegen_path"] == "C:\\Repo\\src"
