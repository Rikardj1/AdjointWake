"""The STAR macros carry a copy of java/WakeLoop.java; it must not drift."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_wakeloop_copy_is_current():
    result = subprocess.run([sys.executable, str(ROOT / "tools" / "embed_wakeloop.py"), "--check"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
