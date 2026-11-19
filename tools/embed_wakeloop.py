"""Copy java/WakeLoop.java into the STAR macros that need it, as a nested static class.

STAR-CCM+ compiles a macro from one file, so the macros carry their own copy of WakeLoop
between the BEGIN/END markers. Run this after changing java/WakeLoop.java:

    python tools/embed_wakeloop.py          # rewrite the macros
    python tools/embed_wakeloop.py --check  # exit 1 if a macro is out of date (used by the tests)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "java" / "WakeLoop.java"
MACROS = [ROOT / "java" / "star" / "CalibrateWake.java"]
BEGIN = re.compile(r"^( *)// BEGIN WakeLoop.*$", re.M)
END = re.compile(r"^ *// END WakeLoop *$", re.M)


def nested_class(indent: str) -> str:
    text = SOURCE.read_text(encoding="utf-8")
    start = text.index("/**\n * Solver-independent driver")
    body = text[start:].rstrip()
    body = body.replace("public abstract class WakeLoop {", "abstract static class WakeLoop {", 1)
    return "\n".join((indent + line) if line.strip() else "" for line in body.splitlines()) + "\n"


def embedded(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    b, e = BEGIN.search(text), END.search(text)
    if not b or not e or e.start() < b.end():
        raise ValueError(f"{path}: WakeLoop markers missing")
    new = text[: b.end()] + "\n" + nested_class(b.group(1)) + text[e.start():]
    return text, new


def main(argv: list[str]) -> int:
    stale = []
    for path in MACROS:
        old, new = embedded(path)
        if old != new:
            if "--check" in argv:
                stale.append(path)
            else:
                path.write_text(new, encoding="utf-8")
                print(f"updated {path.relative_to(ROOT)}")
    for path in stale:
        print(f"out of date: {path.relative_to(ROOT)} (run python tools/embed_wakeloop.py)")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
