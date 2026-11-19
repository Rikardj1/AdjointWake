import shutil
from pathlib import Path

import pytest

from wakegen.config import Case

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def case_dir(tmp_path):
    """A case directory with the example wake and case file."""
    shutil.copy(EXAMPLES / "WakeEdStar.csv", tmp_path)
    shutil.copy(EXAMPLES / "case.properties", tmp_path)
    return tmp_path


@pytest.fixture
def case(case_dir):
    return Case.from_properties(case_dir / "case.properties")
