import tomllib
from pathlib import Path

import provenance_api


def test_installed_version_matches_pyproject() -> None:
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text())["project"]["version"]

    assert provenance_api.__version__ == declared
