from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser):
    parser.addoption("--mesh-filepath", default=Path.cwd() / "data" / "cube.1.node")
