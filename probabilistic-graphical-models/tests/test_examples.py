"""Smoke tests: every example script runs end-to-end (their asserts must hold)."""

from __future__ import annotations

import importlib.util
import runpy
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).resolve().parents[1] / "examples").glob("0*.py"))

# Examples that need an optional dependency, keyed by a substring of the file name.
_REQUIRES = {"bnn": "torch"}


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.name)
def test_example_runs(script, capsys):
    for key, module in _REQUIRES.items():
        if key in script.name and importlib.util.find_spec(module) is None:
            pytest.skip(f"{script.name} requires {module}")
    runpy.run_path(str(script), run_name="__main__")
    assert capsys.readouterr().out.strip()
