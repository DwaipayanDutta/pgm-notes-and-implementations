"""Execute the demo notebook end-to-end (skipped if nbclient/nbformat are missing)."""

from __future__ import annotations

from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")
pytest.importorskip("ipykernel")
pytest.importorskip("matplotlib")
pytest.importorskip("scipy")

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "pgm_handbook_demo.ipynb"


def test_demo_notebook_executes():
    nb = nbformat.read(str(NOTEBOOK), as_version=4)
    nbclient.NotebookClient(nb, timeout=600, kernel_name="python3").execute()
