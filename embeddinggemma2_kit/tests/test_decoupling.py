"""The package must import and work without pyvsag installed.

This is the machine-checkable form of the "not coupled to pyvsag" requirement:
pyvsag is an optional dependency, only one module may import it, and the numpy
fallback must be reachable through an explicit opt-in.
"""

from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC = PACKAGE_ROOT / "src"
ONLY_ALLOWED_IMPORTER = "index/backends/pyvsag_hgraph.py"


def test_only_one_module_imports_pyvsag():
    """Static check: pyvsag may be imported in exactly one file."""
    offenders = []
    for path in SRC.rglob("*.py"):
        text = path.read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import pyvsag", "from pyvsag")):
                rel = path.relative_to(SRC).as_posix()
                if not rel.endswith(ONLY_ALLOWED_IMPORTER):
                    offenders.append(f"{rel}: {stripped}")
    assert not offenders, (
        "pyvsag must only be imported by "
        f"{ONLY_ALLOWED_IMPORTER}; found: {offenders}"
    )


def test_import_works_without_pyvsag():
    """Import the package in a subprocess where pyvsag is blocked."""
    code = (
        "import sys\n"
        "class Blocker:\n"
        "    def find_module(self, name, path=None):\n"
        "        return self if name == 'pyvsag' else None\n"
        "    def load_module(self, name):\n"
        "        raise ImportError('pyvsag blocked for test')\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "sys.path.insert(0, %r)\n"
        "import embeddinggemma2_kit as k\n"
        "print(k.__version__)\n"
        "from embeddinggemma2_kit.index.backends import pyvsag_hgraph\n"
        "assert pyvsag_hgraph.is_available() is False\n"
        "print('OK')\n" % str(SRC)
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, f"import failed without pyvsag:\n{result.stderr}"
    assert "OK" in result.stdout


def test_missing_backend_raises_unless_fallback():
    """Absent pyvsag, AutoIndex must refuse rather than silently downgrade."""
    from embeddinggemma2_kit import AutoIndex
    from embeddinggemma2_kit.errors import IndexBackendUnavailable
    from embeddinggemma2_kit.index.backends import pyvsag_hgraph

    if pyvsag_hgraph.is_available():
        pytest.skip("pyvsag is installed here; the refusal path cannot be tested")

    with pytest.raises(IndexBackendUnavailable) as excinfo:
        AutoIndex(dim=8)
    # The message must tell the user how to fix it.
    assert "pip install" in str(excinfo.value) or "uv venv" in str(excinfo.value)

    # Explicit opt-in downgrades instead.
    idx = AutoIndex(dim=8, fallback=True)
    assert idx.backend == "numpy_flat"


def test_numpy_backend_is_exact():
    """The fallback backend must agree with a hand-computed cosine ranking."""
    from embeddinggemma2_kit import AutoIndex

    rng = np.random.default_rng(0)
    corpus = rng.standard_normal((50, 8), dtype=np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    queries = rng.standard_normal((5, 8), dtype=np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    ids = [f"d{i}" for i in range(len(corpus))]
    idx = AutoIndex(dim=8, backend="numpy")
    idx.build(corpus, ids)

    hits = idx.search(queries, topk=5)
    for row, q in enumerate(queries):
        expected = np.argsort(-(corpus @ q))[:5]
        got = [int(h.id[1:]) for h in hits[row]]
        assert got == expected.tolist(), f"row {row}: {got} != {expected.tolist()}"
        # Scores must be similarities, i.e. match the raw dot products. A backend
        # that returned distances would produce ~0 for the best hit instead.
        true_best = (corpus @ q)[expected[0]]
        assert hits[row][0].score == pytest.approx(float(true_best), abs=1e-5)


def test_index_save_load_roundtrip(tmp_path):
    """A saved index must reload and return identical results."""
    from embeddinggemma2_kit import AutoIndex

    rng = np.random.default_rng(1)
    corpus = rng.standard_normal((40, 8), dtype=np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    query = corpus[3]

    idx = AutoIndex(dim=8, backend="numpy")
    idx.build(corpus, [f"d{i}" for i in range(len(corpus))])
    before = [h.id for h in idx.search(query, topk=5)[0]]

    idx.save(str(tmp_path))
    reloaded = AutoIndex(dim=8, backend="numpy")
    reloaded.load(str(tmp_path))
    after = [h.id for h in reloaded.search(query, topk=5)[0]]
    assert before == after

def test_matmul_ignores_spurious_blas_warnings():
    """Unit-norm scores must not emit BLAS warnings; real overflow still must.

    Accelerate on macOS arm64 raises divide/overflow/invalid from inside a
    float32 matmul whose outputs are all finite and correct. Those warnings are
    suppressed, but a genuinely non-finite result has to stay visible -- hence
    both halves of this test.
    """
    from embeddinggemma2_kit.index.backends.numpy_flat import _matmul

    unit = np.ones((4, 768), dtype=np.float32) / np.sqrt(768)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _matmul(unit, unit.T)
    assert np.isfinite(result).all()
    assert not [w for w in caught if "matmul" in str(w.message)], \
        "unit-norm matmul must not warn; scores are finite and in range"

    big = np.full((4, 8), 1e30, dtype=np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        overflowed = _matmul(big, big.T)
    assert not np.isfinite(overflowed).all()
    assert [w for w in caught if "non-finite" in str(w.message)], \
        "a genuinely non-finite matmul result must still be reported"
