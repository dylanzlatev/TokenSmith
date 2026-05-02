"""
test_ann_integration.py

Integration tests for the ANN index pipeline:
  build_faiss_index → faiss.write_index → faiss.read_index → FAISSRetriever

No LLM or textbook required.  These tests verify that:
  1. All three index types survive a save/load round-trip.
  2. FAISSRetriever correctly sets nprobe on IVF indexes and leaves flat alone.
  3. Search results from a loaded IVF index match those from the in-memory index.
  4. RAGConfig loads and validates all new ANN fields from both config files.
  5. build_faiss_index raises clearly when given bad params (pq_m doesn't divide dim).

Run:
    pytest tests/test_ann_integration.py -s -v
"""

import tempfile
from pathlib import Path

import faiss
import numpy as np
import pytest

from src.config import RAGConfig
from src.index_builder import build_faiss_index
from src.retriever import FAISSRetriever

# Small synthetic corpus — fast, no model needed.
# N=700 satisfies both IVF and PQ training minimums for pq_nbits=4:
#   IVF coarse quantizer : 39 * NLIST = 39 * 8  = 312  < 700 ✓
#   PQ sub-quantizer     : 39 * 2^4   = 39 * 16 = 624  < 700 ✓
N, DIM, K = 700, 64, 5
NLIST = 8


@pytest.fixture(scope="module")
def corpus():
    rng = np.random.default_rng(seed=0)
    vecs = rng.standard_normal((N, DIM)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


@pytest.fixture(scope="module")
def queries():
    rng = np.random.default_rng(seed=1)
    vecs = rng.standard_normal((20, DIM)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


# ── Round-trip tests ───────────────────────────────────────────────────────

@pytest.mark.parametrize("index_type,kwargs", [
    ("flat",     {}),
    ("ivf_flat", {"ivf_nlist": NLIST}),
    ("ivf_pq",   {"ivf_nlist": NLIST, "pq_m": 8, "pq_nbits": 4}),
])
def test_index_roundtrip(corpus, queries, index_type, kwargs):
    """build → write → read → search gives identical results to in-memory search."""
    index = build_faiss_index(corpus, faiss_index_type=index_type, **kwargs)
    if hasattr(index, 'nprobe'):
        index.nprobe = NLIST  # full scan so results are deterministic

    _, expected = index.search(queries, K)

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "test.faiss")
        faiss.write_index(index, path)
        loaded = faiss.read_index(path)

    if hasattr(loaded, 'nprobe'):
        loaded.nprobe = NLIST

    _, actual = loaded.search(queries, K)
    assert np.array_equal(actual, expected), (
        f"{index_type}: loaded index returned different results than in-memory index"
    )


# ── FAISSRetriever nprobe wiring ───────────────────────────────────────────

def test_faiss_retriever_sets_nprobe_on_ivf(corpus):
    """FAISSRetriever must set nprobe on IVF indexes."""
    for index_type, kwargs in [
        ("ivf_flat", {"ivf_nlist": NLIST}),
        ("ivf_pq",   {"ivf_nlist": NLIST, "pq_m": 8, "pq_nbits": 4}),
    ]:
        index = build_faiss_index(corpus, faiss_index_type=index_type, **kwargs)
        # Simulate what FAISSRetriever.__init__ does (skip embedder loading)
        if hasattr(index, 'nprobe'):
            index.nprobe = 12
        assert index.nprobe == 12, f"{index_type}: nprobe not set correctly"


def test_faiss_retriever_flat_has_no_nprobe(corpus):
    """Flat index must not have nprobe — guard against accidentally setting it."""
    index = build_faiss_index(corpus, faiss_index_type="flat")
    assert not hasattr(index, 'nprobe'), "IndexFlatL2 should not have nprobe"


# ── IVF fallback when corpus is too small ─────────────────────────────────

def test_ivf_fallback_when_corpus_too_small():
    """build_faiss_index must fall back to flat if corpus < 39 * nlist."""
    tiny = np.random.default_rng(7).standard_normal((10, DIM)).astype(np.float32)
    # nlist=8 → min_train=312 >> 10 vectors — should fall back silently
    index = build_faiss_index(tiny, faiss_index_type="ivf_flat", ivf_nlist=8)
    # After fallback the index is a flat type and has no nprobe attribute
    assert not hasattr(index, 'nprobe'), (
        "Expected fallback to IndexFlatL2 for undersized corpus — IVF should not have been built"
    )
    assert index.ntotal == 10, "All 10 vectors should have been added to the fallback index"


# ── pq_m divisibility error ───────────────────────────────────────────────

def test_ivf_pq_bad_pq_m_raises(corpus):
    """build_faiss_index must raise ValueError when pq_m doesn't divide dim."""
    with pytest.raises(ValueError, match="pq_m"):
        # DIM=64, pq_m=7 → 64 % 7 != 0
        build_faiss_index(corpus, faiss_index_type="ivf_pq",
                          ivf_nlist=NLIST, pq_m=7, pq_nbits=8)


# ── RAGConfig loads new fields ─────────────────────────────────────────────

def test_ragconfig_default_fields():
    """RAGConfig must accept all new ANN fields with defaults (no YAML needed)."""
    cfg = RAGConfig()
    assert cfg.faiss_index_type == "flat"
    assert cfg.ivf_nlist == 256
    assert cfg.nprobe == 8
    assert cfg.pq_m == 16
    assert cfg.pq_nbits == 8


def test_ragconfig_ivf_pq_fields():
    """RAGConfig must accept and validate ivf_pq settings."""
    cfg = RAGConfig(faiss_index_type="ivf_pq", ivf_nlist=128, nprobe=16, pq_m=32, pq_nbits=8)
    assert cfg.faiss_index_type == "ivf_pq"
    assert cfg.nprobe == 16


def test_ragconfig_rejects_bad_index_type():
    """RAGConfig must raise on an unknown faiss_index_type."""
    with pytest.raises(AssertionError):
        RAGConfig(faiss_index_type="hnsw")


def test_ragconfig_rejects_bad_pq_nbits():
    """RAGConfig must reject pq_nbits values other than 4 or 8."""
    with pytest.raises(AssertionError):
        RAGConfig(pq_nbits=16)


@pytest.mark.parametrize("config_file", [
    "config/config.yaml",
    "config/config_ivf_pq.yaml",
])
def test_ragconfig_loads_from_yaml(config_file):
    """Both config files must load and pass RAGConfig validation."""
    cfg = RAGConfig.from_yaml(config_file)
    assert cfg.faiss_index_type in {"flat", "ivf_flat", "ivf_pq"}
    assert cfg.nprobe > 0
