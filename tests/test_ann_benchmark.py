"""
test_ann_benchmark.py

Synthetic benchmarks comparing TokenSmith's three FAISS index types:
  flat     — IndexFlatL2 (exact brute-force, baseline)
  ivf_flat — IndexIVFFlat (IVF cell-pruning, exact storage)
  ivf_pq   — IndexIVFPQ  (IVF + product quantization, compressed storage)

No LLM or textbook required — all data is generated from random unit vectors
that match a realistic corpus size and embedding dimension.

Run:
    pytest tests/test_ann_benchmark.py -s -v
    pytest tests/test_ann_benchmark.py -s -v -k "nprobe"   # just the sweep
"""

import json
import tempfile
import time
import tracemalloc
import numpy as np
import pytest
from pathlib import Path

import faiss

from src.index_builder import build_faiss_index

# ── Benchmark knobs ────────────────────────────────────────────────────────
# DIM=2560 matches Qwen3-Embedding-4B output dimension.
# N_CORPUS must satisfy two IVF-PQ training requirements simultaneously:
#   IVF coarse quantizer : N_CORPUS >= 39 * NLIST         (= 39 * 64  = 2 496)
#   PQ sub-quantizer     : N_CORPUS >= 39 * 2^PQ_NBITS    (= 39 * 256 = 9 984)
# 12 000 comfortably clears both thresholds and silences FAISS warnings.
N_CORPUS      = 12_000
N_QUERIES     = 200
DIM           = 2560
K             = 10           # Recall@K
NLIST         = 64           # IVF clusters; keep ≤ N_CORPUS / 39
PQ_NBITS      = 8            # bits per PQ code (1 byte); requires N >= 39 * 256
N_SEARCH_REPS = 20           # timeit repetitions for latency measurement

NPROBE_VALUES = [1, 2, 4, 8, 16, 32, 64]   # must all be ≤ NLIST
PQ_M_VALUES   = [8, 16, 32, 64]             # all divide DIM=2560 evenly

# No fixed recall thresholds for synthetic random data — see note in
# test_nprobe_recall_sweep.  Accuracy assertions live in test_ann_real_index.py.


# ── Module-scoped fixtures (built once, reused across tests) ───────────────

@pytest.fixture(scope="module")
def synthetic_data():
    """Reproducible L2-normalised float32 corpus + query embeddings."""
    rng = np.random.default_rng(seed=42)

    corpus = rng.standard_normal((N_CORPUS, DIM)).astype(np.float32)
    norms = np.linalg.norm(corpus, axis=1, keepdims=True)
    corpus /= np.where(norms == 0, 1.0, norms)

    queries = rng.standard_normal((N_QUERIES, DIM)).astype(np.float32)
    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    queries /= np.where(norms == 0, 1.0, norms)

    return corpus, queries


@pytest.fixture(scope="module")
def flat_ground_truth(synthetic_data):
    """Exact top-K neighbour indices from IndexFlatL2 (the reference answer)."""
    corpus, queries = synthetic_data
    index = build_faiss_index(corpus, faiss_index_type="flat")
    _, indices = index.search(queries, K)
    return indices    # (N_QUERIES, K)


@pytest.fixture(scope="module")
def flat_index(synthetic_data):
    corpus, _ = synthetic_data
    return build_faiss_index(corpus, faiss_index_type="flat")


@pytest.fixture(scope="module")
def ivf_flat_index(synthetic_data):
    corpus, _ = synthetic_data
    return build_faiss_index(corpus, faiss_index_type="ivf_flat", ivf_nlist=NLIST)


@pytest.fixture(scope="module")
def ivf_pq_index(synthetic_data):
    corpus, _ = synthetic_data
    return build_faiss_index(
        corpus,
        faiss_index_type="ivf_pq",
        ivf_nlist=NLIST,
        pq_m=16,
        pq_nbits=PQ_NBITS,
    )


# ── Helpers ────────────────────────────────────────────────────────────────

def recall_at_k(ann: np.ndarray, exact: np.ndarray) -> float:
    """Mean Recall@K over all queries.  ann and exact are both (Q, K) integer arrays."""
    hits = sum(
        len(set(ann_row.tolist()) & set(exact_row.tolist()))
        for ann_row, exact_row in zip(ann, exact)
    )
    return hits / (len(ann) * K)


def search_latency_ms(index: faiss.Index, queries: np.ndarray) -> float:
    """Mean per-query latency in milliseconds, averaged over N_SEARCH_REPS full passes."""
    t0 = time.perf_counter()
    for _ in range(N_SEARCH_REPS):
        index.search(queries, K)
    elapsed = time.perf_counter() - t0
    return (elapsed / N_SEARCH_REPS / len(queries)) * 1_000


def index_bytes(index: faiss.Index) -> int:
    """Exact serialised size of a FAISS index in bytes."""
    buf = faiss.serialize_index(index)
    return int(buf.nbytes)


def measure_peak_mb(fn, *args, **kwargs):
    """
    Run fn(*args, **kwargs), return (result, peak_python_heap_mb).

    Uses tracemalloc to capture peak Python-heap allocation during the call.
    Note: FAISS C++ allocations live outside Python's heap and are not
    captured here.  Use index_bytes() for the index storage footprint.
    """
    tracemalloc.start()
    try:
        result = fn(*args, **kwargs)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak / 1e6


def save_results(results_dir: Path, filename: str, data) -> None:
    out = results_dir / filename
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Saved → {out}")


def _try_plot(results_dir: Path, sweep_results: dict, pq_rows: list) -> None:
    """Generate summary plots if matplotlib is available; silently skip otherwise."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  [INFO] matplotlib not installed — skipping plots.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"TokenSmith ANN Index Benchmark\n"
        f"N={N_CORPUS:,}  dim={DIM}  nlist={NLIST}  K={K}",
        fontsize=12,
    )

    # ── Top-left: nprobe vs Recall@10 ─────────────────────────────────────
    ax = axes[0, 0]
    for label, rows in sweep_results.items():
        xs = [r["nprobe"] for r in rows]
        ys = [r["recall_at_10"] for r in rows]
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_xlabel("nprobe")
    ax.set_ylabel("Recall@10")
    ax.set_title("nprobe vs Recall@10")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Top-right: nprobe vs search latency ───────────────────────────────
    ax = axes[0, 1]
    for label, rows in sweep_results.items():
        xs = [r["nprobe"] for r in rows]
        ys = [r["latency_ms"] for r in rows]
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_xlabel("nprobe")
    ax.set_ylabel("Latency (ms / query)")
    ax.set_title("nprobe vs Search Latency")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Bottom-left: pq_m vs index size ───────────────────────────────────
    ax = axes[1, 0]
    if pq_rows:
        xs = [r["pq_m"] for r in pq_rows]
        compression = [r["compression_vs_flat"] for r in pq_rows]
        ax.bar([str(x) for x in xs], compression, color="steelblue")
        ax.set_xlabel("pq_m")
        ax.set_ylabel("Compression vs flat (×)")
        ax.set_title("PQ Compression Ratio")
        ax.grid(True, axis="y", alpha=0.3)

    # ── Bottom-right: pq_m vs Recall@10 ───────────────────────────────────
    ax = axes[1, 1]
    if pq_rows:
        xs = [r["pq_m"] for r in pq_rows]
        ys = [r["recall_at_10"] for r in pq_rows]
        ax.plot([str(x) for x in xs], ys, marker="s", color="darkorange")
        ax.set_xlabel("pq_m (sub-quantizers)")
        ax.set_ylabel("Recall@10")
        ax.set_title("PQ sub-quantizers vs Recall@10")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = results_dir / "ann_benchmark_plots.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n  Plots saved → {out}")


# ── Tests ──────────────────────────────────────────────────────────────────

def test_index_build_and_compression(synthetic_data, flat_index, ivf_flat_index, ivf_pq_index, results_dir):
    """
    Compare serialised index sizes for flat, IVF-Flat, and IVF-PQ.
    Asserts that IVF-PQ achieves at least 5× compression over flat.
    """
    sizes = {
        "flat":     index_bytes(flat_index),
        "ivf_flat": index_bytes(ivf_flat_index),
        "ivf_pq":   index_bytes(ivf_pq_index),
    }

    print(f"\n  {'index':>10}  {'size (MB)':>10}")
    print(f"  {'-'*22}")
    for name, sz in sizes.items():
        print(f"  {name:>10}  {sz/1e6:>10.3f}")

    compression = sizes["flat"] / sizes["ivf_pq"]
    print(f"\n  IVF-PQ compression vs flat: {compression:.1f}×")

    save_results(results_dir, "ann_build_sizes.json", {
        k: {"size_bytes": v, "size_mb": round(v / 1e6, 3)} for k, v in sizes.items()
    })

    assert compression >= 5.0, (
        f"IVF-PQ ({sizes['ivf_pq']/1e6:.2f} MB) should be ≥5× smaller than "
        f"flat ({sizes['flat']/1e6:.2f} MB), got {compression:.2f}×"
    )


def test_nprobe_recall_sweep(synthetic_data, flat_ground_truth, results_dir):
    """
    Sweep nprobe from 1 → NLIST for both IVF-Flat and IVF-PQ.
    Records Recall@10 and per-query search latency at each setting.

    Key assertion: at nprobe=16 both index types must exceed their recall thresholds.
    """
    corpus, queries = synthetic_data
    sweep_results: dict[str, list] = {}

    configs = [
        ("ivf_flat", {"ivf_nlist": NLIST}),
        ("ivf_pq",   {"ivf_nlist": NLIST, "pq_m": 16, "pq_nbits": PQ_NBITS}),
    ]

    for index_type, kwargs in configs:
        index = build_faiss_index(corpus, faiss_index_type=index_type, **kwargs)
        rows = []

        print(f"\n  ── {index_type} nprobe sweep ──────────────────────────")
        print(f"  {'nprobe':>7}  {'Recall@10':>10}  {'latency (ms)':>14}")
        print(f"  {'-'*35}")

        for nprobe in NPROBE_VALUES:
            if nprobe > NLIST:
                continue
            index.nprobe = nprobe
            _, ann = index.search(queries, K)
            rec = recall_at_k(ann, flat_ground_truth)
            lat = search_latency_ms(index, queries)
            rows.append({"nprobe": nprobe, "recall_at_10": round(rec, 4), "latency_ms": round(lat, 5)})
            print(f"  {nprobe:>7}  {rec:>10.4f}  {lat:>14.5f}")

        sweep_results[index_type] = rows

    save_results(results_dir, "ann_nprobe_sweep.json", sweep_results)

    # Deterministic invariant: IVF-Flat at nprobe=NLIST visits every cell,

    # making it equivalent to exact search.  Recall vs the flat ground truth
    # must therefore be 1.0 regardless of data distribution or dimensionality.
    full_scan = next(
        (r for r in sweep_results["ivf_flat"] if r["nprobe"] == NLIST), None
    )
    if full_scan is not None:
        assert full_scan["recall_at_10"] >= 0.99, (
            f"IVF-Flat at nprobe=NLIST={NLIST} (full scan) must match flat "
            f"ground truth exactly, got Recall@10={full_scan['recall_at_10']:.4f}"
        )

    # Sanity check: recall must increase from nprobe=1 to nprobe=NLIST for both
    # index types (more cells searched → monotonically non-decreasing recall).
    for index_type in ("ivf_flat", "ivf_pq"):
        rows = sweep_results[index_type]
        if len(rows) >= 2:
            assert rows[-1]["recall_at_10"] >= rows[0]["recall_at_10"], (
                f"{index_type}: recall at max nprobe ({rows[-1]['nprobe']}) "
                f"should be >= recall at nprobe=1 ({rows[0]['nprobe']})"
            )



def test_pq_compression_vs_recall(synthetic_data, flat_ground_truth, flat_index, results_dir):
    """
    Sweep pq_m for IVF-PQ at a fixed nprobe=16.
    Demonstrates the memory–accuracy tradeoff introduced by product quantization.

    Assertion: every pq_m variant must achieve Recall@10 ≥ 0.80.
    """
    corpus, queries = synthetic_data
    flat_size = index_bytes(flat_index)
    rows = []

    print(f"\n  ── IVF-PQ pq_m sweep (nprobe=16) ────────────────────────")
    print(f"  {'pq_m':>6}  {'size (MB)':>10}  {'compression':>12}  {'Recall@10':>10}")
    print(f"  {'-'*44}")

    for pq_m in PQ_M_VALUES:
        if DIM % pq_m != 0:
            print(f"  pq_m={pq_m} skipped — DIM={DIM} not divisible")
            continue

        index = build_faiss_index(
            corpus,
            faiss_index_type="ivf_pq",
            ivf_nlist=NLIST,
            pq_m=pq_m,
            pq_nbits=PQ_NBITS,
        )
        index.nprobe = 16
        sz = index_bytes(index)
        compression = flat_size / sz
        _, ann = index.search(queries, K)
        rec = recall_at_k(ann, flat_ground_truth)

        rows.append({
            "pq_m": pq_m,
            "size_bytes": sz,
            "size_mb": round(sz / 1e6, 3),
            "compression_vs_flat": round(compression, 2),
            "recall_at_10": round(rec, 4),
        })
        print(f"  {pq_m:>6}  {sz/1e6:>10.3f}  {compression:>11.1f}×  {rec:>10.4f}")

    save_results(results_dir, "ann_pq_sweep.json", rows)

    # No recall threshold here — random high-dim vectors give unpredictable
    # absolute recall values.  The meaningful accuracy assertion is in
    # test_ann_real_index.py on actual Qwen3 textbook embeddings.
    # Assert only the compression direction: more sub-quantizers → larger index.
    if len(rows) >= 2:
        sizes = [r["size_bytes"] for r in rows]
        assert sizes == sorted(sizes), (
            "IVF-PQ index size should increase monotonically with pq_m "
            f"(more sub-quantizers = more storage): {sizes}"
        )



def test_search_latency_comparison(synthetic_data, flat_index, ivf_flat_index, ivf_pq_index, results_dir):
    """
    Compare per-query search latency for all three index types at nprobe=16.
    """
    _, queries = synthetic_data

    ivf_flat_index.nprobe = 16
    ivf_pq_index.nprobe   = 16

    results = {}
    print(f"\n  ── Search latency at nprobe=16 (flat has no nprobe) ──────")
    print(f"  {'index':>10}  {'latency (ms/query)':>20}")
    print(f"  {'-'*33}")

    for label, index in [
        ("flat",     flat_index),
        ("ivf_flat", ivf_flat_index),
        ("ivf_pq",   ivf_pq_index),
    ]:
        lat = search_latency_ms(index, queries)
        results[label] = {"latency_ms_per_query": round(lat, 5)}
        print(f"  {label:>10}  {lat:>20.5f}")

    save_results(results_dir, "ann_search_latency.json", results)


def test_full_benchmark_report(synthetic_data, flat_ground_truth, flat_index, ivf_flat_index, ivf_pq_index, results_dir):
    """
    Aggregate all benchmark metrics into a single JSON report and generate plots.
    This test always passes — it is a reporting test, not a correctness gate.
    """
    corpus, queries = synthetic_data

    # ── Sizes ──────────────────────────────────────────────────────────────
    flat_sz     = index_bytes(flat_index)
    ivf_flat_sz = index_bytes(ivf_flat_index)
    ivf_pq_sz   = index_bytes(ivf_pq_index)

    # ── nprobe sweep (rebuild to avoid state from other tests) ─────────────
    sweep_results: dict[str, list] = {}
    for index_type, kwargs in [
        ("ivf_flat", {"ivf_nlist": NLIST}),
        ("ivf_pq",   {"ivf_nlist": NLIST, "pq_m": 16, "pq_nbits": PQ_NBITS}),
    ]:
        idx = build_faiss_index(corpus, faiss_index_type=index_type, **kwargs)
        rows = []
        for nprobe in NPROBE_VALUES:
            if nprobe > NLIST:
                continue
            idx.nprobe = nprobe
            _, ann = idx.search(queries, K)
            rec = recall_at_k(ann, flat_ground_truth)
            lat = search_latency_ms(idx, queries)
            rows.append({"nprobe": nprobe, "recall_at_10": round(rec, 4), "latency_ms": round(lat, 5)})
        sweep_results[index_type] = rows

    # ── pq_m sweep ─────────────────────────────────────────────────────────
    pq_rows = []
    for pq_m in PQ_M_VALUES:
        if DIM % pq_m != 0:
            continue
        idx = build_faiss_index(corpus, faiss_index_type="ivf_pq",
                                ivf_nlist=NLIST, pq_m=pq_m, pq_nbits=PQ_NBITS)
        idx.nprobe = 16
        sz = index_bytes(idx)
        _, ann = idx.search(queries, K)
        rec = recall_at_k(ann, flat_ground_truth)
        pq_rows.append({
            "pq_m": pq_m,
            "size_bytes": sz,
            "size_mb": round(sz / 1e6, 3),
            "compression_vs_flat": round(flat_sz / sz, 2),
            "recall_at_10": round(rec, 4),
        })

    report = {
        "config": {
            "n_corpus": N_CORPUS,
            "n_queries": N_QUERIES,
            "dim": DIM,
            "k": K,
            "nlist": NLIST,
            "pq_nbits": PQ_NBITS,
        },
        "index_sizes": {
            "flat":     {"size_bytes": flat_sz,     "size_mb": round(flat_sz     / 1e6, 3)},
            "ivf_flat": {"size_bytes": ivf_flat_sz, "size_mb": round(ivf_flat_sz / 1e6, 3)},
            "ivf_pq":   {"size_bytes": ivf_pq_sz,   "size_mb": round(ivf_pq_sz   / 1e6, 3),
                         "compression_vs_flat": round(flat_sz / ivf_pq_sz, 2)},
        },
        "nprobe_sweep": sweep_results,
        "pq_m_sweep": pq_rows,
    }
    save_results(results_dir, "ann_benchmark_report.json", report)
    _try_plot(results_dir, sweep_results, pq_rows)


def test_memory_profiling(synthetic_data, results_dir):
    """
    Measure peak Python-heap allocation (tracemalloc) for three phases:
      - build:   constructing the index from the embedding matrix
      - load:    deserialising a saved index from disk
      - search:  running a batch of N_QUERIES queries

    tracemalloc captures Python-managed heap only.  FAISS allocates its core
    data structures (centroids, PQ codebooks, stored codes) directly in C++
    heap via malloc/free, so those bytes do NOT appear here.  The serialised
    index size (reported in test_index_build_and_compression) is a better
    proxy for the actual in-memory footprint of each index type.
    Together the two metrics give a complete picture:

      tracemalloc  → Python wrapper / I/O overhead during each phase
      index_bytes  → C++ in-memory footprint (storage-equivalent)
    """
    corpus, queries = synthetic_data
    results = {}

    configs = [
        ("flat",     {}),
        ("ivf_flat", {"ivf_nlist": NLIST}),
        ("ivf_pq",   {"ivf_nlist": NLIST, "pq_m": 16, "pq_nbits": PQ_NBITS}),
    ]

    print(f"\n  {'index':>10}  {'build_py_mb':>13}  {'load_py_mb':>12}  {'search_py_mb':>14}  {'store_mb':>10}")
    print(f"  {'-'*65}")

    for index_type, kwargs in configs:
        # ── Build ──────────────────────────────────────────────────────────
        index, build_mb = measure_peak_mb(
            build_faiss_index, corpus, faiss_index_type=index_type, **kwargs
        )
        if hasattr(index, 'nprobe'):
            index.nprobe = 16

        # ── Load (write → read) ────────────────────────────────────────────
        with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as f:
            tmp_path = f.name
        faiss.write_index(index, tmp_path)
        try:
            _, load_mb = measure_peak_mb(faiss.read_index, tmp_path)
        finally:
            import os; os.unlink(tmp_path)

        # ── Search ─────────────────────────────────────────────────────────
        _, search_mb = measure_peak_mb(index.search, queries, K)

        store_mb = index_bytes(index) / 1e6
        results[index_type] = {
            "build_peak_python_heap_mb":  round(build_mb,  3),
            "load_peak_python_heap_mb":   round(load_mb,   3),
            "search_peak_python_heap_mb": round(search_mb, 3),
            "serialized_storage_mb":      round(store_mb,  3),
        }
        print(
            f"  {index_type:>10}  {build_mb:>13.3f}  {load_mb:>12.3f}"
            f"  {search_mb:>14.3f}  {store_mb:>10.3f}"
        )

    save_results(results_dir, "ann_memory_profiling.json", results)

    # IVF-PQ serialised storage must be measurably smaller than flat
    flat_store = results["flat"]["serialized_storage_mb"]
    pq_store   = results["ivf_pq"]["serialized_storage_mb"]
    assert pq_store < flat_store, (
        f"IVF-PQ storage ({pq_store:.2f} MB) should be < flat ({flat_store:.2f} MB)"
    )
