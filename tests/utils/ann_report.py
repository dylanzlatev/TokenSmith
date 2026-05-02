"""
ann_report.py

Aggregates all ANN evaluation JSON files produced by the test suite and
prints a formatted multi-section comparison table suitable for copy-paste
into the final report.

Generated files read (all under tests/results/):
  ann_benchmark_report.json      — synthetic-corpus benchmark
  ann_memory_profiling.json      — tracemalloc memory measurements
  ann_real_index_results.json    — real-corpus nprobe sweep
  ann_per_question_recall.json   — per-question breakdown (real corpus)

Run:
    python tests/utils/ann_report.py [--results-dir <path>]
"""

import argparse
import json
import sys
from pathlib import Path


# ── Formatting helpers ─────────────────────────────────────────────────────

def _col(value: str, width: int, align: str = "<") -> str:
    return f"{value:{align}{width}}"


def _row(*cells) -> str:
    return "  ".join(cells)


def _sep(widths) -> str:
    return "  ".join("-" * w for w in widths)


def _header(*labels_and_widths) -> str:
    labels = [lw[0] for lw in labels_and_widths]
    widths = [lw[1] for lw in labels_and_widths]
    aligns = [lw[2] if len(lw) > 2 else "<" for lw in labels_and_widths]
    hdr = "  ".join(_col(l, w, a) for l, w, a in zip(labels, widths, aligns))
    sep = "  ".join("-" * w for w in widths)
    return hdr + "\n" + sep


def _load(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _section(title: str) -> str:
    bar = "=" * (len(title) + 4)
    return f"\n{bar}\n  {title}\n{bar}"


# ── Section renderers ──────────────────────────────────────────────────────

def _render_sizes(bench, real):
    lines = [_section("1. Index Sizes & Compression")]
    widths = [12, 10, 10]

    def _table(label, sizes_mb):
        lines.append(f"\n  {label}")
        lines.append("  " + _header(
            ("index",     widths[0]),
            ("size (MB)", widths[1], ">"),
            ("vs flat",   widths[2], ">"),
        ))
        flat_mb = sizes_mb.get("flat")
        for key in ("flat", "ivf_flat", "ivf_pq"):
            mb = sizes_mb.get(key)
            if mb is None:
                continue
            ratio = f"{flat_mb / mb:.2f}×" if flat_mb and key != "flat" else "—"
            lines.append("  " + _row(
                _col(key,          widths[0]),
                _col(f"{mb:.3f}",  widths[1], ">"),
                _col(ratio,        widths[2], ">"),
            ))
        pq_ratio = sizes_mb.get("pq_compression_vs_flat")
        if pq_ratio:
            lines.append(f"    IVF-PQ compression vs flat (stored): {pq_ratio:.2f}×")

    if bench:
        # ann_benchmark_report.json stores nested dicts: {key: {"size_mb": ...}}
        raw = bench.get("index_sizes", {})
        sizes_mb = {}
        for k, v in raw.items():
            if isinstance(v, dict):
                sizes_mb[k] = v.get("size_mb")
            else:
                sizes_mb[k] = v
        _table("Synthetic corpus", sizes_mb)
    if real:
        # ann_real_index_results.json stores flat floats: {key: float}
        _table("Real corpus", real.get("index_sizes_mb", {}))
    return "\n".join(lines)


def _render_memory(mem_data):
    lines = [_section("2. Memory Profiling  (Python heap via tracemalloc)")]
    lines.append("  NOTE: FAISS C++ allocations are not captured by tracemalloc.")
    lines.append("        Serialized size is the authoritative storage metric.\n")

    if not mem_data:
        lines.append("  [ann_memory_profiling.json not found — run test_ann_benchmark.py first]")
        return "\n".join(lines)

    # ann_memory_profiling.json structure: {index_type: {metric: value, ...}}
    widths = [12, 14, 14, 18, 14]
    lines.append("  " + _header(
        ("index",             widths[0]),
        ("build peak (MB)",   widths[1], ">"),
        ("load peak (MB)",    widths[2], ">"),
        ("search peak (MB)",  widths[3], ">"),
        ("stored (MB)",       widths[4], ">"),
    ))
    for key in ("flat", "ivf_flat", "ivf_pq"):
        entry = mem_data.get(key, {})
        b  = entry.get("build_peak_python_heap_mb")
        l  = entry.get("load_peak_python_heap_mb")
        s  = entry.get("search_peak_python_heap_mb")
        st = entry.get("serialized_storage_mb")
        lines.append("  " + _row(
            _col(key,                              widths[0]),
            _col(f"{b:.2f}"  if b  is not None else "—", widths[1], ">"),
            _col(f"{l:.2f}"  if l  is not None else "—", widths[2], ">"),
            _col(f"{s:.4f}"  if s  is not None else "—", widths[3], ">"),
            _col(f"{st:.3f}" if st is not None else "—", widths[4], ">"),
        ))
    return "\n".join(lines)


def _render_nprobe_sweep(sweep_rows, flat_lat, label):
    lines = []
    if not sweep_rows:
        lines.append("  [no nprobe sweep data]")
        return "\n".join(lines)

    widths = [10, 8, 12, 16]
    lines.append("  " + _header(
        ("index",        widths[0]),
        ("nprobe",       widths[1], ">"),
        ("Recall@10",    widths[2], ">"),
        ("latency (ms)", widths[3], ">"),
    ))
    for row in sweep_rows:
        lines.append("  " + _row(
            _col(row["index_type"],                   widths[0]),
            _col(str(row["nprobe"]),                  widths[1], ">"),
            _col(f"{row['recall_at_10']:.4f}",        widths[2], ">"),
            _col(f"{row['latency_ms_per_query']:.5f}", widths[3], ">"),
        ))
    if flat_lat is not None:
        lines.append("  " + _row(
            _col("flat",       widths[0]),
            _col("—",          widths[1], ">"),
            _col("1.0000",     widths[2], ">"),
            _col(f"{flat_lat:.5f}", widths[3], ">"),
        ))
    return "\n".join(lines)


def _render_latency_comparison(bench, latency_data=None):
    lines = [_section("3. Search Latency — Synthetic Corpus")]
    if not bench:
        lines.append("  [ann_benchmark_report.json not found]")
        return "\n".join(lines)

    # flat latency comes from ann_search_latency.json if available
    flat_lat = None
    if latency_data:
        flat_entry = latency_data.get("flat", {})
        flat_lat = flat_entry.get("latency_ms_per_query")

    # nprobe_sweep is {index_type: [{nprobe, recall_at_10, latency_ms}, ...]}
    sweep_dict = bench.get("nprobe_sweep", {})
    sweep = []
    for idx_type, rows in sweep_dict.items():
        for row in rows:
            sweep.append({
                "index_type": idx_type,
                "nprobe": row["nprobe"],
                "recall_at_10": row["recall_at_10"],
                "latency_ms_per_query": row["latency_ms"],
            })
    lines.append(_render_nprobe_sweep(sweep, flat_lat, "Synthetic"))
    return "\n".join(lines)


def _render_real_sweep(real):
    lines = [_section("4. nprobe Sweep — Real Corpus")]
    if not real:
        lines.append("  [ann_real_index_results.json not found]")
        return "\n".join(lines)

    corpus = real.get("corpus", {})
    cfg    = real.get("index_config", {})
    lines.append(f"  Corpus: {corpus.get('n_chunks','?'):,} chunks,"
                 f" dim={corpus.get('embedding_dim','?')},"
                 f" index_dir={corpus.get('index_dir','?')}")
    lines.append(f"  Config: nlist={cfg.get('nlist','?')},"
                 f" pq_m={cfg.get('pq_m','?')},"
                 f" pq_nbits={cfg.get('pq_nbits','?')}\n")

    sweep    = real.get("nprobe_sweep", [])
    flat_lat = real.get("flat_latency_ms_per_query")
    lines.append(_render_nprobe_sweep(sweep, flat_lat, "Real"))

    # load memory
    load_mem = real.get("load_peak_python_heap_mb", {})
    if load_mem:
        lines.append("\n  Load-time Python heap (tracemalloc peak):")
        for k, v in load_mem.items():
            lines.append(f"    {k:<12} {v:.3f} MB")
    return "\n".join(lines)


def _render_per_question(pq_data):
    lines = [_section("5. Per-Question Recall — Real Corpus (IVF-PQ, nprobe=16)")]
    if not pq_data:
        lines.append("  [ann_per_question_recall.json not found]")
        return "\n".join(lines)

    widths = [65, 8, 10]
    lines.append("  " + _header(
        ("question (truncated)",  widths[0]),
        ("R@10",                  widths[1], ">"),
        ("missed",                widths[2], ">"),
    ))
    avg = sum(r["recall_at_10"] for r in pq_data) / len(pq_data)
    for row in pq_data:
        q      = row["question"][:62] + "..." if len(row["question"]) > 65 else row["question"]
        r      = row["recall_at_10"]
        missed = len(row.get("missed_by_ivf_pq", []))
        lines.append("  " + _row(
            _col(q,           widths[0]),
            _col(f"{r:.2f}",  widths[1], ">"),
            _col(str(missed), widths[2], ">"),
        ))
    lines.append(f"\n  Average Recall@10: {avg:.4f}  ({len(pq_data)} questions)")
    return "\n".join(lines)


def _render_pq_compression(bench):
    lines = [_section("6. PQ Compression vs Recall — Synthetic Corpus")]
    if not bench:
        lines.append("  [ann_benchmark_report.json not found]")
        return "\n".join(lines)

    # ann_benchmark_report.json uses key "pq_m_sweep"
    pq_rows = bench.get("pq_m_sweep", [])
    if not pq_rows:
        lines.append("  [no pq_m_sweep data in report]")
        return "\n".join(lines)

    widths = [6, 12, 14, 12]
    lines.append("  " + _header(
        ("pq_m",        widths[0]),
        ("size (MB)",   widths[1], ">"),
        ("compression", widths[2], ">"),
        ("Recall@10",   widths[3], ">"),
    ))
    for row in pq_rows:
        lines.append("  " + _row(
            _col(str(row["pq_m"]),                              widths[0]),
            _col(f"{row['size_mb']:.3f}",                       widths[1], ">"),
            _col(f"{row['compression_vs_flat']:.2f}×",          widths[2], ">"),
            _col(f"{row['recall_at_10']:.4f}",                  widths[3], ">"),
        ))
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Aggregate ANN evaluation results for final report.")
    parser.add_argument(
        "--results-dir",
        default=str(Path(__file__).parent.parent / "results"),
        help="Directory containing ann_*.json files (default: tests/results/)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional file path to write the report (default: stdout only)",
    )
    args = parser.parse_args()

    rd = Path(args.results_dir)
    bench   = _load(rd / "ann_benchmark_report.json")
    mem     = _load(rd / "ann_memory_profiling.json")
    real    = _load(rd / "ann_real_index_results.json")
    per_q   = _load(rd / "ann_per_question_recall.json")
    latency = _load(rd / "ann_search_latency.json")

    found = [
        f"ann_benchmark_report.json       {'OK' if bench   else 'MISSING'}",
        f"ann_memory_profiling.json        {'OK' if mem     else 'MISSING'}",
        f"ann_search_latency.json          {'OK' if latency else 'MISSING'}",
        f"ann_real_index_results.json      {'OK' if real    else 'MISSING'}",
        f"ann_per_question_recall.json     {'OK' if per_q   else 'MISSING'}",
    ]

    report_parts = [
        "TokenSmith — ANN Index Evaluation Report",
        "=" * 42,
        f"Results directory: {rd.resolve()}",
        "",
        "Input files:",
    ] + ["  " + f for f in found] + [
        "",
        _render_sizes(bench, real),
        _render_memory(mem),
        _render_latency_comparison(bench, latency),
        _render_real_sweep(real),
        _render_per_question(per_q),
        _render_pq_compression(bench),
        "",
    ]

    report = "\n".join(report_parts)
    print(report)

    if args.out:
        out = Path(args.out)
        out.write_text(report)
        print(f"\nReport written to: {out.resolve()}")


if __name__ == "__main__":
    main()
