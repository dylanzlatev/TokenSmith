#!/usr/bin/env python3
"""
eval_chunk_buffer.py

Evaluates the session-scoped ChunkFrequencyTracker (chunk buffer pool) on
real textbook retrieval sessions.  No LLM required — only the FAISS index
and embedding model.

For each topic-focused session (5 related questions), every query is run in
two modes:
  baseline  — raw FAISS similarity scores only
  buffered  — FAISS scores boosted by hot-chunk frequency from prior queries

Metrics recorded per query (follow-up queries only):
  overlap_raw       |top_K(q_i, raw)  ∩ top_K(q_{i-1}, raw)|  / K
  overlap_buffered  |top_K(q_i, buf)  ∩ top_K(q_{i-1}, buf)|  / K
  chunks_promoted   chunks not in raw top-K that entered buffered top-K
  avg_rank_gain     mean (raw_rank - buffered_rank) for hot chunks in pool

Output files:
  tests/results/chunk_buffer_eval.json    — full per-query detail (JSON)
  tests/results/chunk_buffer_report.txt   — human-readable summary

Run:
  python tests/eval_chunk_buffer.py
  python tests/eval_chunk_buffer.py --config config/config.yaml
"""

import argparse
import json
import pathlib
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import RAGConfig
from src.retriever import load_artifacts, FAISSRetriever
from src.chunk_buffer import ChunkFrequencyTracker

class _NpEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalars and arrays to native Python types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ── Paths ──────────────────────────────────────────────────────────────────
_INDEX_DIRS = [
    pathlib.Path("index/sections"),
    pathlib.Path("index/partial_sections"),
]
_INDEX_PREFIX = "textbook_index"
_RESULTS_DIR  = pathlib.Path("tests/results")

# ── Eval knobs ─────────────────────────────────────────────────────────────
TOP_K        = 10
POOL_SIZE    = 80    # candidates retrieved from FAISS before buffer rerank
BOOST_WEIGHT = 0.15  # default from config
WINDOW_SIZE  = 20    # default from config

# ── Topic-focused sessions ─────────────────────────────────────────────────
# Each session is a sequence of closely related questions.
# For realistic multi-turn behaviour, questions within a session deliberately
# share vocabulary and concepts so that chunks from early queries are likely
# to remain relevant in later ones — this is exactly when the buffer helps.
SESSIONS = [
    {
        "name": "Transactions & Concurrency Control",
        "questions": [
            "What are the ACID properties of a database transaction and why does each matter?",
            "Explain the two-phase locking protocol and how it guarantees serializability.",
            "What is the difference between a shared lock and an exclusive lock in databases?",
            "What is a deadlock in a database system and how is it detected and resolved?",
            "How does multiversion concurrency control allow reads without blocking writes?",
        ],
    },
    {
        "name": "Storage & Indexing",
        "questions": [
            "How does a B+ tree index work and why is it preferred over a B-tree for databases?",
            "What is the difference between a dense index and a sparse index?",
            "What is the difference between a clustered index and an unclustered index?",
            "How does extendible hashing work as a dynamic file organization method?",
            "What role does the buffer manager play in database storage management?",
        ],
    },
    {
        "name": "Query Processing & Optimization",
        "questions": [
            "How does the nested-loop join algorithm work and what is its time complexity?",
            "What is a hash join and when is it more efficient than a sort-merge join?",
            "How does pipelining differ from materialization in query evaluation?",
            "What statistics does the query optimizer use to estimate the cost of a query plan?",
            "How does a sort-merge join algorithm work for joining two large relations?",
        ],
    },
    {
        "name": "Relational Model & Normalization",
        "questions": [
            "What is a foreign key constraint and how does it enforce referential integrity?",
            "What is a functional dependency and how is it used to determine normal forms?",
            "Explain Boyce-Codd Normal Form and how it differs from Third Normal Form.",
            "What is a multivalued dependency and how does it lead to Fourth Normal Form?",
            "Describe the process of converting an ER diagram into a relational schema.",
        ],
    },
    {
        "name": "Recovery",
        "questions": [
            "How does write-ahead logging ensure atomicity and durability in databases?",
            "What are the three phases of the ARIES recovery algorithm?",
            "What is the difference between undo and redo operations in crash recovery?",
            "How does checkpointing reduce the work needed during database recovery?",
            "What is a log record and what information does it contain for recovery?",
        ],
    },
]


# ── Helpers ────────────────────────────────────────────────────────────────

def find_index_dir() -> Optional[pathlib.Path]:
    for d in _INDEX_DIRS:
        if (d / f"{_INDEX_PREFIX}.faiss").exists():
            return d
    return None


def _scores_to_ranked(score_dict: Dict[int, float]) -> Tuple[List[int], List[float]]:
    """Sort a {chunk_id: score} dict into parallel (ids, scores) lists, descending."""
    ordered = [int(k) for k in sorted(score_dict, key=score_dict.__getitem__, reverse=True)]
    scores  = [float(score_dict[i]) for i in ordered]
    return ordered, scores


# ── Session evaluation ─────────────────────────────────────────────────────

def evaluate_session(
    session: dict,
    retriever: FAISSRetriever,
    chunks: List[str],
) -> dict:
    """
    Run all queries in a session in both baseline and buffered modes.
    Returns per-query metrics and a session-level summary.
    """
    buffer = ChunkFrequencyTracker(window_size=WINDOW_SIZE, max_chunks=500)

    query_results    = []
    prev_raw_topk    = None
    prev_buf_topk    = None

    for q_idx, question in enumerate(session["questions"]):

        # ── Raw FAISS retrieval ────────────────────────────────────────────
        score_dict              = retriever.get_scores(question, POOL_SIZE, chunks)
        raw_ordered, raw_scores = _scores_to_ranked(score_dict)
        raw_topk                = set(raw_ordered[:TOP_K])

        # ── Snapshot buffer state BEFORE this query is recorded ───────────
        hot_before    = buffer.get_hot_scores(raw_ordered)   # {id: normalised_freq}
        hot_n_before  = buffer.hot_chunk_count

        # ── Buffered retrieval ─────────────────────────────────────────────
        buf_ordered, buf_scores = buffer.rerank_with_boost(
            raw_ordered, raw_scores, BOOST_WEIGHT
        )
        buf_topk = set(buf_ordered[:TOP_K])

        # ── Rank-change analysis (for all hot chunks in the pool) ──────────
        raw_rank = {cid: r + 1 for r, cid in enumerate(raw_ordered)}
        buf_rank = {cid: r + 1 for r, cid in enumerate(buf_ordered)}

        rank_changes = []
        for cid, hs in hot_before.items():
            rr = raw_rank.get(cid)
            br = buf_rank.get(cid)
            if rr is None or br is None:
                continue
            rank_changes.append({
                "chunk_id":    cid,
                "hot_score":   round(hs, 4),
                "raw_rank":    rr,
                "buf_rank":    br,
                "rank_gain":   rr - br,          # positive  = moved up
                "in_raw_topk": cid in raw_topk,
                "in_buf_topk": cid in buf_topk,
                "preview":     chunks[cid][:120].replace("\n", " ")
                               if 0 <= cid < len(chunks) else "",
            })
        rank_changes.sort(key=lambda x: -x["rank_gain"])

        # ── Set-level differences ──────────────────────────────────────────
        promoted = sorted(buf_topk - raw_topk)   # entered top-K only via boost
        demoted  = sorted(raw_topk - buf_topk)   # left top-K due to others being boosted

        # ── Overlap with previous query ────────────────────────────────────
        overlap_raw = (
            len(raw_topk & prev_raw_topk) / TOP_K
            if prev_raw_topk is not None else None
        )
        overlap_buf = (
            len(buf_topk & prev_buf_topk) / TOP_K
            if prev_buf_topk is not None else None
        )

        avg_rank_gain = (
            sum(r["rank_gain"] for r in rank_changes) / len(rank_changes)
            if rank_changes else 0.0
        )

        # ── Update buffer with buffered top-K (mirrors pipeline behaviour) -
        buffer.record(list(buf_ordered[:TOP_K]))

        query_results.append({
            "query_idx":          q_idx,
            "question":           question,
            "hot_chunks_before":  hot_n_before,
            "raw_topk":           sorted(raw_topk),
            "buf_topk":           sorted(buf_topk),
            "promoted_into_topk": promoted,
            "demoted_from_topk":  demoted,
            "n_promoted":         len(promoted),
            "overlap_raw_prev":   overlap_raw,
            "overlap_buf_prev":   overlap_buf,
            "buffer_hits_topk":   len(buf_topk & set(hot_before)),
            "avg_rank_gain":      round(avg_rank_gain, 3),
            "top_rank_changes":   rank_changes[:5],
        })

        prev_raw_topk = raw_topk
        prev_buf_topk = buf_topk

    # ── Session-level aggregates (follow-up queries only) ─────────────────
    followup = [r for r in query_results if r["query_idx"] > 0]

    def _avg(key):
        vals = [r[key] for r in followup if r[key] is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    summary = {
        "avg_overlap_raw":       _avg("overlap_raw_prev"),
        "avg_overlap_buffered":  _avg("overlap_buf_prev"),
        "avg_rank_gain":         _avg("avg_rank_gain"),
        "avg_buffer_hits_topk":  _avg("buffer_hits_topk"),
        "avg_promoted_per_query": _avg("n_promoted"),
    }

    return {
        "session_name": session["name"],
        "n_queries":    len(query_results),
        "queries":      query_results,
        "summary":      summary,
    }


# ── Report formatter ───────────────────────────────────────────────────────

_W = 80   # report width

def _bar(char="═"):
    return char * _W

def _section_head(title):
    return f"\n{_bar('─')}\n  {title}\n{_bar('─')}"

def _pct(v):
    return f"{v * 100:.1f}%" if v is not None else "—"

def _gain(v):
    return f"{v:+.2f}" if v != 0 else " 0.00"


def format_report(all_results: list, n_chunks: int, index_dir: str) -> str:
    lines = []

    lines += [
        _bar("="),
        "  TokenSmith — Chunk Buffer Pool Evaluation",
        _bar("="),
        "",
        f"  Index:          {index_dir}  ({n_chunks:,} chunks)",
        f"  Config:         top_k={TOP_K}  pool_size={POOL_SIZE}  "
        f"boost_weight={BOOST_WEIGHT}  window_size={WINDOW_SIZE}",
        f"  Timestamp:      {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"  Sessions:       {len(all_results)}",
        f"  Total queries:  {sum(r['n_queries'] for r in all_results)}",
    ]

    for sess in all_results:
        lines.append(_section_head(sess["session_name"]))

        for qr in sess["queries"]:
            q_label = f"Q{qr['query_idx'] + 1}"
            lines.append(f"\n  {q_label}: {qr['question']}")
            lines.append(f"     Buffer before: {qr['hot_chunks_before']} hot chunk(s)")

            if qr["query_idx"] == 0:
                lines.append(f"     ↳ First query — buffer empty, no boost applied")
                lines.append(f"     Raw top-{TOP_K}: {qr['raw_topk']}")
            else:
                ov_r = _pct(qr["overlap_raw_prev"])
                ov_b = _pct(qr["overlap_buf_prev"])
                delta = (
                    (qr["overlap_buf_prev"] - qr["overlap_raw_prev"]) * 100
                    if qr["overlap_raw_prev"] is not None else 0.0
                )
                lines.append(
                    f"     Top-{TOP_K} overlap with prev:  "
                    f"raw={ov_r}  buffered={ov_b}  (Δ {delta:+.1f} pp)"
                )
                lines.append(
                    f"     Chunks promoted into top-{TOP_K}:  {qr['n_promoted']}"
                    + (f"  → chunk ids {qr['promoted_into_topk']}" if qr["promoted_into_topk"] else "")
                )
                lines.append(
                    f"     Buffer hits in top-{TOP_K}:       {qr['buffer_hits_topk']}"
                )
                lines.append(
                    f"     Avg rank gain (hot chunks):    {_gain(qr['avg_rank_gain'])} positions"
                )

                if qr["top_rank_changes"]:
                    lines.append(f"     Top rank movers:")
                    for rc in qr["top_rank_changes"]:
                        arrow = "↑" if rc["rank_gain"] > 0 else ("↓" if rc["rank_gain"] < 0 else "=")
                        in_buf = "✓" if rc["in_buf_topk"] else " "
                        lines.append(
                            f"       {arrow} [{in_buf}] chunk #{rc['chunk_id']:5d}"
                            f"  rank {rc['raw_rank']:3d} → {rc['buf_rank']:3d}"
                            f"  (gain {rc['rank_gain']:+3d})"
                            f"  hot={rc['hot_score']:.2f}"
                            f"  \"{rc['preview'][:55]}…\""
                        )

        # Session summary
        s = sess["summary"]
        ov_r  = s["avg_overlap_raw"]
        ov_b  = s["avg_overlap_buffered"]
        delta = (ov_b - ov_r) * 100
        lines += [
            f"\n  {'─'*60}",
            f"  Session summary  ({sess['n_queries']} queries, {sess['n_queries']-1} follow-ups evaluated)",
            f"  {'─'*60}",
            f"    Avg top-{TOP_K} overlap:         "
            f"raw={_pct(ov_r)}  buffered={_pct(ov_b)}  (Δ {delta:+.1f} pp)",
            f"    Avg rank gain (hot chunks):   {_gain(s['avg_rank_gain'])} positions",
            f"    Avg buffer hits in top-{TOP_K}:  {s['avg_buffer_hits_topk']:.1f} / {TOP_K}  "
            f"({s['avg_buffer_hits_topk'] / TOP_K * 100:.0f}%)",
            f"    Avg chunks promoted / query:  {s['avg_promoted_per_query']:.1f}",
        ]

    # ── Aggregate summary ──────────────────────────────────────────────────
    all_followup = [
        qr
        for sess in all_results
        for qr in sess["queries"]
        if qr["query_idx"] > 0
    ]
    n_fu = len(all_followup)

    def _agg(key):
        vals = [r[key] for r in all_followup if r[key] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    agg_ov_r  = _agg("overlap_raw_prev")
    agg_ov_b  = _agg("overlap_buf_prev")
    agg_gain  = _agg("avg_rank_gain")
    agg_hits  = _agg("buffer_hits_topk")
    agg_promo = _agg("n_promoted")

    lines += [
        "",
        _bar("═"),
        f"  AGGREGATE SUMMARY  ({len(all_results)} sessions · {n_fu} follow-up queries)",
        _bar("═"),
        "",
        f"  Avg top-{TOP_K} overlap (raw baseline):   {_pct(agg_ov_r)}",
        f"  Avg top-{TOP_K} overlap (with buffer):    {_pct(agg_ov_b)}"
        f"  (Δ {(agg_ov_b - agg_ov_r) * 100:+.1f} pp)",
        f"  Avg rank gain for hot chunks:     {_gain(agg_gain)} positions",
        f"  Avg buffer hits in top-{TOP_K}:        "
        f"{agg_hits:.1f} / {TOP_K}  ({agg_hits / TOP_K * 100:.0f}%)",
        f"  Avg chunks promoted per query:    {agg_promo:.1f}",
        "",
        f"  Interpretation:",
        f"    • A positive Δ overlap means the buffer makes retrieval more",
        f"      stable across related queries in a session.",
        f"    • Positive rank gain means hot chunks surfaced higher in follow-ups,",
        f"      increasing the chance the LLM sees contextually consistent chunks.",
        f"    • Buffer hits / top-{TOP_K} shows how much of the final result set",
        f"      was influenced by session memory.",
        "",
        _bar("═"),
    ]

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    global TOP_K, BOOST_WEIGHT

    parser = argparse.ArgumentParser(
        description="Evaluate ChunkFrequencyTracker (buffer pool) on textbook sessions."
    )
    parser.add_argument(
        "--config", default="config/config.yaml",
        help="Path to config YAML (default: config/config.yaml)"
    )
    parser.add_argument(
        "--out-dir", default=str(_RESULTS_DIR),
        help="Directory to write output files (default: tests/results)"
    )
    parser.add_argument(
        "--boost-weight", type=float, default=BOOST_WEIGHT,
        help=f"Buffer boost weight (default: {BOOST_WEIGHT})"
    )
    parser.add_argument(
        "--top-k", type=int, default=TOP_K,
        help=f"Top-K chunks to evaluate (default: {TOP_K})"
    )
    args = parser.parse_args()

    TOP_K        = args.top_k
    BOOST_WEIGHT = args.boost_weight

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load config ────────────────────────────────────────────────────────
    config_path = pathlib.Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)
    cfg = RAGConfig.from_yaml(config_path)

    # ── Find index ─────────────────────────────────────────────────────────
    index_dir = find_index_dir()
    if index_dir is None:
        print("ERROR: Textbook FAISS index not found.")
        print("       Run 'make run-index' first, then re-run this script.")
        sys.exit(1)

    model_path = pathlib.Path(cfg.embed_model)
    if not model_path.exists():
        print(f"ERROR: Embedding model not found at {model_path}")
        sys.exit(1)

    # ── Load artifacts ─────────────────────────────────────────────────────
    print(f"Loading index from {index_dir} ...")
    faiss_idx, _, chunks, _, _ = load_artifacts(index_dir, _INDEX_PREFIX)
    print(f"  {faiss_idx.ntotal:,} chunks  dim={faiss_idx.d}")

    print(f"Loading embedder (this may take a moment) ...")
    retriever = FAISSRetriever(faiss_idx, cfg.embed_model, nprobe=cfg.nprobe)
    print(f"  Ready.\n")

    # ── Run evaluation ─────────────────────────────────────────────────────
    all_results = []
    total_t0 = time.perf_counter()

    for i, session in enumerate(SESSIONS):
        print(f"[{i + 1}/{len(SESSIONS)}] {session['name']}  ({len(session['questions'])} queries)")
        t0 = time.perf_counter()
        result = evaluate_session(session, retriever, chunks)
        elapsed = time.perf_counter() - t0

        s = result["summary"]
        print(
            f"  done in {elapsed:.1f}s  |  "
            f"overlap raw={s['avg_overlap_raw']:.0%}  buf={s['avg_overlap_buffered']:.0%}  "
            f"Δ={( s['avg_overlap_buffered'] - s['avg_overlap_raw'])*100:+.1f}pp  |  "
            f"avg_rank_gain={s['avg_rank_gain']:+.2f}  |  "
            f"promoted/query={s['avg_promoted_per_query']:.1f}"
        )
        all_results.append(result)

    total_elapsed = time.perf_counter() - total_t0
    print(f"\nAll sessions complete in {total_elapsed:.1f}s\n")

    # ── Save JSON ──────────────────────────────────────────────────────────
    payload = {
        "config": {
            "index_dir":    str(index_dir),
            "n_chunks":     faiss_idx.ntotal,
            "embed_model":  cfg.embed_model,
            "top_k":        TOP_K,
            "pool_size":    POOL_SIZE,
            "boost_weight": BOOST_WEIGHT,
            "window_size":  WINDOW_SIZE,
        },
        "sessions": all_results,
    }
    json_path = out_dir / "chunk_buffer_eval.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, cls=_NpEncoder)
    print(f"JSON  → {json_path}")

    # ── Save and print text report ─────────────────────────────────────────
    report = format_report(all_results, faiss_idx.ntotal, str(index_dir))
    txt_path = out_dir / "chunk_buffer_report.txt"
    with open(txt_path, "w") as f:
        f.write(report)
    print(f"Report → {txt_path}\n")
    print(report)


if __name__ == "__main__":
    main()
