"""
chunk_buffer.py

Session-scoped hot-chunk tracker modeled after the database buffer pool.

In a DBMS, a buffer pool keeps frequently-accessed disk pages in memory so
subsequent accesses are served from RAM rather than disk.  ChunkFrequencyTracker
applies the same principle to retrieved text chunks: chunks surfaced repeatedly
across queries in a session are considered "hot" and receive a proportional score
boost at ranking time, biasing the retriever toward them in future queries.

Eviction policy mirrors FIFO buffer pool replacement: the tracker maintains a
fixed-width sliding window over recent queries.  When the window is full, the
oldest query's chunk set is evicted and its frequency counts are decremented —
guaranteeing that the hot set reflects only recent access patterns.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Dict, Iterable, List, Tuple


class ChunkFrequencyTracker:
    """
    Sliding-window frequency tracker for chunk IDs retrieved during a session.

    Args:
        window_size:  Number of most-recent queries whose results are retained
                      in the hot set.  Older entries are evicted FIFO.
        max_chunks:   Cap on distinct tracked chunk IDs.  When exceeded the
                      least-frequent entries are dropped to bound memory use.
    """

    def __init__(self, window_size: int = 20, max_chunks: int = 500):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if max_chunks < 1:
            raise ValueError("max_chunks must be >= 1")
        self.window_size = window_size
        self.max_chunks  = max_chunks
        self._window: deque[frozenset] = deque()
        self._freq:   Counter          = Counter()

    # ── Write ──────────────────────────────────────────────────────────────

    def record(self, chunk_ids: List[int]) -> None:
        """Record the chunk IDs returned by the most-recent query."""
        if not chunk_ids:
            return

        entry = frozenset(chunk_ids)
        self._window.append(entry)
        for cid in entry:
            self._freq[cid] += 1

        # Evict oldest entry when the window is full
        if len(self._window) > self.window_size:
            evicted = self._window.popleft()
            for cid in evicted:
                self._freq[cid] -= 1
                if self._freq[cid] <= 0:
                    del self._freq[cid]

        # Trim to max_chunks by dropping the least-frequent entries
        while len(self._freq) > self.max_chunks:
            least = min(self._freq, key=self._freq.__getitem__)
            del self._freq[least]

    # ── Read ───────────────────────────────────────────────────────────────

    def get_hot_scores(self, candidates: Iterable[int]) -> Dict[int, float]:
        """
        Return a [0, 1]-normalised frequency score for each hot candidate.

        Only candidates that appear in the current hot set are included;
        cold candidates are omitted (implicit score of zero).
        """
        if not self._freq:
            return {}

        hits = {cid: self._freq[cid] for cid in candidates if cid in self._freq}
        if not hits:
            return {}

        max_freq = max(self._freq.values())
        return {cid: cnt / max_freq for cid, cnt in hits.items()}

    def rerank_with_boost(
        self,
        ordered: List[int],
        scores: List[float],
        boost_weight: float,
    ) -> Tuple[List[int], List[float]]:
        """
        Re-order candidates by adding a proportional hot-chunk bonus.

        Each hot chunk's score is scaled up by (1 + boost_weight * hot_score),
        where hot_score is its normalised access frequency in [0, 1].  The
        boost is multiplicative so that it scales correctly regardless of
        whether the underlying scores come from RRF (small, ~0.01-0.02) or
        linear fusion (range [0, 1]).

        Chunks not in the hot set are unaffected.  Cold candidates with a
        score of exactly 0.0 receive a small additive floor so that a non-zero
        boost can still move them.
        """
        if boost_weight <= 0:
            return ordered, scores

        hot = self.get_hot_scores(ordered)
        if not hot:
            return ordered, scores

        # For zero-scored hot chunks, use max(scores) as the floor so the
        # multiplicative boost can compete with non-zero cold candidates.
        max_score = max(scores) if scores else 1.0

        score_map: Dict[int, float] = {}
        for cid, s in zip(ordered, scores):
            h = hot.get(cid, 0.0)
            if h > 0:
                base = s if s > 0 else max_score
                score_map[cid] = base * (1.0 + boost_weight * h)
            else:
                score_map[cid] = s

        new_ordered = sorted(score_map, key=score_map.__getitem__, reverse=True)
        new_scores  = [score_map[cid] for cid in new_ordered]
        return new_ordered, new_scores

    # ── Introspection ──────────────────────────────────────────────────────

    @property
    def hot_chunk_count(self) -> int:
        """Number of distinct chunks currently in the hot set."""
        return len(self._freq)

    @property
    def query_count(self) -> int:
        """Number of queries currently tracked in the window."""
        return len(self._window)

    def top_chunks(self, n: int = 10) -> List[Tuple[int, int]]:
        """Return the n most-frequent (chunk_id, access_count) pairs."""
        return self._freq.most_common(n)

    def clear(self) -> None:
        """Reset all state — use when starting a fresh session."""
        self._window.clear()
        self._freq.clear()
