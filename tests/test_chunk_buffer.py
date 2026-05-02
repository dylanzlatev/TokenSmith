"""
test_chunk_buffer.py

Unit tests for ChunkFrequencyTracker — the session-scoped hot-chunk buffer
modeled after the database buffer pool.

Run:
    pytest tests/test_chunk_buffer.py -s -v
"""

import pytest
from src.chunk_buffer import ChunkFrequencyTracker


# ── record() and basic frequency counting ─────────────────────────────────

def test_record_increments_frequency():
    t = ChunkFrequencyTracker(window_size=10)
    t.record([1, 2, 3])
    t.record([2, 3])
    t.record([3])
    assert t._freq[3] == 3
    assert t._freq[2] == 2
    assert t._freq[1] == 1


def test_record_empty_list_is_noop():
    t = ChunkFrequencyTracker(window_size=5)
    t.record([])
    assert t.query_count == 0
    assert t.hot_chunk_count == 0


def test_record_duplicate_ids_within_query_counted_once():
    """A chunk appearing multiple times in one query result is only counted once."""
    t = ChunkFrequencyTracker(window_size=5)
    t.record([7, 7, 7])
    assert t._freq[7] == 1


# ── Sliding window / LRU eviction ─────────────────────────────────────────

def test_window_eviction_decrements_frequency():
    """After window_size queries the oldest entry is evicted and its counts drop."""
    t = ChunkFrequencyTracker(window_size=3)
    t.record([10])   # query 1 — will be evicted after 3 more
    t.record([20])   # query 2
    t.record([30])   # query 3  ← window full
    assert t._freq[10] == 1

    t.record([40])   # query 4  ← evicts query 1
    assert 10 not in t._freq, "chunk 10 should be evicted"
    assert t._freq[20] == 1
    assert t._freq[30] == 1
    assert t._freq[40] == 1


def test_window_size_respected():
    t = ChunkFrequencyTracker(window_size=4)
    for i in range(10):
        t.record([i])
    assert t.query_count == 4


def test_shared_chunk_survives_partial_eviction():
    """A chunk in multiple queries survives until ALL its queries are evicted."""
    t = ChunkFrequencyTracker(window_size=2)
    t.record([99])   # query 1
    t.record([99])   # query 2  ← window full, freq[99]=2
    t.record([1])    # query 3  ← evicts query 1, freq[99]=1 (still alive)
    assert 99 in t._freq
    assert t._freq[99] == 1

    t.record([2])    # query 4  ← evicts query 2, freq[99]=0 → deleted
    assert 99 not in t._freq


# ── get_hot_scores() ───────────────────────────────────────────────────────

def test_hot_scores_returns_empty_when_tracker_empty():
    t = ChunkFrequencyTracker()
    assert t.get_hot_scores([1, 2, 3]) == {}


def test_hot_scores_only_includes_hot_candidates():
    t = ChunkFrequencyTracker(window_size=5)
    t.record([1, 2])
    scores = t.get_hot_scores([1, 3, 5])   # 3 and 5 are cold
    assert set(scores.keys()) == {1}


def test_hot_scores_are_normalised_to_one():
    """The most-frequent chunk must have a score of exactly 1.0."""
    t = ChunkFrequencyTracker(window_size=10)
    t.record([1])
    t.record([1])
    t.record([1, 2])
    scores = t.get_hot_scores([1, 2])
    assert scores[1] == pytest.approx(1.0)
    assert 0.0 < scores[2] < 1.0


def test_hot_scores_proportional_to_frequency():
    t = ChunkFrequencyTracker(window_size=10)
    for _ in range(3):
        t.record([1])
    t.record([2])
    scores = t.get_hot_scores([1, 2])
    # chunk 1 appeared 3×, chunk 2 appeared 1× → ratio should be 3:1
    assert scores[1] == pytest.approx(1.0)
    assert scores[2] == pytest.approx(1 / 3)


# ── rerank_with_boost() ────────────────────────────────────────────────────

def test_rerank_hot_chunk_moves_up():
    """A hot chunk ranked below a cold chunk should move above it after boost."""
    t = ChunkFrequencyTracker(window_size=5)
    t.record([42])   # make chunk 42 hot

    # Initial ranking: chunk 10 scores higher than chunk 42
    ordered = [10, 42, 99]
    scores  = [1.0, 0.8, 0.5]

    new_ordered, new_scores = t.rerank_with_boost(ordered, scores, boost_weight=0.5)
    # Chunk 42 (hot, score 0.8 * 1.5 = 1.2) should now beat chunk 10 (score 1.0)
    assert new_ordered[0] == 42, "hot chunk should rise to first place"


def test_rerank_zero_boost_weight_no_change():
    t = ChunkFrequencyTracker(window_size=5)
    t.record([1])
    ordered = [1, 2, 3]
    scores  = [0.9, 0.95, 0.8]
    new_ordered, new_scores = t.rerank_with_boost(ordered, scores, boost_weight=0.0)
    assert new_ordered == ordered


def test_rerank_no_hot_chunks_no_change():
    """When none of the candidates are hot, ordering must be unchanged."""
    t = ChunkFrequencyTracker(window_size=5)
    t.record([999])  # a completely different chunk
    ordered = [1, 2, 3]
    scores  = [0.9, 0.7, 0.5]
    new_ordered, new_scores = t.rerank_with_boost(ordered, scores, boost_weight=0.5)
    assert new_ordered == ordered


def test_rerank_cold_chunks_scores_unchanged():
    """Cold chunks must keep their exact original scores."""
    t = ChunkFrequencyTracker(window_size=5)
    t.record([1])
    ordered = [1, 2, 3]
    scores  = [0.5, 0.9, 0.8]
    _, new_scores = t.rerank_with_boost(ordered, scores, boost_weight=0.2)
    score_map = dict(zip(ordered, scores))
    new_map   = dict(zip(*t.rerank_with_boost(ordered, scores, boost_weight=0.2)))
    assert new_map[2] == pytest.approx(score_map[2])
    assert new_map[3] == pytest.approx(score_map[3])


def test_rerank_zero_score_chunk_still_receives_boost():
    """A chunk with score 0.0 must still move up when hot (floor prevents mult-by-zero)."""
    t = ChunkFrequencyTracker(window_size=5)
    t.record([7])
    ordered = [1, 7]
    scores  = [0.1, 0.0]
    new_ordered, _ = t.rerank_with_boost(ordered, scores, boost_weight=0.5)
    assert new_ordered[0] == 7, "hot chunk with score 0 should overtake cold chunk with score 0.1"


# ── max_chunks cap ─────────────────────────────────────────────────────────

def test_max_chunks_cap():
    t = ChunkFrequencyTracker(window_size=100, max_chunks=5)
    for i in range(20):
        t.record([i])
    assert t.hot_chunk_count <= 5


# ── clear() ────────────────────────────────────────────────────────────────

def test_clear_resets_state():
    t = ChunkFrequencyTracker(window_size=5)
    t.record([1, 2, 3])
    t.clear()
    assert t.hot_chunk_count == 0
    assert t.query_count == 0
    assert t.get_hot_scores([1, 2, 3]) == {}


# ── top_chunks() ──────────────────────────────────────────────────────────

def test_top_chunks_ordering():
    t = ChunkFrequencyTracker(window_size=20)
    t.record([1, 2, 3])
    t.record([1, 2])
    t.record([1])
    top = t.top_chunks(3)
    ids = [cid for cid, _ in top]
    assert ids[0] == 1   # chunk 1 accessed 3×
    assert ids[1] == 2   # chunk 2 accessed 2×
    assert ids[2] == 3   # chunk 3 accessed 1×


# ── constructor validation ─────────────────────────────────────────────────

def test_invalid_window_size():
    with pytest.raises(ValueError, match="window_size"):
        ChunkFrequencyTracker(window_size=0)


def test_invalid_max_chunks():
    with pytest.raises(ValueError, match="max_chunks"):
        ChunkFrequencyTracker(max_chunks=0)
