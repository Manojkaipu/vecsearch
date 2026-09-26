import numpy as np
import pytest

import vecsearch as vs


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    return rng.standard_normal((3000, 64)).astype(np.float32), rng.standard_normal((100, 64)).astype(np.float32)


def test_recall_vs_bruteforce(data):
    x, q = data
    h = vs.HNSWIndex(64, len(x), M=16, ef_construction=200)
    h.add(x)
    bf = vs.BruteForceIndex(64)
    bf.add(x)
    ids, dists = h.search(q, k=10, ef=100)
    gt, _ = bf.search(q, k=10)
    assert ids.shape == (100, 10) and ids.dtype == np.int64
    assert np.all(np.diff(dists, axis=1) >= 0)
    assert vs.recall_at_k(ids, gt) >= 0.95


def test_cosine_and_forcecast(data):
    x, q = data
    xn = vs.normalize(x).astype(np.float64)
    h = vs.HNSWIndex(64, len(x), metric="cosine")
    h.add(xn)
    ids, d = h.search(vs.normalize(x[:5]), k=1, ef=50)
    assert list(ids[:, 0]) == [0, 1, 2, 3, 4]
    assert np.allclose(d[:, 0], 0, atol=1e-5)


def test_save_load(tmp_path, data):
    x, q = data
    h = vs.HNSWIndex(64, len(x))
    h.add(x, num_threads=2)
    p = str(tmp_path / "idx.bin")
    h.save(p)
    h2 = vs.HNSWIndex.load(p)
    assert len(h2) == len(h)
    a, _ = h.search(q, 10, 50)
    b, _ = h2.search(q, 10, 50)
    assert np.array_equal(a, b)


def test_bad_shape_raises():
    h = vs.HNSWIndex(8, 10)
    with pytest.raises(ValueError):
        h.add(np.zeros((3, 9), np.float32))


def test_missing_results_are_minus_one():
    h = vs.HNSWIndex(4, 10)
    h.add(np.eye(4, dtype=np.float32))
    ids, d = h.search(np.ones(4, np.float32), k=6)
    assert (ids[0, 4:] == -1).all() and np.isinf(d[0, 4:]).all()


def test_hnsw_filter(data):
    x, q = data
    h = vs.HNSWIndex(64, len(x))
    h.add(x)
    mask = np.zeros(len(x), bool)
    mask[::3] = True
    ids, _ = h.search(q, k=10, ef=100, filter=mask)
    assert mask[ids].all()
    exact, _ = h.search(q, k=10, filter=mask, exact_below=len(x))
    bf = vs.BruteForceIndex(64)
    bf.add(x[mask])
    gt, _ = bf.search(q, k=10)
    assert np.array_equal(exact, np.flatnonzero(mask)[gt])
    with pytest.raises(ValueError):
        h.search(q, filter=np.ones(5, bool))


def test_bm25_roundtrip(tmp_path):
    docs = ["iPhone battery drains after update", "refund for cancelled flight",
            "battery replacement cost", "app crashes on launch"]
    idx = vs.BM25Index()
    idx.add(docs)
    assert len(idx) == 4 and vs.BM25Index.tokenize("It's the iOS-11 update") == ["ios", "11", "update"]
    ids, scores = idx.search("battery", k=3)
    assert set(ids[0, :2]) == {0, 2} and ids[0, 2] == -1 and (np.diff(scores[0, :2]) <= 0).all()
    ids, _ = idx.search(["battery", "refund"], k=1, filter=np.array([0, 1, 1, 1], bool))
    assert ids[:, 0].tolist() == [2, 1]
    p = str(tmp_path / "bm25.bin")
    idx.save(p)
    assert np.array_equal(vs.BM25Index.load(p).search("battery", k=3)[0], idx.search("battery", k=3)[0])


def test_rrf():
    ids, scores = vs.rrf([[1, 2, 3], [3, 1, -1]], k=3)
    assert ids == [1, 3, 2]  # 1: 1/61+1/62, 3: 1/63+1/61, 2: 1/62
    assert scores[0] > scores[1] > scores[2]
