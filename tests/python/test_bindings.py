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
