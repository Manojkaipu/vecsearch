"""GpuBruteForceIndex against the CPU BruteForceIndex. Skipped unless built with CUDA and a GPU is visible."""
import numpy as np
import pytest

import vecsearch as vs

pytestmark = pytest.mark.skipif(not vs.cuda_available(), reason="no CUDA build or no GPU")

METHODS = ["naive", "skinny", "tiled", "auto"]


def data(n, d, seed, normalize=False):
    x = np.random.default_rng(seed).standard_normal((n, d), dtype=np.float32)
    return vs.normalize(x) if normalize else x


@pytest.mark.parametrize("metric", ["l2", "ip"])
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("d", [37, 128])  # 37 is padded to 40 on the device
@pytest.mark.parametrize("nq", [1, 5, 300])
def test_matches_cpu(metric, method, d, nq):
    x, q, k = data(3000, d, 1, metric == "ip"), data(nq, d, 2, metric == "ip"), 10
    cpu = vs.BruteForceIndex(d, metric)
    cpu.add(x)
    gpu = vs.GpuBruteForceIndex(d, metric)
    gpu.add(x[:1000])  # two adds: the device buffer grows and keeps the first part
    gpu.add(x[1000:])
    ids_c, d_c = cpu.search(q, k)
    ids_g, d_g = gpu.search(q, k, method=method)
    assert ids_g.shape == (nq, k)
    np.testing.assert_allclose(d_g, d_c, rtol=1e-4, atol=1e-3)  # tiled L2 uses |q|^2+|x|^2-2q.x
    assert (ids_g == ids_c).mean() > 0.99  # near-ties may swap places
    assert np.all(np.diff(d_g, axis=1) >= 0)


@pytest.mark.parametrize("k", [1, 16, 17, 100, 128])
def test_k_buckets(k):
    x, q = data(20000, 64, 3), data(40, 64, 4)
    cpu = vs.BruteForceIndex(64)
    cpu.add(x)
    gpu = vs.GpuBruteForceIndex(64)
    gpu.add(x)
    ids_c, _ = cpu.search(q, k)
    ids_g, _ = gpu.search(q, k)
    assert vs.recall_at_k(ids_g, ids_c, k) > 0.99


def test_many_chunks_and_rounds():
    # 5000 queries: two query rounds, and the data is split into chunks inside each round.
    x, q = data(70000, 16, 5), data(5000, 16, 6)
    cpu = vs.BruteForceIndex(16)
    cpu.add(x)
    gpu = vs.GpuBruteForceIndex(16)
    gpu.add(x)
    ids_c, _ = cpu.search(q, 10, num_threads=4)
    ids_g, _ = gpu.search(q, 10)
    assert vs.recall_at_k(ids_g, ids_c, 10) > 0.99


def test_missing_results_and_limits():
    gpu = vs.GpuBruteForceIndex(8)
    ids, dists = gpu.search(np.zeros((2, 8), np.float32), 5)  # empty index
    assert (ids == -1).all() and np.isinf(dists).all()
    gpu.add(data(3, 8, 7))
    ids, dists = gpu.search(np.zeros(8, np.float32), 5)
    assert (ids[0, :3] >= 0).all() and (ids[0, 3:] == -1).all() and np.isinf(dists[0, 3:]).all()
    with pytest.raises(ValueError):
        gpu.search(np.zeros(8, np.float32), 129)
    with pytest.raises(ValueError):
        gpu.search(np.zeros(8, np.float32), 5, method="fast")
