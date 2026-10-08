"""OpenCLBruteForceIndex against the CPU BruteForceIndex. Skipped unless built with OpenCL and a
device is visible; CI runs it on POCL, which executes the same kernels on a CPU.

The comparison is test_gpu.py's, so CUDA and OpenCL are held to the same bar: distances within
rtol 1e-4 / atol 1e-3 of the CPU's, at least 99% of ids equal (near-ties may swap places), rows sorted.
"""
import numpy as np
import pytest

import vecsearch as vs

pytestmark = pytest.mark.skipif(not vs.opencl_available(), reason="no OpenCL build or no device")

METHODS = ["naive", "skinny", "tiled", "auto"]
SUBGROUPS = [None, False]  # None: sub-group reductions where the device has them; False: local-memory fallback


def data(n, d, seed, normalize=False):
    x = np.random.default_rng(seed).standard_normal((n, d), dtype=np.float32)
    return vs.normalize(x) if normalize else x


def cpu_search(x, q, k, metric):
    cpu = vs.BruteForceIndex(x.shape[1], metric)
    cpu.add(x)
    return cpu.search(q, k, num_threads=2)


def assert_matches_cpu(ids_g, d_g, x, q, k, metric, min_agreement=0.99):
    ids_c, d_c = cpu_search(x, q, k, metric)
    assert ids_g.shape == (len(q), k)
    np.testing.assert_allclose(d_g, d_c, rtol=1e-4, atol=1e-3)  # tiled L2 uses |q|^2+|x|^2-2q.x
    assert (ids_g == ids_c).mean() > min_agreement  # near-ties may swap places
    assert np.all(np.diff(d_g, axis=1) >= 0)
    # Whatever id came back, its distance on the CPU must be the one the device reported.
    ok = ids_g >= 0
    rows = np.repeat(np.arange(len(q)), k).reshape(len(q), k)
    a, b = q[rows[ok]], x[ids_g[ok]]
    truth = ((a - b) ** 2).sum(1) if metric == "l2" else 1.0 - (a * b).sum(1)
    np.testing.assert_allclose(d_g[ok], truth, rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("subgroups", SUBGROUPS)
@pytest.mark.parametrize("metric", ["l2", "ip"])
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("d", [37, 128])  # 37 is padded to 40 on the device
@pytest.mark.parametrize("nq", [1, 5, 300])
def test_matches_cpu(metric, method, d, nq, subgroups):
    x, q, k = data(3000, d, 1, metric == "ip"), data(nq, d, 2, metric == "ip"), 10
    gpu = vs.OpenCLBruteForceIndex(d, metric, subgroups=subgroups)
    gpu.add(x[:1000])  # two adds: the device buffer grows and keeps the first part
    gpu.add(x[1000:])
    assert len(gpu) == 3000
    ids_g, d_g = gpu.search(q, k, method=method)
    assert_matches_cpu(ids_g, d_g, x, q, k, metric)


def test_devices_and_subgroup_flag():
    devs = vs.opencl_devices()
    assert devs and all(d["name"] and d["max_alloc"] > 0 for d in devs)
    assert vs.OpenCLBruteForceIndex(8, subgroups=False).uses_subgroups is False
    assert vs.OpenCLBruteForceIndex(8).uses_subgroups == devs[0]["subgroups"]
    assert vs.OpenCLBruteForceIndex(8).device_name == devs[0]["name"]
    with pytest.raises(ValueError):
        vs.OpenCLBruteForceIndex(8, device=len(devs))


@pytest.mark.parametrize("k", [1, 16, 17, 32, 33, 64, 65, 100, 128])
def test_k_buckets(k):
    x, q = data(20000, 64, 3), data(40, 64, 4)
    gpu = vs.OpenCLBruteForceIndex(64)
    gpu.add(x)
    ids_c, _ = cpu_search(x, q, k, "l2")
    ids_g, _ = gpu.search(q, k)
    assert vs.recall_at_k(ids_g, ids_c, k) > 0.99


def test_k_larger_than_dataset():
    x = data(3, 8, 7)
    for method in ["naive", "skinny", "tiled"]:
        gpu = vs.OpenCLBruteForceIndex(8)
        gpu.add(x)
        ids, dists = gpu.search(np.zeros((2, 8), np.float32), 5, method=method)
        assert (ids[:, :3] >= 0).all() and (ids[:, 3:] == -1).all() and np.isinf(dists[:, 3:]).all()
        assert sorted(ids[0, :3]) == [0, 1, 2]
    one = vs.OpenCLBruteForceIndex(8)
    one.add(x[:1])
    ids, dists = one.search(x[:1], 128)  # a single vector, the largest k
    assert ids[0, 0] == 0 and (ids[0, 1:] == -1).all() and dists[0, 0] < 1e-5


@pytest.mark.parametrize("metric", ["l2", "ip"])
@pytest.mark.parametrize("method", METHODS)
def test_batch_size_one(metric, method):
    x, q = data(5000, 64, 8, metric == "ip"), data(1, 64, 9, metric == "ip")
    gpu = vs.OpenCLBruteForceIndex(64, metric)
    gpu.add(x)
    ids_g, d_g = gpu.search(q[0], 10, method=method)  # a 1-D query is one row
    assert_matches_cpu(ids_g, d_g, x, q, 10, metric)


@pytest.mark.parametrize("n", [1, 9, 127, 128, 129, 257, 1000])
@pytest.mark.parametrize("nq", [1, 127, 129])
@pytest.mark.parametrize("d", [1, 9, 130])
def test_sizes_that_dont_divide_into_tiles(n, nq, d):
    # 128 is the tile edge of the tiled kernel; tiny dims have many near-ties, hence 0.9 agreement.
    x, q = data(n, d, 10), data(nq, d, 11)
    gpu = vs.OpenCLBruteForceIndex(d)
    gpu.add(x)
    k = min(7, n)
    ids_g, d_g = gpu.search(q, k, method="tiled")
    assert_matches_cpu(ids_g, d_g, x, q, k, "l2", min_agreement=0.9)


def test_very_large_batch_rounds_and_chunks():
    # 5000 queries: two query rounds, and the data is split into chunks inside each round.
    x, q = data(70000, 16, 5), data(5000, 16, 6)
    gpu = vs.OpenCLBruteForceIndex(16)
    gpu.add(x)
    ids_c, _ = cpu_search(x, q, 10, "l2")
    ids_g, _ = gpu.search(q, 10)
    assert vs.recall_at_k(ids_g, ids_c, 10) > 0.99


def test_paging_when_one_allocation_is_limited(monkeypatch):
    # 4 MB allocation limit: 40,000 x 128 floats become 5 pages, ids continue across pages.
    monkeypatch.setenv("VECSEARCH_OPENCL_MAX_ALLOC_MB", "4")
    x, q = data(40000, 128, 12), data(20, 128, 13)
    for method in ["skinny", "tiled", "naive"]:
        gpu = vs.OpenCLBruteForceIndex(128)
        gpu.add(x[:15000])
        gpu.add(x[15000:])
        ids_g, d_g = gpu.search(q, 10, method=method)
        assert_matches_cpu(ids_g, d_g, x, q, 10, "l2")
        assert ids_g.max() > 32768  # winners come from the later pages too


def test_missing_results_and_limits():
    gpu = vs.OpenCLBruteForceIndex(8)
    ids, dists = gpu.search(np.zeros((2, 8), np.float32), 5)  # empty index
    assert (ids == -1).all() and np.isinf(dists).all()
    gpu.add(data(3, 8, 7))
    ids, dists = gpu.search(np.zeros(8, np.float32), 5)
    assert (ids[0, :3] >= 0).all() and (ids[0, 3:] == -1).all() and np.isinf(dists[0, 3:]).all()
    with pytest.raises(ValueError):
        gpu.search(np.zeros(8, np.float32), 129)
    with pytest.raises(ValueError):
        gpu.search(np.zeros(8, np.float32), 5, method="fast")
    with pytest.raises(ValueError):
        gpu.add(np.zeros((2, 9), np.float32))  # wrong dim
