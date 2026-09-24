"""p50/p95/p99 latency, QPS and recall vs shard count.

For each N in --shards: build (or reuse) the shard indexes, start a local
cluster and run a closed loop with --concurrency requests in flight. The
in-process row (same index, no RPC) separates search time from gRPC overhead.
"""
import argparse
import asyncio
import csv
import os
import sys
import time

import grpc
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protogen  # noqa: E402,F401
import search_pb2  # noqa: E402
import search_pb2_grpc  # noqa: E402
import vecsearch as vs  # noqa: E402
from build_shards import build, shard_path  # noqa: E402
from cluster import LocalCluster  # noqa: E402


async def run_load(addr, xq, k, ef, concurrency):
    ch = grpc.aio.insecure_channel(addr)
    stub = search_pb2_grpc.SearcherStub(ch)
    reqs = [search_pb2.SearchRequest(query=q.tobytes(), k=k, ef=ef) for q in xq]
    for r in reqs[:50]:  # warm-up
        await stub.Search(r)
    lat = np.zeros(len(reqs))
    ids = np.full((len(reqs), k), -1, dtype=np.int64)
    nxt = 0

    async def worker():
        nonlocal nxt
        while nxt < len(reqs):
            i = nxt
            nxt += 1
            t = time.perf_counter()
            resp = await stub.Search(reqs[i])
            lat[i] = (time.perf_counter() - t) * 1e3
            ids[i, : len(resp.ids)] = resp.ids

    t0 = time.perf_counter()
    await asyncio.gather(*[worker() for _ in range(concurrency)])
    wall = time.perf_counter() - t0
    await ch.close()
    return lat, ids, len(reqs) / wall


def summarize(label, shards, lat, ids, gt, qps, k):
    return dict(mode=label, shards=shards, p50_ms=np.percentile(lat, 50), p95_ms=np.percentile(lat, 95),
                p99_ms=np.percentile(lat, 99), qps=qps, recall=vs.recall_at_k(ids, gt, k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tweets")
    ap.add_argument("--shards", default="1,2,4")
    ap.add_argument("--n-queries", type=int, default=5000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ef", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=1, help="1 = pure latency; >1 = latency under load")
    ap.add_argument("--pin", action="store_true", help="pin each shard to its own CPU cores")
    ap.add_argument("--name", default="tweets")
    a = ap.parse_args()

    xq = np.load(os.path.join(a.data, "queries.npy"))[: a.n_queries]
    gt = np.load(os.path.join(a.data, "gt.npy"))[: a.n_queries]
    rows = []

    one = os.path.join("indexes", f"{os.path.basename(a.data)}_1shards")
    m = build(a.data, 1, one)
    idx = vs.HNSWIndex.load(shard_path(os.path.join(one, "manifest.json"), m["shards"][0]))
    for q in xq[:50]:
        idx.search(q, a.k, a.ef)
    lat = np.zeros(len(xq))
    ids = np.zeros((len(xq), a.k), dtype=np.int64)
    for i, q in enumerate(xq):
        t = time.perf_counter()
        ids[i] = idx.search(q, a.k, a.ef)[0][0]
        lat[i] = (time.perf_counter() - t) * 1e3
    rows.append(summarize("in-process", 1, lat, ids, gt, len(xq) / (lat.sum() / 1e3), a.k))
    del idx
    print(rows[-1], flush=True)

    for n in map(int, a.shards.split(",")):
        mpath = os.path.join("indexes", f"{os.path.basename(a.data)}_{n}shards")
        build(a.data, n, mpath)
        with LocalCluster(os.path.join(mpath, "manifest.json"), pin=a.pin) as addr:
            lat, ids, qps = asyncio.run(run_load(addr, xq, a.k, a.ef, a.concurrency))
        rows.append(summarize("grpc", n, lat, ids, gt, qps, a.k))
        print(rows[-1], flush=True)

    out = os.path.join("results", a.name)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"distributed_c{a.concurrency}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
