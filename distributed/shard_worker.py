"""gRPC server for one shard. A thread pool is fine here because search()
releases the GIL."""
import argparse
import json
import os
import time
from concurrent import futures

import grpc
import numpy as np

import protogen  # noqa: F401  (generates search_pb2 if missing)
import search_pb2
import search_pb2_grpc
import vecsearch as vs


class ShardServicer(search_pb2_grpc.SearcherServicer):
    def __init__(self, index_path, offset, shard_id):
        t = time.time()
        self.idx = vs.HNSWIndex.load(index_path)
        self.offset = offset
        self.shard_id = shard_id
        print(f"[shard {shard_id}] loaded {len(self.idx):,} vectors in {time.time() - t:.1f}s", flush=True)

    def Search(self, req, ctx):
        t = time.perf_counter()
        q = np.frombuffer(req.query, dtype=np.float32)
        if q.shape[0] != self.idx.dim:
            ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, f"expected dim {self.idx.dim}, got {q.shape[0]}")
        ids, d = self.idx.search(q, k=req.k, ef=req.ef)
        ids, d = ids[0], d[0]
        keep = ids >= 0
        return search_pb2.SearchResponse(
            ids=(ids[keep] + self.offset).tolist(), distances=d[keep].tolist(),
            timings_ms={f"shard_{self.shard_id}": (time.perf_counter() - t) * 1e3})

    def Info(self, req, ctx):
        return search_pb2.InfoResponse(num_vectors=len(self.idx), dim=self.idx.dim, num_shards=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--shard-id", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--threads", type=int, default=4, help="gRPC handler threads")
    ap.add_argument("--cpus", default="", help="pin to CPUs, e.g. '2,3' (Linux)")
    a = ap.parse_args()
    if a.cpus and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {int(c) for c in a.cpus.split(",")})
    shard = json.load(open(a.manifest))["shards"][a.shard_id]
    path = os.path.join(os.path.dirname(os.path.abspath(a.manifest)), shard["file"])
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=a.threads),
                         options=[("grpc.so_reuseport", 0)])  # fail on port clash instead of sharing
    search_pb2_grpc.add_SearcherServicer_to_server(ShardServicer(path, shard["offset"], a.shard_id), server)
    server.add_insecure_port(f"[::]:{a.port}")
    server.start()
    print(f"[shard {a.shard_id}] serving on :{a.port}", flush=True)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
