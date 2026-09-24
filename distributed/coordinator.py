"""Scatter-gather coordinator: queries every shard concurrently and merges the
per-shard top-k lists. Shards that fail or time out are skipped, so a partial
answer beats no answer.
"""
import argparse
import asyncio
import heapq
import time

import grpc

import protogen  # noqa: F401  (generates search_pb2 if missing)
import search_pb2
import search_pb2_grpc


class Coordinator(search_pb2_grpc.SearcherServicer):
    def __init__(self, addrs, timeout_ms):
        self.addrs = addrs
        self.timeout = timeout_ms / 1e3
        self.stubs = [search_pb2_grpc.SearcherStub(grpc.aio.insecure_channel(a)) for a in addrs]

    async def Search(self, req, ctx):
        t = time.perf_counter()
        calls = [s.Search(req, timeout=self.timeout) for s in self.stubs]
        results = await asyncio.gather(*calls, return_exceptions=True)
        lists, timings = [], {}
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                timings[f"shard_{i}_failed"] = 1.0
                continue
            lists.append(zip(r.distances, r.ids))
            timings.update(r.timings_ms)
        if not lists:
            await ctx.abort(grpc.StatusCode.UNAVAILABLE, "all shards failed")
        # shard lists are sorted, so a lazy k-way merge is enough
        top = heapq.nsmallest(req.k, heapq.merge(*lists))
        timings["coordinator_total"] = (time.perf_counter() - t) * 1e3
        return search_pb2.SearchResponse(ids=[i for _, i in top], distances=[d for d, _ in top],
                                         timings_ms=timings)

    async def Info(self, req, ctx):
        infos = await asyncio.gather(*[s.Info(req) for s in self.stubs])
        return search_pb2.InfoResponse(num_vectors=sum(i.num_vectors for i in infos),
                                       dim=infos[0].dim, num_shards=len(infos))


async def serve(addrs, port, timeout_ms):
    server = grpc.aio.server(options=[("grpc.so_reuseport", 0)])
    search_pb2_grpc.add_SearcherServicer_to_server(Coordinator(addrs, timeout_ms), server)
    server.add_insecure_port(f"[::]:{port}")
    await server.start()
    print(f"[coordinator] :{port} -> {addrs}", flush=True)
    await server.wait_for_termination()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", required=True, help="host:port,host:port,...")
    ap.add_argument("--port", type=int, default=50050)
    ap.add_argument("--timeout-ms", type=float, default=1000)
    a = ap.parse_args()
    asyncio.run(serve(a.workers.split(","), a.port, a.timeout_ms))
