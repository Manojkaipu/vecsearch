"""Run N shard workers and a coordinator as local processes.

    python distributed/cluster.py --manifest indexes/4shards/manifest.json
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time

import grpc

import protogen  # noqa: F401  (generates search_pb2 if missing)
import search_pb2
import search_pb2_grpc

HERE = os.path.dirname(os.path.abspath(__file__))


def free_port():
    # Fixed ports collided across back-to-back runs: with SO_REUSEPORT (gRPC's
    # default) a new server could silently share a port with a dying one.
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class LocalCluster:
    def __init__(self, manifest_path, base_port=None, coord_port=None, pin=False, worker_threads=4):
        self.manifest_path = os.path.abspath(manifest_path)
        self.m = json.load(open(manifest_path))
        self.base_port, self.coord_port = base_port, coord_port
        self.pin, self.worker_threads = pin, worker_threads
        self.procs = []

    def __enter__(self):
        ncpu = os.cpu_count() or 1
        n = len(self.m["shards"])
        per = max(1, ncpu // n)
        addrs = []
        self.coord_port = self.coord_port or free_port()
        for i, s in enumerate(self.m["shards"]):
            port = self.base_port + i if self.base_port else free_port()
            cmd = [sys.executable, os.path.join(HERE, "shard_worker.py"), "--manifest", self.manifest_path,
                   "--shard-id", str(i), "--port", str(port),
                   "--threads", str(self.worker_threads)]
            if self.pin:  # disjoint cores per shard
                cmd += ["--cpus", ",".join(str((i * per + c) % ncpu) for c in range(per))]
            self.procs.append(subprocess.Popen(cmd))
            addrs.append(f"localhost:{port}")
        self.procs.append(subprocess.Popen([sys.executable, os.path.join(HERE, "coordinator.py"),
                                            "--workers", ",".join(addrs), "--port", str(self.coord_port)]))
        self._wait_ready([f"localhost:{self.coord_port}"] + addrs)
        return f"localhost:{self.coord_port}"

    def _wait_ready(self, addrs, timeout=600):
        deadline = time.time() + timeout
        for a in addrs:
            while True:
                try:
                    search_pb2_grpc.SearcherStub(grpc.insecure_channel(a)).Info(
                        search_pb2.InfoRequest(), timeout=2)
                    break
                except grpc.RpcError:
                    if time.time() > deadline:
                        raise TimeoutError(f"{a} did not come up")
                    time.sleep(0.5)

    def __exit__(self, *exc):
        for p in self.procs:
            p.terminate()
        for p in self.procs:
            p.wait()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pin", action="store_true")
    a = ap.parse_args()
    with LocalCluster(a.manifest, pin=a.pin) as addr:
        print(f"cluster ready at {addr}; Ctrl-C to stop", flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
