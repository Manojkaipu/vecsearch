"""Generates search_pb2*.py on first import. They aren't committed, so they always
match the installed grpcio-tools. Import before search_pb2."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(HERE, "search_pb2_grpc.py")):
    from grpc_tools import protoc

    rc = protoc.main(["protoc", f"-I{HERE}", f"--python_out={HERE}", f"--grpc_python_out={HERE}",
                      os.path.join(HERE, "search.proto")])
    if rc != 0:
        raise RuntimeError("protoc failed")
