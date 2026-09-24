FROM python:3.11-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
# x86-64-v3 (AVX2+FMA) rather than -march=native so the image runs on other hosts
RUN cmake -S . -B build -DVECSEARCH_ARCH=x86-64-v3 && cmake --build build -j \
    && ctest --test-dir build --output-on-failure
ENV CMAKE_ARGS="-DVECSEARCH_ARCH=x86-64-v3"
# hnswlib ships no wheels, so every dependency is built here where a compiler exists
RUN pip wheel -w /wheels . -r requirements-bench.txt

FROM python:3.11-slim
WORKDIR /app
COPY requirements-bench.txt .
RUN --mount=type=bind,from=build,source=/wheels,target=/wheels \
    pip install --no-cache-dir --no-index --find-links /wheels -r requirements-bench.txt vecsearch
COPY bench bench
COPY distributed distributed
COPY tests/python tests/python
RUN python -m grpc_tools.protoc -I distributed --python_out=distributed \
      --grpc_python_out=distributed distributed/search.proto \
 && python -m pytest tests/python -q
EXPOSE 50050
# expects a prebuilt sharded index mounted at /app/indexes
CMD ["python", "distributed/cluster.py", "--manifest", "/app/indexes/tweets_4shards/manifest.json"]
