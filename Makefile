PY ?= python3
# Plain `=` so a NAME/DATA environment variable (WSL exports NAME=<hostname>)
# can't override these; `make NAME=foo` still works.
DATA = data/tweets
NAME = tweets

.PHONY: help build test pytest install tsan proto smoke smoke-dist data gt bench plot ablate dist dist-load plot-dist docker all

help:
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*## "}{printf "  %-10s %s\n",$$1,$$2}'

build:  ## C++ library + GoogleTest binary
	cmake -S . -B build/cpp -DCMAKE_BUILD_TYPE=Release && cmake --build build/cpp -j

test: build  ## run C++ unit tests
	ctest --test-dir build/cpp --output-on-failure

tsan:  ## race-check the parallel graph build
	cmake -S . -B build/tsan -DCMAKE_BUILD_TYPE=RelWithDebInfo -DVECSEARCH_TSAN=ON && cmake --build build/tsan -j
	ctest --test-dir build/tsan --output-on-failure

install:  ## pip-install the Python package (compiles the extension)
	$(PY) -m pip install -e . -r requirements-bench.txt

pytest:  ## run Python binding tests
	$(PY) -m pytest tests/python -q

proto:  ## generate gRPC stubs
	$(PY) -m grpc_tools.protoc -I distributed --python_out=distributed --grpc_python_out=distributed distributed/search.proto

smoke: proto  ## tiny end-to-end run on synthetic data (what CI runs)
	$(PY) bench/prepare_data.py --synthetic 20000 --n-queries 500 --out data/smoke
	$(PY) bench/ground_truth.py --data data/smoke
	$(PY) bench/run_benchmarks.py --data data/smoke --name smoke --algos vecsearch,faiss-hnsw --n-queries 500 --runs 1
	$(PY) bench/plot.py --name smoke

smoke-dist: proto  ## tiny 1/2-shard gRPC latency run (needs `make smoke` data)
	$(PY) distributed/latency_bench.py --data data/smoke --name smoke --shards 1,2 --n-queries 300

data:  ## embed tweets (CSV=path/to/twcs.csv)
	$(PY) bench/prepare_data.py --csv $(CSV) --out $(DATA)

gt:  ## exact ground truth
	$(PY) bench/ground_truth.py --data $(DATA)

bench:  ## recall@10 vs QPS for all libraries
	$(PY) bench/run_benchmarks.py --data $(DATA) --name $(NAME)

plot:  ## chart + summary table
	$(PY) bench/plot.py --name $(NAME)

ablate:  ## heuristic / M / efC ablations
	$(PY) bench/ablation.py --data $(DATA) --name $(NAME)

dist: proto  ## p50/p99 vs shards (1,2,4), pure latency
	$(PY) distributed/latency_bench.py --data $(DATA) --name $(NAME) --shards 1,2,4 --pin --concurrency 1

dist-load: proto  ## same, under load (16 in-flight requests)
	$(PY) distributed/latency_bench.py --data $(DATA) --name $(NAME) --shards 1,2,4 --pin --concurrency 16

plot-dist:  ## latency chart
	$(PY) distributed/plot_latency.py --name $(NAME)

docker:  ## build container image
	docker build -t vecsearch .

all: test install gt bench plot ablate dist dist-load plot-dist
