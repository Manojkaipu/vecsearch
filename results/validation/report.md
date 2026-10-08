# OpenCL and OpenVINO validation

- commit: unknown (not a git checkout)
- stages: correctness, benchmark, embeddings, llm
- thresholds: validation_thresholds.json (set 2026-10-08 from runs on an Intel Core Ultra 7 256V laptop with an Arc 140V (WSL2, AC power), a little below the measured values; the measured value is in each comment)
- wall time: 89 s

**PASS**: 35/35 checks pass.

| stage | check | measured | threshold | result |
|---|---|---|---|---|
| correctness | pytest: OpenCL results vs the CPU index, tests failed | 0 | <= 0 | PASS |
| correctness | pytest: tests passed | 181 | >= 170 | PASS |
| correctness | pytest: tests skipped (no device would skip them all) | 0 | <= 0 | PASS |
| correctness | GoogleTest: tests failed | 0 | <= 0 | PASS |
| correctness | GoogleTest: tests passed | 104 | >= 100 | PASS |
| correctness | GoogleTest: tests skipped | 0 | <= 0 | PASS |
| benchmark | OpenCL recall vs the CPU index (lowest over batch sizes) | 1 | >= 0.999 | PASS |
| benchmark | OpenCL queries/s at batch 1 (baseline 59) | 53.7 | >= 47.12 | PASS |
| benchmark | OpenCL queries/s at batch 8 (baseline 434) | 395.1 | >= 347.3 | PASS |
| benchmark | OpenCL queries/s at batch 128 (baseline 1,991) | 1,902 | >= 1,593 | PASS |
| benchmark | OpenCL queries/s at batch 1024 (baseline 1,806) | 1,775 | >= 1,444 | PASS |
| benchmark | OpenCL speedup over the AVX2 CPU index at batch 128 | 7.752x | >= 3x | PASS |
| benchmark | OpenCL speedup over the AVX2 CPU index at batch 1024 | 5.743x | >= 3x | PASS |
| embeddings | FP16 top-10 overlap with FP32 (chunks, 96 questions) | 1 | >= 0.99 | PASS |
| embeddings | INT8 top-10 overlap with FP32 (chunks, 96 questions) | 0.975 | >= 0.95 | PASS |
| embeddings | INT4 top-10 overlap with FP32 (chunks, 96 questions) | 0.7917 | >= 0.75 | PASS |
| embeddings | FP16 mean cosine to the FP32 embedding | 1 | >= 0.9999 | PASS |
| embeddings | INT8 mean cosine to the FP32 embedding | 0.9992 | >= 0.998 | PASS |
| embeddings | INT4 mean cosine to the FP32 embedding | 0.9664 | >= 0.95 | PASS |
| embeddings | FP16 source-in-top-10 drop vs FP32 | 0 | <= 0.01 | PASS |
| embeddings | INT8 source-in-top-10 drop vs FP32 | 0 | <= 0.02 | PASS |
| embeddings | INT4 source-in-top-10 drop vs FP32 | 0.03488 | <= 0.08 | PASS |
| embeddings | INT4 peak RSS / FP16 peak RSS | 1.223 | <= 1.4 | PASS |
| embeddings | INT4 chunks/s / FP16 chunks/s | 0.8153 | >= 0.7 | PASS |
| llm | INT8 verdicts agree with FP16's | 0.8605 | >= 0.75 | PASS |
| llm | INT4 verdicts agree with FP16's | 0.6279 | >= 0.5 | PASS |
| llm | FP16 kappa vs the original verifier | 0.03237 | >= -0.1 | PASS |
| llm | INT8 kappa vs the original verifier | 0.03808 | >= -0.1 | PASS |
| llm | INT4 kappa vs the original verifier | -0.05009 | >= -0.15 | PASS |
| llm | FP16 replies with no parsable verdict | 0 | <= 0 | PASS |
| llm | INT8 replies with no parsable verdict | 0 | <= 0 | PASS |
| llm | INT4 replies with no parsable verdict | 0 | <= 0 | PASS |
| llm | INT4 decode tokens/s / FP16 decode tokens/s | 2.399x | >= 1.8x | PASS |
| llm | INT4 peak RSS / FP16 peak RSS | 0.8828 | <= 1 | PASS |
| llm | INT4 GPU allocation / FP16 GPU allocation | 0.3206 | <= 0.45 | PASS |

## pytest

`181 passed in 12.98s`

## benchmark

```
 opencl batch     1:     18.64 ms         54 QPS  recall 1.0
 opencl batch     8:     20.25 ms        395 QPS  recall 1.0
 opencl batch   128:     67.31 ms      1,902 QPS  recall 1.0
 opencl batch  1024:    576.87 ms      1,775 QPS  recall 1.0
    cpu batch     1:     32.24 ms         31 QPS  recall 1.0
    cpu batch     8:     54.05 ms        148 QPS  recall 1.0
    cpu batch   128:    521.80 ms        245 QPS  recall 1.0
    cpu batch  1024:   3312.35 ms        309 QPS  recall 1.0
```

## embeddings on GPU

| precision | top-10 overlap with FP32 (chunks) | (conversations) | source in top 10 | mean cosine to FP32 | query p50 (ms) | chunks/s | peak RSS (MB) | GPU alloc (MB) | model (MB) |
|---|---|---|---|---|---|---|---|---|---|
| FP32 (PyTorch, CPU; reference) | 1.000 | 1.000 | 0.547 | 1.00000 | 7.94 | 45 | 818 | — | — |
| FP16 | 1.000 | 1.000 | 0.547 | 1.00000 | 2.84 | 830 | 840 | 99 | 43 |
| INT8 | 0.975 | 0.973 | 0.547 | 0.99919 | 3.53 | 792 | 1,080 | 98 | 22 |
| INT4 | 0.792 | 0.796 | 0.512 | 0.96640 | 3.47 | 677 | 1,027 | 93 | 17 |

## LLM verifier on GPU

| precision | agrees with original verifier | kappa | flagged answers caught | false alarms | agrees with FP16 | TTFT, verifier prompt (ms) | decode (tok/s) | prefill (tok/s) | peak RSS (MB) | GPU alloc (MB) | model (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FP16 | 29/86 (34%) | 0.03 | 6/7 | 56 | 100% | 280 | 27.6 | 6,556 | 1,227 | 3,201 | 2,946 |
| INT8 | 39/86 (45%) | 0.04 | 5/7 | 45 | 86% | 248 | 47.0 | 7,420 | 1,175 | 1,626 | 1,477 |
| INT4 | 47/86 (55%) | -0.05 | 2/7 | 34 | 63% | 274 | 66.2 | 6,716 | 1,083 | 1,026 | 935 |
