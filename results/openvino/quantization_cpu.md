### Embeddings: all-MiniLM-L6-v2 on CPU

| precision | top-10 overlap with FP32 (chunks) | (conversations) | source in top 10 | mean cosine to FP32 | query p50 (ms) | chunks/s | peak RSS (MB) | GPU alloc (MB) | model (MB) |
|---|---|---|---|---|---|---|---|---|---|
| FP32 (PyTorch, CPU; reference) | 1.000 | 1.000 | 0.547 | 1.00000 | 7.94 | 45 | 818 | — | — |
| FP16 | 0.999 | 0.999 | 0.547 | 1.00000 | 4.85 | 56 | 981 | — | 43 |
| INT8 | 0.983 | 0.982 | 0.547 | 0.99959 | 4.96 | 87 | 1,019 | — | 22 |
| INT4 | 0.795 | 0.800 | 0.523 | 0.96750 | 4.59 | 74 | 993 | — | 17 |

### LLM verifier: Qwen2.5-1.5B-Instruct on CPU

| precision | agrees with original verifier | kappa | flagged answers caught | false alarms | agrees with FP16 | TTFT, verifier prompt (ms) | decode (tok/s) | prefill (tok/s) | peak RSS (MB) | GPU alloc (MB) | model (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FP16 | — | — | — | — | — | 14,097 | 12.4 | 130 | 5,493 | — | 2,946 |
| INT8 | — | — | — | — | — | 5,284 | 24.4 | 348 | 4,087 | — | 1,477 |
| INT4 | — | — | — | — | — | 7,476 | 26.6 | 246 | 3,048 | — | 935 |
