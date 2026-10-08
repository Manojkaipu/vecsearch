### Embeddings: all-MiniLM-L6-v2 on GPU

| precision | top-10 overlap with FP32 (chunks) | (conversations) | source in top 10 | mean cosine to FP32 | query p50 (ms) | chunks/s | peak RSS (MB) | GPU alloc (MB) | model (MB) |
|---|---|---|---|---|---|---|---|---|---|
| FP32 (PyTorch, CPU; reference) | 1.000 | 1.000 | 0.547 | 1.00000 | 7.94 | 45 | 818 | — | — |
| FP16 | 1.000 | 1.000 | 0.547 | 1.00000 | 2.84 | 830 | 840 | 99 | 43 |
| INT8 | 0.975 | 0.973 | 0.547 | 0.99919 | 3.53 | 792 | 1,080 | 98 | 22 |
| INT4 | 0.792 | 0.796 | 0.512 | 0.96640 | 3.47 | 677 | 1,027 | 93 | 17 |

### LLM verifier: Qwen2.5-1.5B-Instruct on GPU

| precision | agrees with original verifier | kappa | flagged answers caught | false alarms | agrees with FP16 | TTFT, verifier prompt (ms) | decode (tok/s) | prefill (tok/s) | peak RSS (MB) | GPU alloc (MB) | model (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FP16 | 29/86 (34%) | 0.03 | 6/7 | 56 | 100% | 280 | 27.6 | 6,556 | 1,227 | 3,201 | 2,946 |
| INT8 | 39/86 (45%) | 0.04 | 5/7 | 45 | 86% | 248 | 47.0 | 7,420 | 1,175 | 1,626 | 1,477 |
| INT4 | 47/86 (55%) | -0.05 | 2/7 | 34 | 63% | 274 | 66.2 | 6,716 | 1,083 | 1,026 | 935 |
