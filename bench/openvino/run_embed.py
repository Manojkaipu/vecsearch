"""Time one MiniLM configuration and save its question embeddings.

    python bench/openvino/run_embed.py --precision fp16 --device GPU --models ~/ov-models --bundle data/quant/support_rag_bundle.json --out results/openvino/runs
    python bench/openvino/run_embed.py --precision fp32 --device CPU ...      # the PyTorch reference

One process per configuration, so the peak-memory reading belongs to that model alone. It measures
  * query latency: each of the 96 evaluation questions embedded on its own (batch 1), 3 passes
  * throughput: 512 corpus chunks in batches of 32, median of 3 passes
  * peak resident memory, and OpenVINO's GPU allocation count on the GPU device
and writes <out>/embed_<precision>_<device>.json plus the questions' embeddings as .npy.
"""
import argparse
import statistics
import time
from pathlib import Path

import numpy as np

from common import MINILM, gpu_memory_mb, load_bundle, peak_rss_mb, rss_mb, write_json


def preload(precision):
    """Import what the configuration needs, so the baseline memory reading includes the libraries."""
    if precision == "fp32":
        import sentence_transformers  # noqa: F401
    else:
        import torch  # noqa: F401
        import optimum.intel  # noqa: F401
        import transformers  # noqa: F401


def load_encoder(precision, device, models):
    """Returns (encode(list[str]) -> normalized float32 array, load seconds)."""
    t = time.perf_counter()
    if precision == "fp32":  # the PyTorch model the index was built with
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(MINILM, device="cpu")

        def encode(texts):
            return model.encode(texts, batch_size=len(texts), normalize_embeddings=True, convert_to_numpy=True)

        return encode, time.perf_counter() - t

    import torch
    from optimum.intel import OVModelForFeatureExtraction
    from transformers import AutoTokenizer

    path = Path(models) / f"minilm-{precision}"
    tok = AutoTokenizer.from_pretrained(path)
    model = OVModelForFeatureExtraction.from_pretrained(path, device=device)

    def encode(texts):  # mean pooling over real tokens, then L2 normalization: all-MiniLM-L6-v2's recipe
        enc = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(pooled, dim=1).float().numpy()

    return encode, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--precision", required=True, choices=["fp32", "fp16", "int8", "int4"])
    ap.add_argument("--device", default="GPU", choices=["GPU", "CPU"])
    ap.add_argument("--models", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--passes", type=int, default=3)
    args = ap.parse_args()

    bundle = load_bundle(args.bundle)
    questions = [q["question"] for q in bundle["questions"]]
    chunks = bundle["chunk_sample"]
    preload(args.precision)
    baseline = rss_mb()  # Python + torch + the libraries, before the model is loaded
    encode, load_s = load_encoder(args.precision, args.device, args.models)

    for q in questions[:8]:  # warm-up: kernel compilation, caches
        encode([q])
    encode(chunks[:32])

    lat = []
    for _ in range(args.passes):
        for q in questions:
            t = time.perf_counter()
            encode([q])
            lat.append((time.perf_counter() - t) * 1e3)
    rates = []
    for _ in range(args.passes):
        t = time.perf_counter()
        for i in range(0, len(chunks), 32):
            encode(chunks[i:i + 32])
        rates.append(len(chunks) / (time.perf_counter() - t))

    emb = np.concatenate([encode(questions[i:i + 32]) for i in range(0, len(questions), 32)])
    out = Path(args.out)
    tag = f"{args.precision}_{args.device}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"embed_{tag}.npy", emb.astype(np.float32))
    result = dict(model="all-MiniLM-L6-v2", precision=args.precision, device=args.device,
                  load_s=round(load_s, 2), queries=len(questions),
                  query_p50_ms=round(statistics.median(lat), 3),
                  query_p95_ms=round(float(np.percentile(lat, 95)), 3),
                  chunks_note="512 corpus chunks, batch 32",
                  chunks_per_s=round(statistics.median(rates), 1),
                  peak_rss_mb=round(peak_rss_mb(), 1), baseline_rss_mb=round(baseline, 1),
                  gpu_mem_mb=None if args.device != "GPU" or args.precision == "fp32" else round(gpu_memory_mb(), 1))
    write_json(out / f"embed_{tag}.json", result)
    print(result)


if __name__ == "__main__":
    main()
