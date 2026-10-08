"""Quality, speed and memory of the quantized models, as markdown tables.

    python bench/openvino/quant_report.py --runs results/openvino/runs --bundle data/quant/support_rag_bundle.json \
        --index ~/support-rag/data --out results/openvino/quantization.md

Reads what run_embed.py and run_llm.py wrote into --runs (and runs them first with --run).

Embeddings: the 96 evaluation questions are embedded at each precision and searched in support-rag's
own HNSW index (892,800 chunk vectors from the PyTorch FP32 model, ef=128, as in rag/retrieval.py).
"top-10 overlap" is the share of the FP32 model's top 10 chunks that the quantized model also returns,
averaged over questions. The conversation-level number collapses chunks to their best conversation
first (what the agent sees); hit@10 asks whether the question's source conversation is in that top 10.

LLM: the model checks each stored answer the way support-rag's verifier does. Agreement is with the
original verifier's verdict (grok-4.7); because 79 of the 86 verdicts are "supported", always saying
"supported" already agrees 92%, so Cohen's kappa and the number of the 7 flagged answers caught are
reported next to it.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import PRECISIONS, dir_size_mb, load_bundle  # noqa: E402


def cohen_kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def collapse(chunk_ids, chunk_conv, k):
    """Chunk ranking -> conversation ranking, keeping each conversation's best chunk (rag/retrieval.py)."""
    seen, out = set(), []
    for c in chunk_ids:
        if c < 0:
            continue
        conv = int(chunk_conv[c])
        if conv not in seen:
            seen.add(conv)
            out.append(conv)
            if len(out) == k:
                break
    return out


def load_run(runs, kind, precision, device):
    p = Path(runs) / f"{kind}_{precision}_{device}.json"
    return json.loads(p.read_text()) if p.exists() else None


# ---- embeddings ----------------------------------------------------------------------------

def embedding_quality(runs, bundle, index_dir, device, ef=128, k=10, pool=100):
    """{precision: metrics} for every embedding run on `device` found in runs, against the FP32 reference."""
    import vecsearch as vs

    index_dir = Path(index_dir)
    hnsw = vs.HNSWIndex.load(str(index_dir / "hnsw.bin"))
    chunk_conv = np.load(index_dir / "chunk_conversation.npy")
    ref = np.load(Path(runs) / "embed_fp32_CPU.npy")
    gold = [set(q["gold"]) for q in bundle["questions"]]
    answerable = [q["answerable"] for q in bundle["questions"]]

    def search(emb):
        ids, _ = hnsw.search(np.ascontiguousarray(emb, np.float32), k=pool, ef=max(ef, pool))
        return ids

    ref_ids = search(ref)
    out = {}
    for prec in ["fp32"] + PRECISIONS:
        f = Path(runs) / f"embed_{prec}_{'CPU' if prec == 'fp32' else device}.npy"
        if not f.exists():
            continue
        emb = np.load(f)
        ids = search(emb)
        chunk = np.mean([len(set(a[:k]) & set(b[:k])) / k for a, b in zip(ids, ref_ids)])
        conv = np.mean([len(set(collapse(a, chunk_conv, k)) & set(collapse(b, chunk_conv, k))) / k
                        for a, b in zip(ids, ref_ids)])
        hits = [bool(gold[i] & set(collapse(ids[i], chunk_conv, k))) for i in range(len(ids)) if answerable[i]]
        ref_hits = [bool(gold[i] & set(collapse(ref_ids[i], chunk_conv, k))) for i in range(len(ids)) if answerable[i]]
        out[prec] = dict(top10_chunk_overlap=float(chunk), top10_conversation_overlap=float(conv),
                         gold_hit_at_10=float(np.mean(hits)), gold_hit_at_10_fp32=float(np.mean(ref_hits)),
                         mean_cosine_to_fp32=float(np.mean((emb * ref).sum(1))),
                         min_cosine_to_fp32=float(np.min((emb * ref).sum(1))))
    return out


# ---- LLM verifier --------------------------------------------------------------------------

def verifier_quality(runs, device):
    """{precision: metrics}: the model's verdicts against the original verifier's, and against FP16's."""
    runs_by_prec = {p: load_run(runs, "llm", p, device) for p in PRECISIONS}
    fp16 = runs_by_prec.get("fp16")
    out = {}
    for prec, r in runs_by_prec.items():
        if r is None or not r.get("verdicts"):  # a --speed-only run has no verdicts
            continue
        v = r["verdicts"]
        # No parsable verdict counts as unsupported, which is what Agent.verify does.
        pred = np.array([bool(x["supported"]) if x["supported"] is not None else False for x in v])
        orig = np.array([x["original_supported"] for x in v])
        flagged = ~orig
        row = dict(n=len(v), agreement=float((pred == orig).mean()), agree_count=int((pred == orig).sum()),
                   always_supported_agreement=float(orig.mean()), kappa=float(cohen_kappa(pred, orig)),
                   flagged=int(flagged.sum()), flagged_caught=int((~pred & flagged).sum()),
                   false_alarms=int((~pred & orig).sum()), unparsed=int(sum(x["supported"] is None for x in v)),
                   said_supported=int(pred.sum()))
        if fp16 is not None:
            ref = np.array([bool(x["supported"]) if x["supported"] is not None else False for x in fp16["verdicts"]])
            n = min(len(ref), len(pred))
            row["agreement_with_fp16"] = float((pred[:n] == ref[:n]).mean())
        out[prec] = row
    return out


# ---- tables --------------------------------------------------------------------------------

def fmt(x, spec=".1f"):
    return "—" if x is None else format(x, spec)


def embedding_table(runs, quality, device, models):
    rows = ["| precision | top-10 overlap with FP32 (chunks) | (conversations) | source in top 10 | mean cosine to FP32 "
            "| query p50 (ms) | chunks/s | peak RSS (MB) | GPU alloc (MB) | model (MB) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for prec in ["fp32"] + PRECISIONS:
        r = load_run(runs, "embed", prec, "CPU" if prec == "fp32" else device)
        q = quality.get(prec)
        if r is None or q is None:
            continue
        size = None if prec == "fp32" else dir_size_mb(Path(models) / f"minilm-{prec}")
        label = "FP32 (PyTorch, CPU; reference)" if prec == "fp32" else prec.upper()
        rows.append(f"| {label} | {q['top10_chunk_overlap']:.3f} | {q['top10_conversation_overlap']:.3f} "
                    f"| {q['gold_hit_at_10']:.3f} | {q['mean_cosine_to_fp32']:.5f} | {fmt(r['query_p50_ms'], '.2f')} "
                    f"| {fmt(r['chunks_per_s'], ',.0f')} | {fmt(r['peak_rss_mb'], ',.0f')} "
                    f"| {fmt(r['gpu_mem_mb'], ',.0f')} | {fmt(size, '.0f')} |")
    return "\n".join(rows)


def llm_table(runs, quality, device, models):
    rows = ["| precision | agrees with original verifier | kappa | flagged answers caught | false alarms "
            "| agrees with FP16 | TTFT, verifier prompt (ms) | decode (tok/s) | prefill (tok/s) | peak RSS (MB) "
            "| GPU alloc (MB) | model (MB) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for prec in PRECISIONS:
        r = load_run(runs, "llm", prec, device)
        q = quality.get(prec)
        if r is None:
            continue
        v, s = r["speed"]["verifier"], r["speed"]["short"]
        if q is None:  # speed only: no verdicts were collected on this device
            head = "| — | — | — | — | —"
        else:
            head = (f"| {q['agree_count']}/{q['n']} ({q['agreement']:.0%}) | {q['kappa']:.2f} "
                    f"| {q['flagged_caught']}/{q['flagged']} | {q['false_alarms']} "
                    f"| {fmt(q.get('agreement_with_fp16'), '.0%')}")
        rows.append(f"| {prec.upper()} {head} | {v['ttft_ms']:,.0f} "
                    f"| {s['decode_tokens_per_s']:.1f} | {v['prefill_tokens_per_s']:,.0f} | {r['peak_rss_mb']:,.0f} "
                    f"| {fmt(r['gpu_mem_mb'], ',.0f')} | {dir_size_mb(Path(models) / f'qwen2.5-1.5b-{prec}'):,.0f} |")
    return "\n".join(rows)


def run_missing(runs, models, bundle, device, which):
    """Run each configuration that has no result yet in its own process."""
    configs = []
    if "embed" in which:
        configs += [("run_embed.py", "embed", "fp32", "CPU")] + [("run_embed.py", "embed", p, device) for p in PRECISIONS]
    if "llm" in which:
        configs += [("run_llm.py", "llm", p, device) for p in PRECISIONS]
    for script, kind, prec, dev in configs:
        if (Path(runs) / f"{kind}_{prec}_{dev}.json").exists():
            continue
        print(f"== {kind} {prec} on {dev}", flush=True)
        subprocess.run([sys.executable, str(HERE / script), "--precision", prec, "--device", dev,
                        "--models", str(models), "--bundle", str(bundle), "--out", str(runs)], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--index", required=True, help="support-rag's data directory (hnsw.bin, chunk_conversation.npy)")
    ap.add_argument("--models", required=True, help="where export_models.sh wrote the OpenVINO models")
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--run", nargs="*", choices=["embed", "llm"], help="run the configurations that have no result yet")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.run:
        run_missing(args.runs, args.models, args.bundle, args.device, args.run)
    bundle = load_bundle(args.bundle)
    eq = embedding_quality(args.runs, bundle, args.index, args.device)
    lq = verifier_quality(args.runs, args.device)
    text = (f"### Embeddings: all-MiniLM-L6-v2 on {args.device}\n\n{embedding_table(args.runs, eq, args.device, args.models)}\n\n"
            f"### LLM verifier: Qwen2.5-1.5B-Instruct on {args.device}\n\n{llm_table(args.runs, lq, args.device, args.models)}\n")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    Path(args.out).with_suffix(".json").write_text(json.dumps({"embedding": eq, "verifier": lq}, indent=2))
    print(text)


if __name__ == "__main__":
    main()
