"""Time one Qwen2.5-1.5B-Instruct configuration, then run it as support-rag's verifier.

    python bench/openvino/run_llm.py --precision int4 --device GPU --models ~/ov-models --bundle data/quant/support_rag_bundle.json --out results/openvino/runs

One process per configuration, so the peak-memory reading belongs to that model alone.
  * speed: greedy decoding of 128 tokens after (a) a short prompt and (b) the median-length verifier
    prompt from the bundle; time to first token and decode tokens/second, median of 3 runs
  * quality: every stored answer in the bundle is checked the way support-rag's Agent.verify does,
    with its prompt and verdict schema (the schema is enforced during decoding), temperature 0
and writes <out>/llm_<precision>_<device>.json with the speed numbers and every verdict.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

from common import QWEN, gpu_memory_mb, load_bundle, parse_verdict, peak_rss_mb, rss_mb, write_json

FORMAT_NOTE = ("\n\nReply with a JSON object with the fields supported (boolean), unsupported_claims "
               "(list of strings) and feedback (string).")
SHORT_PROMPT = ("Explain, in a few paragraphs, how an HNSW graph index finds the nearest neighbours of a query "
                "vector, and why it needs far fewer distance computations than a linear scan.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--precision", required=True, choices=["fp16", "int8", "int4"])
    ap.add_argument("--device", default="GPU", choices=["GPU", "CPU"])
    ap.add_argument("--models", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="check only the first N bundle items (dry runs)")
    ap.add_argument("--speed-only", action="store_true", help="time the model but skip the verifier items")
    ap.add_argument("--max-new-tokens", type=int, default=256, help="cap on a verdict; `supported` comes first")
    ap.add_argument("--passes", type=int, default=3)
    args = ap.parse_args()

    import openvino_genai as ov_genai
    from transformers import AutoTokenizer  # imported before the baseline memory reading

    bundle = load_bundle(args.bundle)
    v = bundle["verifier"]
    all_items = v["items"][:args.limit] if args.limit else v["items"]
    items = [] if args.speed_only else all_items
    path = Path(args.models) / f"qwen2.5-1.5b-{args.precision}"
    tok = AutoTokenizer.from_pretrained(path)
    baseline = rss_mb()

    t = time.perf_counter()
    pipe = ov_genai.LLMPipeline(str(path), args.device)
    load_s = time.perf_counter() - t

    def render(system, user):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def config(max_new, min_new=0, schema=None):
        c = ov_genai.GenerationConfig()
        c.max_new_tokens, c.min_new_tokens, c.do_sample = max_new, min_new, False
        c.apply_chat_template = False  # the prompt is already rendered
        if schema is not None:
            c.structured_output_config = ov_genai.StructuredOutputConfig(json_schema=json.dumps(schema))
        return c

    def timed(prompt, cfg):
        res = pipe.generate(prompt, cfg)
        m = res.perf_metrics
        return dict(text=res.texts[0], input_tokens=m.get_num_input_tokens(),
                    new_tokens=m.get_num_generated_tokens(), ttft_ms=m.get_ttft().mean,
                    tpot_ms=m.get_tpot().mean, tokens_per_s=m.get_throughput().mean,
                    total_ms=m.get_generate_duration().mean)

    # ---- speed ----
    lengths = sorted((len(tok(it["prompt"]).input_ids), i) for i, it in enumerate(all_items))
    median_item = all_items[lengths[len(lengths) // 2][1]]
    workloads = {
        "short": render("You are a helpful assistant.", SHORT_PROMPT),
        "verifier": render(v["system"] + FORMAT_NOTE, median_item["prompt"]),
    }
    speed = {}
    cfg = config(128, 128)  # exactly 128 new tokens, so runs are comparable
    for name, prompt in workloads.items():
        timed(prompt, cfg)  # warm-up (kernel compilation, caches)
        runs = [timed(prompt, cfg) for _ in range(args.passes)]
        speed[name] = dict(input_tokens=runs[0]["input_tokens"], new_tokens=runs[0]["new_tokens"],
                           ttft_ms=round(statistics.median(r["ttft_ms"] for r in runs), 1),
                           tpot_ms=round(statistics.median(r["tpot_ms"] for r in runs), 2),
                           decode_tokens_per_s=round(1000 / statistics.median(r["tpot_ms"] for r in runs), 1),
                           prefill_tokens_per_s=round(runs[0]["input_tokens"] * 1000
                                                      / statistics.median(r["ttft_ms"] for r in runs), 1))
        print(name, speed[name], flush=True)

    # ---- quality: the model as support-rag's verifier ----
    system = v["system"] + FORMAT_NOTE
    constrained = True
    verdicts = []
    t_all = time.perf_counter()
    for i, it in enumerate(items):
        prompt = render(system, it["prompt"])
        try:
            r = timed(prompt, config(args.max_new_tokens, 0, v["schema"] if constrained else None))
        except Exception as e:  # constrained decoding unavailable on this build/device: fall back to the prompt alone
            if not constrained:
                raise
            print("structured output failed, continuing without it:", str(e)[:200], flush=True)
            constrained = False
            r = timed(prompt, config(args.max_new_tokens))
        supported, parsed = parse_verdict(r["text"])
        verdicts.append(dict(id=it["id"], supported=supported, parsed=parsed is not None,
                             original_supported=it["original_supported"], reply=r["text"][:1500],
                             input_tokens=r["input_tokens"], new_tokens=r["new_tokens"],
                             total_ms=round(r["total_ms"], 1)))
        print(f"{i + 1}/{len(items)} {it['id']} supported={supported} (original {it['original_supported']}) "
              f"{r['input_tokens']} in / {r['new_tokens']} out, {r['total_ms'] / 1000:.1f} s", flush=True)

    result = dict(model="Qwen2.5-1.5B-Instruct", precision=args.precision, device=args.device,
                  load_s=round(load_s, 2), speed=speed, constrained_decoding=constrained,
                  verifier_items=len(items), verifier_wall_s=round(time.perf_counter() - t_all, 1),
                  peak_rss_mb=round(peak_rss_mb(), 1), baseline_rss_mb=round(baseline, 1),
                  gpu_mem_mb=round(gpu_memory_mb(), 1) if args.device == "GPU" else None,
                  verdicts=verdicts)
    out = Path(args.out)
    write_json(out / f"llm_{args.precision}_{args.device}.json", result)
    print({k: v for k, v in result.items() if k != "verdicts"})


if __name__ == "__main__":
    main()
