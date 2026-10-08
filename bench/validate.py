"""One command that runs the OpenCL and OpenVINO validation (Steps 2-5) and writes a markdown report.

    python bench/validate.py --out results/validation/report.md                      # everything
    python bench/validate.py --stages correctness --out /tmp/report.md              # what CI can run
    python bench/validate.py --stages correctness,benchmark --opencl-device Intel   # the OpenCL half

Stages (each is skipped, and says so, if its inputs are missing):
  correctness  the OpenCL pytest suite and the GoogleTest binary, against the CPU index
  benchmark    exact search on --n vectors through OpenCL (and the CPU index), against the recorded baseline
  embeddings   MiniLM at FP16/INT8/INT4 on support-rag's 96 questions: top-10 overlap with FP32 in its HNSW index
  llm          Qwen2.5-1.5B-Instruct at FP16/INT8/INT4 as support-rag's verifier: agreement, speed, memory

Every check has a threshold in bench/validation_thresholds.json. The report lists each check as PASS or
FAIL with the measured value, and the exit code is 1 if any check fails, so a change that hurts quality
or speed fails the build. A stage that is skipped is not a pass: use --require to fail on a skip.

The thresholds were set from a run on this laptop (Intel Arc 140V, WSL2), a little below the measured
values so run-to-run noise doesn't fail them; see "basis" in the JSON. Other hardware needs its own
"benchmark" baseline (or it only gets the recall check).
"""
import argparse
import csv
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench" / "openvino"))

STAGES = ["correctness", "benchmark", "embeddings", "llm"]


@dataclass
class Check:
    stage: str
    name: str
    value: float
    op: str  # ">=" or "<="
    threshold: float
    unit: str = ""

    @property
    def passed(self):
        return self.value >= self.threshold if self.op == ">=" else self.value <= self.threshold


class Report:
    def __init__(self):
        self.checks, self.skipped, self.sections, self.errors = [], {}, [], []

    def check(self, stage, name, value, op, threshold, unit=""):
        self.checks.append(Check(stage, name, float(value), op, float(threshold), unit))

    def skip(self, stage, why):
        self.skipped[stage] = why

    def markdown(self, meta):
        failed = [c for c in self.checks if not c.passed]
        lines = ["# OpenCL and OpenVINO validation", "", *[f"- {k}: {v}" for k, v in meta.items()], ""]
        verdict = "FAIL" if failed or self.errors else "PASS"
        lines += [f"**{verdict}**: {len(self.checks) - len(failed)}/{len(self.checks)} checks pass"
                  + (f", {len(self.skipped)} stage(s) skipped" if self.skipped else "") + ".", ""]
        for stage, why in self.skipped.items():
            lines.append(f"- SKIPPED `{stage}`: {why}")
        for e in self.errors:
            lines.append(f"- ERROR: {e}")
        if self.skipped or self.errors:
            lines.append("")
        lines += ["| stage | check | measured | threshold | result |", "|---|---|---|---|---|"]
        for c in self.checks:
            lines.append(f"| {c.stage} | {c.name} | {c.value:,.4g}{c.unit} | {c.op} {c.threshold:,.4g}{c.unit} "
                         f"| {'PASS' if c.passed else '**FAIL**'} |")
        for title, body in self.sections:
            lines += ["", f"## {title}", "", body]
        return "\n".join(lines) + "\n"


def run(cmd, **kw):
    print("$", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


# ---- correctness -----------------------------------------------------------------------------

def stage_correctness(rep, th, args):
    t = th["correctness"]
    r = run([sys.executable, "-m", "pytest", str(ROOT / "tests/python/test_opencl.py"), "-q", "-p", "no:cacheprovider"],
            cwd=ROOT)
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:]
    counts = {k: int(n) for n, k in re.findall(r"(\d+) (passed|failed|skipped|error)", tail)}
    passed, failed, skipped = counts.get("passed", 0), counts.get("failed", 0) + counts.get("error", 0), counts.get("skipped", 0)
    rep.check("correctness", "pytest: OpenCL results vs the CPU index, tests failed", failed, "<=", 0)
    rep.check("correctness", "pytest: tests passed", passed, ">=", t["min_pytest_passed"])
    rep.check("correctness", "pytest: tests skipped (no device would skip them all)", skipped, "<=", 0)
    rep.sections.append(("pytest", f"`{tail}`"))

    if args.gtest and Path(args.gtest).exists():
        g = run([args.gtest, "--gtest_filter=*OpenCL*"])
        text = g.stdout
        gp = re.search(r"\[  PASSED  \] (\d+) tests?", text)
        gf = re.search(r"\[  FAILED  \] (\d+) tests?, listed", text)
        gs = re.findall(r"\[  SKIPPED \] (\d+) tests?", text)
        rep.check("correctness", "GoogleTest: tests failed", int(gf.group(1)) if gf else (0 if g.returncode == 0 else 1), "<=", 0)
        rep.check("correctness", "GoogleTest: tests passed", int(gp.group(1)) if gp else 0, ">=", t["min_gtest_passed"])
        rep.check("correctness", "GoogleTest: tests skipped", int(gs[0]) if gs else 0, "<=", 0)
    else:
        rep.sections.append(("GoogleTest", "not run (pass --gtest build/vecsearch_tests)"))


# ---- benchmark -------------------------------------------------------------------------------

def stage_benchmark(rep, th, args):
    t = th["benchmark"]
    out = Path(args.work) / "validate_bench.csv"
    backends = "opencl,cpu" if args.with_cpu else "opencl"
    cmd = [sys.executable, str(ROOT / "bench/gpu_bench.py"), "--backends", backends, "--opencl-device", args.opencl_device,
           "--n", str(args.n), "--batches", ",".join(map(str, t["batches"])), "--min-time", "0.5", "--check", "64",
           "--out", str(out)]
    r = run(cmd, cwd=ROOT)
    if r.returncode != 0 or not out.exists():
        rep.errors.append(f"gpu_bench.py failed: {r.stderr[-400:] or r.stdout[-400:]}")
        return
    rows = list(csv.DictReader(open(out)))
    ocl = [x for x in rows if x["backend"] == "opencl"]
    rep.check("benchmark", "OpenCL recall vs the CPU index (lowest over batch sizes)",
              min(float(x["recall_vs_cpu"]) for x in ocl), ">=", t["min_recall"])
    device = ocl[0]["device"].split(",")[0]
    base = t["baselines"].get(device)
    if base and base["n"] == args.n:
        for x in ocl:
            b = base["qps"].get(x["batch"])
            if b:
                rep.check("benchmark", f"OpenCL queries/s at batch {x['batch']} (baseline {b:,.0f})", float(x["qps"]),
                          ">=", t["min_fraction_of_baseline"] * b)
    else:
        rep.sections.append(("benchmark baseline", f"no recorded baseline for '{device}' at n={args.n:,}: "
                             "only recall is checked. Record one in bench/validation_thresholds.json."))
    cpu = {x["batch"]: float(x["qps"]) for x in rows if x["backend"] == "cpu"}
    for x in ocl:
        if x["batch"] in cpu and x["batch"] in t.get("min_speedup_over_cpu", {}):
            rep.check("benchmark", f"OpenCL speedup over the AVX2 CPU index at batch {x['batch']}",
                      float(x["qps"]) / cpu[x["batch"]], ">=", t["min_speedup_over_cpu"][x["batch"]], "x")
    rep.sections.append(("benchmark", "```\n" + "\n".join(f"{x['backend']:>7} batch {x['batch']:>5}: {float(x['ms_per_batch']):9.2f} ms "
                                                       f"{float(x['qps']):10,.0f} QPS  recall {x['recall_vs_cpu']}" for x in rows) + "\n```"))


# ---- OpenVINO stages -------------------------------------------------------------------------

def openvino_inputs(rep, stage, args):
    need = [Path(args.bundle), Path(args.models)]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        rep.skip(stage, "missing " + ", ".join(missing))
        return False
    return True


def stage_embeddings(rep, th, args):
    from quant_report import embedding_quality, embedding_table, load_bundle, run_missing

    if not openvino_inputs(rep, "embeddings", args):
        return
    t = th["embeddings"]
    if args.run_models:
        run_missing(args.runs, args.models, args.bundle, args.device, ["embed"])
    bundle = load_bundle(args.bundle)
    q = embedding_quality(args.runs, bundle, args.index, args.device)
    if not q:
        rep.skip("embeddings", f"no embedding runs in {args.runs} (use --run-models)")
        return
    for prec, lim in t["min_top10_chunk_overlap"].items():
        if prec in q:
            rep.check("embeddings", f"{prec.upper()} top-10 overlap with FP32 (chunks, {len(bundle['questions'])} questions)",
                      q[prec]["top10_chunk_overlap"], ">=", lim)
    for prec, lim in t["min_mean_cosine"].items():
        if prec in q:
            rep.check("embeddings", f"{prec.upper()} mean cosine to the FP32 embedding", q[prec]["mean_cosine_to_fp32"], ">=", lim)
    for prec, lim in t["max_hit10_drop"].items():
        if prec in q:
            rep.check("embeddings", f"{prec.upper()} source-in-top-10 drop vs FP32", q["fp32"]["gold_hit_at_10"] - q[prec]["gold_hit_at_10"], "<=", lim)
    from quant_report import load_run

    f16, i4 = load_run(args.runs, "embed", "fp16", args.device), load_run(args.runs, "embed", "int4", args.device)
    if f16 and i4:
        rep.check("embeddings", "INT4 peak RSS / FP16 peak RSS", i4["peak_rss_mb"] / f16["peak_rss_mb"], "<=", t["max_int4_rss_ratio"])
        rep.check("embeddings", "INT4 chunks/s / FP16 chunks/s", i4["chunks_per_s"] / f16["chunks_per_s"], ">=", t["min_int4_speed_ratio"])
    rep.sections.append((f"embeddings on {args.device}", embedding_table(args.runs, q, args.device, args.models)))


def stage_llm(rep, th, args):
    from quant_report import llm_table, load_run, run_missing, verifier_quality

    if not openvino_inputs(rep, "llm", args):
        return
    t = th["llm"]
    if args.run_models:
        run_missing(args.runs, args.models, args.bundle, args.device, ["llm"])
    q = verifier_quality(args.runs, args.device)
    if not q:
        rep.skip("llm", f"no LLM runs in {args.runs} (use --run-models)")
        return
    for prec, lim in t["min_agreement_with_fp16"].items():
        if prec in q and "agreement_with_fp16" in q[prec]:
            rep.check("llm", f"{prec.upper()} verdicts agree with FP16's", q[prec]["agreement_with_fp16"], ">=", lim)
    for prec, lim in t["min_kappa_vs_original"].items():
        if prec in q:
            rep.check("llm", f"{prec.upper()} kappa vs the original verifier", q[prec]["kappa"], ">=", lim)
    for prec in q:
        rep.check("llm", f"{prec.upper()} replies with no parsable verdict", q[prec]["unparsed"], "<=", t["max_unparsed"])
    r = {p: load_run(args.runs, "llm", p, args.device) for p in ("fp16", "int8", "int4")}
    if r["fp16"] and r["int4"]:
        d16, d4 = (r[p]["speed"]["short"]["decode_tokens_per_s"] for p in ("fp16", "int4"))
        rep.check("llm", "INT4 decode tokens/s / FP16 decode tokens/s", d4 / d16, ">=", t["min_int4_decode_speed_ratio"], "x")
        rep.check("llm", "INT4 peak RSS / FP16 peak RSS", r["int4"]["peak_rss_mb"] / r["fp16"]["peak_rss_mb"], "<=", t["max_int4_rss_ratio"])
        if r["int4"]["gpu_mem_mb"] and r["fp16"]["gpu_mem_mb"]:
            rep.check("llm", "INT4 GPU allocation / FP16 GPU allocation", r["int4"]["gpu_mem_mb"] / r["fp16"]["gpu_mem_mb"], "<=", t["max_int4_gpu_mem_ratio"])
    rep.sections.append((f"LLM verifier on {args.device}", llm_table(args.runs, q, args.device, args.models)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresholds", default=str(ROOT / "bench/validation_thresholds.json"))
    ap.add_argument("--require", action="store_true", help="a skipped stage counts as a failure")
    ap.add_argument("--gtest", default=str(ROOT / "build/vecsearch_tests"))
    ap.add_argument("--opencl-device", default="0")
    ap.add_argument("--n", type=int, default=1_000_000, help="vectors in the benchmark stage")
    ap.add_argument("--with-cpu", action="store_true", help="also time the CPU index (slow at 1M) for the speedup checks")
    ap.add_argument("--work", default="/tmp")
    ap.add_argument("--runs", default=str(ROOT / "results/openvino/runs"))
    ap.add_argument("--bundle", default=str(ROOT / "data/quant/support_rag_bundle.json"))
    ap.add_argument("--models", default=str(Path.home() / "ov-models"))
    ap.add_argument("--index", default=str(Path.home() / "support-rag/data"))
    ap.add_argument("--device", default="GPU", help="OpenVINO device")
    ap.add_argument("--run-models", action="store_true", help="run the model configurations that have no result yet")
    args = ap.parse_args()

    th = json.loads(Path(args.thresholds).read_text())
    stages = [s for s in args.stages.split(",") if s]
    unknown = set(stages) - set(STAGES)
    if unknown:
        ap.error(f"unknown stages {sorted(unknown)}")
    rep = Report()
    t0 = time.time()
    fns = dict(correctness=stage_correctness, benchmark=stage_benchmark, embeddings=stage_embeddings, llm=stage_llm)
    for s in stages:
        print(f"\n=== {s}", flush=True)
        try:
            fns[s](rep, th, args)
        except Exception as e:  # a crashed stage is a failure, not a pass
            rep.errors.append(f"stage {s} crashed: {type(e).__name__}: {e}")
    git = run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).stdout.strip()
    meta = {"commit": git or "unknown (not a git checkout)", "stages": ", ".join(stages), "thresholds": f"{Path(args.thresholds).name} ({th.get('basis', '')})",
            "wall time": f"{time.time() - t0:.0f} s"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(rep.markdown(meta), encoding="utf-8")
    print(rep.markdown(meta))
    bad = any(not c.passed for c in rep.checks) or bool(rep.errors) or (args.require and bool(rep.skipped))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
