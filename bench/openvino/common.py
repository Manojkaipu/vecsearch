"""Shared bits of the OpenVINO benchmarks: the support-rag bundle, memory readings, JSON output."""
import json
import re
import resource
from pathlib import Path

MINILM = "sentence-transformers/all-MiniLM-L6-v2"
QWEN = "Qwen/Qwen2.5-1.5B-Instruct"
PRECISIONS = ["fp16", "int8", "int4"]


def load_bundle(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def peak_rss_mb():
    """Peak resident memory of this process so far (VmHWM), in MB."""
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def rss_mb():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return 0.0


def gpu_memory_mb(core=None):
    """OpenVINO's own count of GPU allocations (device + host-visible USM + cl_mem), in MB.
    On an integrated GPU the weights usually live in host-visible USM, so both are summed."""
    import openvino as ov

    core = core or ov.Core()
    try:
        stats = core.get_property("GPU", "GPU_MEMORY_STATISTICS")
    except Exception:
        return None
    return sum(stats.values()) / 2**20


def dir_size_mb(path):
    return sum(f.stat().st_size for f in Path(path).glob("openvino_model.*")) / 2**20


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_verdict(text):
    """The `supported` flag from a model reply: JSON if it parses, else the first "supported": true/false.
    Returns (supported or None, parsed_json or None). None means no verdict (the original code treats
    that as unsupported)."""
    text = text.strip()
    start = text.find("{")
    while start != -1:
        for end in range(len(text), start, -1):
            if text[end - 1] != "}":
                continue
            try:
                obj = json.loads(text[start:end])
            except ValueError:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("supported"), bool):
                return obj["supported"], obj
        start = text.find("{", start + 1)
    m = re.search(r'"supported"\s*:\s*(true|false)', text)
    if m:
        return m.group(1) == "true", None
    return None, None
