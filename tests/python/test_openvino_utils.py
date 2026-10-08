"""The small helpers the OpenVINO quality numbers rest on. They need only numpy, so CI runs them."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench" / "openvino"))
from common import parse_verdict  # noqa: E402
from quant_report import cohen_kappa, collapse, verifier_quality  # noqa: E402


@pytest.mark.parametrize("text,expected", [
    ('{"supported": true, "unsupported_claims": [], "feedback": "ok"}', True),
    ('{"supported": false, "unsupported_claims": ["x"], "feedback": "drop x"}', False),
    ('Sure! Here is the verdict:\n```json\n{"supported": false, "unsupported_claims": [], "feedback": ""}\n```', False),
    ('{"unsupported_claims": [], "supported": true, "feedback": "braces } inside"}', True),
    ('{"supported": true, "unsupported_claims": ["a claim that was cut off by the token lim', True),  # truncated
    ('{"supported": "yes"}', None),  # not a boolean
    ("I cannot verify this.", None),
    ("", None),
])
def test_parse_verdict(text, expected):
    assert parse_verdict(text)[0] is expected


def test_cohen_kappa():
    a = [True] * 9 + [False]
    assert cohen_kappa(a, a) == pytest.approx(1.0)
    assert cohen_kappa(a, [True] * 10) == pytest.approx(0.0)  # always "supported" earns no credit
    assert cohen_kappa([True, False] * 5, [False, True] * 5) == pytest.approx(-1.0)
    assert cohen_kappa([True] * 4, [True] * 4) == 1.0  # no variance: defined as agreement


def test_collapse_keeps_best_chunk_per_conversation():
    chunk_conv = np.array([5, 5, 7, 9, 7, 5])
    assert collapse([1, 0, 2, 4, 3], chunk_conv, 10) == [5, 7, 9]
    assert collapse([-1, 3, 2], chunk_conv, 1) == [9]  # missing ids are skipped, k is respected


def test_verifier_quality_counts(tmp_path):
    import json

    def verdicts(preds):
        orig = [True, True, True, False, False]
        return [dict(supported=p, original_supported=o) for p, o in zip(preds, orig)]

    for prec, preds in {"fp16": [True, True, False, False, None], "int8": [True, True, True, True, True],
                        "int4": [True, False, True, False, False]}.items():
        (tmp_path / f"llm_{prec}_GPU.json").write_text(json.dumps({"verdicts": verdicts(preds)}))
    q = verifier_quality(tmp_path, "GPU")
    # A missing verdict counts as unsupported, so slot 4 (original: unsupported) agrees.
    assert q["fp16"]["agree_count"] == 4 and q["fp16"]["unparsed"] == 1
    assert q["fp16"]["flagged_caught"] == 2 and q["fp16"]["false_alarms"] == 1
    assert q["int8"]["agreement"] == pytest.approx(0.6) and q["int8"]["kappa"] == pytest.approx(0.0)
    assert q["int8"]["always_supported_agreement"] == pytest.approx(0.6)
    assert q["int4"]["agreement_with_fp16"] == pytest.approx(0.6)
