"""Bundle what the quantization checks need from support-rag into one JSON file.

Run it with support-rag's own Python environment and from its directory (it reads the database
through support-rag's code and its .env):

    cd ~/support-rag && .venv/bin/python ~/vecsearch/bench/openvino/export_support_rag.py \
        --out ~/vecsearch/data/quant/support_rag_bundle.json

The bundle has the approved evaluation questions, and for every stored answer the exact text
support-rag's verifier was shown (rendered the way Agent.verify renders it) together with the
verdict the original verifier gave, plus a sample of corpus chunks for embedding-speed tests. It
stays out of git: the conversations are tweets from the Kaggle "Customer Support on Twitter" dataset.
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--support-rag", default=".", help="the support-rag checkout (default: current directory)")
    ap.add_argument("--chunks", type=int, default=512, help="corpus chunks to include for throughput tests")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.support_rag).resolve()
    os.chdir(root)  # settings reads .env from here
    sys.path.insert(0, str(root))
    from rag.agent import VERDICT_SCHEMA, VERIFIER_PROMPT, load_conversation
    from rag.config import settings
    from rag.db.session import make_db

    questions = [json.loads(line) for line in (root / "eval/questions.jsonl").read_text(encoding="utf-8").splitlines()]
    questions = [q for q in questions if q.get("review") == "approved"]
    rows = [json.loads(line) for line in (root / "results/answers.jsonl").read_text(encoding="utf-8").splitlines()]
    approved = {q["id"] for q in questions}

    engine, sessionmaker = make_db()
    verifier_items = []
    for r in rows:
        a = r["answer"]
        # Mirrors Agent._execute: a "no answer" submission with no citations is accepted without a
        # verifier call, so there is no original verdict to compare with.
        if r["id"] not in approved or a is None or (not a["answerable"] and not a["cited_conversation_ids"]):
            continue
        docs = []
        for cid in a["cited_conversation_ids"]:
            try:
                docs.append((await load_conversation(sessionmaker, cid))[0])
            except LookupError:
                docs.append(f"Conversation #{cid} doesn't exist.")
        prompt = (f"Question: {r['question']}\n\nAnswer to check:\n{a['answer']}\n\nCited conversations:\n\n"
                  + "\n\n---\n\n".join(docs or ["(none cited)"]))
        verifier_items.append({"id": r["id"], "prompt": prompt, "answerable": a["answerable"],
                               "original_supported": bool(a["verification"]["supported"]),
                               "original_unsupported_claims": a["verification"]["unsupported_claims"]})
    await engine.dispose()

    # Corpus chunks, evenly spaced through the file: the embedding throughput test needs realistic text.
    import pandas as pd

    texts = pd.read_parquet(Path(settings.data_dir) / "chunks.parquet", columns=["text"])["text"].to_numpy()
    step = len(texts) // args.chunks
    chunk_sample = [str(t) for t in texts[::step][:args.chunks]]

    bundle = {
        "embed_model": settings.embed_model,
        "chunk_sample": chunk_sample,
        "questions": [{k: q[k] for k in ("id", "question", "company", "answerable", "gold", "gold_root_tweet_ids")}
                      for q in questions],
        "verifier": {"system": VERIFIER_PROMPT, "schema": VERDICT_SCHEMA, "items": verifier_items},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
    flagged = sum(not i["original_supported"] for i in verifier_items)
    print(f"{len(bundle['questions'])} questions; {len(verifier_items)} verifier items "
          f"({flagged} the original verifier called unsupported) -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
