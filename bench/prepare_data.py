"""Build base.npy / queries.npy (L2-normalized float32) and base_tweet_ids.npy.

Sources:
  --csv twcs.csv      Kaggle "Customer Support on Twitter", embedded with all-MiniLM-L6-v2 (384-d).
                      Embeddings go to a memmap in 50k chunks, so an interrupted run resumes.
  --embeddings X.npy  precomputed embeddings
  --synthetic N       clustered Gaussian data, for CI

Queries are the first --n-queries rows after shuffling and are excluded from base.
"""
import argparse
import os
import re

import numpy as np


def normalize(x):
    return (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def clean(text: str) -> str:
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def from_csv(args):
    import pandas as pd

    df = pd.read_csv(args.csv, usecols=["tweet_id", "text"], dtype={"tweet_id": np.int64, "text": str})
    df["clean"] = df["text"].fillna("").map(clean)
    df = df[df["clean"].str.len() >= 5].drop_duplicates("clean")
    if args.limit:
        df = df.sample(n=min(args.limit + args.n_queries, len(df)), random_state=0)
    df = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    print(f"{len(df):,} unique tweets after cleaning")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model)
    d = model.get_sentence_embedding_dimension()
    texts = df["clean"].tolist()
    os.makedirs(args.out, exist_ok=True)
    mm_path = os.path.join(args.out, "all_emb.f32")
    progress_path = mm_path + ".done"
    mm = np.memmap(mm_path, dtype=np.float32, mode="r+" if os.path.exists(mm_path) else "w+",
                   shape=(len(texts), d))
    start = int(open(progress_path).read()) if os.path.exists(progress_path) else 0
    chunk = 50_000
    for s in range(start, len(texts), chunk):
        e = min(s + chunk, len(texts))
        mm[s:e] = model.encode(texts[s:e], batch_size=512, normalize_embeddings=True,
                               show_progress_bar=False, convert_to_numpy=True)
        mm.flush()
        open(progress_path, "w").write(str(e))
        print(f"embedded {e:,}/{len(texts):,}")
    return np.asarray(mm), df["tweet_id"].to_numpy()


def synthetic(n, d, n_clusters=2000, latent=24, spread=0.5, seed=0):
    """Gaussian clusters in a `latent`-dim space, projected to d dims. Low intrinsic
    dimension like real sentence embeddings; uniform noise would be unrealistically hard."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, latent)).astype(np.float32)
    z = centers[rng.integers(0, n_clusters, n)] + spread * rng.standard_normal((n, latent)).astype(np.float32)
    proj = rng.standard_normal((latent, d)).astype(np.float32) / np.sqrt(latent)
    return z @ proj + 0.05 * rng.standard_normal((n, d)).astype(np.float32), np.arange(n)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv")
    src.add_argument("--embeddings")
    src.add_argument("--synthetic", type=int)
    ap.add_argument("--dim", type=int, default=384, help="synthetic only")
    ap.add_argument("--latent", type=int, default=24, help="synthetic only: intrinsic dimension (higher = harder)")
    ap.add_argument("--spread", type=float, default=0.5, help="synthetic only: within-cluster std (higher = harder)")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--limit", type=int, default=0, help="subsample tweets (0 = all)")
    ap.add_argument("--n-queries", type=int, default=10_000)
    ap.add_argument("--out", default="data/tweets")
    args = ap.parse_args()

    if args.csv:
        x, ids = from_csv(args)
    elif args.embeddings:
        x = np.load(args.embeddings, mmap_mode="r")
        ids = np.arange(len(x))
    else:
        x, ids = synthetic(args.synthetic + args.n_queries, args.dim, latent=args.latent, spread=args.spread)

    x = normalize(np.asarray(x, dtype=np.float32))
    q = args.n_queries
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "queries.npy"), x[:q])
    np.save(os.path.join(args.out, "base.npy"), x[q:])
    np.save(os.path.join(args.out, "base_tweet_ids.npy"), ids[q:])
    print(f"base {x[q:].shape}, queries {x[:q].shape} -> {args.out}")


if __name__ == "__main__":
    main()
