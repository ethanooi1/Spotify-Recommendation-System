# Evaluates a trained model with full-catalog retrieval. For each eval playlist it encodes the
# visible context, scores that against every track in the MPD, and checks the top K against the
# hidden targets. The metric definitions match run_baselines.py so the numbers line up with the
# popularity and co-occurrence baselines.
#
#     python3 scripts/twotower/evaluate.py --checkpoint artifacts/twotower/checkpoints/best.pt --split validation

# Import Libraries
import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1])) # so "twotower" resolves when run as a script
from twotower.vocab import Vocabulary  # noqa: E402
from twotower.model import build_model_from_vocab_sizes, PAD_INDEX  # noqa: E402

ENTITIES = ("track", "artist", "album")
UNK_INDEX = Vocabulary.UNK_INDEX


def select_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Rebuilds the model from the config saved in the checkpoint.
def load_model(checkpoint_path, device):
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = build_model_from_vocab_sizes(
        cfg["vocab_sizes"], embedding_dim=cfg["embedding_dim"],
        temperature=cfg["temperature"], hidden_dims=cfg.get("hidden_dims"),
    )
    model.load_state_dict(ck["model_state_dict"])
    return model.to(device).eval(), cfg


# Encodes every track into a normalized item vector, so retrieval is one matmul.
# A track's artist and album come from the first time it appears in the train cache.
def build_item_index(model, cache_path, device, encode_chunk=200000):
    data = np.load(cache_path)
    track_idx, artist_idx, album_idx = data["track_idx"], data["artist_idx"], data["album_idx"]
    num_tracks = int(track_idx.max()) + 1

    track_to_artist = np.zeros(num_tracks, dtype=np.int64)
    track_to_album = np.zeros(num_tracks, dtype=np.int64)
    uniq, first = np.unique(track_idx, return_index=True)
    track_to_artist[uniq] = artist_idx[first]
    track_to_album[uniq] = album_idx[first]

    item_vecs = torch.empty(num_tracks, model.embedding_dim, device=device)
    with torch.no_grad():
        for start in range(0, num_tracks, encode_chunk):
            end = min(start + encode_chunk, num_tracks)
            vecs = model.encode_item(
                torch.arange(start, end, device=device),
                torch.from_numpy(track_to_artist[start:end]).to(device),
                torch.from_numpy(track_to_album[start:end]).to(device),
            )
            item_vecs[start:end] = F.normalize(vecs, dim=-1)
    return item_vecs


def _encode_column(series, id_to_index):
    return series.map(id_to_index).fillna(UNK_INDEX).astype(np.int64).to_numpy()


# The gold tables can be a Spark directory of parts or a single file, so handle both.
def _read_parquet_path(path):
    path = Path(path)
    if path.is_dir():
        files = sorted(glob.glob(str(path / "*.parquet")))
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return pd.read_parquet(path)


# Reads a split's context and targets and groups them per playlist. Targets get deduplicated
# the same way run_baselines does, so recall divides by the same denominator.
def load_eval_playlists(gold_dir, split, vocabs):
    gold_dir = Path(gold_dir)
    ctx = _read_parquet_path(gold_dir / f"{split}_context.parquet")
    tgt = _read_parquet_path(gold_dir / f"{split}_targets.parquet")

    ctx = ctx[ctx["track_id"].notna()].sort_values(["pid", "pos"])
    ctx["t"] = _encode_column(ctx["track_id"], vocabs["track"].id_to_index)
    ctx["a"] = _encode_column(ctx["artist_id"], vocabs["artist"].id_to_index)
    ctx["al"] = _encode_column(ctx["album_id"], vocabs["album"].id_to_index)

    tgt = tgt[tgt["track_id"].notna()][["pid", "track_id"]].drop_duplicates()
    tgt["ti"] = _encode_column(tgt["track_id"], vocabs["track"].id_to_index)
    targets = {pid: set(g["ti"]) for pid, g in tgt.groupby("pid")}

    playlists = []
    for pid, g in ctx.groupby("pid"):
        if pid not in targets:
            continue
        playlists.append({
            "context_track": g["t"].to_numpy(),
            "context_artist": g["a"].to_numpy(),
            "context_album": g["al"].to_numpy(),
            "target_indices": targets[pid],
            "target_count": len(targets[pid]),
        })
    return playlists


def _pad_chunk(playlists, device):
    max_len = max(len(p["context_track"]) for p in playlists)
    tensors = {e: torch.full((len(playlists), max_len), PAD_INDEX, dtype=torch.long) for e in ENTITIES}
    mask = torch.zeros((len(playlists), max_len), dtype=torch.bool)

    for i, p in enumerate(playlists):
        n = len(p["context_track"])
        for entity in ENTITIES:
            tensors[entity][i, :n] = torch.from_numpy(p[f"context_{entity}"])
        mask[i, :n] = True

    return (tensors["track"].to(device), tensors["artist"].to(device),
            tensors["album"].to(device), mask.to(device))


# Recall, precision and NDCG at each K for one playlist. Same definitions as run_baselines.
def playlist_metrics(ranked_indices, target_indices, target_count, k_values):
    out = {}
    for k in k_values:
        hits, dcg = 0, 0.0
        for rank, idx in enumerate(ranked_indices[:k], start=1):
            if idx in target_indices:
                hits += 1
                dcg += 1.0 / math.log2(rank + 1)
        idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(target_count, k) + 1))
        out[k] = {"recall": hits / target_count, "precision": hits / k,
                  "ndcg": (dcg / idcg) if idcg > 0 else 0.0}
    return out


# Scores every playlist against the whole catalog and averages the metrics.
def evaluate(model, item_vecs, playlists, k_values, device, chunk_size=256):
    max_k = max(k_values)
    sums = {k: {"recall": 0.0, "precision": 0.0, "ndcg": 0.0} for k in k_values}
    n = 0

    with torch.no_grad():
        for start in range(0, len(playlists), chunk_size):
            chunk = playlists[start:start + chunk_size]
            track, artist, album, mask = _pad_chunk(chunk, device)

            playlist_vec = F.normalize(model.encode_playlist(track, artist, album, mask), dim=-1)
            scores = playlist_vec @ item_vecs.t()
            scores[:, PAD_INDEX] = float("-inf")
            scores[:, UNK_INDEX] = float("-inf")
            for i, p in enumerate(chunk):
                scores[i, torch.from_numpy(p["context_track"]).to(device)] = float("-inf")

            topk = scores.topk(max_k, dim=1).indices.cpu().numpy()

            for i, p in enumerate(chunk):
                m = playlist_metrics(topk[i], p["target_indices"], p["target_count"], k_values)
                for k in k_values:
                    for name in ("recall", "precision", "ndcg"):
                        sums[k][name] += m[k][name]
                n += 1

    metrics = {}
    for k in k_values:
        for name, label in (("recall", "Recall"), ("precision", "Precision"), ("ndcg", "NDCG")):
            metrics[f"{label}@{k}"] = sums[k][name] / n
    return metrics, n


# Prints the model next to the baselines from run_baselines, when that file is around.
def print_comparison(twotower_metrics, baseline_path, k_values):
    rows = [("twotower", twotower_metrics)]
    path = Path(baseline_path)
    if path.exists():
        baselines = json.loads(path.read_text()).get("baselines", {})
        rows += [(name, baselines[name]) for name in ("popularity", "cooccurrence") if name in baselines]

    for metric in ("Recall", "Precision", "NDCG"):
        header = "  ".join(f"{metric}@{k:<4}" for k in k_values)
        print(f"\n{'model':<14} {header}")
        print("-" * (14 + len(header) + 2))
        for name, m in rows:
            print(f"{name:<14} " + "  ".join(f"{m.get(f'{metric}@{k}', float('nan')):<9.4f}" for k in k_values))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the two-tower model with full-catalog retrieval.")
    parser.add_argument("--checkpoint", default="artifacts/twotower/checkpoints/best.pt")
    parser.add_argument("--cache", default="artifacts/twotower/train_playlists.npz")
    parser.add_argument("--gold-dir", default="data/gold")
    parser.add_argument("--vocab-dir", default="artifacts/vocab")
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--k-values", default="10,50,100")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--baseline-metrics", default="artifacts/baselines/sample/validation_metrics.json")
    parser.add_argument("--out", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    k_values = sorted({int(v) for v in args.k_values.split(",") if v.strip()})
    device = select_device(args.device)

    vocabs = {e: Vocabulary.load(Path(args.vocab_dir) / f"{e}_vocab.json") for e in ENTITIES}
    model, cfg = load_model(args.checkpoint, device)

    item_vecs = build_item_index(model, args.cache, device)
    playlists = load_eval_playlists(args.gold_dir, args.split, vocabs)
    metrics, n = evaluate(model, item_vecs, playlists, k_values, device)

    print_comparison(metrics, args.baseline_metrics, k_values)

    summary = {"split": args.split, "playlists": n, "metrics": metrics}
    print("\n" + json.dumps(summary, indent=2, sort_keys=True))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
