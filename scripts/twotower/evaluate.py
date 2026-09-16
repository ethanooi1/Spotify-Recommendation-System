# Evaluates a trained model with full-catalog retrieval. For each eval playlist it encodes the
# visible context, scores that against every track in the MPD, and checks the top K against the
# hidden targets. The metric definitions match run_baselines.py so the numbers line up with the
# popularity and co-occurrence baselines.
#
#     python3 scripts/twotower/evaluate.py --split validation

# Import Libraries
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from twotower.vocab import Vocabulary
from twotower.model import build_model_from_vocab_sizes
from twotower.train import select_device

CHECKPOINT_PATH = 'artifacts/twotower/checkpoints/best.pt'
CACHE_PATH = 'artifacts/twotower/train_playlists.npz'
GOLD_DIR = 'data/gold'
VOCAB_DIR = 'artifacts/vocab'
BASELINE_DIR = 'artifacts/baselines'

ENTITIES = ['track', 'artist', 'album']
K_VALUES = [10, 50, 100]
UNK_INDEX = Vocabulary.UNK_INDEX
PAD_INDEX = Vocabulary.PAD_INDEX

# Rebuilds the model from the config saved in best.pt
def load_model(checkpoint_path, device):
    pt = torch.load(checkpoint_path, map_location=device)
    cfg = pt['config']
    model = build_model_from_vocab_sizes(cfg['vocab_sizes'], embedding_dim=cfg['embedding_dim'], temperature=cfg['temperature'], hidden_dims=cfg['hidden_dims'])
    model.load_state_dict(pt['model_state_dict'])
    model.eval()

    return model.to(device), cfg

# Encodes every track into an item vector for easy cosine similarity retrieval
def build_item_index(model, cache_path, device, encode_chunk_size=200000):
    data = np.load(cache_path)
    track_idx, artist_idx, album_idx = data['track_idx'], data['artist_idx'], data['album_idx']
    num_tracks = int(track_idx.max()) + 1

    # Build track to artist/album mappings
    track_to_artist = np.zeros(num_tracks, dtype=np.int64)
    track_to_album = np.zeros(num_tracks, dtype=np.int64)
    uniq, first = np.unique(track_idx, return_index=True)
    track_to_artist[uniq] = artist_idx[first]
    track_to_album[uniq] = album_idx[first]

    item_vecs = torch.empty(num_tracks, model.embedding_dim, device=device)
    with torch.no_grad():
        for start in range(0, num_tracks, encode_chunk_size):
            end = min(start + encode_chunk_size, num_tracks)
            vecs = model.encode_item(
                torch.arange(start, end, device=device),
                torch.from_numpy(track_to_artist[start:end]).to(device),
                torch.from_numpy(track_to_album[start:end]).to(device)
            )
            item_vecs[start:end] = F.normalize(vecs, dim=-1)

    return item_vecs # shape: (num_tracks, embedding_dim)

# Encodes split's (val/test) context & target playlists (track, artist, album)
# Groups context & target tracks per playlist for eval
def load_eval_playlists(split, vocabs):
    gold_dir = Path(GOLD_DIR)
    ctx = pd.read_parquet(gold_dir / f'{split}_context.parquet')
    tgt = pd.read_parquet(gold_dir / f'{split}_targets.parquet')

    ctx = ctx.sort_values(['pid', 'pos'])
    ctx['t'] = vocabs['track'].encode_batch(ctx['track_id'])
    ctx['a'] = vocabs['artist'].encode_batch(ctx['artist_id'])
    ctx['al'] = vocabs['album'].encode_batch(ctx['album_id'])

    tgt['ti'] = vocabs['track'].encode_batch(tgt['track_id'])
    
    # Both loops len(validation_playlists) times, each iteration is one playlist (pid)
    # Builds a dict: key: pid, value: set of hidden target for that pid
    targets = {}
    for pid, g in tgt.groupby('pid'):
        targets[pid] = set(g['ti'])

    # Builds a list of dicts of KV pairs below 
    playlists = []
    for pid, g in ctx.groupby('pid'):
        p = {
            'context_track': g['t'].to_numpy(),
            'context_artist': g['a'].to_numpy(),
            'context_album': g['al'].to_numpy(),
            'target_indices': targets[pid],
            'target_count': len(targets[pid])
        }
        
        playlists.append(p)

    return playlists

# Similar to dataset.py, collate_playlist(). Needs to pad all tensors to the same length with PAD_INDEX
def pad_chunk(playlists, device):
    max_len = max(len(p['context_track']) for p in playlists)

    tensors = {}
    for e in ENTITIES:
        tensors[e] = torch.full((len(playlists), max_len), PAD_INDEX, dtype=torch.long)

    mask = torch.zeros((len(playlists), max_len), dtype=torch.bool)

    for i, p in enumerate(playlists):
        n = len(p['context_track'])
        for entity in ENTITIES:
            tensors[entity][i, :n] = torch.from_numpy(p[f'context_{entity}'])
        mask[i, :n] = True

    return (tensors['track'].to(device), tensors['artist'].to(device), tensors['album'].to(device), mask.to(device))

# Recall, precision and NDCG at each K for one playlist. Same eval as run_baselines.
def playlist_metrics(ranked_indices, target_indices, target_count):
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, target_count + 1))

    m = {}
    for k in K_VALUES:
        hits, dcg = 0, 0.0
        for rank, idx in enumerate(ranked_indices[:k], start=1):
            if idx in target_indices:
                hits += 1
                dcg += 1.0 / math.log2(rank + 1)
        m[k] = {
            'recall': hits / target_count,
            'precision': hits / k,
            'ndcg': dcg / idcg
        }

    return m

# Scores every playlist against the whole catalog and averages the metrics
def evaluate(model, item_vecs, playlists, device, chunk_size=256):
    max_k = max(K_VALUES)
    sums = {}
    for k in K_VALUES:
        sums[k] = {'recall': 0.0, 'precision': 0.0, 'ndcg': 0.0}
        
    n = 0

    with torch.no_grad():
        for start in range(0, len(playlists), chunk_size):
            chunk = playlists[start : start + chunk_size]
            track, artist, album, mask = pad_chunk(chunk, device)

            # Score the playlists mean-pooled context tracks against every track in the catalog. Shape: (chunk_size, num_tracks)
            playlist_vec = F.normalize(model.encode_playlist(track, artist, album, mask), dim=-1)
            scores = playlist_vec @ item_vecs.t()

            # PAD_INDEX & UNK_INDEX aren't valid recommendations, so set those scores to -inf.
            scores[:, PAD_INDEX] = float('-inf')
            scores[:, UNK_INDEX] = float('-inf')
            
            # Don't let a playlist recommend a song it already has in its context. Set those scores to -inf so they won't be in the top K.
            for i, p in enumerate(chunk):
                scores[i, torch.from_numpy(p['context_track']).to(device)] = float('-inf')

            # Pick top-K tracks that had the highest dot product from playlist context to tracks.
            topk = scores.topk(max_k, dim=1).indices.tolist()

            # Loop through every playlist in the chunk and compute recall, precision, ndcg at each K. Sum those metrics across all playlists.
            for i, p in enumerate(chunk):
                m = playlist_metrics(topk[i], p['target_indices'], p['target_count'])
                for k in K_VALUES:
                    for name in ('recall', 'precision', 'ndcg'):
                        sums[k][name] += m[k][name]
                n += 1
    # After all playlists are scored, average the eval metrics
    metrics = {}
    for k in K_VALUES:
        for name, label in (('recall', 'Recall'), ('precision', 'Precision'), ('ndcg', 'NDCG')):
            metrics[f'{label}@{k}'] = sums[k][name] / n

    return metrics, n

# Clean output table comparing the trained model to the baselines on the same splits.
def print_comparison(twotower_metrics, split):
    baselines = json.loads((Path(BASELINE_DIR) / f'{split}_metrics.json').read_text())['baselines']
    rows = [('twotower', twotower_metrics), ('popularity', baselines['popularity']), ('cooccurrence', baselines['cooccurrence'])]

    for metric in ['Recall', 'Precision', 'NDCG']:
        labels = [f'{metric}@{k}' for k in K_VALUES]
        print(f'\n{"model":<14}' + ''.join(f'{label:<14}' for label in labels))
        print('-' * (14 * (len(labels) + 1)))
        for name, m in rows:
            print(f'{name:<14}' + ''.join(f'{m[label]:<14.4f}' for label in labels))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='validation', choices=['validation', 'test'])

    return parser.parse_args()

def main():
    args = parse_args()
    device = select_device()

    vocabs = {}
    for e in ENTITIES:
        vocabs[e] = Vocabulary.load(Path(VOCAB_DIR) / f'{e}_vocab.json')
        
    model, _ = load_model(CHECKPOINT_PATH, device)
    item_vecs = build_item_index(model, CACHE_PATH, device)
    playlists = load_eval_playlists(args.split, vocabs)
    
    metrics, n = evaluate(model, item_vecs, playlists, device)

    print_comparison(metrics, args.split)
    print('\n' + json.dumps({'split': args.split, 'playlists': n, 'metrics': metrics}, indent=2))

if __name__ == '__main__':
    main()
