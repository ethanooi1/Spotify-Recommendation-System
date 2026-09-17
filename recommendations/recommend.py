# Recommends tracks for a playlist encoded by prepare_playlist.py

# Import Libraries
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

RECS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RECS_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
from twotower.evaluate import load_model, build_item_index, select_device, PAD_INDEX, UNK_INDEX
from prepare_playlist import song_key

CHECKPOINT_PATH = PROJECT_ROOT / 'artifacts/twotower/checkpoints/best.pt'
CACHE_PATH = PROJECT_ROOT / 'artifacts/twotower/train_playlists.npz'
VOCAB_PATH = PROJECT_ROOT / 'artifacts/vocab/track_vocab.json'
TRACKS_PATH = PROJECT_ROOT / 'data/silver/tracks.parquet'

# Turns track indices back into ids and names, using the vocab in reverse
def name_lookup(indices):
    index_to_id = {}
    for k, v in json.loads(VOCAB_PATH.read_text())['id_to_index'].items():
        index_to_id[v] = k

    tracks = pd.read_parquet(TRACKS_PATH, columns=['track_id', 'track_name', 'artist_name']).set_index('track_id')

    rows = []
    for i in indices:
        track_id = index_to_id[i]
        row = tracks.loc[track_id]
        rows.append((track_id, row['track_name'], row['artist_name']))

    return rows

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--playlist')
    parser.add_argument('--top-k', type=int, default=20)

    return parser.parse_args()

def main():
    args = parse_args()
    device = select_device()
    model = load_model(CHECKPOINT_PATH, device)

    playlist = pd.read_csv(RECS_ROOT / 'artifacts' / f'{args.playlist}.csv')
    print(f'playlist: {args.playlist}, {len(playlist)} tracks')

    # Same as evaluate.py, score the playlists mean-pooled context tracks against every track in the catalog.
    with torch.no_grad():
        track = torch.tensor(playlist['track_idx'], device=device).unsqueeze(0)
        artist = torch.tensor(playlist['artist_idx'], device=device).unsqueeze(0)
        album = torch.tensor(playlist['album_idx'], device=device).unsqueeze(0)
        mask = torch.ones_like(track, dtype=torch.bool)
        playlist_vec = F.normalize(model.encode_playlist(track, artist, album, mask), dim=-1)

    item_vecs = build_item_index(model, CACHE_PATH, device)
    scores = (playlist_vec @ item_vecs.t()).squeeze(0)
    
    # PAD_INDEX & UNK_INDEX aren't valid recommendations, so set those scores to -inf.
    scores[PAD_INDEX] = float('-inf')
    scores[UNK_INDEX] = float('-inf')
    scores[torch.tensor(playlist['track_idx'].to_numpy(), device=device)] = float('-inf')

    # Over-recommend the top K songs because we may drop some that are already in the playlist
    candidates = scores.topk(args.top_k * 5)
    rows = name_lookup(candidates.indices.tolist())
    seen = set()
    for t, a in zip(playlist['track_name'], playlist['artist_name']):
        key = song_key(t, a)
        if key:
            seen.add(key)

    kept = []
    for (track_id, name, artist), score in zip(rows, candidates.values.tolist()):
        if song_key(name, artist) not in seen:
            kept.append((track_id, name, artist, round(score, 3)))
        if len(kept) == args.top_k:
            break

    out = pd.DataFrame(kept, columns=['mpd_track_id', 'track_name', 'artist_name', 'score'])
    out.insert(0, 'rank', range(1, len(out) + 1))

    print(f'\ntop {len(out)} recommendations\n')
    print(out.to_string(index=False))

    out_path = RECS_ROOT / 'artifacts' / f'{args.playlist}_recommendations.csv'
    out.to_csv(out_path, index=False)
    print(f'\nwrote to {out_path}')

if __name__ == '__main__':
    main()
