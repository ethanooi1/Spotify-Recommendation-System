# Turns an Exportify CSV of one of my playlists into the track/artist/album indices recommend.py expects.
# Drops anything released after 2017, since the MPD was collected before then, and anything not in the vocabulary.

# Import Libraries
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RECS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RECS_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
from twotower.vocab import Vocabulary

VOCAB_DIR = PROJECT_ROOT / 'artifacts/vocab'
CACHE_PATH = PROJECT_ROOT / 'artifacts/twotower/train_playlists.npz'
TRACKS_PATH = PROJECT_ROOT / 'data/silver/tracks.parquet'
CUTOFF_YEAR = 2017

# Standardizes song names like "- Remastered" and "(feat. Artist)", etc.
# "Champagne Supernova - 2009 Remastered" by Oasis -> "champagnesupernova|oasis"
def song_key(title, artist):
    title = re.split(r'\s+-\s+', str(title).lower())[0]
    title = re.sub(r'[\(\[][^)\]]*[)\]]', '', title)
    title = re.sub(r'[^a-z0-9]', '', title)
    if not title:
        return None

    return title + '|' + re.sub(r'[^a-z0-9]', '', str(artist).lower())

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--playlist')

    return parser.parse_args()

def main():
    args = parse_args()
    raw = pd.read_csv(RECS_ROOT / 'data' / f'{args.playlist}.csv')

    # Local files show up as spotify:local:... with no release date, so they can't be in the MPD
    year = pd.to_numeric(raw['Release Date'].astype(str).str[:4], errors='coerce')
    raw = raw[raw['Track URI'].str.startswith('spotify:track:') & (year <= CUTOFF_YEAR)]
    df = pd.DataFrame({
        'track_id': raw['Track URI'].str.replace('spotify:track:', ''),
        'track_name': raw['Track Name'],
        'artist_name': raw['Artist Name(s)'].str.split(';').str[0]
    })

    vocabs = {}
    for e in ['track', 'artist', 'album']:
        vocabs[e] = Vocabulary.load(VOCAB_DIR / f'{e}_vocab.json')

    # One song can carry several Spotify ids, so match on name and artist when the id isn't the one the MPD has, and take the copy that appears in the most training playlists
    mpd = pd.read_parquet(TRACKS_PATH, columns=['track_id', 'track_name', 'artist_name', 'artist_id', 'album_id'])
    mpd = mpd[mpd['track_id'].isin(vocabs['track'].id_to_index)]
    mpd['key'] = [song_key(n, a) for n, a in zip(mpd['track_name'], mpd['artist_name'])]
    mpd['train_count'] = np.bincount(np.load(CACHE_PATH)['track_idx'])[vocabs['track'].encode_batch(mpd['track_id'])]
    by_name = mpd.dropna(subset=['key']).sort_values('train_count', ascending=False).drop_duplicates('key').set_index('key')['track_id']

    df['key'] = [song_key(n, a) for n, a in zip(df['track_name'], df['artist_name'])]
    in_vocab = df['track_id'].isin(vocabs['track'].id_to_index)
    df['mpd_track_id'] = df['track_id'].where(in_vocab, df['key'].map(by_name))
    usable = df[df['mpd_track_id'].notna()].join(mpd.set_index('track_id')[['artist_id', 'album_id']], on='mpd_track_id')

    usable['track_idx'] = vocabs['track'].encode_batch(usable['mpd_track_id'])
    usable['artist_idx'] = vocabs['artist'].encode_batch(usable['artist_id'])
    usable['album_idx'] = vocabs['album'].encode_batch(usable['album_id'])

    out_path = RECS_ROOT / 'artifacts' / f'{args.playlist}.csv'
    usable[['track_name', 'artist_name', 'track_idx', 'artist_idx', 'album_idx']].to_csv(out_path, index=False)
    print(f'{len(usable)} of {len(raw)} tracks released by {CUTOFF_YEAR} are in the MPD, wrote {out_path}')

if __name__ == '__main__':
    main()
