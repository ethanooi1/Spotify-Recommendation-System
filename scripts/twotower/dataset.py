# Import Libraries
import json
import sys
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset
from pyspark.sql import SparkSession, functions as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from twotower.vocab import Vocabulary

INPUT_PATH = 'data/gold/train_playlist_tracks.parquet'
OUTPUT_PATH = 'artifacts/twotower'
VOCAB_DIR = 'artifacts/vocab'

ENTITIES = ['track', 'artist', 'album']
PAD_INDEX = Vocabulary.PAD_INDEX
MIN_PLAYLIST_LENGTH = 2 # anything shorter can't be split into a context and a positive

def create_spark_session():
    spark = (
        SparkSession.builder.appName('spotify-mpd-twotower-cache')
        .master('local[*]')
        .config('spark.driver.bindAddress', '127.0.0.1')
        .config('spark.driver.memory', '8g')
        .config('spark.sql.shuffle.partitions', '200')
        .config('spark.sql.session.timeZone', 'UTC')
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel('ERROR')

    return spark

# Run scripts/build_vocab.py first
def load_vocabularies():
    vocabs = {}
    for entity in ENTITIES:
        path = Path(VOCAB_DIR) / f'{entity}_vocab.json'
        vocabs[entity] = Vocabulary.load(path)

    return vocabs

# Encodes the train split into CSR style arrays and saves them as an .npz.
# Each playlist[i] has a slice [offsets[i], offset[i+1]] to mark beginning and end of its occurrences in the track_idx/artist_idx/album_idx arrays.
def build_cache(spark):
    vocabs = load_vocabularies()
    train_df = spark.read.parquet(INPUT_PATH).select('pid', 'pos', 'track_id', 'artist_id', 'album_id')

    lengths = train_df.groupBy('pid').count().where(F.col('count') >= F.lit(MIN_PLAYLIST_LENGTH))
    kept = train_df.join(lengths.select('pid'), on='pid', how='inner')
    num_occurrences = kept.count()
    num_playlists = lengths.count()

    # sort_array orders the structs by their first field, pos, so each playlist keeps its original track order
    grouped = (kept
        .groupBy('pid')
        .agg(F.sort_array(F.collect_list(F.struct('pos', 'track_id', 'artist_id', 'album_id'))).alias('items'))
    )

    track_idx = np.empty(num_occurrences, dtype=np.int32)
    artist_idx = np.empty(num_occurrences, dtype=np.int32)
    album_idx = np.empty(num_occurrences, dtype=np.int32)
    offsets = np.empty(num_playlists + 1, dtype=np.int64)
    pids = np.empty(num_playlists, dtype=np.int64)
    offsets[0] = 0

    # toLocalIterator streams one partition at a time instead of all 60M rows at once
    ptr = 0
    for i, row in enumerate(grouped.toLocalIterator()):
        items = row['items']
        n = len(items)
        track_idx[ptr:ptr + n] = vocabs['track'].encode_batch([x['track_id'] for x in items])
        artist_idx[ptr:ptr + n] = vocabs['artist'].encode_batch([x['artist_id'] for x in items])
        album_idx[ptr:ptr + n] = vocabs['album'].encode_batch([x['album_id'] for x in items])
        ptr += n
        offsets[i + 1] = ptr
        pids[i] = row['pid']

    output_path = Path(OUTPUT_PATH)
    output_path.mkdir(parents=True, exist_ok=True)
    np.savez(output_path / 'train_playlists.npz', track_idx=track_idx, artist_idx=artist_idx, album_idx=album_idx, offsets=offsets, pids=pids)

    return {
        'output_path': str(output_path),
        'num_playlists': int(num_playlists),
        'num_occurrences': int(num_occurrences)
    }

# Each instance is a playlist
class PlaylistDataset(Dataset):
    def __init__(self, cache_path, max_context_len):
        data = np.load(cache_path)
        self.track_idx = data['track_idx']
        self.artist_idx = data['artist_idx']
        self.album_idx = data['album_idx']
        self.offsets = data['offsets']
        self.pids = data['pids']
        self.max_context_len = max_context_len

    # torch.Dataset requires
    def __len__(self):
        return len(self.offsets) - 1

    # torch.Dataset requires
    # Each __getitem__ picks a random "positive" track, and the rest of the playlist is the "context" to predict it from.
    def __getitem__(self, index):
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        pos = start + int(np.random.randint(end - start))
        context = np.concatenate([np.arange(start, pos), np.arange(pos + 1, end)])

        # Playlists with > 100 tracks are sampled down to 100 tracks for their context pool
        if context.shape[0] > self.max_context_len:
            context = np.random.choice(context, size=self.max_context_len, replace=False)

        return {
            'context_track': torch.from_numpy(self.track_idx[context].astype(np.int64)),
            'context_artist': torch.from_numpy(self.artist_idx[context].astype(np.int64)),
            'context_album': torch.from_numpy(self.album_idx[context].astype(np.int64)),
            'pos_track': torch.tensor(int(self.track_idx[pos]), dtype=torch.long),
            'pos_artist': torch.tensor(int(self.artist_idx[pos]), dtype=torch.long),
            'pos_album': torch.tensor(int(self.album_idx[pos]), dtype=torch.long),
            'pid': torch.tensor(int(self.pids[index]), dtype=torch.long)
        }

# All playlists need to be the same length within a batch for in-batch negatives to work, so short playlists get padded with an empty row from the embedding table
def collate_playlists(batch):
    lengths = [s['context_track'].shape[0] for s in batch]
    max_len = max(lengths)

    out = {}
    for e in ENTITIES:
        out[f'context_{e}'] = torch.full((len(batch), max_len), PAD_INDEX, dtype=torch.long)

    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)

    # Fill the context tensors and create the mask. Shape: (4096, max_len)
    for i, sample in enumerate(batch):
        n = lengths[i]
        for entity in ENTITIES:
            out[f'context_{entity}'][i, :n] = sample[f'context_{entity}']
        mask[i, :n] = True

    # The mask is used to ignore the padded cells. Shape: (4096, max_len)
    out['context_mask'] = mask

    # Turn the "positive" scalars into tensors Shape: (4096, )
    for key in ['pos_track', 'pos_artist', 'pos_album', 'pid']:
        out[key] = torch.stack([s[key] for s in batch])

    return out

# Reset the RNG for each worker per epoch, but keep the seed deterministic
def seed_worker(worker_id):
    np.random.seed(torch.initial_seed() % (2 ** 32))

def make_dataloader(dataset, batch_size, num_workers, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_playlists,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=True # ~219.76 batches of 4096 playlists in train, drops the last .76 batch
    )

def main():
    spark = create_spark_session()

    try:
        print(json.dumps(build_cache(spark), indent=2))

    finally:
        spark.stop()

if __name__ == '__main__':
    main()
