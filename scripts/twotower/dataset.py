# Two halves live here. The cache builder is a one-off Spark job that encodes the gold train
# table into a flat .npz, so training never has to re-encode 60M rows. PlaylistDataset then
# serves (context, positive) pairs out of that cache, one random leave-one-out split per
# playlist per epoch.
#
#     python3 scripts/twotower/dataset.py --input data/gold/train_playlist_tracks.parquet

# Import Libraries
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1])) # so "twotower" resolves when run as a script
from twotower.vocab import Vocabulary  # noqa: E402

ENTITIES = ("track", "artist", "album")
PAD_INDEX = Vocabulary.PAD_INDEX


def create_spark_session(driver_memory):
    from pyspark.sql import SparkSession # lazy, so training doesn't need Spark installed
    return (
        SparkSession.builder.appName("spotify-mpd-twotower-cache")
        .master("local[*]")
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def load_vocabularies(vocab_dir):
    vocabs = {}
    for entity in ENTITIES:
        path = Path(vocab_dir) / f"{entity}_vocab.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}, run scripts/build_vocab.py first.")
        vocabs[entity] = Vocabulary.load(path)
    return vocabs


# Encodes the train split into CSR-style arrays and saves them as one .npz.
# track_idx/artist_idx/album_idx hold every occurrence back to back, and playlist i owns the
# slice [offsets[i], offsets[i+1]) in each of them. Playlists shorter than 2 get dropped since
# they can't be split into a context and a positive.
def build_cache(train_path, vocab_dir, output_path, spark, min_playlist_length=2):
    from pyspark.sql import functions as F

    vocabs = load_vocabularies(vocab_dir)
    train_df = spark.read.parquet(str(train_path)).select("pid", "pos", "track_id", "artist_id", "album_id")

    lengths = train_df.groupBy("pid").count().where(F.col("count") >= F.lit(min_playlist_length))
    kept = train_df.join(lengths.select("pid"), on="pid", how="inner")
    num_occurrences = kept.count()
    num_playlists = lengths.count()

    # sort_array orders the structs by their first field, pos, so each playlist keeps its
    # original track order.
    grouped = kept.groupBy("pid").agg(
        F.sort_array(F.collect_list(F.struct("pos", "track_id", "artist_id", "album_id"))).alias("items")
    )

    track_idx = np.empty(num_occurrences, dtype=np.int32)
    artist_idx = np.empty(num_occurrences, dtype=np.int32)
    album_idx = np.empty(num_occurrences, dtype=np.int32)
    offsets = np.empty(num_playlists + 1, dtype=np.int64)
    pids = np.empty(num_playlists, dtype=np.int64)
    offsets[0] = 0

    # toLocalIterator streams one partition at a time instead of collecting all 60M rows.
    cursor = 0
    for i, row in enumerate(grouped.toLocalIterator()):
        items = row["items"]
        n = len(items)
        track_idx[cursor:cursor + n] = vocabs["track"].encode_batch([it["track_id"] for it in items])
        artist_idx[cursor:cursor + n] = vocabs["artist"].encode_batch([it["artist_id"] for it in items])
        album_idx[cursor:cursor + n] = vocabs["album"].encode_batch([it["album_id"] for it in items])
        cursor += n
        offsets[i + 1] = cursor
        pids[i] = row["pid"]

    if cursor != num_occurrences:
        raise RuntimeError(f"Filled {cursor} occurrences but expected {num_occurrences}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, track_idx=track_idx, artist_idx=artist_idx, album_idx=album_idx,
             offsets=offsets, pids=pids, min_playlist_length=np.int64(min_playlist_length))

    return {"output_path": str(output_path), "num_playlists": int(num_playlists),
            "num_occurrences": int(num_occurrences)}


# One item is one playlist. Each __getitem__ picks a different random positive, so the same
# playlist trains on a different held-out track every epoch.
class PlaylistDataset(Dataset):
    def __init__(self, cache_path, max_context_len=100):
        data = np.load(cache_path)
        self.track_idx = data["track_idx"]
        self.artist_idx = data["artist_idx"]
        self.album_idx = data["album_idx"]
        self.offsets = data["offsets"]
        self.pids = data["pids"]
        self.max_context_len = max_context_len

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, index):
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        pos = start + int(np.random.randint(end - start))
        context = np.concatenate([np.arange(start, pos), np.arange(pos + 1, end)])

        if context.shape[0] > self.max_context_len:
            context = np.random.choice(context, size=self.max_context_len, replace=False)

        return {
            "context_track": torch.from_numpy(self.track_idx[context].astype(np.int64)),
            "context_artist": torch.from_numpy(self.artist_idx[context].astype(np.int64)),
            "context_album": torch.from_numpy(self.album_idx[context].astype(np.int64)),
            "pos_track": torch.tensor(int(self.track_idx[pos]), dtype=torch.long),
            "pos_artist": torch.tensor(int(self.artist_idx[pos]), dtype=torch.long),
            "pos_album": torch.tensor(int(self.album_idx[pos]), dtype=torch.long),
            "pid": torch.tensor(int(self.pids[index]), dtype=torch.long),
        }


# Pads every context in the batch out to the longest one and marks the real slots in a mask.
def collate_playlists(batch, pad_index=PAD_INDEX):
    lengths = [s["context_track"].shape[0] for s in batch]
    max_len = max(lengths)

    out = {f"context_{e}": torch.full((len(batch), max_len), pad_index, dtype=torch.long) for e in ENTITIES}
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)

    for i, sample in enumerate(batch):
        n = lengths[i]
        for entity in ENTITIES:
            out[f"context_{entity}"][i, :n] = sample[f"context_{entity}"]
        mask[i, :n] = True

    out["context_mask"] = mask
    for key in ("pos_track", "pos_artist", "pos_album", "pid"):
        out[key] = torch.stack([s[key] for s in batch])
    return out


# Forked workers copy the parent RNG, so without this they all draw the same positives.
def seed_worker(worker_id):
    np.random.seed(torch.initial_seed() % (2 ** 32))


def make_dataloader(dataset, batch_size, shuffle=True, num_workers=0, seed=None):
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_playlists,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
        drop_last=True, # keeps every batch the same size, which the in-batch negatives rely on
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Build the cached train playlists for the two-tower model.")
    parser.add_argument("--input", default="data/gold/train_playlist_tracks.parquet")
    parser.add_argument("--vocab-dir", default="artifacts/vocab")
    parser.add_argument("--output", default="artifacts/twotower/train_playlists.npz")
    parser.add_argument("--driver-memory", default="8g")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Cache already exists and --overwrite was not set: {output_path}")

    spark = create_spark_session(args.driver_memory)
    try:
        print(json.dumps(build_cache(args.input, args.vocab_dir, output_path, spark), indent=2))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
