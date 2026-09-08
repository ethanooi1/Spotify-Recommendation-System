# Import Libraries
import argparse
import json
import sys
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

sys.path.insert(0, str(Path(__file__).resolve().parent)) # so "twotower" resolves when run as a script
from twotower.vocab import Vocabulary  # noqa: E402  (import after sys.path setup)



# Embedding dimensions we report parameter counts and GPU memory for. The
# two-tower model shares one embedding table per entity across both towers, so
# the total row count is track_size + artist_size + album_size.
REPORT_DIMS = (64, 128)

# params + Adam state (first and second moment) at fp32 = 4 + 4 + 4 = 12 bytes,
# but we budget 16 bytes/param as requested to leave headroom for gradients and
# fp32 master copies when training on the 16GB P100.
BYTES_PER_PARAM = 16


# Create the Spark session that does the heavy dataframe work for this script.
# Mirrors the config used in ingest_mpd.py and build_splits.py.
def create_spark_session(driver_memory):
    return (
        SparkSession.builder.appName("spotify-mpd-vocab")  # name of Spark app (spotify-mpd-vocab)
        .master("local[*]")  # where the Spark Session will run (local)
        .config("spark.driver.memory", driver_memory)  # amount of RAM to give process (4 - 8GB)
        .config("spark.sql.shuffle.partitions", "64")  # split groupBy/agg into 64 partitions instead of the default 200
        .config("spark.sql.session.timeZone", "UTC")  # sets timezone to UTC
        .getOrCreate()  # if SparkSession already exists, return it, otherwise create a new one
    )


# Build a single Vocabulary for one ID column, applying the frequency cutoff.
# Returns the Vocabulary plus stats used in the metadata file:
#   raw_distinct    -> number of distinct non-null IDs before the cutoff
#   kept            -> number of distinct IDs kept after the cutoff
# Only the final list of kept IDs is collected to the driver; the per-ID counts
# stay distributed.
def build_entity_vocabulary(train_df, column, min_freq):
    # groupBy(id).count() gives one row per distinct ID with its occurrence
    # count. Nulls are dropped first: a null ID is not a real entity and must
    # not become a vocab entry.
    counts = (
        train_df.select(F.col(column).alias("id"))
        .where(F.col("id").isNotNull())
        .groupBy("id")
        .count()
    )
    # Cache so the raw distinct count and the filtered collect below do not each
    # re-run the (expensive) shuffle/aggregation from scratch.
    counts = counts.cache()

    try:
        raw_distinct = counts.count()  # distinct IDs before the frequency cutoff

        # Keep only IDs appearing at least min_freq times. With the default
        # min_freq == 1 this keeps everything. Below-cutoff IDs are simply
        # absent from the vocab, so they encode to the unknown index at runtime.
        kept_rows = (
            counts.where(F.col("count") >= F.lit(min_freq))
            .select("id")
            .collect()  # collect ONLY the final ID list to the driver
        )
    finally:
        counts.unpersist()

    kept_ids = [row["id"] for row in kept_rows]

    # Vocabulary.build() does sorted(set(ids)) internally, so the index
    # assignment is fully deterministic and independent of collect() ordering.
    vocab = Vocabulary.build(kept_ids)
    return vocab, raw_distinct, len(kept_ids)


# Reserved indices per entity = vocab_size - kept_after_cutoff. Recorded in the
# metadata so downstream code knows how many rows (PAD / UNK) precede real IDs.
def reserved_indices_per_entity(vocab_sizes, entity_stats):
    return {
        name: vocab_sizes[name] - entity_stats[name]["kept_after_cutoff"]
        for name in vocab_sizes
    }


# Given the per-entity vocab sizes, compute total embedding parameters and the
# estimated GPU memory (params + optimizer state) at each reported dimension.
def compute_embedding_budget(vocab_sizes):
    total_rows = sum(vocab_sizes.values())  # shared tables: track + artist + album rows
    budget = {"total_embedding_rows": total_rows}
    for dim in REPORT_DIMS:
        params = total_rows * dim
        memory_bytes = params * BYTES_PER_PARAM
        budget[f"dim_{dim}"] = {
            "params": params,
            "memory_bytes": memory_bytes,
            "memory_gb": round(memory_bytes / (1024 ** 3), 3),
        }
    return budget


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build track/artist/album vocabularies from the gold train split for the two-tower model."
    )
    # --input must be the gold TRAIN table, val/test would leak labels into the vocab.
    parser.add_argument("--input", default="data/gold/train_playlist_tracks.parquet")
    parser.add_argument("--output", default="artifacts/vocab")
    parser.add_argument("--driver-memory", default="4g")
    # Frequency cutoffs: IDs appearing fewer than N times in train are excluded
    # from the vocab and encode to the unknown index at runtime. Default 1 keeps
    # everything.
    parser.add_argument("--min-track-freq", type=int, default=1)
    parser.add_argument("--min-artist-freq", type=int, default=1)
    parser.add_argument("--min-album-freq", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


# Reads the gold train table, builds the three vocabularies, saves them plus a
# metadata file, and prints a summary. Fails loudly on a missing input or on
# existing outputs when --overwrite is not set.
def main():
    args = parse_args()

    # Fail loudly if the input path is missing.
    input_path = Path(args.input)

    # Map each entity's min-freq flag once so the loop below stays simple.
    min_freqs = {
        "track_id": args.min_track_freq,
        "artist_id": args.min_artist_freq,
        "album_id": args.min_album_freq,
    }

    output_root = Path(args.output)
    vocab_paths = {name: output_root / f"{name}_vocab.json" for name in ENTITY_NAMES.values()}
    metadata_path = output_root / "vocab_metadata.json"

    # Fail loudly if any output already exists and --overwrite was not passed.
    if not args.overwrite:
        existing = [p for p in (*vocab_paths.values(), metadata_path) if p.exists()]
        if existing:
            existing_str = ", ".join(str(p) for p in existing)
            raise FileExistsError(
                f"Outputs already exist and --overwrite was not set: {existing_str}"
            )

    output_root.mkdir(parents=True, exist_ok=True)

    spark = create_spark_session(args.driver_memory)

    try:
        train_df = spark.read.parquet(str(input_path))

        entity_stats = {}  # name -> stats dict, in ENTITY_COLUMNS order
        vocab_sizes = {}   # name -> vocab.size, for the embedding budget
        for column in ENTITY_COLUMNS:
            name = ENTITY_NAMES[column]
            min_freq = min_freqs[column]

            vocab, raw_distinct, kept = build_entity_vocabulary(train_df, column, min_freq)
            vocab.save(vocab_paths[name])

            # vocab.size is the authoritative table height (kept IDs + reserved
            # slots). Using it here keeps this script correct regardless of how
            # many indices Vocabulary reserves (PAD-only vs. PAD+UNK).
            entity_stats[name] = {
                "raw_distinct": raw_distinct,
                "kept_after_cutoff": kept,
                "vocab_size": vocab.size,
                "min_freq": min_freq,
            }
            vocab_sizes[name] = vocab.size

        embedding_budget = compute_embedding_budget(vocab_sizes)

        # Write the metadata file combining per-entity stats and the budget.
        metadata = {
            "input_path": str(input_path),
            "output_path": str(output_root),
            "reserved_indices": reserved_indices_per_entity(vocab_sizes, entity_stats),
            "entities": entity_stats,
            "embedding_budget": embedding_budget,
            "bytes_per_param": BYTES_PER_PARAM,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

        summary = {
            "output_path": str(output_root),
            "vocab_sizes": vocab_sizes,
            "embedding_rows": embedding_budget["total_embedding_rows"],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        spark.stop()  # Closes the spark connection


if __name__ == "__main__":
    main()
