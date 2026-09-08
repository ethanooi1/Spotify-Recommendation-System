# Import Libraries
import argparse
import json
import math
from pathlib import Path

from pyspark import StorageLevel
from pyspark.sql import SparkSession, Window, functions as F


# Baseline settings, fixed rather than exposed as flags.
SEED = 42
K_VALUES = [10, 50, 100]
TRAIN_SAMPLE_SIZE = 50000 # sample mode only, --full-data uses everything
VALIDATION_SAMPLE_SIZE = 5000
MIN_PAIR_SUPPORT = 3 # a co-occurrence pair needs this many playlists behind it
MIN_CANDIDATE_POPULARITY = 5


# Create the Spark session that does the heavy dataframe work for this script.
def create_spark_session(driver_memory):
    return (
        SparkSession.builder.appName("spotify-mpd-baselines") # name of Spark app (spotify-mpd-ingestion)
        .master("local[*]") # where the Spark Session will run (local)
        .config("spark.driver.memory", driver_memory) # amount of RAM to give process (4 - 8GB)
        .config("spark.sql.shuffle.partitions", "64") # when dealing with groupBy, joins, aggregations, we split into 64 partitions instead of the default 200 (easier on local device)
        .config("spark.sql.session.timeZone", "UTC") # sets timezone to UTC
        .getOrCreate() # if SparkSession already exists, return it, otherwise create a new one
    )


# Reads parquet tables from data/gold
def read_gold_tables(input_dir, spark):
    input_root = Path(input_dir)
    train_path = input_root / "train_playlist_tracks.parquet"
    validation_context_path = input_root / "validation_context.parquet"
    validation_targets_path = input_root / "validation_targets.parquet"


    train_df = spark.read.parquet(str(train_path))
    validation_context_df = spark.read.parquet(str(validation_context_path))
    validation_targets_df = spark.read.parquet(str(validation_targets_path))
    return train_df, validation_context_df, validation_targets_df


# Deterministically samples pid values by hashing pid and taking the lowest hash scores.
# main() uses this for sample mode so local reruns always pick the same playlists.
def select_sample_pids(pid_df, split_name, seed, sample_size, full_data):
    distinct_pids = pid_df.select("pid").distinct()

    if full_data:
        return distinct_pids

    return (
        distinct_pids
        .withColumn(
            "sample_score",
            F.pmod(
                F.xxhash64(
                    F.lit(str(seed)),
                    F.lit(split_name),
                    F.col("pid").cast("string"),
                ),
                F.lit(1000000),
            ),
        )
        .orderBy(F.col("sample_score"), F.col("pid"))
        .limit(sample_size)
        .select("pid")
    )


# Filters the train and validation tables down to either the sampled playlists or the full data.
# main() uses this once so every later baseline runs on the exact same sampled or full subset.
def build_run_tables(train_df, validation_context_df, validation_targets_df, seed, train_sample_size, validation_sample_size, full_data):
    sampled_train_pids = select_sample_pids(train_df.select("pid"), "train", seed, train_sample_size, full_data)
    sampled_validation_pids = select_sample_pids(validation_context_df.select("pid"), "validation", seed, validation_sample_size, full_data)

    sampled_train_df = (
        train_df.join(sampled_train_pids, on="pid", how="inner")
        .where(F.col("track_id").isNotNull())
        .select("pid", "pos", "track_id", "artist_id", "album_id", "duration_ms")
    )
    sampled_validation_context_df = (
        validation_context_df.join(sampled_validation_pids, on="pid", how="inner")
        .where(F.col("track_id").isNotNull())
        .select("pid", "pos", "track_id", "artist_id", "album_id", "duration_ms")
    )
    sampled_validation_targets_df = (
        validation_targets_df.join(sampled_validation_pids, on="pid", how="inner")
        .where(F.col("track_id").isNotNull())
        .select("pid", "pos", "track_id", "artist_id", "album_id", "duration_ms")
    )

    return sampled_train_df, sampled_validation_context_df, sampled_validation_targets_df


# Counts how popular each track is in the train split and ranks tracks from most common to least common.
# main() builds this once because both baselines use global train popularity in different ways.
def build_track_popularity(train_df):
    popularity_window = Window.orderBy(F.desc("train_frequency"), F.asc("track_id"))

    return (
        train_df.groupBy("track_id")
        .agg(F.count(F.lit(1)).cast("long").alias("train_frequency"))
        .withColumn("popularity_rank", F.row_number().over(popularity_window))
        .select("track_id", "train_frequency", "popularity_rank")
    )


# Builds the popularity baseline by recommending the globally most common tracks not already in each playlist.
# main() uses this to create the easiest benchmark before the stronger co-occurrence baseline.
def build_popularity_recommendations(validation_context_df, track_popularity_df, max_k):
    playlists_df = validation_context_df.select("pid").distinct()
    context_tracks_df = validation_context_df.select("pid", "track_id").distinct()
    max_unique_context_tracks = (
        context_tracks_df.groupBy("pid")
        .agg(F.count(F.lit(1)).alias("unique_context_tracks"))
        .agg(F.max("unique_context_tracks").alias("max_unique_context_tracks"))
        .first()["max_unique_context_tracks"]
    )
    max_unique_context_tracks = 0 if max_unique_context_tracks is None else int(max_unique_context_tracks)

    candidate_pool_size = max_k + max_unique_context_tracks + 50
    candidate_pool_df = (
        track_popularity_df.orderBy(F.asc("popularity_rank"))
        .limit(candidate_pool_size)
        .select("track_id", "train_frequency", "popularity_rank")
    )

    popularity_rank_window = Window.partitionBy("pid").orderBy(
        F.desc("train_frequency"),
        F.asc("popularity_rank"),
        F.asc("track_id"),
    )

    return (
        playlists_df.crossJoin(F.broadcast(candidate_pool_df))
        .join(context_tracks_df, on=["pid", "track_id"], how="left_anti")
        .withColumn("rank", F.row_number().over(popularity_rank_window))
        .where(F.col("rank") <= F.lit(max_k))
        .withColumn("baseline_name", F.lit("popularity"))
        .withColumn("score", F.col("train_frequency").cast("double"))
        .select("baseline_name", "pid", "rank", "track_id", "score")
    )


# Builds exact track-to-track co-occurrence counts from the train playlists.
# main() uses this as the stronger simple baseline that later PyTorch models should beat.
def build_cooccurrence_pairs(train_df, track_popularity_df, min_pair_support, min_candidate_popularity):
    train_source_df = train_df.select("pid", "track_id").where(F.col("track_id").isNotNull())
    recommendable_tracks_df = (
        track_popularity_df.where(F.col("train_frequency") >= F.lit(min_candidate_popularity))
        .select(
            F.col("track_id").alias("candidate_track_id"),
            F.col("train_frequency").alias("candidate_popularity"),
        )
    )

    context_rows_df = train_source_df.select(
        F.col("pid"),
        F.col("track_id").alias("context_track_id"),
    )
    candidate_rows_df = (
        train_source_df.join(
            recommendable_tracks_df.select("candidate_track_id"),
            train_source_df.track_id == F.col("candidate_track_id"),
            how="inner",
        )
        .select(
            train_source_df.pid.alias("pid"),
            F.col("candidate_track_id"),
        )
    )

    return (
        context_rows_df.join(candidate_rows_df, on="pid", how="inner")
        .where(F.col("context_track_id") != F.col("candidate_track_id"))
        .groupBy("context_track_id", "candidate_track_id")
        .agg(F.count(F.lit(1)).cast("long").alias("pair_support"))
        .where(F.col("pair_support") >= F.lit(min_pair_support))
        .join(recommendable_tracks_df, on="candidate_track_id", how="inner")
        .select("context_track_id", "candidate_track_id", "pair_support", "candidate_popularity")
    )


# Builds the co-occurrence baseline by scoring candidate tracks against all context tracks in each playlist.
# main() uses this after the pair table is built so validation playlists can get top-K ranked candidates.
def build_cooccurrence_recommendations(validation_context_df, cooccurrence_pairs_df, max_k):
    context_tracks_df = validation_context_df.select("pid", "track_id").where(F.col("track_id").isNotNull())
    context_filter_df = context_tracks_df.select("pid", "track_id").distinct()

    cooccurrence_rank_window = Window.partitionBy("pid").orderBy(
        F.desc("score"),
        F.desc("candidate_popularity"),
        F.asc("track_id"),
    )

    return (
        context_tracks_df.select(
            "pid",
            F.col("track_id").alias("context_track_id"),
        )
        .join(cooccurrence_pairs_df, on="context_track_id", how="inner")
        .groupBy("pid", F.col("candidate_track_id").alias("track_id"))
        .agg(
            F.sum("pair_support").cast("double").alias("score"),
            F.first("candidate_popularity", ignorenulls=True).alias("candidate_popularity"),
        )
        .join(context_filter_df, on=["pid", "track_id"], how="left_anti")
        .withColumn("rank", F.row_number().over(cooccurrence_rank_window))
        .where(F.col("rank") <= F.lit(max_k))
        .withColumn("baseline_name", F.lit("cooccurrence"))
        .select("baseline_name", "pid", "rank", "track_id", "score")
    )


# Builds a tiny lookup table for ideal DCG so NDCG can be computed without a UDF.
# compute_metrics() uses this for each K value when it turns playlist-level hits into NDCG.
def build_idcg_lookup(spark, max_target_count, k_value):
    rows = []
    for target_count in range(1, max_target_count + 1):
        ideal_rank_count = min(target_count, k_value)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_rank_count + 1))
        rows.append((target_count, float(idcg)))

    return spark.createDataFrame(rows, ["target_count", "idcg"])


# Evaluates all baseline recommendations against the deduplicated validation targets.
# main() uses this after recommendations are built so the milestone ends with real benchmark metrics.
def compute_metrics(spark, recommendations_df, validation_targets_df, k_values):
    target_relevance_df = validation_targets_df.select("pid", "track_id").distinct().persist(StorageLevel.MEMORY_AND_DISK)
    target_counts_df = target_relevance_df.groupBy("pid").agg(F.count(F.lit(1)).cast("int").alias("target_count")).persist(StorageLevel.MEMORY_AND_DISK)
    evaluation_pids_df = target_counts_df.select("pid")
    baseline_names_df = spark.createDataFrame(
        [("popularity",), ("cooccurrence",)],
        ["baseline_name"],
    )
    evaluation_grid_df = baseline_names_df.crossJoin(evaluation_pids_df).persist(StorageLevel.MEMORY_AND_DISK)
    max_target_count = target_counts_df.agg(F.max("target_count").alias("max_target_count")).first()["max_target_count"]

    metrics_by_baseline = {
        "popularity": {},
        "cooccurrence": {},
    }

    for k_value in k_values:
        idcg_lookup_df = build_idcg_lookup(spark, max_target_count, k_value)
        hits_df = (
            recommendations_df.where(F.col("rank") <= F.lit(k_value))
            .join(target_relevance_df.withColumn("is_relevant", F.lit(1.0)), on=["pid", "track_id"], how="left")
            .withColumn("is_hit", F.when(F.col("is_relevant").isNotNull(), F.lit(1.0)).otherwise(F.lit(0.0)))
            .withColumn("dcg_contribution", F.col("is_hit") / F.log2(F.col("rank") + F.lit(1.0)))
            .groupBy("baseline_name", "pid")
            .agg(
                F.sum("is_hit").alias("hit_count"),
                F.sum("dcg_contribution").alias("dcg"),
            )
        )

        metric_rows = (
            evaluation_grid_df.join(target_counts_df, on="pid", how="inner")
            .join(idcg_lookup_df, on="target_count", how="left")
            .join(hits_df, on=["baseline_name", "pid"], how="left")
            .fillna({"hit_count": 0.0, "dcg": 0.0})
            .withColumn("recall", F.col("hit_count") / F.col("target_count"))
            .withColumn("precision", F.col("hit_count") / F.lit(float(k_value)))
            .withColumn(
                "ndcg",
                F.when(F.col("idcg") > F.lit(0.0), F.col("dcg") / F.col("idcg")).otherwise(F.lit(0.0)),
            )
            .groupBy("baseline_name")
            .agg(
                F.avg("recall").alias("recall"),
                F.avg("precision").alias("precision"),
                F.avg("ndcg").alias("ndcg"),
            )
            .collect()
        )

        for row in metric_rows:
            metrics_by_baseline[row["baseline_name"]][f"Recall@{k_value}"] = float(row["recall"])
            metrics_by_baseline[row["baseline_name"]][f"Precision@{k_value}"] = float(row["precision"])
            metrics_by_baseline[row["baseline_name"]][f"NDCG@{k_value}"] = float(row["ndcg"])

    target_relevance_df.unpersist()
    target_counts_df.unpersist()
    evaluation_grid_df.unpersist()
    return metrics_by_baseline


# Writes the metrics summary as JSON so the notebook and Azure runs both have one simple artifact to inspect.
# main() calls this after Spark metrics are collected back to Python dictionaries.
def write_metrics_json(metrics_by_baseline, metrics_path):
    output_payload = {
        "baselines": metrics_by_baseline,
    }
    metrics_path.write_text(json.dumps(output_payload, indent=2, sort_keys=True))


# ArgumentParser lets us run the same script locally on a sample or later on Azure with full data.
# main() calls this first so both execution modes share one interface.
def parse_args():
    parser = argparse.ArgumentParser(description="Run popularity and co-occurrence baselines on the gold Parquet tables.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--driver-memory", default="8g")
    parser.add_argument("--full-data", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


# Uses the helper functions to sample data, build baselines, score recommendations, and write artifacts.
# This is the one place where the local sample workflow and later Azure full-data workflow stay identical.
def main():
    args = parse_args()
    max_k = max(K_VALUES)
    
    output_root = Path(args.output) # creates a Path object to data/silver where the Parquet tables will be stored
    output_root.mkdir(parents=True, exist_ok=True) # creates the output directory data/silver if it doesn't exist already
    
    # Creates path for each parquet table write
    popularity_path = output_root / "track_popularity.parquet"
    cooccurrence_pairs_path = output_root / "cooccurrence_pairs.parquet"
    recommendations_path = output_root / "validation_recommendations.parquet"
    metrics_path = output_root / "validation_metrics.json"
    write_mode = "overwrite" if args.overwrite else "errorifexists" # if we use --overwrite in CLI, then MPD file ingestion overwrites all old files in data/silver/{path}

    # Creates SparkSession
    spark = create_spark_session(args.driver_memory)

    try:
        train_df, validation_context_df, validation_targets_df = read_gold_tables(args.input, spark)
        run_train_df, run_validation_context_df, run_validation_targets_df = build_run_tables(
            train_df,
            validation_context_df,
            validation_targets_df,
            SEED, TRAIN_SAMPLE_SIZE, VALIDATION_SAMPLE_SIZE,
            args.full_data, # defaults to False, unless add --full_data in CLI script run
        )
        # .persist(StorageLevel.MEMORY_AND_DISK) -> persist/cache the DataFrame so it doesn't recompute it on every run, use memory first and spill into disk if need be.
        run_train_df = run_train_df.persist(StorageLevel.MEMORY_AND_DISK)
        run_validation_context_df = run_validation_context_df.persist(StorageLevel.MEMORY_AND_DISK)
        run_validation_targets_df = run_validation_targets_df.persist(StorageLevel.MEMORY_AND_DISK)

        track_popularity_df = build_track_popularity(run_train_df).persist(StorageLevel.MEMORY_AND_DISK)
        popularity_recommendations_df = build_popularity_recommendations(run_validation_context_df, track_popularity_df, max_k)
        cooccurrence_pairs_df = build_cooccurrence_pairs(
            run_train_df,
            track_popularity_df,
            MIN_PAIR_SUPPORT, MIN_CANDIDATE_POPULARITY,
        ).persist(StorageLevel.MEMORY_AND_DISK)
        cooccurrence_recommendations_df = build_cooccurrence_recommendations(
            run_validation_context_df, 
            cooccurrence_pairs_df,
            max_k,
        )
        recommendations_df = popularity_recommendations_df.unionByName(cooccurrence_recommendations_df).persist(StorageLevel.MEMORY_AND_DISK)
        metrics_by_baseline = compute_metrics(spark, recommendations_df, run_validation_targets_df, K_VALUES)
        
        # Writes train/val/test tables to parquet in artifacts/baselines/sample
        track_popularity_df.write.mode(write_mode).parquet(str(popularity_path))
        cooccurrence_pairs_df.write.mode(write_mode).parquet(str(cooccurrence_pairs_path))
        recommendations_df.write.mode(write_mode).parquet(str(recommendations_path))
        
        # Validation summary
        write_metrics_json(metrics_by_baseline, metrics_path)

        sampled_train_playlists = run_train_df.select("pid").distinct().count()
        sampled_validation_playlists = run_validation_context_df.select("pid").distinct().count()
        summary = {
            "full_data_mode": bool(args.full_data),
            "input_path": str(Path(args.input)),
            "output_path": str(output_root),
            "k_values": K_VALUES,
            "sampled_train_playlists": sampled_train_playlists,
            "sampled_validation_playlists": sampled_validation_playlists,
            "track_popularity_path": str(popularity_path),
            "cooccurrence_pairs_path": str(cooccurrence_pairs_path),
            "recommendations_path": str(recommendations_path),
            "metrics_path": str(metrics_path),
            "metrics": metrics_by_baseline,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        spark.stop()  # Closes spark connection


if __name__ == "__main__":
    main()
