# Import Libraries
import json
from pathlib import Path

from pyspark.sql import SparkSession, Window, functions as F

INPUT_DIR = 'data/silver'
OUTPUT_DIR = 'data/gold'

SEED = 42
TRAIN_RATIO = 0.90
VALIDATION_RATIO = 0.05
MIN_PLAYLIST_LENGTH = 5
MASK_FRACTION = 0.20 # hide 20% of each eval playlist
MAX_HIDDEN = 10 # hide up to 10 songs

# Spark session that does the heavy dataframe work.
def create_spark_session():
    spark = (
        SparkSession.builder.appName('spotify-mpd-splits')
        .master('local[*]')
        .config('spark.driver.bindAddress', '127.0.0.1')
        .config('spark.driver.memory', '4g')
        .config('spark.sql.shuffle.partitions', '200')
        .config('spark.sql.session.timeZone', 'UTC')
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel('ERROR') # the aggregation buffers warn once per batch otherwise

    return spark

def read_silver_tables(spark):
    input_root = Path(INPUT_DIR)
    playlists_df = spark.read.parquet(str(input_root / 'playlists.parquet'))
    playlist_tracks_df = spark.read.parquet(str(input_root / 'playlist_tracks.parquet'))

    return playlists_df, playlist_tracks_df

# Count number of tracks in each playlist, fallback to num_tracks from MPD
def build_playlist_lengths(playlists_df, playlist_tracks_df):
    track_counts = (playlist_tracks_df
        .groupBy('pid')
        .agg(F.count(F.lit(1)).cast('int').alias('playlist_length_from_tracks'))
    )

    return (playlists_df
        .select(
            F.col('pid').cast('long').alias('pid'),
            F.col('num_tracks').cast('int').alias('num_tracks'))
        .join(track_counts, on='pid', how='left')
        .select(
            F.col('pid'),
            F.coalesce(F.col('playlist_length_from_tracks'), F.col('num_tracks'), F.lit(0)).cast('int').alias('playlist_length'))
    )

# Use hash splitting for train/val/test splits so that it's reproducible when new playlists are added
def assign_playlist_splits(playlist_lengths_df):
    # Each playlist has the same split_score
    split_score = (F.pmod(
        F.xxhash64(F.lit(str(SEED)), F.col('pid').cast('string')), 
        F.lit(1000000)) / F.lit(1000000.0))

    return (playlist_lengths_df
        .withColumn('split_score', split_score)
        .withColumn('split',
            F.when(F.col('playlist_length') < F.lit(MIN_PLAYLIST_LENGTH), F.lit('train'))
            .when(F.col('split_score') < F.lit(TRAIN_RATIO), F.lit('train'))
            .when(F.col('split_score') < F.lit(TRAIN_RATIO + VALIDATION_RATIO), F.lit('validation'))
            .otherwise(F.lit('test')))
        .withColumn('eligible_for_eval', F.col('split').isin('validation', 'test'))
        .select('pid', 'split', 'playlist_length', 'eligible_for_eval')
    )

# Build training set if split = 'train'
def build_train_tracks(playlist_tracks_df, playlist_splits_df):
    train_pids = playlist_splits_df.where(F.col('split') == 'train').select('pid')

    return (playlist_tracks_df
        .join(train_pids, on='pid', how='inner')
        .select('pid', 'pos', 'track_id', 'artist_id', 'album_id', 'duration_ms')
    )

# Each eval playlist hides 20% of its tracks based off hash splitting, capped at 10 tracks
def build_masked_split(playlist_tracks_df, playlist_splits_df, split_name):
    eval_playlists = (playlist_splits_df
        .where(F.col('split') == split_name)
        .select('pid', 'playlist_length')
        .withColumn('hidden_count',
            F.least(F.lit(MAX_HIDDEN), F.greatest(F.lit(1), F.ceil(F.col('playlist_length') * F.lit(MASK_FRACTION)).cast('int'))))
    )
    # Within each eval playlist, each track gets a random score based on the pid and its position.
    masked_rows = (playlist_tracks_df
        .join(eval_playlists, on='pid', how='inner')
        .withColumn('mask_score',
            F.pmod(
                F.xxhash64(F.lit(str(SEED)), F.lit(split_name), F.col('pid').cast('string'), F.col('pos').cast('string')), 
                F.lit(1000000.0)))
    )

    # Order by mask_score and take the "hidden_count" # of tracks as hidden tracks. The rest are used as context to predict the hidden tracks.
    # A hidden track is a "positive" recommendation for the context tracks to recommend.
    rank_window = (Window
        .partitionBy('pid')
        .orderBy(F.col('mask_score'), F.col('pos'))
    )
    ranked_rows = masked_rows.withColumn('mask_rank', F.row_number().over(rank_window))

    columns = ['pid', 'pos', 'track_id', 'artist_id', 'album_id', 'duration_ms']
    context_df = ranked_rows.where(F.col('mask_rank') > F.col('hidden_count')).select(*columns)
    targets_df = ranked_rows.where(F.col('mask_rank') <= F.col('hidden_count')).select(*columns)

    return context_df, targets_df

def main():
    output_root = Path(OUTPUT_DIR)
    output_root.mkdir(parents=True, exist_ok=True)
    
    playlist_splits_path = output_root / 'playlist_splits.parquet'
    train_playlist_tracks_path = output_root / 'train_playlist_tracks.parquet'
    validation_context_path = output_root / 'validation_context.parquet'
    validation_targets_path = output_root / 'validation_targets.parquet'
    test_context_path = output_root / 'test_context.parquet'
    test_targets_path = output_root / 'test_targets.parquet'

    spark = create_spark_session()

    try:
        playlists_df, playlist_tracks_df = read_silver_tables(spark)
        playlist_lengths_df = build_playlist_lengths(playlists_df, playlist_tracks_df)
        playlist_splits_df = assign_playlist_splits(playlist_lengths_df)

        train_playlist_tracks_df = build_train_tracks(playlist_tracks_df, playlist_splits_df)
        validation_context_df, validation_targets_df = build_masked_split(playlist_tracks_df, playlist_splits_df, 'validation')
        test_context_df, test_targets_df = build_masked_split(playlist_tracks_df, playlist_splits_df, 'test')

        playlist_splits_df.write.mode('overwrite').parquet(str(playlist_splits_path))
        train_playlist_tracks_df.write.mode('overwrite').parquet(str(train_playlist_tracks_path))
        validation_context_df.write.mode('overwrite').parquet(str(validation_context_path))
        validation_targets_df.write.mode('overwrite').parquet(str(validation_targets_path))
        test_context_df.write.mode('overwrite').parquet(str(test_context_path))
        test_targets_df.write.mode('overwrite').parquet(str(test_targets_path))

        written_splits_df = spark.read.parquet(str(playlist_splits_path))
        split_counts = {row['split']: row['count'] for row in written_splits_df.groupBy('split').count().collect()}

        summary = {
            'output_path': str(output_root),
            'eligible_eval_playlists': split_counts.get('validation') + split_counts.get('test'),
            'split_counts': split_counts
        }
        print(json.dumps(summary, indent=2))
        
    finally:
        spark.stop()

if __name__ == '__main__':
    main()
