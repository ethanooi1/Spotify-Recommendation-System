# Import Libraries
import json
import math
from pathlib import Path

from pyspark import StorageLevel
from pyspark.sql import SparkSession, Window, functions as F

from build_splits import MAX_HIDDEN

INPUT_DIR = 'data/gold'
OUTPUT_DIR = 'artifacts/baselines'

SEED = 42
K_VALUES = [10, 50, 100]
SPLITS = ['validation', 'test']
BASELINES = ['popularity', 'cooccurrence']
MIN_PAIR_SUPPORT = 3
MIN_CANDIDATE_POPULARITY = 5
COOCCURRENCE_SAMPLE_SIZE = 50000 # Cooccurrence sample size is capped for memory reasons, would need 6.5B pairs which is too much to compute
MAX_PAIRS_PER_TRACK = 200 # capped for memory reasons

def create_spark_session():
    spark = (
        SparkSession.builder.appName('spotify-mpd-baselines')
        .master('local[8]')
        .config('spark.driver.bindAddress', '127.0.0.1')
        .config('spark.driver.memory', '12g')
        .config('spark.sql.shuffle.partitions', '800')
        .config('spark.sql.session.timeZone', 'UTC')
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel('ERROR')
    
    return spark

# Training data, val for context+target, and test for context+target
def read_gold_tables(spark):
    input_root = Path(INPUT_DIR)
    train_df = spark.read.parquet(str(input_root / 'train_playlist_tracks.parquet'))

    eval_tables = {}
    for split in SPLITS:
        context_df = spark.read.parquet(str(input_root / f'{split}_context.parquet'))
        targets_df = spark.read.parquet(str(input_root / f'{split}_targets.parquet'))
        eval_tables[split] = (context_df, targets_df)

    return train_df, eval_tables

# Samples playlists determinisitcally using hash splitting for cooccurrence split
def select_sample_pids(train_df, sample_size):
    return (train_df
        .select('pid').distinct()
        .withColumn('sample_score',
            F.pmod(
                F.xxhash64(F.lit(str(SEED)),F.lit('train'),F.col('pid').cast('string')), 
                F.lit(1000000)))
        .orderBy(F.col('sample_score'), F.col('pid'))
        .limit(sample_size)
        .select('pid')
    )

# Frequency of tracks in the train split ranked desc
def build_track_popularity(train_df):
    popularity_window = Window.orderBy(F.desc('train_frequency'), F.asc('track_id'))

    return (train_df
        .groupBy('track_id')
        .agg(F.count(F.lit(1)).cast('long').alias('train_frequency'))
        .withColumn('popularity_rank', F.row_number().over(popularity_window))
        .select('track_id', 'train_frequency', 'popularity_rank')
    )

# Popularity Baseline: recommend the most frequent tracks a playlist doesn't already have
def build_popularity_recommendations(context_df, track_popularity_df, max_k):
    playlists_df = context_df.select('pid').distinct()
    context_tracks_df = context_df.select('pid', 'track_id').distinct()
    max_unique_context_tracks = (context_tracks_df
        .groupBy('pid')
        .agg(F.count(F.lit(1)).alias('unique_context_tracks'))
        .agg(F.max('unique_context_tracks').alias('max_unique_context_tracks'))
        .first()['max_unique_context_tracks']
    )
    max_unique_context_tracks = 0 if max_unique_context_tracks is None else int(max_unique_context_tracks)

    candidate_pool_df = (track_popularity_df
        .orderBy(F.asc('popularity_rank'))
        .limit(max_k + max_unique_context_tracks + 50)
        .select('track_id', 'train_frequency', 'popularity_rank')
    )

    popularity_rank_window = (Window
        .partitionBy('pid')
        .orderBy(F.desc('train_frequency'), F.asc('popularity_rank'), F.asc('track_id'))
    )

    return (playlists_df
        .crossJoin(F.broadcast(candidate_pool_df))
        .join(context_tracks_df, on=['pid', 'track_id'], how='left_anti')
        .withColumn('rank', F.row_number().over(popularity_rank_window))
        .where(F.col('rank') <= F.lit(max_k))
        .withColumn('baseline_name', F.lit('popularity'))
        .withColumn('score', F.col('train_frequency').cast('double'))
        .select('baseline_name', 'pid', 'rank', 'track_id', 'score')
    )

# Builds track-track pairs across sampled train playlists. Capped at 200 pairs per context track
def build_cooccurrence_pairs(train_df, track_popularity_df):
    train_source_df = train_df.select('pid', 'track_id').where(F.col('track_id').isNotNull())
    recommendable_tracks_df = (track_popularity_df
        .where(F.col('train_frequency') >= F.lit(MIN_CANDIDATE_POPULARITY))
        .select(
            F.col('track_id').alias('candidate_track_id'),
            F.col('train_frequency').alias('candidate_popularity'))
    )

    context_rows_df = (train_source_df
        .select(
            'pid', 
            F.col('track_id').alias('context_track_id'))
    )
    
    candidate_rows_df = (train_source_df
        .join(recommendable_tracks_df, F.col('track_id') == F.col('candidate_track_id'), how='inner')
        .select('pid', 'candidate_track_id')
    )

    pairs_df = (context_rows_df
        .join(candidate_rows_df, on='pid', how='inner')
        .where(F.col('context_track_id') != F.col('candidate_track_id'))
        .groupBy('context_track_id', 'candidate_track_id')
        .agg(F.count(F.lit(1)).cast('long').alias('pair_support'))
        .where(F.col('pair_support') >= F.lit(MIN_PAIR_SUPPORT))
    )
    
    # Popular tracks will explode the join, so limit to the 200 strongest pairs per context track.
    top_pairs_window = (Window
        .partitionBy('context_track_id')
        .orderBy(F.desc('pair_support'), F.asc('candidate_track_id'))
    )

    return (pairs_df
        .withColumn('pair_rank', F.row_number().over(top_pairs_window))
        .where(F.col('pair_rank') <= F.lit(MAX_PAIRS_PER_TRACK))
        .join(recommendable_tracks_df, on='candidate_track_id', how='inner')
        .select('context_track_id', 'candidate_track_id', 'pair_support', 'candidate_popularity')
    )

# Cooccurrence Baseline: Every track-track pair counts from the sampled train playlists. Capped at 200 pairs for memory reasons.
def build_cooccurrence_recommendations(context_df, cooccurrence_pairs_df, max_k):
    context_tracks_df = context_df.select('pid', 'track_id').where(F.col('track_id').isNotNull())
    context_filter_df = context_tracks_df.select('pid', 'track_id').distinct()

    cooccurrence_rank_window = (Window
        .partitionBy('pid')
        .orderBy(F.desc('score'), F.desc('candidate_popularity'), F.asc('track_id'))
    )

    return (context_tracks_df
        .select('pid', F.col('track_id').alias('context_track_id'))
        .join(cooccurrence_pairs_df, on='context_track_id', how='inner')
        .groupBy('pid', F.col('candidate_track_id').alias('track_id'))
        .agg(
            F.sum('pair_support').cast('double').alias('score'),
            F.first('candidate_popularity', ignorenulls=True).alias('candidate_popularity'))
        .join(context_filter_df, on=['pid', 'track_id'], how='left_anti')
        .withColumn('rank', F.row_number().over(cooccurrence_rank_window))
        .where(F.col('rank') <= F.lit(max_k))
        .withColumn('baseline_name', F.lit('cooccurrence'))
        .select('baseline_name', 'pid', 'rank', 'track_id', 'score')
    )

# Ideal DCG for every target, used as denom for NDCG metric
def build_idcg_lookup(spark, k_value):
    rows = []
    for target_count in range(1, MAX_HIDDEN + 1):
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(target_count, k_value) + 1))
        rows.append((target_count, float(idcg)))

    return spark.createDataFrame(rows, ['target_count', 'idcg'])

# Evaluates on Recall@K, Precison@K, and NDCG@K for each baseline. Evaluate.py uses these same definitions
def compute_metrics(spark, recommendations_df, targets_df):
    target_relevance_df = targets_df.select('pid', 'track_id').distinct().persist(StorageLevel.MEMORY_AND_DISK)
    target_counts_df = target_relevance_df.groupBy('pid').agg(F.count(F.lit(1)).cast('int').alias('target_count')).persist(StorageLevel.MEMORY_AND_DISK)
    baseline_names_df = spark.createDataFrame([(name,) for name in BASELINES], ['baseline_name'])

    # Every baseline crossed with every playlist
    evaluation_grid_df = baseline_names_df.crossJoin(target_counts_df.select('pid')).persist(StorageLevel.MEMORY_AND_DISK)

    metrics_by_baseline = {name: {} for name in BASELINES}

    for k_value in K_VALUES:
        idcg_lookup_df = build_idcg_lookup(spark, k_value)
        hits_df = (recommendations_df
            .where(F.col('rank') <= F.lit(k_value))
            .join(target_relevance_df.withColumn('is_relevant', F.lit(1.0)), on=['pid', 'track_id'], how='left')
            .withColumn('is_hit', F.when(F.col('is_relevant').isNotNull(), F.lit(1.0)).otherwise(F.lit(0.0)))
            .withColumn('dcg_contribution', F.col('is_hit') / F.log2(F.col('rank') + F.lit(1.0)))
            .groupBy('baseline_name', 'pid')
            .agg(
                F.sum('is_hit').alias('hit_count'),
                F.sum('dcg_contribution').alias('dcg'))
        )

        metric_rows = (evaluation_grid_df
            .join(target_counts_df, on='pid', how='inner')
            .join(idcg_lookup_df, on='target_count', how='left')
            .join(hits_df, on=['baseline_name', 'pid'], how='left')
            .fillna({'hit_count': 0.0, 'dcg': 0.0})
            .withColumn('recall', F.col('hit_count') / F.col('target_count'))
            .withColumn('precision', F.col('hit_count') / F.lit(float(k_value)))
            .withColumn('ndcg',
                F.when(F.col('idcg') > F.lit(0.0), F.col('dcg') / F.col('idcg')).otherwise(F.lit(0.0)))
            .groupBy('baseline_name')
            .agg(
                F.avg('recall').alias('recall'),
                F.avg('precision').alias('precision'),
                F.avg('ndcg').alias('ndcg'))
            .collect()
        )

        for row in metric_rows:
            metrics_by_baseline[row['baseline_name']][f'Recall@{k_value}'] = float(row['recall'])
            metrics_by_baseline[row['baseline_name']][f'Precision@{k_value}'] = float(row['precision'])
            metrics_by_baseline[row['baseline_name']][f'NDCG@{k_value}'] = float(row['ndcg'])

    target_relevance_df.unpersist()
    target_counts_df.unpersist()
    evaluation_grid_df.unpersist()

    return metrics_by_baseline

# Scores a split (val/test) on both baselines (popularity/cooccurrence) and writes its metrics 
def score_split(spark, split, context_df, targets_df, track_popularity_df, cooccurrence_pairs_df, output_root, max_k):
    context_df = context_df.where(F.col('track_id').isNotNull()).select('pid', 'track_id')
    scored_playlists = context_df.select('pid').distinct().count()

    popularity_recommendations_df = build_popularity_recommendations(context_df, track_popularity_df, max_k)
    cooccurrence_recommendations_df = build_cooccurrence_recommendations(context_df, cooccurrence_pairs_df, max_k)

    # Land the recommendations on disk and read them back. Holding 10M rows in memory while the metric joins run over them is what blows the heap.
    recommendations_path = output_root / f'{split}_recommendations.parquet'
    recommendations_df = popularity_recommendations_df.unionByName(cooccurrence_recommendations_df)
    recommendations_df.write.mode('overwrite').parquet(str(recommendations_path))
    recommendations_df = spark.read.parquet(str(recommendations_path))

    metrics_by_baseline = compute_metrics(spark, recommendations_df, targets_df)

    (output_root / f'{split}_metrics.json').write_text(json.dumps({'split': split, 'baselines': metrics_by_baseline}, indent=2, sort_keys=True))

    return metrics_by_baseline, scored_playlists

def main():
    max_k = max(K_VALUES)

    output_root = Path(OUTPUT_DIR)
    output_root.mkdir(parents=True, exist_ok=True)

    spark = create_spark_session()

    try:
        train_df, eval_tables = read_gold_tables(spark)
        train_df = (train_df
            .where(F.col('track_id').isNotNull())
            .select('pid', 'track_id')
            .persist(StorageLevel.MEMORY_AND_DISK)
        )

        # popularity on 900k train playlists
        popularity_path = output_root / 'track_popularity.parquet'
        build_track_popularity(train_df).write.mode('overwrite').parquet(str(popularity_path))
        track_popularity_df = spark.read.parquet(str(popularity_path))

        # cooccurrence on 50k sampled train playlists
        cooccurrence_train_df = train_df.join(select_sample_pids(train_df, COOCCURRENCE_SAMPLE_SIZE), on='pid', how='inner')
        pairs_path = output_root / 'cooccurrence_pairs.parquet'
        build_cooccurrence_pairs(cooccurrence_train_df, track_popularity_df).write.mode('overwrite').parquet(str(pairs_path))
        cooccurrence_pairs_df = spark.read.parquet(str(pairs_path))

        summary = {
            'output_path': str(output_root),
            'k_values': K_VALUES,
            'popularity_train_playlists': train_df.select('pid').distinct().count(),
            'cooccurrence_train_playlists': COOCCURRENCE_SAMPLE_SIZE,
            'splits': {}
        }
        
        train_df.unpersist()

        for split in SPLITS:
            context_df, targets_df = eval_tables[split]
            metrics_by_baseline, scored_playlists = score_split(spark, split, context_df, targets_df, track_popularity_df, cooccurrence_pairs_df, output_root, max_k)
            summary['splits'][split] = {'scored_playlists': scored_playlists, 'metrics': metrics_by_baseline}

        print(json.dumps(summary, indent=2))
        
    finally:
        spark.stop()

if __name__ == '__main__':
    main()
