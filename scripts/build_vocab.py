# Import Libraries
import json
import sys
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from twotower.vocab import Vocabulary

INPUT_PATH = 'data/gold/train_playlist_tracks.parquet'
OUTPUT_DIR = 'artifacts/vocab'

ENTITIES = {
    'track_id': 'track',
    'artist_id': 'artist',
    'album_id': 'album'
}

def create_spark_session():
    spark = (
        SparkSession.builder.appName('spotify-mpd-vocab')
        .master('local[*]')
        .config('spark.driver.bindAddress', '127.0.0.1')
        .config('spark.driver.memory', '4g')
        .config('spark.sql.shuffle.partitions', '200')
        .config('spark.sql.session.timeZone', 'UTC')
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel('ERROR')

    return spark

# Builds an ID to index mapping for every distinct non-null entity in the training set
def build_entity_vocabulary(train_df, column):
    rows = (train_df
        .select(F.col(column).alias('id'))
        .where(F.col('id').isNotNull())
        .distinct()
        .collect()
    )

    return Vocabulary.build([row['id'] for row in rows])

def main():
    output_root = Path(OUTPUT_DIR)
    output_root.mkdir(parents=True, exist_ok=True)

    spark = create_spark_session()

    try:
        train_df = spark.read.parquet(INPUT_PATH)

        vocab_sizes = {}
        for column, name in ENTITIES.items():
            vocab = build_entity_vocabulary(train_df, column)
            vocab.save(output_root / f'{name}_vocab.json')
            vocab_sizes[name] = vocab.size 
            print(f'Built {name} vocab with {vocab.size} entries, saved to {output_root / f"{name}_vocab.json"}')

        (output_root / 'vocab_metadata.json').write_text(json.dumps({'vocab_sizes': vocab_sizes}, indent=2))
        
    finally:
        spark.stop()

if __name__ == '__main__':
    main()
