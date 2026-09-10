# Import Libraries
import json
import re
from pathlib import Path

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import ArrayType, BooleanType, LongType, StringType, StructField, StructType

INPUT_DIR = 'data/bronze'
OUTPUT_DIR = 'data/silver'

MPD_FILE_PATTERN = re.compile(r'mpd\.slice\.(\d+)-(\d+).*\.json$')
SLICE_RANGE_PATTERN = r'mpd\.slice\.(\d+)-(\d+)' # captures the start and end of the slice from the file name

def create_spark_session():
    spark = (
        SparkSession.builder.appName('spotify-mpd-ingestion')
        .master('local[*]')
        .config('spark.driver.bindAddress', '127.0.0.1')
        .config('spark.driver.memory', '4g')
        .config('spark.sql.shuffle.partitions', '200')
        .config('spark.sql.session.timeZone', 'UTC')
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel('ERROR')

    return spark

# Mirror the MPD schema for ingestion 
TRACK_SCHEMA = StructType(
    [
        StructField('track_uri', StringType(), True),
        StructField('track_name', StringType(), True),
        StructField('artist_uri', StringType(), True),
        StructField('artist_name', StringType(), True),
        StructField('album_uri', StringType(), True),
        StructField('album_name', StringType(), True),
        StructField('duration_ms', LongType(), True),
        StructField('pos', LongType(), True)
    ]
)

PLAYLIST_SCHEMA = StructType(
    [
        StructField('pid', LongType(), True),
        StructField('name', StringType(), True),
        StructField('collaborative', BooleanType(), True),
        StructField('modified_at', LongType(), True),
        StructField('num_albums', LongType(), True),
        StructField('num_artists', LongType(), True),
        StructField('num_edits', LongType(), True),
        StructField('num_followers', LongType(), True),
        StructField('num_tracks', LongType(), True),
        StructField('duration_ms', LongType(), True),
        StructField('tracks', ArrayType(TRACK_SCHEMA), True)
    ]
)

MPD_SCHEMA = StructType(
    [
        StructField('info', StructType([]), True),
        StructField('playlists', ArrayType(PLAYLIST_SCHEMA), True)
    ]
)

# Every JSON file under the input directory whose name matches an MPD slice, sorted for a consistent read order
def find_mpd_files(input_dir):
    mpd_files = []
    for path in sorted(Path(input_dir).rglob('*.json')):
        if MPD_FILE_PATTERN.search(path.name):
            mpd_files.append(str(path.resolve()))

    return mpd_files

# Trims whitespace and turns empty strings into nulls
def clean_text(column):
    trimmed = F.trim(column)

    return F.when(column.isNull() | (trimmed == ''), F.lit(None)).otherwise(trimmed)

# URIs look like "spotify:track:15twB7zTglmu0Bg8gW4Mrm", extract to -> "15twB7zTglmu0Bg8gW4Mrm"
def extract_id_from_uri(column):
    extracted = F.regexp_extract(column, r'([^:]+)$', 1)

    return F.when(column.isNull() | (extracted == ''), F.lit(None)).otherwise(extracted)

# Every playlist is a row
def build_playlists(playlist_rows):
    return (playlist_rows
        .select(
            F.col('playlist.pid').cast('long').alias('pid'),
            clean_text(F.col('playlist.name')).alias('name'),
            F.col('playlist.collaborative').cast('boolean').alias('collaborative'),
            F.col('playlist.modified_at').cast('long').alias('modified_at'),
            F.col('playlist.num_albums').cast('long').alias('num_albums'),
            F.col('playlist.num_artists').cast('long').alias('num_artists'),
            F.col('playlist.num_edits').cast('long').alias('num_edits'),
            F.col('playlist.num_followers').cast('long').alias('num_followers'),
            F.col('playlist.num_tracks').cast('long').alias('num_tracks'),
            F.col('playlist.duration_ms').cast('long').alias('duration_ms'),
            F.col('source_file'),
            # 'mpd.slice.1000-1999.json' gives slice_start 1000 and slice_end 1999
            F.regexp_extract('source_file', SLICE_RANGE_PATTERN, 1).cast('int').alias('slice_start'),
            F.regexp_extract('source_file', SLICE_RANGE_PATTERN, 2).cast('int').alias('slice_end'))
        .where(F.col('pid').isNotNull())
    )
# Every unique track URI is a row
def build_tracks(playlist_rows):
    return (playlist_rows
        .select(F.explode_outer('playlist.tracks').alias('track'))
        .where(F.col('track').isNotNull())
        .select(
            clean_text(F.col('track.track_uri')).alias('track_uri'),
            extract_id_from_uri(F.col('track.track_uri')).alias('track_id'),
            clean_text(F.col('track.track_name')).alias('track_name'),
            clean_text(F.col('track.artist_uri')).alias('artist_uri'),
            extract_id_from_uri(F.col('track.artist_uri')).alias('artist_id'),
            clean_text(F.col('track.artist_name')).alias('artist_name'),
            clean_text(F.col('track.album_uri')).alias('album_uri'),
            extract_id_from_uri(F.col('track.album_uri')).alias('album_id'),
            clean_text(F.col('track.album_name')).alias('album_name'),
            F.col('track.duration_ms').cast('long').alias('duration_ms'))
        .where(F.col('track_uri').isNotNull())
        .groupBy('track_uri')
        .agg(
            F.first('track_id', ignorenulls=True).alias('track_id'),
            F.first('track_name', ignorenulls=True).alias('track_name'),
            F.first('artist_uri', ignorenulls=True).alias('artist_uri'),
            F.first('artist_id', ignorenulls=True).alias('artist_id'),
            F.first('artist_name', ignorenulls=True).alias('artist_name'),
            F.first('album_uri', ignorenulls=True).alias('album_uri'),
            F.first('album_id', ignorenulls=True).alias('album_id'),
            F.first('album_name', ignorenulls=True).alias('album_name'),
            F.first('duration_ms', ignorenulls=True).alias('duration_ms'))
    )
# Every playlist-track pair is a row
def build_playlist_tracks(playlist_rows):
    return (playlist_rows
        .select(
            F.col('playlist.pid').cast('long').alias('pid'),
            F.explode_outer('playlist.tracks').alias('track'))
        .where(F.col('pid').isNotNull() & F.col('track').isNotNull())
        .select(
            F.col('pid'),
            F.col('track.pos').cast('int').alias('pos'),
            clean_text(F.col('track.track_uri')).alias('track_uri'),
            extract_id_from_uri(F.col('track.track_uri')).alias('track_id'),
            clean_text(F.col('track.artist_uri')).alias('artist_uri'),
            extract_id_from_uri(F.col('track.artist_uri')).alias('artist_id'),
            clean_text(F.col('track.album_uri')).alias('album_uri'),
            extract_id_from_uri(F.col('track.album_uri')).alias('album_id'),
            F.col('track.duration_ms').cast('long').alias('duration_ms'))
        .where(F.col('track_uri').isNotNull() & F.col('pos').isNotNull())
    )

def main():
    input_files = find_mpd_files(INPUT_DIR)

    output_root = Path(OUTPUT_DIR)
    output_root.mkdir(parents=True, exist_ok=True)
    
    playlists_path = output_root / 'playlists.parquet'
    tracks_path = output_root / 'tracks.parquet'
    playlist_tracks_path = output_root / 'playlist_tracks.parquet'

    spark = create_spark_session()

    try:
        # ("multiLine", True) is important because our MPD files span across multiple lines even though it looks pretty-printed
        raw_df = spark.read.option('multiLine', True).schema(MPD_SCHEMA).json(input_files)

        playlist_rows = (raw_df
            .select(
                F.input_file_name().alias('source_file'),
                F.explode_outer('playlists').alias('playlist'))
            .where(F.col('playlist').isNotNull())
        )

        build_playlists(playlist_rows).write.mode('overwrite').parquet(str(playlists_path))
        build_tracks(playlist_rows).write.mode('overwrite').parquet(str(tracks_path))
        build_playlist_tracks(playlist_rows).write.mode('overwrite').parquet(str(playlist_tracks_path))

        summary = {
            'input_file_count': len(input_files),
            'tracks_path': str(tracks_path),
            'playlists_path': str(playlists_path),
            'playlist_tracks_path': str(playlist_tracks_path)
        }
        
        print(json.dumps(summary, indent=2))
        
    finally:
        spark.stop()

if __name__ == '__main__':
    main()
