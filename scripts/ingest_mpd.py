# Import Libraries
import argparse
import json
import re
from pathlib import Path

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import ArrayType, BooleanType, LongType, StringType, StructField, StructType

# Create the Spark session that does the heavy dataframe work for this script.
def create_spark_session(driver_memory):
    return (
        SparkSession.builder.appName("spotify-mpd-ingestion") # name of Spark app (spotify-mpd-ingestion)
        .master("local[*]") # where the Spark Session will run (local)
        .config("spark.driver.memory", driver_memory) # amount of RAM to give process (4 - 8GB)
        .config("spark.sql.shuffle.partitions", "64") # when dealing with groupBy, joins, aggregations, we split into 64 partitions instead of the default 200 (easier on local device)
        .config("spark.sql.session.timeZone", "UTC") # sets timezone to UTC
        .getOrCreate() # if SparkSession already exists, return it, otherwise create a new one
    )

MPD_FILE_PATTERN = re.compile(r"mpd\.slice\.(\d+)-(\d+).*\.json$") # compiles regex to match MPD data filenames
SLICE_RANGE_PATTERN = r"mpd\.slice\.(\d+)-(\d+)" # raw regex string that matches the slice range, captures "slice_start" and "slice_end"

# Setting the schema of the JSON so it can be ingested into a Spark DataFrame
# Spark reads the JSON more reliably when we tell it the shape up front.
# JSON Track Schema (nested in Playlist)
TRACK_SCHEMA = StructType(
    [
        StructField("track_uri", StringType(), True),
        StructField("track_name", StringType(), True),
        StructField("artist_uri", StringType(), True),
        StructField("artist_name", StringType(), True),
        StructField("album_uri", StringType(), True),
        StructField("album_name", StringType(), True),
        StructField("duration_ms", LongType(), True),
        StructField("pos", LongType(), True),
    ]
)
# JSON Playlist Schema
PLAYLIST_SCHEMA = StructType(
    [
        StructField("pid", LongType(), True),
        StructField("name", StringType(), True),
        StructField("collaborative", BooleanType(), True),
        StructField("modified_at", LongType(), True),
        StructField("num_albums", LongType(), True),
        StructField("num_artists", LongType(), True),
        StructField("num_edits", LongType(), True),
        StructField("num_followers", LongType(), True),
        StructField("num_tracks", LongType(), True),
        StructField("duration_ms", LongType(), True),
        StructField("tracks", ArrayType(TRACK_SCHEMA), True),
    ]
)
# JSON Full Data Schema
MPD_SCHEMA = StructType(
    [
        StructField("info", StructType([]), True),
        StructField("playlists", ArrayType(PLAYLIST_SCHEMA), True),
    ]
)

# Finds all JSON files that match the MPD file pattern set above in an input directory
def find_mpd_files(input_dir):
    input_path = Path(input_dir) # creates a Path object to input_dir

    mpd_files = []
    for path in sorted(input_path.rglob("*.json")): # recursively checks all files in input_dir, sort for consistency
        if MPD_FILE_PATTERN.search(path.name): # if the file name matches the MPD file pattern set above
            mpd_files.append(str(path.resolve())) # appends the full input_path + path.name

    return mpd_files


# Clean text columns by trimming whitespace and turning empty strings into nulls.
# This function is used in all 3 build_() functions later
def clean_text(column): # input is a Spark Column object
    trimmed = F.trim(column) # removes whitespace
    return F.when(column.isNull() | (trimmed == ""), F.lit(None)).otherwise(trimmed) # If column is null, or trimmed value is empty, return null, otherwise return column
    # If value = "    Hello   ", trimmed = "Hello", if value = "    ", trimmed = ""

# URI's are "spotify:track:***", this function removes the "spotify:track:"
# This function is used in all 3 build_() functions later
def extract_id_from_uri(column): # input is a Spark Column object
    extracted = F.regexp_extract(column, r"([^:]+)$", 1) # match the chunk of characters that comes after the last colon, group 1 because there is only 1 group after the last colon
    return F.when(column.isNull() | (extracted == ""), F.lit(None)).otherwise(extracted) # If column is null, or extracted value is empty, return null, otherwise return column
    # If value = "spotify:track:15twB7zTglmu0Bg8gW4Mrm", extracted = "15twB7zTglmu0Bg8gW4Mrm", if value = "spotify:track:", extracted = ""


# Turns each nested playlist from JSON into 1 flat row for playlists.parquet
def build_playlists(playlist_rows):
    return (
        playlist_rows.select(
            F.col("playlist.pid").cast("long").alias("pid"),
            clean_text(F.col("playlist.name")).alias("name"),
            F.col("playlist.collaborative").cast("boolean").alias("collaborative"),
            F.col("playlist.modified_at").cast("long").alias("modified_at"),
            F.col("playlist.num_albums").cast("long").alias("num_albums"),
            F.col("playlist.num_artists").cast("long").alias("num_artists"),
            F.col("playlist.num_edits").cast("long").alias("num_edits"),
            F.col("playlist.num_followers").cast("long").alias("num_followers"),
            F.col("playlist.num_tracks").cast("long").alias("num_tracks"),
            F.col("playlist.duration_ms").cast("long").alias("duration_ms"),
            F.col("source_file"),
            F.regexp_extract("source_file", SLICE_RANGE_PATTERN, 1).cast("int").alias("slice_start"), # Create "slice_start" of the group 1 slice from source_file column 
            F.regexp_extract("source_file", SLICE_RANGE_PATTERN, 2).cast("int").alias("slice_end"), # Create "slice_end" of the group 2 slice from source_file column
            # Ex:
                # source_file = "mpd.slice.1000-1999.json"
                # slice_start = 1000
                # slice_end = 1999
        )
        .where(F.col("pid").isNotNull()) # remove rows where pid is null
    )


# Inside the flattened playlist rows, there is a nested tracks array. Notice how playlist.tracks isn't used in build_playlists
def build_tracks(playlist_rows):
    return (
        playlist_rows.select(F.explode_outer("playlist.tracks").alias("track")) # Explodes that tracks array into individual rows
        .where(F.col("track").isNotNull())
        .select(
            clean_text(F.col("track.track_uri")).alias("track_uri"),
            extract_id_from_uri(F.col("track.track_uri")).alias("track_id"),
            clean_text(F.col("track.track_name")).alias("track_name"),
            clean_text(F.col("track.artist_uri")).alias("artist_uri"),
            extract_id_from_uri(F.col("track.artist_uri")).alias("artist_id"),
            clean_text(F.col("track.artist_name")).alias("artist_name"),
            clean_text(F.col("track.album_uri")).alias("album_uri"),
            extract_id_from_uri(F.col("track.album_uri")).alias("album_id"),
            clean_text(F.col("track.album_name")).alias("album_name"),
            F.col("track.duration_ms").cast("long").alias("duration_ms"),
        )
        .where(F.col("track_uri").isNotNull())
        .groupBy("track_uri")
        # Removes duplicate track_uri's and takes the first appearance of the track
        .agg(
            F.first("track_id", ignorenulls=True).alias("track_id"),
            F.first("track_name", ignorenulls=True).alias("track_name"),
            F.first("artist_uri", ignorenulls=True).alias("artist_uri"),
            F.first("artist_id", ignorenulls=True).alias("artist_id"),
            F.first("artist_name", ignorenulls=True).alias("artist_name"),
            F.first("album_uri", ignorenulls=True).alias("album_uri"),
            F.first("album_id", ignorenulls=True).alias("album_id"),
            F.first("album_name", ignorenulls=True).alias("album_name"),
            F.first("duration_ms", ignorenulls=True).alias("duration_ms"),
        )
    )


# Build the playlist-track interaction table, keeping every track occurrence and its playlist order.
def build_playlist_tracks(playlist_rows):
    return (
        playlist_rows.select(
            F.col("playlist.pid").cast("long").alias("pid"),
            F.explode_outer("playlist.tracks").alias("track"), # Similarly explodes the tracks array into individual rows, but doesn't remove the duplicates this time
        )
        .where(F.col("pid").isNotNull() & F.col("track").isNotNull())
        .select(
            F.col("pid"),
            F.col("track.pos").cast("int").alias("pos"),
            clean_text(F.col("track.track_uri")).alias("track_uri"),
            extract_id_from_uri(F.col("track.track_uri")).alias("track_id"),
            clean_text(F.col("track.artist_uri")).alias("artist_uri"),
            extract_id_from_uri(F.col("track.artist_uri")).alias("artist_id"),
            clean_text(F.col("track.album_uri")).alias("album_uri"),
            extract_id_from_uri(F.col("track.album_uri")).alias("album_id"),
            F.col("track.duration_ms").cast("long").alias("duration_ms"),
        )
        .where(F.col("track_uri").isNotNull() & F.col("pos").isNotNull())
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest Spotify MPD JSON into three Parquet tables.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--driver-memory", default="4g")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


# Creating a pipeline which finds files, reads JSON, flattens tables, and writes to Parquet.
def main():
    args = parse_args()
    input_files = find_mpd_files(args.input) # runs find_mpd_files() with the input_dir of data/bronze
    
    output_root = Path(args.output) # creates a Path object to data/silver where the Parquet tables will be stored
    output_root.mkdir(parents=True, exist_ok=True) # creates the output directory data/silver if it doesn't exist already

    # Creates path for each parquet table write
    playlists_path = output_root / "playlists.parquet"
    tracks_path = output_root / "tracks.parquet"
    playlist_tracks_path = output_root / "playlist_tracks.parquet" 
    write_mode = "overwrite" if args.overwrite else "errorifexists" # if we use --overwrite in CLI, then MPD file ingestion overwrites all old files in data/silver/{path}

    # Creates SparkSession
    spark = create_spark_session(args.driver_memory) # Creates SparkSession with given a name, where to run it, and RAM to allocate it

    try:
        # uses our SparkSession to read a the input_files JSON into the MPD_SCHEMA schema. 
        # ("multiLine", True) is important because our MPD files span across multiple lines even though it looks pretty-printed
        raw_df = spark.read.option("multiLine", True).schema(MPD_SCHEMA).json(input_files)

        # Creates a new dataframe with just source_file and playlist
        playlist_rows = (
            raw_df.select(
                F.input_file_name().alias("source_file"), # creates a column that tracks which slice file each row came from
                F.explode_outer("playlists").alias("playlist"), # `explode_outer()` takes the array of playlists and creates a row for each element in the array
            )
            .where(F.col("playlist").isNotNull()) # essentially removes null playlists
        )

        # Cleans and builds each table before writing to parquet
        playlists_df = build_playlists(playlist_rows)
        tracks_df = build_tracks(playlist_rows)
        playlist_tracks_df = build_playlist_tracks(playlist_rows)

        # Writes each Spark DataFrame into Parquet tables, write_mode determines overriding
        playlists_df.write.mode(write_mode).parquet(str(playlists_path))
        tracks_df.write.mode(write_mode).parquet(str(tracks_path))
        playlist_tracks_df.write.mode(write_mode).parquet(str(playlist_tracks_path))

        # Validation summary to ensure data and paths are correct
        summary = {
            "input_file_count": len(input_files),
            "playlists_path": str(playlists_path),
            "tracks_path": str(tracks_path),
            "playlist_tracks_path": str(playlist_tracks_path),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        spark.stop() # Closes the spark connection

if __name__ == "__main__":
    main()
