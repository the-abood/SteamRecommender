# ============================================================
# FILE: src/preprocessing/prepare_data.py
# PURPOSE: Reusable helpers for loading, cleaning, encoding,
#          and splitting the Steam dataset for ALS training.
# ============================================================

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType, IntegerType
)
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer


# ── Schema ────────────────────────────────────────────────────────────────────

STEAM_SCHEMA = StructType([
    StructField("user_id",   StringType(), True),
    StructField("game_name", StringType(), True),
    StructField("behaviour", StringType(), True),
    StructField("value",     FloatType(),  True),
])


# ── Load ──────────────────────────────────────────────────────────────────────

def load_steam(spark: SparkSession, file_path: str) -> DataFrame:
    """
    Load steam-200k.csv into a Spark DataFrame with an explicit schema.

    The CSV has no header row. Column names are assigned via the schema.

    Parameters
    ----------
    spark     : active SparkSession
    file_path : DBFS or local path to steam-200k.csv

    Returns
    -------
    Raw DataFrame with columns: user_id, game_name, behaviour, value
    """
    return (
        spark.read
        .option("header", "false")
        .schema(STEAM_SCHEMA)
        .csv(file_path)
    )


# ── Filter ────────────────────────────────────────────────────────────────────

def filter_play(df: DataFrame) -> DataFrame:
    """
    Return only 'play' behaviour rows.

    Design decision
    ---------------
    Play hours are a continuous, graded implicit feedback signal — a user
    who logged 300 hours values a game far more than one who logged 1 hour.
    Purchase rows (value always = 1) carry no preference gradient and would
    collapse all bought-but-unplayed titles to the same weight.
    Using play hours with implicitPrefs=True in ALS gives a richer signal.
    """
    return df.filter(F.col("behaviour") == "play") \
             .select("user_id", "game_name", "value")


def filter_purchase(df: DataFrame) -> DataFrame:
    """
    Return only 'purchase' behaviour rows (value always = 1).

    Use this if you want to train a binary implicit feedback model
    (has the user bought this game: yes/no).
    """
    return df.filter(F.col("behaviour") == "purchase") \
             .select("user_id", "game_name", "value")


# ── Transform ─────────────────────────────────────────────────────────────────

def log_transform(df: DataFrame, col: str = "value") -> DataFrame:
    """
    Apply log1p transformation to a numeric column to reduce right skew.

    Playtime is heavily right-skewed (most users: a few hours; a handful:
    thousands of hours). Log-transforming produces a more symmetric
    distribution and improves ALS convergence.

    Adds column: log_hours
    """
    return df.withColumn("log_hours", F.log1p(F.col(col)))


# ── Encode ────────────────────────────────────────────────────────────────────

def encode_ids(df: DataFrame) -> tuple[DataFrame, PipelineModel]:
    """
    Encode string user_id and game_name to integer indices required by ALS.

    Uses MLlib StringIndexer which assigns 0-based integer indices ordered
    by frequency (most frequent label → index 0).

    Parameters
    ----------
    df : DataFrame with columns user_id, game_name

    Returns
    -------
    (indexed_df, pipeline_model)
      indexed_df     : DataFrame with added user_idx (int), game_idx (int)
      pipeline_model : fitted Pipeline — call .transform() on new data to
                       apply the same encoding consistently
    """
    user_indexer = StringIndexer(
        inputCol="user_id",   outputCol="user_idx",  handleInvalid="keep"
    )
    game_indexer = StringIndexer(
        inputCol="game_name", outputCol="game_idx",  handleInvalid="keep"
    )

    pipeline       = Pipeline(stages=[user_indexer, game_indexer])
    pipeline_model = pipeline.fit(df)
    indexed_df     = pipeline_model.transform(df) \
        .withColumn("user_idx", F.col("user_idx").cast(IntegerType())) \
        .withColumn("game_idx", F.col("game_idx").cast(IntegerType()))

    return indexed_df, pipeline_model


def extract_labels(indexed_df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """
    Extract integer-to-string lookup tables for users and games.

    Returns
    -------
    (user_labels, game_labels)
      user_labels : DataFrame(user_idx INT, user_id STRING)
      game_labels : DataFrame(game_idx INT, game_name STRING)
    """
    user_labels = indexed_df.select("user_idx", "user_id").distinct()
    game_labels = indexed_df.select("game_idx", "game_name").distinct()
    return user_labels, game_labels


# ── Split ─────────────────────────────────────────────────────────────────────

def train_test_split(
    indexed_df: DataFrame,
    train_ratio: float = 0.8,
    seed: int = 42
) -> tuple[DataFrame, DataFrame]:
    """
    Split the encoded interaction DataFrame into training and test sets.

    Parameters
    ----------
    indexed_df  : output of encode_ids()
    train_ratio : fraction for training (default 0.8 = 80/20 split)
    seed        : random seed for reproducibility

    Returns
    -------
    (train_df, test_df)
    """
    train_df, test_df = indexed_df.randomSplit(
        [train_ratio, 1.0 - train_ratio], seed=seed
    )
    train_df.cache()
    test_df.cache()

    print(f"Train rows : {train_df.count():,}")
    print(f"Test rows  : {test_df.count():,}")

    return train_df, test_df
