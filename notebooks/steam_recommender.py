# Databricks notebook source
# ============================================================
# SteamRecommender — steam_recommender.py
#
# Collaborative filtering recommender system built with
# PySpark MLlib ALS on the Steam 200k dataset.
#
# Run as a Databricks notebook (import as Python file)
# or locally with a configured PySpark installation.
# ============================================================

# COMMAND ----------
# MAGIC %md
# MAGIC # 🎮 Steam Game Recommender — ALS Collaborative Filtering
# MAGIC
# MAGIC **Dataset:** `steam-200k.csv` — ~200,000 Steam user-game interaction records
# MAGIC
# MAGIC **Pipeline:**
# MAGIC 1. Load & explore
# MAGIC 2. Pre-process (filter, encode, transform)
# MAGIC 3. Train ALS model with grid search
# MAGIC 4. Evaluate (RMSE vs baseline)
# MAGIC 5. Generate and inspect recommendations
# MAGIC 6. Coverage analysis

# COMMAND ----------
# MAGIC %md
# MAGIC ## Setup — Imports

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType, IntegerType
)
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np

# ── Spark session ──────────────────────────────────────────
try:
    spark  # pre-created on Databricks
except NameError:
    spark = SparkSession.builder \
        .appName("SteamRecommender") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()

spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
sns.set_style("whitegrid")
print(f"Spark version: {spark.version}")


# COMMAND ----------
# MAGIC %md
# MAGIC ---
# MAGIC ## 1 — Load Dataset

# COMMAND ----------

FILE_PATH = "/FileStore/tables/steam-200k.csv"
# Local fallback: FILE_PATH = "data/steam-200k.csv"

schema = StructType([
    StructField("user_id",   StringType(),  True),
    StructField("game_name", StringType(),  True),
    StructField("behaviour", StringType(),  True),
    StructField("value",     FloatType(),   True),
])

raw_df = (
    spark.read
    .option("header", "false")
    .schema(schema)
    .csv(FILE_PATH)
)

raw_df.createOrReplaceTempView("steam_raw")

print(f"Total rows   : {raw_df.count():,}")
print(f"Unique users : {raw_df.select('user_id').distinct().count():,}")
print(f"Unique games : {raw_df.select('game_name').distinct().count():,}")
raw_df.show(5, truncate=False)


# COMMAND ----------
# MAGIC %md
# MAGIC ---
# MAGIC ## 2 — Exploratory Data Analysis

# COMMAND ----------
# MAGIC %md
# MAGIC ### 2a — Behaviour Split

# COMMAND ----------

spark.sql("""
    SELECT   behaviour,
             COUNT(*)                        AS row_count,
             COUNT(DISTINCT user_id)         AS unique_users,
             COUNT(DISTINCT game_name)       AS unique_games
    FROM     steam_raw
    GROUP BY behaviour
""").show()


# COMMAND ----------
# MAGIC %md
# MAGIC ### 2b — Top 20 Most-Played Games (by total hours)

# COMMAND ----------

top_games_df = spark.sql("""
    SELECT   game_name,
             COUNT(DISTINCT user_id)          AS unique_players,
             ROUND(SUM(value), 1)             AS total_hours,
             ROUND(AVG(value), 1)             AS avg_hours_per_player
    FROM     steam_raw
    WHERE    behaviour = 'play'
    GROUP BY game_name
    ORDER BY total_hours DESC
    LIMIT    20
""")
top_games_df.show(truncate=False)

# ── Chart ──────────────────────────────────────────────────
top_pd = top_games_df.toPandas()
fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(
    top_pd["game_name"][::-1],
    top_pd["total_hours"][::-1],
    color="steelblue", edgecolor="white"
)
ax.set_title("Top 20 Games by Total Hours Played", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Total Hours Played")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
plt.tight_layout()
plt.savefig("outputs/top_games.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → outputs/top_games.png")


# COMMAND ----------
# MAGIC %md
# MAGIC ### 2c — Playtime Distribution

# COMMAND ----------

play_sample = spark.sql("""
    SELECT value FROM steam_raw WHERE behaviour = 'play'
""").toPandas()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Raw distribution (capped)
axes[0].hist(
    play_sample["value"].clip(upper=200),
    bins=60, color="steelblue", edgecolor="white"
)
axes[0].set_title("Playtime Distribution (capped at 200 hrs)")
axes[0].set_xlabel("Hours Played")
axes[0].set_ylabel("Count")

# Log-transformed
axes[1].hist(
    np.log1p(play_sample["value"]),
    bins=60, color="tomato", edgecolor="white"
)
axes[1].set_title("Log-Transformed Playtime (log1p hours)")
axes[1].set_xlabel("log1p(Hours Played)")
axes[1].set_ylabel("Count")

plt.suptitle("Playtime Distribution — Raw vs Log-Transformed", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("outputs/playtime_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → outputs/playtime_distribution.png")


# COMMAND ----------
# MAGIC %md
# MAGIC ### 2d — User Activity Distribution

# COMMAND ----------

user_activity = spark.sql("""
    SELECT   user_id,
             COUNT(DISTINCT game_name) AS games_played
    FROM     steam_raw
    WHERE    behaviour = 'play'
    GROUP BY user_id
""").toPandas()

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(
    user_activity["games_played"].clip(upper=50),
    bins=50, color="mediumpurple", edgecolor="white"
)
ax.set_title("User Activity — Number of Distinct Games Played (capped at 50)", fontsize=13)
ax.set_xlabel("Distinct Games Played per User")
ax.set_ylabel("Number of Users")
plt.tight_layout()
plt.savefig("outputs/user_activity.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → outputs/user_activity.png")


# COMMAND ----------
# MAGIC %md
# MAGIC ---
# MAGIC ## 3 — Pre-Processing

# COMMAND ----------
# MAGIC %md
# MAGIC ### 3a — Behaviour Selection
# MAGIC
# MAGIC **Decision: Use `play` rows only.**
# MAGIC
# MAGIC Play hours provide a continuous, graded implicit feedback signal. A user who
# MAGIC played 300 hours values a game far more than one who played 1 hour. Purchase
# MAGIC rows (value always = 1) carry no preference gradient and would collapse every
# MAGIC bought-but-unplayed game to the same weight, introducing noise rather than signal.
# MAGIC
# MAGIC We use `implicitPrefs=True` in ALS, treating hours as confidence-weighted
# MAGIC preferences rather than explicit numeric ratings.

# COMMAND ----------

play_df = raw_df.filter(F.col("behaviour") == "play").select(
    "user_id", "game_name", "value"
)

# Log-transform hours to reduce right skew (many users, few hours;
# a small number with thousands of hours would dominate otherwise)
play_df = play_df.withColumn("log_hours", F.log1p(F.col("value")))

print(f"Play rows   : {play_df.count():,}")
play_df.describe("value", "log_hours").show()


# COMMAND ----------
# MAGIC %md
# MAGIC ### 3b — Integer Index Encoding
# MAGIC
# MAGIC ALS requires integer user and item IDs. We use `StringIndexer` from MLlib
# MAGIC to encode `user_id` and `game_name` into integer indices, and store the
# MAGIC reverse mapping for decoding recommendations back to readable names.

# COMMAND ----------

user_indexer = StringIndexer(inputCol="user_id",   outputCol="user_idx",  handleInvalid="keep")
game_indexer = StringIndexer(inputCol="game_name", outputCol="game_idx",  handleInvalid="keep")

index_pipeline = Pipeline(stages=[user_indexer, game_indexer])
index_model    = index_pipeline.fit(play_df)
indexed_df     = index_model.transform(play_df) \
    .withColumn("user_idx", F.col("user_idx").cast(IntegerType())) \
    .withColumn("game_idx", F.col("game_idx").cast(IntegerType()))

# Store label lookups for decoding
user_labels = indexed_df.select("user_idx", "user_id").distinct()
game_labels = indexed_df.select("game_idx", "game_name").distinct()

indexed_df.select("user_idx", "game_idx", "log_hours").show(5)
print(f"Unique users : {indexed_df.select('user_idx').distinct().count():,}")
print(f"Unique games : {indexed_df.select('game_idx').distinct().count():,}")


# COMMAND ----------
# MAGIC %md
# MAGIC ### 3c — Train / Test Split

# COMMAND ----------

train_df, test_df = indexed_df.randomSplit([0.8, 0.2], seed=42)
train_df.cache()
test_df.cache()

print(f"Training rows : {train_df.count():,}")
print(f"Test rows     : {test_df.count():,}")


# COMMAND ----------
# MAGIC %md
# MAGIC ---
# MAGIC ## 4 — ALS Model Training

# COMMAND ----------
# MAGIC %md
# MAGIC ### 4a — Baseline ALS (Default Params)
# MAGIC
# MAGIC Train an initial ALS model with sensible defaults to establish a benchmark
# MAGIC before the full hyperparameter search.

# COMMAND ----------

als_baseline = ALS(
    userCol          = "user_idx",
    itemCol          = "game_idx",
    ratingCol        = "log_hours",
    implicitPrefs    = True,
    coldStartStrategy= "drop",
    nonnegative      = True,
    rank             = 10,
    regParam         = 0.1,
    maxIter          = 10,
    seed             = 42
)

baseline_model = als_baseline.fit(train_df)

evaluator = RegressionEvaluator(
    metricName   = "rmse",
    labelCol     = "log_hours",
    predictionCol= "prediction"
)

baseline_preds = baseline_model.transform(test_df)
baseline_rmse  = evaluator.evaluate(baseline_preds)
print(f"Baseline ALS RMSE: {baseline_rmse:.4f}")


# COMMAND ----------
# MAGIC %md
# MAGIC ### 4b — Hyperparameter Grid Search

# COMMAND ----------

als = ALS(
    userCol          = "user_idx",
    itemCol          = "game_idx",
    ratingCol        = "log_hours",
    implicitPrefs    = True,
    coldStartStrategy= "drop",
    nonnegative      = True,
    seed             = 42
)

param_grid = (
    ParamGridBuilder()
    .addGrid(als.rank,     [10, 20, 50])
    .addGrid(als.regParam, [0.01, 0.1, 1.0])
    .addGrid(als.maxIter,  [10, 20])
    .build()
)

cv = CrossValidator(
    estimator         = als,
    estimatorParamMaps= param_grid,
    evaluator         = evaluator,
    numFolds          = 3,
    seed              = 42
)

print("Running 3-fold cross-validation over parameter grid ...")
print(f"Total configurations: {len(param_grid)}")
cv_model    = cv.fit(train_df)
best_model  = cv_model.bestModel

best_rank    = best_model.rank
best_reg     = best_model._java_obj.parent().getRegParam()
best_maxiter = best_model._java_obj.parent().getMaxIter()

print(f"\nBest rank     : {best_rank}")
print(f"Best regParam : {best_reg}")
print(f"Best maxIter  : {best_maxiter}")


# COMMAND ----------
# MAGIC %md
# MAGIC ---
# MAGIC ## 5 — Evaluation

# COMMAND ----------

# RMSE on held-out test set
best_preds = best_model.transform(test_df)
best_rmse  = evaluator.evaluate(best_preds)

# Trivial baseline: predict the mean log_hours for every interaction
mean_log       = float(train_df.agg(F.avg("log_hours")).collect()[0][0])
mean_baseline  = test_df.withColumn("prediction", F.lit(mean_log))
mean_baseline_rmse = evaluator.evaluate(mean_baseline)

improvement = ((mean_baseline_rmse - best_rmse) / mean_baseline_rmse) * 100

print("=== Evaluation Results ===")
print(f"  Mean baseline RMSE : {mean_baseline_rmse:.4f}")
print(f"  Baseline ALS RMSE  : {baseline_rmse:.4f}")
print(f"  Best ALS RMSE      : {best_rmse:.4f}")
print(f"  Improvement over mean baseline : {improvement:.1f}%")

# ── RMSE comparison bar chart ──────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
labels = ["Mean Baseline", "ALS Defaults", "ALS Best (CV)"]
values = [mean_baseline_rmse, baseline_rmse, best_rmse]
colours = ["#d9534f", "#f0ad4e", "#5cb85c"]
bars = ax.bar(labels, values, color=colours, edgecolor="white", width=0.5)
ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=11)
ax.set_title("RMSE Comparison", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("RMSE (log-hours scale)")
ax.set_ylim(0, max(values) * 1.2)
plt.tight_layout()
plt.savefig("outputs/rmse_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → outputs/rmse_comparison.png")


# COMMAND ----------
# MAGIC %md
# MAGIC ---
# MAGIC ## 6 — Recommendation Generation

# COMMAND ----------
# MAGIC %md
# MAGIC ### 6a — Top-10 Games for All Users

# COMMAND ----------

user_recs = best_model.recommendForAllUsers(10)

# Explode and decode
user_recs_flat = (
    user_recs
    .withColumn("rec", F.explode("recommendations"))
    .select(
        F.col("user_idx"),
        F.col("rec.game_idx").alias("game_idx"),
        F.col("rec.rating").alias("predicted_score")
    )
    .join(game_labels, on="game_idx")
    .join(user_labels, on="user_idx")
)

# Show recommendations for a sample user
sample_user_idx = indexed_df.limit(1).collect()[0]["user_idx"]
print(f"Top-10 recommendations for user_idx={sample_user_idx}:")
user_recs_flat \
    .filter(F.col("user_idx") == sample_user_idx) \
    .orderBy("predicted_score", ascending=False) \
    .select("game_name", "predicted_score") \
    .show(10, truncate=False)


# COMMAND ----------
# MAGIC %md
# MAGIC ### 6b — Top-10 Users for Each Game (Similar Audience)

# COMMAND ----------

game_recs = best_model.recommendForAllItems(10)

game_recs_flat = (
    game_recs
    .withColumn("rec", F.explode("recommendations"))
    .select(
        F.col("game_idx"),
        F.col("rec.user_idx").alias("user_idx"),
        F.col("rec.rating").alias("predicted_score")
    )
    .join(game_labels, on="game_idx")
    .join(user_labels, on="user_idx")
)

# Show target users for a sample game
sample_game = "Dota 2"
print(f"Top-10 target users for '{sample_game}':")
game_recs_flat \
    .filter(F.col("game_name") == sample_game) \
    .orderBy("predicted_score", ascending=False) \
    .select("user_id", "predicted_score") \
    .show(10, truncate=False)


# COMMAND ----------
# MAGIC %md
# MAGIC ### 6c — Recommendation Coverage

# COMMAND ----------

total_games         = indexed_df.select("game_idx").distinct().count()
recommended_games   = user_recs_flat.select("game_idx").distinct().count()
coverage_pct        = (recommended_games / total_games) * 100

total_users         = indexed_df.select("user_idx").distinct().count()
covered_users       = user_recs_flat.select("user_idx").distinct().count()
user_coverage_pct   = (covered_users / total_users) * 100

print("=== Coverage Analysis ===")
print(f"  Games in catalogue   : {total_games:,}")
print(f"  Games recommended    : {recommended_games:,}  ({coverage_pct:.1f}%)")
print(f"  Users in dataset     : {total_users:,}")
print(f"  Users with recs      : {covered_users:,}  ({user_coverage_pct:.1f}%)")


# COMMAND ----------
# MAGIC %md
# MAGIC ### 6d — Recommendation Diversity: Most-Recommended Games

# COMMAND ----------

most_recommended = (
    user_recs_flat
    .groupBy("game_name")
    .agg(F.count("user_idx").alias("times_recommended"))
    .orderBy("times_recommended", ascending=False)
    .limit(15)
    .toPandas()
)

fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(
    most_recommended["game_name"][::-1],
    most_recommended["times_recommended"][::-1],
    color="mediumpurple", edgecolor="white"
)
ax.set_title("Top 15 Most-Recommended Games Across All Users", fontsize=13, fontweight="bold")
ax.set_xlabel("Times Recommended")
plt.tight_layout()
plt.savefig("outputs/most_recommended_games.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → outputs/most_recommended_games.png")


# COMMAND ----------
# MAGIC %md
# MAGIC ---
# MAGIC ## 7 — Save Model

# COMMAND ----------

best_model.save("outputs/als_model/")
print("Model saved → outputs/als_model/")

# Reload example:
# loaded_model = ALSModel.load("outputs/als_model/")
