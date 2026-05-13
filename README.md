# 🎮 SteamRecommender — Collaborative Filtering on Steam Play Data

A **PySpark MLlib** recommender system trained on 200,000 Steam user-game interaction records. Uses Alternating Least Squares (ALS) collaborative filtering to generate personalised game recommendations, evaluated with RMSE and coverage metrics, with exploratory analysis and visualisations throughout.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
- [Pipeline](#pipeline)
- [Examples](#examples)
- [Project Structure](#project-structure)
- [Design Decisions](#design-decisions)
- [Tech Stack](#tech-stack)

---

## Project Overview

Steam members generate rich implicit feedback simply by purchasing and playing games. This project turns that signal into a collaborative filtering model: given a user's history, which games are they most likely to enjoy next?

**What this project covers:**

- Exploratory data analysis of user behaviour and playtime distributions
- Pre-processing: integer ID encoding, behaviour selection, train/test splitting
- ALS model training with hyperparameter grid search
- RMSE evaluation on held-out test data
- Top-N recommendation generation and coverage analysis
- Visualisations: playtime distribution, top games, recommendation diversity

---

## Dataset

| File | Rows | Description |
|------|------|-------------|
| `steam-200k.csv` | ~200,000 | Steam user-game interaction records |

### Columns

| # | Name | Description |
|---|------|-------------|
| 1 | `user_id` | Unique member identifier |
| 2 | `game_name` | Title of the game |
| 3 | `behaviour` | `'purchase'` or `'play'` |
| 4 | `value` | `1` for purchases; hours played for `'play'` rows |

> **Note:** A purchased game that was never played has only a `purchase` row. Games that were played have both a `purchase` row (value=1) and a `play` row (value=hours).

---

## Getting Started

### Option A — Databricks (recommended)

1. Upload `steam-200k.csv` to DBFS:
   ```
   /FileStore/tables/steam-200k.csv
   ```
2. Import `notebooks/steam_recommender.py` as a Databricks notebook.
3. Run all cells top to bottom.

### Option B — Local PySpark

```bash
git clone https://github.com/yourusername/SteamRecommender.git
cd SteamRecommender

pip install -r requirements.txt

python notebooks/steam_recommender.py
```

> Requires Java 8 or 11 and `JAVA_HOME` set.

### Restore from checkpoint

The trained ALS model is saved to `outputs/als_model/`. Reload without retraining:

```python
from pyspark.ml.recommendation import ALSModel
model = ALSModel.load("outputs/als_model/")
```

---

## Pipeline

```
steam-200k.csv
      │
      ▼
 1. Load & Schema  →  raw_df
      │
      ▼
 2. EDA            →  top games, playtime distribution, user activity charts
      │
      ▼
 3. Pre-processing →  filter to 'play' rows, encode IDs to integers,
      │                log-transform hours, train/test split (80/20)
      ▼
 4. ALS Training   →  grid search over rank, regParam, maxIter
      │
      ▼
 5. Evaluation     →  RMSE on test set, coverage, cold-start handling
      │
      ▼
 6. Recommendations→  Top-N per user, top-N per game (similar users)
      │
      ▼
 7. Outputs        →  charts, model artefacts, recommendation CSVs
```

---

## Examples

### Example 1 — Exploratory Analysis: Top 20 Most-Played Games

```python
top_games = spark.sql("""
    SELECT   game_name,
             COUNT(DISTINCT user_id)          AS unique_players,
             ROUND(SUM(value), 1)             AS total_hours,
             ROUND(AVG(value), 1)             AS avg_hours_per_player
    FROM     steam_play
    GROUP BY game_name
    ORDER BY total_hours DESC
    LIMIT    20
""")
top_games.show(truncate=False)
```

Bar chart of unique players per game saved to `outputs/top_games.png`.

---

### Example 2 — Playtime Distribution

```python
play_df.select("value").toPandas()["value"] \
    .apply(lambda x: min(x, 200)) \
    .hist(bins=50, figsize=(10, 4), color="steelblue", edgecolor="white")
plt.title("Playtime Distribution (capped at 200 hrs)")
plt.xlabel("Hours Played")
plt.ylabel("Number of Sessions")
plt.tight_layout()
plt.savefig("outputs/playtime_distribution.png", dpi=150)
```

The distribution is heavily right-skewed — a log transform is applied before training.

---

### Example 3 — Pre-processing: Behaviour Selection & ID Encoding

**Why `'play'` rows only?**

Play hours provide a continuous, graded signal of preference strength — a user who played 200 hours values a game far more than one who played 1 hour. Purchase rows (always value=1) carry no preference gradient and would collapse all bought-but-unplayed games to the same weight. Using play hours as implicit feedback produces a richer training signal.

```python
from pyspark.ml.feature import StringIndexer

# Keep only play behaviour
play_df = raw_df.filter(F.col("behaviour") == "play")

# Encode string user_id and game_name to integer indices
user_indexer = StringIndexer(inputCol="user_id",  outputCol="user_idx")
game_indexer = StringIndexer(inputCol="game_name", outputCol="game_idx")

pipeline = Pipeline(stages=[user_indexer, game_indexer])
indexed_df = pipeline.fit(play_df).transform(play_df)

# Log-transform hours to reduce skew
indexed_df = indexed_df.withColumn("log_hours", F.log1p(F.col("value")))
```

---

### Example 4 — Train/Test Split & ALS Training

```python
from pyspark.ml.recommendation import ALS

train_df, test_df = indexed_df.randomSplit([0.8, 0.2], seed=42)

als = ALS(
    userCol      = "user_idx",
    itemCol      = "game_idx",
    ratingCol    = "log_hours",
    implicitPrefs= True,           # treat as implicit feedback
    coldStartStrategy = "drop",    # exclude cold-start users from eval
    nonnegative  = True
)
```

---

### Example 5 — Hyperparameter Grid Search

```python
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator

param_grid = (
    ParamGridBuilder()
    .addGrid(als.rank,       [10, 20, 50])
    .addGrid(als.regParam,   [0.01, 0.1, 1.0])
    .addGrid(als.maxIter,    [10, 20])
    .build()
)

evaluator = RegressionEvaluator(
    metricName   = "rmse",
    labelCol     = "log_hours",
    predictionCol= "prediction"
)

cv = CrossValidator(
    estimator         = als,
    estimatorParamMaps= param_grid,
    evaluator         = evaluator,
    numFolds          = 3,
    seed              = 42
)

cv_model  = cv.fit(train_df)
best_model = cv_model.bestModel
print(f"Best rank     : {best_model.rank}")
print(f"Best regParam : {best_model._java_obj.parent().getRegParam()}")
```

---

### Example 6 — RMSE Evaluation

```python
predictions = best_model.transform(test_df)

rmse = evaluator.evaluate(predictions)
print(f"Test RMSE: {rmse:.4f}")

# Baseline: predict the mean log_hours for every user
mean_log = train_df.agg(F.avg("log_hours")).collect()[0][0]
baseline_preds = test_df.withColumn("prediction", F.lit(mean_log))
baseline_rmse  = evaluator.evaluate(baseline_preds)
print(f"Baseline RMSE: {baseline_rmse:.4f}")
print(f"Improvement  : {((baseline_rmse - rmse) / baseline_rmse * 100):.1f}%")
```

---

### Example 7 — Top-10 Game Recommendations for a User

```python
# Recommend top 10 games for every user
user_recs = best_model.recommendForAllUsers(10)

# Decode integer indices back to game names
game_labels = indexed_df.select("game_idx", "game_name").distinct()

# Explode recommendation list and join names
user_recs_exploded = user_recs \
    .withColumn("rec", F.explode("recommendations")) \
    .select(
        F.col("user_idx"),
        F.col("rec.game_idx").alias("game_idx"),
        F.col("rec.rating").alias("predicted_log_hours")
    ) \
    .join(game_labels, on="game_idx")

# Show recommendations for a specific user
example_user = indexed_df.limit(1).collect()[0]["user_idx"]
user_recs_exploded \
    .filter(F.col("user_idx") == example_user) \
    .orderBy("predicted_log_hours", ascending=False) \
    .show(truncate=False)
```

---

### Example 8 — Recommendation Coverage

```python
total_games      = indexed_df.select("game_idx").distinct().count()
recommended_games = user_recs_exploded.select("game_idx").distinct().count()
coverage = (recommended_games / total_games) * 100

print(f"Total unique games in catalogue : {total_games}")
print(f"Games appearing in top-10 recs  : {recommended_games}")
print(f"Catalogue coverage              : {coverage:.1f}%")
```

---

## Project Structure

```
SteamRecommender/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── steam_recommender.py        ← Full end-to-end Databricks notebook
│
├── src/
│   ├── preprocessing/
│   │   └── prepare_data.py         ← Load, filter, encode, split helpers
│   ├── model/
│   │   └── als_trainer.py          ← ALS training + grid search wrapper
│   ├── evaluation/
│   │   └── metrics.py              ← RMSE, baseline comparison, coverage
│   └── visualisation/
│       └── plots.py                ← EDA and results charts
│
├── data/
│   └── README.txt                  ← Place steam-200k.csv here
│
├── outputs/
│   ├── README.txt
│   ├── top_games.png
│   ├── playtime_distribution.png
│   ├── rmse_comparison.png
│   └── als_model/                  ← Saved ALS model artefacts
│
└── docs/
    └── design_notes.md             ← Methodology, design decisions, assumptions
```

---

## Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Behaviour | `play` rows only | Hours played give a graded preference signal; purchases are binary |
| Rating column | `log1p(hours)` | Right-skewed distribution; log transform improves ALS convergence |
| Cold-start | `drop` strategy | Prevents NaN predictions corrupting RMSE on unseen users/items |
| Implicit prefs | `implicitPrefs=True` | Data is implicit feedback (hours), not explicit ratings |
| Evaluation | RMSE vs mean baseline | Quantifies model lift over a trivial predictor |

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Apache Spark 3.x | Distributed data processing |
| PySpark MLlib | ALS collaborative filtering |
| Spark SQL | EDA queries |
| Databricks | Managed Spark environment |
| Matplotlib / Seaborn | Visualisation |
| Pandas | Small-result conversion for plotting |

---

## License

MIT — free to use, extend, and adapt.
