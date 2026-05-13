# 🎮 SteamRecommender — Collaborative Filtering on Steam Play Data

A **PySpark MLlib** recommender system trained on 200,000 Steam user-game interaction records. Uses Alternating Least Squares (ALS) collaborative filtering to generate personalised game recommendations, evaluated with RMSE and coverage metrics, with exploratory analysis and visualisations throughout.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
- [Pipeline](#pipeline)
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
