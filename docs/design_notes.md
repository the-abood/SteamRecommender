# Design Notes — SteamRecommender

## 1. Problem Framing

Collaborative filtering learns from patterns of co-occurrence: if users A and B
have similar play histories, the model surfaces games that B liked but A hasn't
seen yet. It requires no game metadata (no genres, no descriptions) — only the
interaction matrix of who played what for how long.

This makes ALS (Alternating Least Squares) particularly well suited: it
factorises the sparse user-game matrix into low-dimensional latent factor
representations for both users and games, then scores unobserved pairs by
taking the dot product of their factors.

---

## 2. Behaviour Selection: Why Play Hours?

The dataset has two interaction types:

| Behaviour | Value | Signal |
|-----------|-------|--------|
| purchase  | always 1 | binary — game was bought |
| play      | hours played | graded — depth of engagement |

**Decision: use `play` rows only.**

Play hours are a richer implicit signal. A user who invested 300 hours in a
game clearly values it more than one who played for 1 hour. Purchase rows
collapse this gradient: every bought-but-barely-played game has the same weight
as a beloved favourite, introducing noise.

`implicitPrefs=True` in ALS is the appropriate setting here — it treats the
value column as a confidence weight on a binary "did the user engage with this
game?" preference, rather than an explicit numeric rating. This is standard
practice for e-commerce and streaming implicit feedback datasets.

---

## 3. Log Transformation

Raw playtime is extremely right-skewed. The distribution shows many sessions
under 10 hours and a long tail reaching thousands of hours for dedicated
players. Without transformation, the ALS loss function is dominated by the
outlier values.

`log1p(hours)` compresses the scale symmetrically, improving convergence and
producing a more interpretable latent space. `log1p` (log(1 + x)) is used
rather than `log` to handle the case where hours = 0 without producing `-inf`.

---

## 4. Integer ID Encoding

ALS requires integer user and item identifiers. `StringIndexer` maps each
unique `user_id` and `game_name` to a 0-based integer index, ordered by
frequency (most frequent → index 0). The fitted `PipelineModel` is retained
so the same mapping can be applied to new data consistently.

After generating recommendations, `game_labels` and `user_labels` lookup
DataFrames are used to decode integer indices back to human-readable names
before presenting results.

---

## 5. Cold-Start Strategy

Users or games that appear in the test set but not the training set cannot
have latent factors computed for them. ALS would produce `NaN` predictions for
these rows.

`coldStartStrategy="drop"` excludes these rows from the test predictions so
they do not contaminate RMSE. In a production deployment, cold-start users
would be handled separately — for example, by recommending globally popular
games until sufficient interaction history accumulates.

---

## 6. Hyperparameter Grid Search

Three parameters are tuned via 3-fold cross-validation:

| Parameter | Values tried | Effect |
|-----------|-------------|--------|
| `rank` | 10, 20, 50 | Dimensionality of latent factors; higher = more expressive but slower and more prone to overfitting |
| `regParam` | 0.01, 0.1, 1.0 | L2 regularisation strength; higher = less overfitting, potentially more bias |
| `maxIter` | 10, 20 | Number of ALS alternation steps; more = finer convergence |

Cross-validation gives a more reliable performance estimate than a single
train/test split because it averages over three different held-out folds.
`CrossValidator` selects the configuration with the lowest average RMSE.

---

## 7. Evaluation with RMSE

RMSE is computed on the held-out 20% test set, on the log-transformed scale.
This means the unit is "log-hours" rather than raw hours — it penalises large
prediction errors proportionally.

Two baselines are reported alongside the model:

1. **Mean baseline** — predict the global average log-hours for every
   interaction. This is the simplest possible model and establishes a floor
   that any useful recommender must beat.
2. **Default ALS** — ALS with rank=10, regParam=0.1, maxIter=10 before
   hyperparameter tuning.

The improvement percentage quantifies how much lift the grid-searched model
provides over the trivial predictor.

---

## 8. Coverage Analysis

**Catalogue coverage** measures what fraction of all games in the dataset
appear in at least one user's top-10 list. Low coverage (e.g. 20%) would
indicate a popularity bias — the model keeps recommending the same small set
of blockbuster titles to everyone, ignoring the long tail of niche games.
Higher coverage is generally desirable for user satisfaction and diversity.

**User coverage** measures the fraction of users who received at least one
recommendation. Users excluded by `coldStartStrategy="drop"` appear in the
training set but not the recommendation output — tracking this reveals how
many users are left without personalised suggestions.

---

## 9. Performance Notes

- `train_df.cache()` and `test_df.cache()` are called after the split to avoid
  re-reading and re-computing the data for every cross-validation fold.
- `spark.sql.shuffle.partitions` is set to 200 (default is 200; adjust down
  for small clusters to reduce scheduler overhead).
- On a single-node Databricks cluster, the full grid search (18 configs × 3
  folds = 54 ALS fits) takes approximately 20–40 minutes depending on cluster
  size. Narrow the grid for faster iteration during development.

---

## 10. Limitations and Extensions

| Limitation | Possible extension |
|------------|-------------------|
| Cold-start users get no personalised recs | Hybrid model: combine ALS with content-based features (genre, tags) |
| RMSE on log-hours may not reflect ranking quality | Add Precision@K / NDCG evaluation using `spark.ml.evaluation.RankingEvaluator` |
| All behaviours weighted equally | Weight recent play sessions more heavily (time-decay factor) |
| Single-factor model | Experiment with NMF or deep learning (Neural Collaborative Filtering) |
