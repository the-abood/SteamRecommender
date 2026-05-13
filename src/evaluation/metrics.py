# ============================================================
# FILE: src/evaluation/metrics.py
# PURPOSE: Evaluation helpers — RMSE, baseline comparison,
#          and catalogue/user coverage analysis.
# ============================================================

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALSModel
from pyspark.ml.evaluation import RegressionEvaluator


# ── RMSE ──────────────────────────────────────────────────────────────────────

def compute_rmse(
    model:      ALSModel,
    test_df:    DataFrame,
    rating_col: str = "log_hours",
) -> float:
    """
    Generate predictions on test_df and compute RMSE.

    Cold-start rows (users or items unseen during training) are
    automatically dropped by the model's coldStartStrategy="drop"
    setting, so they do not skew the metric.

    Parameters
    ----------
    model      : fitted ALSModel
    test_df    : encoded test DataFrame
    rating_col : column containing ground-truth values

    Returns
    -------
    float RMSE on the log-transformed scale
    """
    evaluator = RegressionEvaluator(
        metricName   = "rmse",
        labelCol     = rating_col,
        predictionCol= "prediction",
    )
    predictions = model.transform(test_df)
    return evaluator.evaluate(predictions)


# ── Baselines ─────────────────────────────────────────────────────────────────

def mean_baseline_rmse(
    train_df:   DataFrame,
    test_df:    DataFrame,
    rating_col: str = "log_hours",
) -> float:
    """
    Compute RMSE of the trivial mean baseline: predict the global mean
    log_hours from the training set for every test row.

    Comparing the model RMSE against this baseline quantifies how much
    the collaborative filtering actually learns vs a zero-information guess.
    """
    evaluator = RegressionEvaluator(
        metricName   = "rmse",
        labelCol     = rating_col,
        predictionCol= "prediction",
    )
    mean_val = float(
        train_df.agg(F.avg(rating_col)).collect()[0][0]
    )
    baseline_df = test_df.withColumn("prediction", F.lit(mean_val))
    return evaluator.evaluate(baseline_df)


def print_rmse_summary(
    model:      ALSModel,
    train_df:   DataFrame,
    test_df:    DataFrame,
    label:      str = "Model",
    rating_col: str = "log_hours",
) -> dict:
    """
    Print a formatted RMSE comparison table and return results as dict.

    Returns
    -------
    {
        "mean_baseline_rmse": float,
        "model_rmse":         float,
        "improvement_pct":    float,
    }
    """
    model_rmse    = compute_rmse(model, test_df, rating_col)
    baseline_rmse = mean_baseline_rmse(train_df, test_df, rating_col)
    improvement   = ((baseline_rmse - model_rmse) / baseline_rmse) * 100

    print(f"{'='*40}")
    print(f"  RMSE Evaluation — {label}")
    print(f"{'='*40}")
    print(f"  Mean baseline RMSE : {baseline_rmse:.4f}")
    print(f"  {label} RMSE      : {model_rmse:.4f}")
    print(f"  Improvement        : {improvement:.1f}%")
    print(f"{'='*40}")

    return {
        "mean_baseline_rmse": baseline_rmse,
        "model_rmse":         model_rmse,
        "improvement_pct":    improvement,
    }


# ── Coverage ──────────────────────────────────────────────────────────────────

def coverage_report(
    indexed_df:     DataFrame,
    user_recs_flat: DataFrame,
) -> dict:
    """
    Compute catalogue and user coverage for a set of recommendations.

    Catalogue coverage: what fraction of all games appear in at least
    one user's top-N recommendation list? Low coverage indicates the
    model is over-recommending a small set of popular items.

    User coverage: what fraction of users received at least one
    recommendation? Users not seen during training are excluded by
    coldStartStrategy="drop" and will have no recommendations.

    Parameters
    ----------
    indexed_df     : full encoded interaction DataFrame
    user_recs_flat : exploded recommendations with game_idx and user_idx

    Returns
    -------
    dict with catalogue_coverage_pct and user_coverage_pct
    """
    total_games       = indexed_df.select("game_idx").distinct().count()
    recommended_games = user_recs_flat.select("game_idx").distinct().count()
    catalogue_pct     = (recommended_games / total_games) * 100

    total_users       = indexed_df.select("user_idx").distinct().count()
    covered_users     = user_recs_flat.select("user_idx").distinct().count()
    user_pct          = (covered_users / total_users) * 100

    print(f"{'='*45}")
    print(f"  Coverage Report")
    print(f"{'='*45}")
    print(f"  Games in catalogue        : {total_games:>7,}")
    print(f"  Games in top-N recs       : {recommended_games:>7,}  ({catalogue_pct:.1f}%)")
    print(f"  Total users               : {total_users:>7,}")
    print(f"  Users with recommendations: {covered_users:>7,}  ({user_pct:.1f}%)")
    print(f"{'='*45}")

    return {
        "total_games":           total_games,
        "recommended_games":     recommended_games,
        "catalogue_coverage_pct":catalogue_pct,
        "total_users":           total_users,
        "covered_users":         covered_users,
        "user_coverage_pct":     user_pct,
    }
