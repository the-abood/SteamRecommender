# ============================================================
# FILE: src/model/als_trainer.py
# PURPOSE: ALS model training wrapper with hyperparameter
#          grid search via CrossValidator.
# ============================================================

from pyspark.sql import DataFrame
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_PARAM_GRID = {
    "rank":      [10, 20, 50],
    "regParam":  [0.01, 0.1, 1.0],
    "maxIter":   [10, 20],
}


# ── Single Model ──────────────────────────────────────────────────────────────

def build_als(
    user_col:   str   = "user_idx",
    item_col:   str   = "game_idx",
    rating_col: str   = "log_hours",
    rank:       int   = 10,
    reg_param:  float = 0.1,
    max_iter:   int   = 10,
    seed:       int   = 42,
) -> ALS:
    """
    Return a configured ALS estimator.

    Key parameters
    --------------
    implicitPrefs=True
        Treats values as confidence weights on a binary preference
        (did the user play this game?) rather than explicit ratings.
        Appropriate here because hours-played is implicit feedback.

    coldStartStrategy="drop"
        Exclude predictions for users/items not seen during training
        from evaluation. Without this, NaN predictions corrupt RMSE.

    nonnegative=True
        Constrain latent factor values to be non-negative, which
        tends to improve interpretability and stability for implicit
        feedback datasets.
    """
    return ALS(
        userCol           = user_col,
        itemCol           = item_col,
        ratingCol         = rating_col,
        implicitPrefs     = True,
        coldStartStrategy = "drop",
        nonnegative       = True,
        rank              = rank,
        regParam          = reg_param,
        maxIter           = max_iter,
        seed              = seed,
    )


def train_als(
    train_df:   DataFrame,
    user_col:   str   = "user_idx",
    item_col:   str   = "game_idx",
    rating_col: str   = "log_hours",
    rank:       int   = 10,
    reg_param:  float = 0.1,
    max_iter:   int   = 10,
    seed:       int   = 42,
) -> ALSModel:
    """
    Fit a single ALS model and return the trained ALSModel.
    Use for a quick baseline before running grid search.
    """
    als = build_als(
        user_col=user_col,
        item_col=item_col,
        rating_col=rating_col,
        rank=rank,
        reg_param=reg_param,
        max_iter=max_iter,
        seed=seed,
    )
    model = als.fit(train_df)
    print(f"ALS trained  rank={rank}  regParam={reg_param}  maxIter={max_iter}")
    return model


# ── Grid Search ───────────────────────────────────────────────────────────────

def grid_search_als(
    train_df:   DataFrame,
    param_grid: dict  = None,
    num_folds:  int   = 3,
    rating_col: str   = "log_hours",
    user_col:   str   = "user_idx",
    item_col:   str   = "game_idx",
    seed:       int   = 42,
) -> tuple[ALSModel, CrossValidator]:
    """
    Run k-fold cross-validation over a hyperparameter grid and return
    the best fitted ALSModel.

    Parameters
    ----------
    train_df    : encoded training DataFrame
    param_grid  : dict with keys 'rank', 'regParam', 'maxIter'
                  (lists of values to try per parameter)
    num_folds   : number of cross-validation folds (default 3)

    Returns
    -------
    (best_model, cv_model)
      best_model : best ALSModel as selected by lowest RMSE
      cv_model   : full CrossValidatorModel (contains all avg metrics)
    """
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID

    als = build_als(
        user_col=user_col,
        item_col=item_col,
        rating_col=rating_col,
        seed=seed,
    )

    grid = (
        ParamGridBuilder()
        .addGrid(als.rank,     param_grid["rank"])
        .addGrid(als.regParam, param_grid["regParam"])
        .addGrid(als.maxIter,  param_grid["maxIter"])
        .build()
    )

    evaluator = RegressionEvaluator(
        metricName   = "rmse",
        labelCol     = rating_col,
        predictionCol= "prediction",
    )

    cv = CrossValidator(
        estimator         = als,
        estimatorParamMaps= grid,
        evaluator         = evaluator,
        numFolds          = num_folds,
        seed              = seed,
    )

    total_configs = len(grid)
    print(f"Grid search: {total_configs} configs × {num_folds} folds "
          f"= {total_configs * num_folds} ALS fits")
    print("Training ...")

    cv_model   = cv.fit(train_df)
    best_model = cv_model.bestModel

    best_rank     = best_model.rank
    best_reg      = best_model._java_obj.parent().getRegParam()
    best_max_iter = best_model._java_obj.parent().getMaxIter()

    print(f"\nBest params found:")
    print(f"  rank     = {best_rank}")
    print(f"  regParam = {best_reg}")
    print(f"  maxIter  = {best_max_iter}")

    return best_model, cv_model
