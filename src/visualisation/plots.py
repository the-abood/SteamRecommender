# ============================================================
# FILE: src/visualisation/plots.py
# PURPOSE: Reusable chart helpers for EDA and model results.
#          All functions save to the outputs/ directory and
#          also render inline on Databricks / Jupyter.
# ============================================================

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame

sns.set_style("whitegrid")
OUTPUT_DIR = "outputs"


# ── EDA ───────────────────────────────────────────────────────────────────────

def plot_top_games(
    top_games_pd: pd.DataFrame,
    name_col:  str = "game_name",
    value_col: str = "total_hours",
    title:     str = "Top Games by Total Hours Played",
    filename:  str = "top_games.png",
) -> None:
    """
    Horizontal bar chart of top games by a numeric metric.

    Parameters
    ----------
    top_games_pd : Pandas DataFrame (already collected from Spark)
    name_col     : column containing game names
    value_col    : column containing the metric to plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(
        top_games_pd[name_col][::-1],
        top_games_pd[value_col][::-1],
        color="steelblue", edgecolor="white"
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(value_col.replace("_", " ").title())
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/{filename}"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {path}")


def plot_playtime_distribution(
    play_pd:  pd.DataFrame,
    cap:      float = 200.0,
    filename: str   = "playtime_distribution.png",
) -> None:
    """
    Side-by-side histograms: raw playtime (capped) vs log1p-transformed.
    Demonstrates the motivation for log-transforming before ALS training.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(
        play_pd["value"].clip(upper=cap),
        bins=60, color="steelblue", edgecolor="white"
    )
    axes[0].set_title(f"Playtime Distribution (capped at {cap:.0f} hrs)")
    axes[0].set_xlabel("Hours Played")
    axes[0].set_ylabel("Count")

    axes[1].hist(
        np.log1p(play_pd["value"]),
        bins=60, color="tomato", edgecolor="white"
    )
    axes[1].set_title("Log-Transformed Playtime (log1p hours)")
    axes[1].set_xlabel("log1p(Hours Played)")
    axes[1].set_ylabel("Count")

    plt.suptitle(
        "Playtime Distribution — Raw vs Log-Transformed",
        fontsize=13, y=1.02
    )
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/{filename}"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {path}")


def plot_user_activity(
    activity_pd: pd.DataFrame,
    cap:         int = 50,
    filename:    str = "user_activity.png",
) -> None:
    """
    Histogram of the number of distinct games played per user.
    Reveals the long-tail nature of user engagement.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        activity_pd["games_played"].clip(upper=cap),
        bins=50, color="mediumpurple", edgecolor="white"
    )
    ax.set_title(
        f"User Activity — Distinct Games Played (capped at {cap})",
        fontsize=13
    )
    ax.set_xlabel("Distinct Games Played per User")
    ax.set_ylabel("Number of Users")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/{filename}"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {path}")


# ── Model Results ─────────────────────────────────────────────────────────────

def plot_rmse_comparison(
    mean_rmse:  float,
    base_rmse:  float,
    best_rmse:  float,
    filename:   str = "rmse_comparison.png",
) -> None:
    """
    Bar chart comparing RMSE across three models:
      1. Mean baseline (trivial predictor)
      2. ALS with default params
      3. Best ALS from grid search
    """
    labels  = ["Mean\nBaseline", "ALS\nDefaults", "ALS Best\n(Grid Search)"]
    values  = [mean_rmse, base_rmse, best_rmse]
    colours = ["#d9534f", "#f0ad4e", "#5cb85c"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colours, edgecolor="white", width=0.5)
    ax.bar_label(bars, fmt="%.4f", padding=5, fontsize=11)
    ax.set_title("RMSE Comparison", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("RMSE (log-hours scale)")
    ax.set_ylim(0, max(values) * 1.25)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/{filename}"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {path}")


def plot_most_recommended(
    recs_pd:  pd.DataFrame,
    name_col: str = "game_name",
    cnt_col:  str = "times_recommended",
    filename: str = "most_recommended_games.png",
) -> None:
    """
    Horizontal bar chart of games most frequently recommended to users.
    High concentration on a few titles indicates a popularity bias in the
    model — a useful diagnostic for real-world deployment.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(
        recs_pd[name_col][::-1],
        recs_pd[cnt_col][::-1],
        color="mediumpurple", edgecolor="white"
    )
    ax.set_title(
        "Most-Recommended Games Across All Users",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Times Recommended")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/{filename}"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {path}")
