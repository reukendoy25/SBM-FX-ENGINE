"""
Anomaly Visualization Utilities
=================================
Generates publication-quality plots for FX anomaly detection results:
- Time series with anomaly overlays
- Anomaly score distributions
- Multi-currency correlation heatmaps
"""

import logging
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

# Professional styling
plt.style.use("seaborn-v0_8-darkgrid")
COLORS = {
    "normal": "#2196F3",
    "anomaly": "#FF1744",
    "background": "#0D1117",
    "grid": "#21262D",
    "text": "#C9D1D9",
    "accent1": "#58A6FF",
    "accent2": "#F78166",
    "accent3": "#7EE787",
    "accent4": "#D2A8FF",
}


def setup_dark_style():
    """Configure dark-mode matplotlib styling."""
    plt.rcParams.update({
        "figure.facecolor": COLORS["background"],
        "axes.facecolor": COLORS["background"],
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["text"],
        "text.color": COLORS["text"],
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
        "grid.color": COLORS["grid"],
        "grid.alpha": 0.3,
        "font.family": "sans-serif",
        "font.size": 11,
    })


def plot_fx_with_anomalies(fx_data: pd.DataFrame,
                            anomalies: pd.DataFrame,
                            currency: str,
                            save_path: str = None,
                            figsize: tuple = (16, 8)):
    """
    Plot FX time series with anomaly markers overlaid.

    Args:
        fx_data: DataFrame with DatetimeIndex and currency columns
        anomalies: DataFrame from AnomalyDetector.predict() with anomaly_label
        currency: Currency code to plot (column in fx_data)
        save_path: Path to save the figure
        figsize: Figure dimensions
    """
    setup_dark_style()
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, height_ratios=[3, 1],
        gridspec_kw={"hspace": 0.15}
    )

    # --- Top panel: FX Rate with Anomaly Markers ---
    ax1.plot(
        fx_data.index, fx_data[currency],
        color=COLORS["accent1"], linewidth=1.2, alpha=0.9,
        label=f"{currency}/EUR Rate"
    )

    # Overlay anomaly points
    anomaly_mask = anomalies["anomaly_label"] == -1
    anomaly_dates = anomalies[anomaly_mask].index
    common_dates = anomaly_dates.intersection(fx_data.index)

    if len(common_dates) > 0:
        ax1.scatter(
            common_dates,
            fx_data.loc[common_dates, currency],
            color=COLORS["anomaly"],
            s=60, zorder=5, alpha=0.8,
            label=f"Anomalies ({len(common_dates)})",
            edgecolors="white", linewidth=0.5,
        )

    ax1.set_title(
        f"{currency}/EUR Exchange Rate — Isolation Forest Anomaly Detection",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax1.set_ylabel("Exchange Rate", fontsize=12)
    ax1.legend(loc="upper left", fontsize=10, framealpha=0.7)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())

    # --- Bottom panel: Anomaly Score ---
    common_idx = anomalies.index.intersection(fx_data.index)
    scores = anomalies.loc[common_idx, "anomaly_score_normalized"]

    ax2.fill_between(
        common_idx, 0, scores,
        alpha=0.4, color=COLORS["accent2"], label="Anomaly Score"
    )
    ax2.axhline(y=0.8, color=COLORS["anomaly"], linestyle="--",
                alpha=0.6, label="Threshold (0.8)")
    ax2.set_ylabel("Anomaly Score", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc="upper left", fontsize=10, framealpha=0.7)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=COLORS["background"])
        logger.info(f"Anomaly plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)
    return fig


def plot_anomaly_score_distribution(anomalies: pd.DataFrame,
                                     save_path: str = None,
                                     figsize: tuple = (12, 6)):
    """
    Plot the distribution of anomaly scores.
    """
    setup_dark_style()
    fig, ax = plt.subplots(figsize=figsize)

    scores = anomalies["anomaly_score_normalized"]

    ax.hist(
        scores, bins=50, color=COLORS["accent1"],
        alpha=0.7, edgecolor=COLORS["background"],
        label="All samples"
    )

    # Highlight anomalous portion
    anomaly_scores = scores[anomalies["anomaly_label"] == -1]
    ax.hist(
        anomaly_scores, bins=50, color=COLORS["anomaly"],
        alpha=0.8, edgecolor=COLORS["background"],
        label=f"Anomalies (n={len(anomaly_scores)})"
    )

    ax.set_title(
        "Distribution of Isolation Forest Anomaly Scores",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.set_xlabel("Normalized Anomaly Score (0=Normal, 1=Anomalous)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend(fontsize=10, framealpha=0.7)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=COLORS["background"])
        logger.info(f"Score distribution plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)
    return fig


def plot_correlation_heatmap(fx_data: pd.DataFrame,
                              save_path: str = None,
                              figsize: tuple = (10, 8)):
    """
    Plot correlation heatmap of FX rates.
    """
    setup_dark_style()
    fig, ax = plt.subplots(figsize=figsize)

    # Compute log-return correlations (more meaningful than price correlations)
    returns = np.log(fx_data / fx_data.shift(1)).dropna()
    corr = returns.corr()

    # Custom colormap
    cmap = plt.cm.RdYlBu_r

    im = ax.imshow(corr.values, cmap=cmap, aspect="auto",
                    vmin=-1, vmax=1)

    # Labels
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, fontsize=11, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns, fontsize=11)

    # Add correlation values
    for i in range(len(corr)):
        for j in range(len(corr)):
            text_color = "white" if abs(corr.values[i, j]) > 0.5 else COLORS["text"]
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", color=text_color, fontsize=10)

    ax.set_title(
        "FX Log-Return Correlation Matrix",
        fontsize=14, fontweight="bold", pad=15,
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation", fontsize=12)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=COLORS["background"])
        logger.info(f"Correlation heatmap saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)
    return fig


def plot_forecast_results(actual: np.ndarray, predicted: np.ndarray,
                           dates: list = None, currency: str = "USD",
                           save_path: str = None,
                           figsize: tuple = (16, 8)):
    """
    Plot LSTM forecast vs actual rates.
    """
    setup_dark_style()
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, height_ratios=[3, 1],
        gridspec_kw={"hspace": 0.15}
    )

    x = dates if dates else range(len(actual))

    # --- Top panel: Actual vs Predicted ---
    ax1.plot(x, actual, color=COLORS["accent1"], linewidth=1.5,
             alpha=0.9, label="Actual Rate")
    ax1.plot(x, predicted, color=COLORS["accent3"], linewidth=1.5,
             alpha=0.8, label="LSTM Forecast", linestyle="--")

    ax1.fill_between(
        x, actual, predicted,
        alpha=0.15, color=COLORS["accent4"]
    )

    ax1.set_title(
        f"{currency}/EUR — LSTM Rate Forecast vs Actual",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax1.set_ylabel("Exchange Rate", fontsize=12)
    ax1.legend(loc="upper left", fontsize=10, framealpha=0.7)

    # --- Bottom panel: Prediction Error ---
    error = actual - predicted
    ax2.bar(x, error, color=np.where(error >= 0, COLORS["accent3"], COLORS["anomaly"]),
            alpha=0.6, width=1)
    ax2.axhline(y=0, color=COLORS["text"], linewidth=0.5)
    ax2.set_ylabel("Error", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=COLORS["background"])
        logger.info(f"Forecast plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)
    return fig


def plot_sentiment_timeline(sentiment_df: pd.DataFrame,
                             save_path: str = None,
                             figsize: tuple = (16, 6)):
    """
    Plot FinBERT sentiment scores over time.
    """
    setup_dark_style()
    fig, ax = plt.subplots(figsize=figsize)

    colors_map = {
        "positive": COLORS["accent3"],
        "negative": COLORS["anomaly"],
        "neutral": COLORS["accent4"],
    }

    for _, row in sentiment_df.iterrows():
        date = row.name if hasattr(row, 'name') else row.get("date")
        color = colors_map.get(row.get("label", "neutral"), COLORS["text"])
        ax.bar(date, row["sentiment"], color=color, alpha=0.7, width=20)

    ax.axhline(y=0, color=COLORS["text"], linewidth=0.5, linestyle="--")
    ax.set_title(
        "Central Bank Communication Sentiment (FinBERT)",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.set_ylabel("Sentiment Score (-1 to +1)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylim(-1.2, 1.2)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["accent3"], alpha=0.7, label="Positive"),
        Patch(facecolor=COLORS["anomaly"], alpha=0.7, label="Negative"),
        Patch(facecolor=COLORS["accent4"], alpha=0.7, label="Neutral"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10,
              framealpha=0.7)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=COLORS["background"])
        logger.info(f"Sentiment timeline saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)
    return fig
