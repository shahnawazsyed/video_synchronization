"""
visualize_results.py
---------------------
Generates publication-ready plots from evaluation results.

Plots:
  1. Error vs True Offset  (grouped bar chart)
  2. Confidence vs Error   (scatter + regression)
  3. Audio-Video Offset Difference Histogram
  4. Runtime Comparison    (bar chart)

Output:
  evaluation/plots/*.png
"""

import os
import logging

import numpy as np
import pandas as pd

# Force non-interactive backend (safe for scripts / background execution)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(BASE_DIR, "results", "results.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# Styling
COLORS = {"audio": "#2196F3", "visual": "#FF9800"}
DPI = 300
FIGSIZE_WIDE = (10, 5)
FIGSIZE_SQUARE = (7, 6)


# ──────────────────────────────────────────────────────────────────────
# Style setup
# ──────────────────────────────────────────────────────────────────────

def _apply_style():
    """Apply a clean, publication-ready matplotlib style."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ──────────────────────────────────────────────────────────────────────
# Plot 1: Error vs True Offset
# ──────────────────────────────────────────────────────────────────────

def plot_error_vs_offset(df: pd.DataFrame, output_dir: str):
    """Grouped bar chart of MAE per true offset, split by method."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    offsets = sorted(df["true_offset_ms"].unique())
    x = np.arange(len(offsets))
    width = 0.35

    for i, method in enumerate(("audio", "visual")):
        mdf = df[df["method_type"] == method]
        maes = []
        for off in offsets:
            subset = mdf[mdf["true_offset_ms"] == off]
            maes.append(subset["absolute_error_ms"].mean() if not subset.empty else 0)
        ax.bar(
            x + (i - 0.5) * width,
            maes,
            width,
            label=method.capitalize(),
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("True Offset (ms)")
    ax.set_ylabel("Mean Absolute Error (ms)")
    ax.set_title("Synchronization Error by Offset Magnitude")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(o):+d}" for o in offsets])
    ax.legend()
    ax.set_ylim(bottom=0)

    path = os.path.join(output_dir, "error_vs_offset.png")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    logger.info("Saved: %s", path)


# ──────────────────────────────────────────────────────────────────────
# Plot 2: Confidence vs Error
# ──────────────────────────────────────────────────────────────────────

def plot_confidence_vs_error(df: pd.DataFrame, output_dir: str):
    """Scatter plot of confidence vs absolute error with regression line."""
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)

    for method in ("audio", "visual"):
        mdf = df[df["method_type"] == method]
        if mdf.empty:
            continue
        ax.scatter(
            mdf["confidence_score"],
            mdf["absolute_error_ms"],
            label=method.capitalize(),
            alpha=0.6,
            s=40,
            color=COLORS[method],
            edgecolors="white",
            linewidth=0.5,
        )

        # Regression line
        confs = mdf["confidence_score"].values.astype(float)
        errors = mdf["absolute_error_ms"].values.astype(float)
        if len(confs) >= 2 and np.std(confs) > 1e-12:
            z = np.polyfit(confs, errors, 1)
            p = np.poly1d(z)
            xs = np.linspace(confs.min(), confs.max(), 50)
            ax.plot(xs, p(xs), "--", color=COLORS[method], alpha=0.7, linewidth=1.5)
            r = np.corrcoef(confs, errors)[0, 1]
            ax.annotate(
                f"{method}: r={r:.3f}",
                xy=(0.02, 0.98 if method == "audio" else 0.93),
                xycoords="axes fraction",
                fontsize=9,
                color=COLORS[method],
                verticalalignment="top",
            )

    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Absolute Error (ms)")
    ax.set_title("Confidence vs Synchronization Error")
    ax.legend()

    path = os.path.join(output_dir, "confidence_vs_error.png")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    logger.info("Saved: %s", path)


# ──────────────────────────────────────────────────────────────────────
# Plot 3: Audio-Video Offset Difference Histogram
# ──────────────────────────────────────────────────────────────────────

def plot_audio_video_diff_histogram(df: pd.DataFrame, output_dir: str):
    """Histogram of |audio_estimate − video_estimate| per test case."""
    audio_df = (
        df[df["method_type"] == "audio"]
        .set_index(["video_id", "true_offset_ms"])["estimated_offset_ms"]
    )
    visual_df = (
        df[df["method_type"] == "visual"]
        .set_index(["video_id", "true_offset_ms"])["estimated_offset_ms"]
    )

    common = audio_df.index.intersection(visual_df.index)
    if len(common) == 0:
        logger.warning("No overlapping cases for audio-video diff histogram; skipping.")
        return

    diffs = np.abs(audio_df.loc[common].values - visual_df.loc[common].values)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.hist(diffs, bins=20, color="#7E57C2", edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.axvline(
        np.mean(diffs), color="#D32F2F", linestyle="--", linewidth=1.5,
        label=f"Mean = {np.mean(diffs):.1f} ms",
    )
    ax.axvline(
        np.median(diffs), color="#388E3C", linestyle="--", linewidth=1.5,
        label=f"Median = {np.median(diffs):.1f} ms",
    )

    ax.set_xlabel("|Audio Estimate − Visual Estimate| (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Audio–Visual Offset Differences")
    ax.legend()

    path = os.path.join(output_dir, "audio_video_diff_histogram.png")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    logger.info("Saved: %s", path)


# ──────────────────────────────────────────────────────────────────────
# Plot 4: Runtime Comparison
# ──────────────────────────────────────────────────────────────────────

def plot_runtime_comparison(df: pd.DataFrame, output_dir: str):
    """Bar chart of mean runtime per method."""
    fig, ax = plt.subplots(figsize=(6, 5))

    methods = []
    means = []
    stds = []
    colors = []

    for method in ("audio", "visual"):
        mdf = df[df["method_type"] == method]
        if mdf.empty:
            continue
        methods.append(method.capitalize())
        means.append(mdf["runtime_seconds"].mean())
        stds.append(mdf["runtime_seconds"].std())
        colors.append(COLORS[method])

    bars = ax.bar(
        methods, means, yerr=stds,
        color=colors, edgecolor="white", linewidth=0.5,
        capsize=5, error_kw={"linewidth": 1.2},
    )

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{mean:.2f}s",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Mean Synchronization Runtime by Method")
    ax.set_ylim(bottom=0)

    path = os.path.join(output_dir, "runtime_comparison.png")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    logger.info("Saved: %s", path)


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def generate_plots(
    results_csv: str = RESULTS_CSV,
    plots_dir: str = PLOTS_DIR,
):
    """Generate all evaluation plots from results.csv."""
    _apply_style()
    os.makedirs(plots_dir, exist_ok=True)

    df = pd.read_csv(results_csv)
    if df.empty:
        raise ValueError(f"No rows in {results_csv}")

    logger.info("Generating plots from %d result rows ...", len(df))

    plot_error_vs_offset(df, plots_dir)
    plot_confidence_vs_error(df, plots_dir)
    plot_audio_video_diff_histogram(df, plots_dir)
    plot_runtime_comparison(df, plots_dir)

    logger.info("All plots saved to %s", plots_dir)


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    generate_plots()
