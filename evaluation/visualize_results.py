"""
visualize_results.py
---------------------
Generates publication-ready plots from evaluation results.

Plots:
  1. Error vs True Offset  (grouped bar chart)
  2. Confidence vs Error   (scatter + regression)
  3. Audio-Video Offset Difference Histogram
  4. Runtime Comparison    (bar chart)
  5. Error Distribution    (boxplot by method and offset)
  6. Resource Usage        (CPU & memory bar charts)
  7. Motion Before/After   (signal overlay per case)
  8. Sync Timelines        (per-video timeline with offset arrows)

Output:
  evaluation/plots/*.png
"""

import os
import glob
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
DIAGNOSTICS_DIR = os.path.join(BASE_DIR, "diagnostics")

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
# Plot 5: Error Distribution Boxplot
# ──────────────────────────────────────────────────────────────────────

def plot_error_distribution(df: pd.DataFrame, output_dir: str):
    """Side-by-side boxplots of absolute error grouped by true offset and method."""
    offsets = sorted(df["true_offset_ms"].unique())
    audio_data = []
    visual_data = []

    for off in offsets:
        a = df[(df["method_type"] == "audio") & (df["true_offset_ms"] == off)]["absolute_error_ms"].values
        v = df[(df["method_type"] == "visual") & (df["true_offset_ms"] == off)]["absolute_error_ms"].values
        audio_data.append(a)
        visual_data.append(v)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    positions = np.arange(len(offsets))
    box_width = 0.3

    bp_audio = ax.boxplot(
        audio_data,
        positions=positions - box_width / 2,
        widths=box_width * 0.8,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 1.5},
    )
    bp_visual = ax.boxplot(
        visual_data,
        positions=positions + box_width / 2,
        widths=box_width * 0.8,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 1.5},
    )

    for box in bp_audio["boxes"]:
        box.set_facecolor(COLORS["audio"])
        box.set_alpha(0.8)
    for box in bp_visual["boxes"]:
        box.set_facecolor(COLORS["visual"])
        box.set_alpha(0.8)

    # Overlay individual data points
    for i, off in enumerate(offsets):
        a = audio_data[i]
        v = visual_data[i]
        if len(a) > 0:
            jitter = np.random.default_rng(42).uniform(-0.06, 0.06, size=len(a))
            ax.scatter(
                np.full_like(a, i - box_width / 2) + jitter,
                a, color=COLORS["audio"], s=18, alpha=0.6,
                edgecolors="white", linewidth=0.3, zorder=3,
            )
        if len(v) > 0:
            jitter = np.random.default_rng(99).uniform(-0.06, 0.06, size=len(v))
            ax.scatter(
                np.full_like(v, i + box_width / 2) + jitter,
                v, color=COLORS["visual"], s=18, alpha=0.6,
                edgecolors="white", linewidth=0.3, zorder=3,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{int(o):+d}" for o in offsets])
    ax.set_xlabel("True Offset (ms)")
    ax.set_ylabel("Absolute Error (ms)")
    ax.set_title("Error Distribution by Method and Offset")
    ax.set_ylim(bottom=0)

    # Legend
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor=COLORS["audio"], label="Audio"),
            Patch(facecolor=COLORS["visual"], label="Visual"),
        ],
        loc="upper right",
    )

    path = os.path.join(output_dir, "error_distribution_boxplot.png")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    logger.info("Saved: %s", path)


# ──────────────────────────────────────────────────────────────────────
# Plot 6: Resource Usage
# ──────────────────────────────────────────────────────────────────────

def plot_resource_usage(df: pd.DataFrame, output_dir: str):
    """Dual bar chart comparing peak CPU% and peak memory between methods."""
    if "peak_cpu_percent" not in df.columns or "peak_memory_mb" not in df.columns:
        logger.info("No resource usage columns in results; skipping resource plot.")
        return

    fig, (ax_cpu, ax_mem) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    for ax, col, ylabel, title, fmt in [
        (ax_cpu, "peak_cpu_percent", "Peak CPU (%)", "Peak CPU Usage by Method", "{:.1f}%"),
        (ax_mem, "peak_memory_mb", "Peak Memory (MB)", "Peak Memory Usage by Method", "{:.0f} MB"),
    ]:
        methods = []
        means = []
        stds = []
        colors = []

        for method in ("audio", "visual"):
            mdf = df[df["method_type"] == method]
            if mdf.empty:
                continue
            vals = mdf[col].values.astype(float)
            methods.append(method.capitalize())
            means.append(np.mean(vals))
            stds.append(np.std(vals))
            colors.append(COLORS[method])

        if not methods:
            continue

        bars = ax.bar(
            methods, means, yerr=stds,
            color=colors, edgecolor="white", linewidth=0.5,
            capsize=5, error_kw={"linewidth": 1.2},
        )

        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                fmt.format(mean),
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(bottom=0)

    path = os.path.join(output_dir, "resource_usage.png")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    logger.info("Saved: %s", path)


# ──────────────────────────────────────────────────────────────────────
# Plot 7: Motion Before/After Overlay
# ──────────────────────────────────────────────────────────────────────

def plot_motion_before_after(df: pd.DataFrame, output_dir: str,
                             diagnostics_dir: str = DIAGNOSTICS_DIR):
    """
    For each .npz in diagnostics/, plot the original and synthetic motion
    signals before alignment (raw) and after alignment (shifted by the
    estimated offset).  Produces one PNG per test case.
    """
    npz_files = sorted(glob.glob(os.path.join(diagnostics_dir, "*.npz")))
    if not npz_files:
        logger.info("No diagnostics .npz files found; skipping before/after plots.")
        return

    ba_dir = os.path.join(output_dir, "before_after")
    os.makedirs(ba_dir, exist_ok=True)

    for npz_path in npz_files:
        case_label = os.path.splitext(os.path.basename(npz_path))[0]

        try:
            data = np.load(npz_path)
            original = data["original"]
            synthetic = data["synthetic"]
            fps = float(data["fps"])
            est_offset_ms = float(data["estimated_offset_ms"])
        except Exception:
            logger.warning("Could not load %s; skipping.", npz_path)
            continue

        # Parse true offset from case_label (e.g. "3_video_b_offset+500")
        true_offset_ms = 0.0
        try:
            parts = case_label.rsplit("_offset", 1)
            if len(parts) == 2:
                true_offset_ms = float(parts[1])
        except ValueError:
            pass

        t_orig = np.arange(len(original)) / fps
        t_synth = np.arange(len(synthetic)) / fps

        # "After" = shift synthetic backwards by the estimated offset
        shift_sec = est_offset_ms / 1000.0
        t_synth_aligned = t_synth - shift_sec

        fig, (ax_before, ax_after) = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

        # Before alignment
        ax_before.plot(t_orig, original, color=COLORS["audio"], linewidth=0.8,
                       label="Original", alpha=0.85)
        ax_before.plot(t_synth, synthetic, color=COLORS["visual"], linewidth=0.8,
                       label=f"Synthetic (true offset {true_offset_ms:+.0f} ms)", alpha=0.85)
        ax_before.set_ylabel("Motion Energy")
        ax_before.set_title(f"{case_label} — Before Alignment")
        ax_before.legend(loc="upper right", fontsize=9)

        # After alignment
        ax_after.plot(t_orig, original, color=COLORS["audio"], linewidth=0.8,
                      label="Original", alpha=0.85)
        ax_after.plot(t_synth_aligned, synthetic, color=COLORS["visual"], linewidth=0.8,
                      label=f"Synthetic (shifted by est. {est_offset_ms:+.0f} ms)", alpha=0.85)
        ax_after.set_xlabel("Time (s)")
        ax_after.set_ylabel("Motion Energy")
        ax_after.set_title(f"{case_label} — After Alignment")
        ax_after.legend(loc="upper right", fontsize=9)

        path = os.path.join(ba_dir, f"{case_label}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=DPI)
        plt.close(fig)

    logger.info("Saved %d before/after plots to %s", len(npz_files), ba_dir)


# ──────────────────────────────────────────────────────────────────────
# Plot 8: Sync Timelines
# ──────────────────────────────────────────────────────────────────────

def plot_sync_timelines(df: pd.DataFrame, output_dir: str):
    """
    For each video_id, produce a timeline diagram showing:
      - A horizontal bar for the original video
      - A bar for the synthetic (shifted by true offset)
      - Arrows showing where audio and visual sync estimated the offset
      - Annotation of padding / trimming applied
    """
    tl_dir = os.path.join(output_dir, "timelines")
    os.makedirs(tl_dir, exist_ok=True)

    # Group by video_id + true_offset_ms for per-case timelines
    grouped = df.groupby(["video_id", "true_offset_ms"])

    for (video_id, true_offset_ms), gdf in grouped:
        true_offset_sec = true_offset_ms / 1000.0

        # Get per-method estimated offsets
        method_ests = {}
        for _, row in gdf.iterrows():
            method_ests[row["method_type"]] = row["estimated_offset_ms"] / 1000.0

        # Assume ~30s video length for bar width (use actual if available)
        vid_len = 30.0
        if "video_length_sec" in gdf.columns:
            vl = gdf["video_length_sec"].iloc[0]
            if pd.notna(vl) and float(vl) > 0:
                vid_len = float(vl)

        fig, ax = plt.subplots(figsize=(12, 3.5))

        bar_height = 0.35
        y_orig = 1.0
        y_synth = 0.0

        # Original bar (always starts at 0)
        ax.barh(y_orig, vid_len, height=bar_height, left=0,
                color="#78909C", edgecolor="white", linewidth=0.5, label="Original")
        ax.text(vid_len / 2, y_orig, "Original", ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")

        # Synthetic bar (shifted by true offset)
        synth_start = true_offset_sec
        ax.barh(y_synth, vid_len, height=bar_height, left=synth_start,
                color="#B0BEC5", edgecolor="white", linewidth=0.5, label="Synthetic")
        ax.text(synth_start + vid_len / 2, y_synth, "Synthetic", ha="center",
                va="center", fontsize=9, fontweight="bold", color="#37474F")

        # Arrow for true offset
        if abs(true_offset_sec) > 0.001:
            ax.annotate(
                "", xy=(synth_start, y_synth + bar_height / 2 + 0.05),
                xytext=(0, y_synth + bar_height / 2 + 0.05),
                arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=2),
            )
            mid_arrow = synth_start / 2
            action = "pad" if true_offset_sec > 0 else "trim"
            ax.text(mid_arrow, y_synth + bar_height / 2 + 0.15,
                    f"True: {true_offset_ms:+.0f} ms ({action})",
                    ha="center", va="bottom", fontsize=8, color="#D32F2F",
                    fontweight="bold")

        # Arrows for estimated offsets (audio & visual)
        y_arrow_base = y_orig + bar_height / 2 + 0.05
        for i, (method, est_sec) in enumerate(method_ests.items()):
            color = COLORS.get(method, "#555")
            y_arrow = y_arrow_base + i * 0.2

            if abs(est_sec) > 0.001:
                ax.annotate(
                    "", xy=(est_sec, y_arrow), xytext=(0, y_arrow),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.8,
                                    linestyle="--"),
                )
                action = "pad" if est_sec > 0 else "trim"
                ax.text(est_sec / 2, y_arrow + 0.06,
                        f"{method.capitalize()}: {est_sec * 1000:+.0f} ms ({action})",
                        ha="center", va="bottom", fontsize=8, color=color)
            else:
                ax.text(0.5, y_arrow + 0.06,
                        f"{method.capitalize()}: 0 ms (no shift)",
                        ha="center", va="bottom", fontsize=8, color=color,
                        transform=ax.get_yaxis_transform())

        # Styling
        ax.set_yticks([y_synth, y_orig])
        ax.set_yticklabels(["Synthetic", "Original"])
        ax.set_xlabel("Time (s)")
        ax.set_title(f"Sync Timeline — {video_id} (true offset {true_offset_ms:+.0f} ms)")

        # Extend x-axis to show negative offsets
        x_min = min(0, synth_start, *method_ests.values()) - 1
        x_max = max(vid_len, synth_start + vid_len,
                    *[v + vid_len for v in method_ests.values()]) + 1
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.5, y_orig + bar_height / 2 + 0.2 * len(method_ests) + 0.4)

        # Zero line
        ax.axvline(0, color="#999", linewidth=0.8, linestyle=":", alpha=0.6)

        case_label = f"{video_id}_offset{int(true_offset_ms):+d}"
        path = os.path.join(tl_dir, f"{case_label}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=DPI)
        plt.close(fig)

    logger.info("Saved %d timeline diagrams to %s", len(grouped), tl_dir)


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
    plot_error_distribution(df, plots_dir)
    plot_resource_usage(df, plots_dir)
    plot_motion_before_after(df, plots_dir)
    plot_sync_timelines(df, plots_dir)

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
