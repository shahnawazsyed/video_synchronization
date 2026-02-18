"""
compute_metrics.py
-------------------
Reads results.csv and computes accuracy, cross-method, confidence,
efficiency, and grouped metrics.

Output:
  evaluation/metrics/metrics_summary.json
  Printed summary table to stdout
"""

import os
import json
import logging
import math

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(BASE_DIR, "results", "results.csv")
METRICS_DIR = os.path.join(BASE_DIR, "metrics")

AGREEMENT_THRESHOLD_MS = 100.0  # threshold for "within X ms" reporting
CONFIDENCE_FILTER_QUANTILE = 0.20  # bottom 20 % confidence


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient, returning 0.0 for degenerate cases."""
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _safe_float(val) -> float:
    """Convert to float, replacing NaN/Inf with 0.0."""
    v = float(val)
    return 0.0 if (math.isnan(v) or math.isinf(v)) else round(v, 4)


# ──────────────────────────────────────────────────────────────────────
# Core metrics computation
# ──────────────────────────────────────────────────────────────────────

def compute_metrics(
    results_csv: str = RESULTS_CSV,
    metrics_dir: str = METRICS_DIR,
) -> dict:
    """
    Compute all evaluation metrics from results.csv.

    Returns the metrics dict and writes metrics_summary.json.
    """
    os.makedirs(metrics_dir, exist_ok=True)
    df = pd.read_csv(results_csv)

    if df.empty:
        raise ValueError(f"No rows in {results_csv}")

    metrics = {}

    # ── 1. Accuracy metrics ──────────────────────────────────────────
    accuracy = {}

    for method in ("audio", "visual"):
        mdf = df[df["method_type"] == method]
        if mdf.empty:
            continue
        errors = mdf["absolute_error_ms"].values
        accuracy[method] = {
            "mae_ms": _safe_float(np.mean(errors)),
            "rmse_ms": _safe_float(np.sqrt(np.mean(errors ** 2))),
            "median_error_ms": _safe_float(np.median(errors)),
            "max_error_ms": _safe_float(np.max(errors)),
            "count": int(len(errors)),
        }

        # MAE per offset magnitude
        per_offset = {}
        for offset_val, group in mdf.groupby("true_offset_ms"):
            per_offset[str(int(offset_val))] = _safe_float(
                group["absolute_error_ms"].mean()
            )
        accuracy[method]["mae_per_offset_ms"] = per_offset

    metrics["accuracy"] = accuracy

    # ── 2. Cross-method agreement ────────────────────────────────────
    cross_method = {}
    audio_df = df[df["method_type"] == "audio"].set_index(["video_id", "true_offset_ms"])
    visual_df = df[df["method_type"] == "visual"].set_index(["video_id", "true_offset_ms"])

    common_idx = audio_df.index.intersection(visual_df.index)
    if len(common_idx) > 0:
        audio_est = audio_df.loc[common_idx, "estimated_offset_ms"].values.astype(float)
        visual_est = visual_df.loc[common_idx, "estimated_offset_ms"].values.astype(float)
        diffs = np.abs(audio_est - visual_est)

        cross_method["mean_audio_video_diff_ms"] = _safe_float(np.mean(diffs))
        cross_method["median_audio_video_diff_ms"] = _safe_float(np.median(diffs))
        cross_method["pct_within_threshold"] = _safe_float(
            100.0 * np.mean(diffs < AGREEMENT_THRESHOLD_MS)
        )
        cross_method["threshold_ms"] = AGREEMENT_THRESHOLD_MS
        cross_method["n_pairs"] = int(len(common_idx))
    else:
        cross_method["note"] = "No overlapping audio/visual cases for comparison"

    metrics["cross_method_agreement"] = cross_method

    # ── 3. Confidence validation ─────────────────────────────────────
    confidence_metrics = {}

    for method in ("audio", "visual"):
        mdf = df[df["method_type"] == method]
        if mdf.empty or mdf["confidence_score"].isna().all():
            continue

        confs = mdf["confidence_score"].values.astype(float)
        errors = mdf["absolute_error_ms"].values.astype(float)

        corr = _safe_pearson(confs, errors)

        # MAE before filtering
        mae_all = _safe_float(np.mean(errors))

        # MAE after removing bottom 20 % confidence
        threshold = np.quantile(confs, CONFIDENCE_FILTER_QUANTILE)
        mask = confs >= threshold
        if mask.sum() > 0:
            mae_filtered = _safe_float(np.mean(errors[mask]))
            n_filtered = int(mask.sum())
        else:
            mae_filtered = mae_all
            n_filtered = int(len(errors))

        confidence_metrics[method] = {
            "pearson_confidence_vs_error": _safe_float(corr),
            "mae_all_ms": mae_all,
            "mae_filtered_ms": mae_filtered,
            "confidence_threshold": _safe_float(threshold),
            "n_after_filter": n_filtered,
            "n_total": int(len(errors)),
        }

    metrics["confidence_validation"] = confidence_metrics

    # ── 4. Efficiency ────────────────────────────────────────────────
    efficiency = {}

    for method in ("audio", "visual"):
        mdf = df[df["method_type"] == method]
        if mdf.empty:
            continue
        runtimes = mdf["runtime_seconds"].values.astype(float)
        efficiency[method] = {
            "mean_runtime_seconds": _safe_float(np.mean(runtimes)),
            "median_runtime_seconds": _safe_float(np.median(runtimes)),
            "total_runtime_seconds": _safe_float(np.sum(runtimes)),
        }

        # Runtime per minute of video
        if "video_length_sec" in mdf.columns:
            lengths = mdf["video_length_sec"].values.astype(float)
            valid = lengths > 0
            if valid.any():
                per_minute = runtimes[valid] / (lengths[valid] / 60.0)
                efficiency[method]["runtime_per_video_minute"] = _safe_float(
                    np.mean(per_minute)
                )

    metrics["efficiency"] = efficiency

    # ── 5. Grouped metrics (sensitivity tags) ────────────────────────
    grouped = {}
    tag_columns = ["video_length_sec", "motion_level", "audio_energy_level"]

    for tag_col in tag_columns:
        if tag_col not in df.columns or df[tag_col].isna().all():
            continue

        tag_group = {}
        # Bin continuous values into categories
        values = df[tag_col].astype(float)
        if tag_col == "video_length_sec":
            bins = [0, 30, 60, 120, float("inf")]
            labels = ["<30s", "30-60s", "60-120s", ">120s"]
        else:
            bins = [0, 0.25, 0.5, 0.75, float("inf")]
            labels = ["low", "medium", "high", "very_high"]

        df["_tag_bin"] = pd.cut(values, bins=bins, labels=labels, right=False)

        for method in ("audio", "visual"):
            mdf = df[df["method_type"] == method]
            method_group = {}
            for bin_label, group in mdf.groupby("_tag_bin", observed=True):
                errors = group["absolute_error_ms"].values
                method_group[str(bin_label)] = {
                    "mae_ms": _safe_float(np.mean(errors)),
                    "rmse_ms": _safe_float(np.sqrt(np.mean(errors ** 2))),
                    "count": int(len(errors)),
                }
            tag_group[method] = method_group

        grouped[tag_col] = tag_group
        df.drop(columns=["_tag_bin"], inplace=True)

    metrics["grouped_by_tag"] = grouped

    # ── Write output ─────────────────────────────────────────────────
    output_path = os.path.join(metrics_dir, "metrics_summary.json")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Metrics written to %s", output_path)

    # ── Print summary table ──────────────────────────────────────────
    _print_summary(metrics)

    return metrics


def _print_summary(metrics: dict):
    """Print a human-readable summary table."""
    print("\n" + "=" * 70)
    print("  EVALUATION METRICS SUMMARY")
    print("=" * 70)

    # Accuracy
    if "accuracy" in metrics:
        print("\n── Accuracy ───────────────────────────────────────────")
        for method, vals in metrics["accuracy"].items():
            print(f"  [{method.upper()}]")
            print(f"    MAE:    {vals['mae_ms']:.2f} ms")
            print(f"    RMSE:   {vals['rmse_ms']:.2f} ms")
            print(f"    Median: {vals['median_error_ms']:.2f} ms")
            print(f"    Max:    {vals['max_error_ms']:.2f} ms")
            print(f"    Count:  {vals['count']}")
            if "mae_per_offset_ms" in vals:
                print("    Per-offset MAE:")
                for offset, mae in sorted(vals["mae_per_offset_ms"].items(), key=lambda x: float(x[0])):
                    print(f"      {offset:>6s} ms → {mae:.2f} ms error")

    # Cross-method
    if "cross_method_agreement" in metrics:
        cm = metrics["cross_method_agreement"]
        print("\n── Cross-Method Agreement ─────────────────────────────")
        if "mean_audio_video_diff_ms" in cm:
            print(f"  Mean |audio − visual| diff: {cm['mean_audio_video_diff_ms']:.2f} ms")
            print(f"  % within {cm['threshold_ms']:.0f}ms:         {cm['pct_within_threshold']:.1f}%")
        else:
            print(f"  {cm.get('note', 'N/A')}")

    # Confidence
    if "confidence_validation" in metrics:
        print("\n── Confidence Validation ──────────────────────────────")
        for method, vals in metrics["confidence_validation"].items():
            print(f"  [{method.upper()}]")
            print(f"    Pearson(confidence, error): {vals['pearson_confidence_vs_error']:.4f}")
            print(f"    MAE (all):                  {vals['mae_all_ms']:.2f} ms")
            print(f"    MAE (filtered, top 80%):    {vals['mae_filtered_ms']:.2f} ms")

    # Efficiency
    if "efficiency" in metrics:
        print("\n── Efficiency ─────────────────────────────────────────")
        for method, vals in metrics["efficiency"].items():
            print(f"  [{method.upper()}]")
            print(f"    Mean runtime: {vals['mean_runtime_seconds']:.2f}s")
            if "runtime_per_video_minute" in vals:
                print(f"    Per video-min: {vals['runtime_per_video_minute']:.2f}s")

    print("\n" + "=" * 70 + "\n")


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    compute_metrics()
