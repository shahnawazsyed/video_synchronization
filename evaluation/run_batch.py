"""
run_batch.py
-------------
Batch synchronization runner for the evaluation suite.

For each synthetic test case (original + offset-shifted pair):
  1. Run audio-based synchronization (GCC-PHAT)
  2. Run visual-based synchronization (motion correlation)

Records estimated offset, confidence score, runtime, and resource usage for each.

Output:
  evaluation/results/results.csv
  evaluation/diagnostics/*.npz  (motion signals for before/after plots)
"""

import os
import sys
import csv
import time
import shutil
import logging
import tempfile
import threading

import subprocess

import numpy as np
import psutil

# Ensure the project root is on sys.path so we can import `src.*`
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import extract_audio_from_videos
from src.audio_sync import estimate_offsets_robust
from src.visual_sync import sync_videos_by_motion
from src.visual_sync import extract_motion_energy, smooth_motion_signal

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_CSV = os.path.join(BASE_DIR, "metadata", "synthetic_metadata.csv")
ORIGINALS_DIR = os.path.join(BASE_DIR, "originals")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DIAGNOSTICS_DIR = os.path.join(BASE_DIR, "diagnostics")

RESULT_FIELDNAMES = [
    "video_id",
    "true_offset_ms",
    "method_type",
    "estimated_offset_ms",
    "absolute_error_ms",
    "confidence_score",
    "runtime_seconds",
    "peak_cpu_percent",
    "peak_memory_mb",
    "video_length_sec",
    "motion_level",
    "audio_energy_level",
]


# ──────────────────────────────────────────────────────────────────────
# Resource monitoring
# ──────────────────────────────────────────────────────────────────────

class ResourceMonitor:
    """Context manager that samples CPU and memory in a background thread."""

    def __init__(self, interval: float = 0.2):
        self._interval = interval
        self._process = psutil.Process(os.getpid())
        self._peak_cpu: float = 0.0
        self._peak_memory_mb: float = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_loop(self):
        while not self._stop_event.is_set():
            try:
                cpu = self._process.cpu_percent(interval=None)
                mem_mb = self._process.memory_info().rss / (1024 * 1024)
                self._peak_cpu = max(self._peak_cpu, cpu)
                self._peak_memory_mb = max(self._peak_memory_mb, mem_mb)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            self._stop_event.wait(self._interval)

    def __enter__(self):
        # Prime cpu_percent so the first real sample is meaningful
        self._process.cpu_percent(interval=None)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    @property
    def peak_cpu_percent(self) -> float:
        return round(self._peak_cpu, 1)

    @property
    def peak_memory_mb(self) -> float:
        return round(self._peak_memory_mb, 1)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _read_metadata(csv_path: str) -> list:
    """Read the synthetic metadata CSV and return list of dicts."""
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _find_original_video(video_id: str, originals_dir: str) -> str:
    """
    Find the original video file matching *video_id* in originals_dir.
    Returns the full path.
    """
    for ext in (".mp4", ".mov", ".avi"):
        candidate = os.path.join(originals_dir, video_id + ext)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"No original video found for '{video_id}' in {originals_dir}"
    )


def _has_audio_stream(video_path: str) -> bool:
    """Check whether the video file contains an audio stream."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return len(result.stdout.strip()) > 0
    except Exception:
        return False


def _run_audio_sync(original_path: str, synthetic_path: str) -> dict:
    """
    Run audio-based synchronization between two videos.

    Returns dict with keys: estimated_offset_ms, confidence_score, runtime_seconds
    """
    # Pre-check: both files must have audio streams
    if not _has_audio_stream(original_path):
        raise RuntimeError(f"Original video has no audio stream: {original_path}")
    if not _has_audio_stream(synthetic_path):
        raise RuntimeError(f"Synthetic video has no audio stream: {synthetic_path}")

    workdir = tempfile.mkdtemp(prefix="eval_audio_")
    try:
        video_dir = os.path.join(workdir, "videos")
        audio_dir = os.path.join(workdir, "audio")
        os.makedirs(video_dir)
        os.makedirs(audio_dir)

        # Copy both files into the working directory
        orig_fname = "original.mp4"
        synth_fname = "synthetic.mp4"
        shutil.copy2(original_path, os.path.join(video_dir, orig_fname))
        shutil.copy2(synthetic_path, os.path.join(video_dir, synth_fname))

        with ResourceMonitor() as monitor:
            t0 = time.time()

            # Step 1: Extract audio
            extract_audio_from_videos(video_dir, audio_dir)

            # Step 2: Run GCC-PHAT
            offsets = estimate_offsets_robust(audio_dir, max_offset_sec=15.0, window_sec=None)
            runtime = time.time() - t0

        # offsets maps wav filenames → offset in seconds.
        # These are *alignment* offsets: "add this to the file's timestamps to align it."
        # To recover the *applied synthetic offset*, we negate the alignment offset.
        orig_wav = "original.wav"
        synth_wav = "synthetic.wav"

        if synth_wav in offsets and orig_wav in offsets:
            alignment_offset_sec = offsets[synth_wav] - offsets[orig_wav]
        elif synth_wav in offsets:
            alignment_offset_sec = offsets[synth_wav]
        else:
            alignment_offset_sec = 0.0

        # Audio sync note: compute_gcc_phat already negates the raw lag
        # (line 118: return -offset_seconds), so the optimized offsets from
        # estimate_offsets_robust are already in the same sign convention
        # as the true synthetic offset. Do NOT negate here.
        estimated_offset_ms = alignment_offset_sec * 1000.0

        # Extract confidence via direct GCC-PHAT on the pair
        from src.audio_sync import compute_gcc_phat
        from src.utils import load_audio

        orig_audio_path = os.path.join(audio_dir, orig_wav)
        synth_audio_path = os.path.join(audio_dir, synth_wav)

        if os.path.exists(orig_audio_path) and os.path.exists(synth_audio_path):
            sig_a, sr_a = load_audio(orig_audio_path)
            sig_b, sr_b = load_audio(synth_audio_path)
            _, confidence = compute_gcc_phat(sig_a, sig_b, sr_a, max_offset_sec=15.0)
        else:
            confidence = 0.0

        return {
            "estimated_offset_ms": round(estimated_offset_ms, 2),
            "confidence_score": round(confidence, 4),
            "runtime_seconds": round(runtime, 3),
            "peak_cpu_percent": monitor.peak_cpu_percent,
            "peak_memory_mb": monitor.peak_memory_mb,
        }

    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _run_visual_sync(original_path: str, synthetic_path: str,
                     diagnostics_dir: str | None = None,
                     case_label: str = "") -> dict:
    """
    Run visual-based synchronization between two videos.

    Returns dict with keys: estimated_offset_ms, confidence_score, runtime_seconds,
    peak_cpu_percent, peak_memory_mb.

    If *diagnostics_dir* is given, saves the motion signals as a .npz file for
    later before/after visualization.
    """
    workdir = tempfile.mkdtemp(prefix="eval_visual_")
    try:
        video_dir = os.path.join(workdir, "videos")
        os.makedirs(video_dir)

        orig_fname = "original.mp4"
        synth_fname = "synthetic.mp4"
        shutil.copy2(original_path, os.path.join(video_dir, orig_fname))
        shutil.copy2(synthetic_path, os.path.join(video_dir, synth_fname))

        with ResourceMonitor() as monitor:
            t0 = time.time()

            offsets = sync_videos_by_motion(
                video_dir,
                selected_files=[orig_fname, synth_fname],
                max_offset_sec=20.0,
                output_dir=None,
            )
            runtime = time.time() - t0

        # Offsets are *alignment* offsets: "shift this file by X to align it."
        # To recover the *applied synthetic offset*, we negate.
        if synth_fname in offsets and orig_fname in offsets:
            alignment_offset_sec = offsets[synth_fname] - offsets[orig_fname]
        elif synth_fname in offsets:
            alignment_offset_sec = offsets[synth_fname]
        else:
            alignment_offset_sec = 0.0

        # Negate: alignment offset is the inverse of the synthetic shift
        estimated_offset_ms = -alignment_offset_sec * 1000.0

        # For confidence, extract motion signals (also used for diagnostics)
        from src.visual_sync import correlate_motion_signals
        from scipy.signal import resample

        m1 = m2 = None
        target_fps = 10.0
        confidence = 0.0

        try:
            m1_raw, fps1 = extract_motion_energy(
                os.path.join(video_dir, orig_fname), step=3
            )
            m2_raw, fps2 = extract_motion_energy(
                os.path.join(video_dir, synth_fname), step=3
            )
            m1_raw = smooth_motion_signal(m1_raw, fps1)
            m2_raw = smooth_motion_signal(m2_raw, fps2)

            m1 = resample(m1_raw, int(len(m1_raw) * target_fps / fps1)) if fps1 > 0 else m1_raw
            m2 = resample(m2_raw, int(len(m2_raw) * target_fps / fps2)) if fps2 > 0 else m2_raw

            _, confidence = correlate_motion_signals(m1, m2, target_fps)
        except Exception:
            confidence = 0.0

        # Save motion signals for before/after visualization
        if diagnostics_dir and m1 is not None and m2 is not None:
            try:
                os.makedirs(diagnostics_dir, exist_ok=True)
                npz_path = os.path.join(diagnostics_dir, f"{case_label}.npz")
                np.savez_compressed(
                    npz_path,
                    original=m1,
                    synthetic=m2,
                    fps=target_fps,
                    estimated_offset_ms=estimated_offset_ms,
                )
            except Exception:
                logger.debug("Failed to save diagnostics for %s", case_label)

        return {
            "estimated_offset_ms": round(estimated_offset_ms, 2),
            "confidence_score": round(confidence, 4),
            "runtime_seconds": round(runtime, 3),
            "peak_cpu_percent": monitor.peak_cpu_percent,
            "peak_memory_mb": monitor.peak_memory_mb,
        }

    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def run_batch(
    metadata_csv: str = METADATA_CSV,
    originals_dir: str = ORIGINALS_DIR,
    results_dir: str = RESULTS_DIR,
    diagnostics_dir: str = DIAGNOSTICS_DIR,
) -> str:
    """
    Run audio and visual synchronization on every synthetic test case.

    Returns the path to the output results CSV.
    """
    os.makedirs(results_dir, exist_ok=True)
    metadata = _read_metadata(metadata_csv)

    if not metadata:
        raise ValueError(f"No entries found in {metadata_csv}")

    results_csv = os.path.join(results_dir, "results.csv")
    total = len(metadata)
    results_rows = []

    logger.info("Starting batch synchronization: %d test cases × 2 methods", total)

    for idx, entry in enumerate(metadata, 1):
        video_id = entry["video_id"]
        true_offset_ms = float(entry["true_offset_ms"])
        synthetic_path = entry["synthetic_file_path"]

        # Sensitivity tags (pass through from metadata)
        tags = {
            "video_length_sec": entry.get("video_length_sec", ""),
            "motion_level": entry.get("motion_level", ""),
            "audio_energy_level": entry.get("audio_energy_level", ""),
        }

        if not os.path.exists(synthetic_path):
            logger.warning("Synthetic file missing, skipping: %s", synthetic_path)
            continue

        original_path = _find_original_video(video_id, originals_dir)

        logger.info(
            "[%d/%d] %s | offset=%+dms",
            idx, total, video_id, int(true_offset_ms),
        )

        # ── Audio sync ──────────────────────────────────────────────
        try:
            logger.info("  Running audio sync ...")
            audio_result = _run_audio_sync(original_path, synthetic_path)
            audio_error = abs(audio_result["estimated_offset_ms"] - true_offset_ms)
            results_rows.append({
                "video_id": video_id,
                "true_offset_ms": true_offset_ms,
                "method_type": "audio",
                "estimated_offset_ms": audio_result["estimated_offset_ms"],
                "absolute_error_ms": round(audio_error, 2),
                "confidence_score": audio_result["confidence_score"],
                "runtime_seconds": audio_result["runtime_seconds"],
                "peak_cpu_percent": audio_result["peak_cpu_percent"],
                "peak_memory_mb": audio_result["peak_memory_mb"],
                **tags,
            })
            logger.info(
                "    Audio: est=%+.1fms  err=%.1fms  conf=%.3f  time=%.2fs  cpu=%.1f%%  mem=%.1fMB",
                audio_result["estimated_offset_ms"],
                audio_error,
                audio_result["confidence_score"],
                audio_result["runtime_seconds"],
                audio_result["peak_cpu_percent"],
                audio_result["peak_memory_mb"],
            )
        except Exception as e:
            logger.error("  Audio sync failed for %s: %s", video_id, e, exc_info=True)

        # ── Visual sync ─────────────────────────────────────────────
        try:
            logger.info("  Running visual sync ...")
            case_label = f"{video_id}_offset{int(true_offset_ms):+d}"
            visual_result = _run_visual_sync(
                original_path, synthetic_path,
                diagnostics_dir=diagnostics_dir,
                case_label=case_label,
            )
            visual_error = abs(visual_result["estimated_offset_ms"] - true_offset_ms)
            results_rows.append({
                "video_id": video_id,
                "true_offset_ms": true_offset_ms,
                "method_type": "visual",
                "estimated_offset_ms": visual_result["estimated_offset_ms"],
                "absolute_error_ms": round(visual_error, 2),
                "confidence_score": visual_result["confidence_score"],
                "runtime_seconds": visual_result["runtime_seconds"],
                "peak_cpu_percent": visual_result["peak_cpu_percent"],
                "peak_memory_mb": visual_result["peak_memory_mb"],
                **tags,
            })
            logger.info(
                "    Visual: est=%+.1fms  err=%.1fms  conf=%.3f  time=%.2fs  cpu=%.1f%%  mem=%.1fMB",
                visual_result["estimated_offset_ms"],
                visual_error,
                visual_result["confidence_score"],
                visual_result["runtime_seconds"],
                visual_result["peak_cpu_percent"],
                visual_result["peak_memory_mb"],
            )
        except Exception as e:
            logger.error("  Visual sync failed for %s: %s", video_id, e, exc_info=True)

    # Write results
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(results_rows)

    logger.info("Wrote %d result rows to %s", len(results_rows), results_csv)
    logger.info("Batch synchronization complete.")
    return results_csv


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    run_batch()
