"""
offset_generation.py
---------------------
Generates synthetic offset-shifted videos from original source videos.

For each original video and each offset in the offset list:
  - Positive offset: prepends black frames + silent audio
  - Negative offset: trims from the start

Outputs:
  - Shifted video files in evaluation/synthetic/
  - Metadata CSV in evaluation/metadata/synthetic_metadata.csv
"""

import os
import csv
import shlex
import subprocess
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGINALS_DIR = os.path.join(BASE_DIR, "originals")
SYNTHETIC_DIR = os.path.join(BASE_DIR, "synthetic")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")

DEFAULT_OFFSETS_MS = [-1000, -500, -100, 100, 500, 1000]

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}


# ──────────────────────────────────────────────────────────────────────
# Sensitivity tag helpers
# ──────────────────────────────────────────────────────────────────────

def _get_video_duration(path: str) -> float:
    """Return video duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return float(result.stdout.decode().strip())


def _compute_motion_level(path: str, max_frames: int = 300) -> float:
    """
    Estimate motion level as the mean frame-to-frame pixel difference
    over the first *max_frames* frames.  Returns a value in [0, 1].
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0.0

    prev_gray = None
    diffs = []
    count = 0

    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 120))
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            diffs.append(np.mean(diff) / 255.0)
        prev_gray = gray
        count += 1

    cap.release()
    return float(np.mean(diffs)) if diffs else 0.0


def _compute_audio_energy(path: str) -> float:
    """
    Extract a short WAV snippet with ffmpeg and compute RMS energy.
    Returns a normalised energy value in [0, 1].
    """
    import tempfile
    from scipy.io import wavfile

    tmp_wav = os.path.join(tempfile.gettempdir(), "_eval_energy.wav")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", path,
        "-ac", "1", "-ar", "16000",
        "-t", "10",           # first 10 seconds only
        "-vn", tmp_wav,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        sr, data = wavfile.read(tmp_wav)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2**31
        else:
            data = data.astype(np.float32)
        rms = float(np.sqrt(np.mean(data ** 2)))
        # Normalise: typical speech RMS is ~0.05-0.15; cap at 1.0
        return min(rms / 0.15, 1.0)
    except Exception:
        return 0.0
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)


# ──────────────────────────────────────────────────────────────────────
# Core generation logic
# ──────────────────────────────────────────────────────────────────────

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


def _apply_positive_offset(input_path: str, output_path: str, offset_sec: float):
    """
    Prepend *offset_sec* seconds of black video + silent audio
    then concatenate with the original.
    """
    has_audio = _has_audio_stream(input_path)

    # Get original video properties for the black leader
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "csv=p=0",
        input_path,
    ]
    probe = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    parts = probe.stdout.decode().strip().split(",")
    width, height = int(parts[0]), int(parts[1])
    fps_str = parts[2]  # e.g. "30/1"

    if has_audio:
        # Use filter_complex to prepend black+silence then concat
        filter_complex = (
            f"color=c=black:s={width}x{height}:r={fps_str}:d={offset_sec}[vblack];"
            f"aevalsrc=0:d={offset_sec}[ablack];"
            f"[vblack][ablack][0:v][0:a]concat=n=2:v=1:a=1[v][a]"
        )
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", input_path,
            "-filter_complex", filter_complex,
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            output_path,
        ]
    else:
        # Video only — no audio concat needed
        filter_complex = (
            f"color=c=black:s={width}x{height}:r={fps_str}:d={offset_sec}[vblack];"
            f"[vblack][0:v]concat=n=2:v=1:a=0[v]"
        )
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", input_path,
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            output_path,
        ]

    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _apply_negative_offset(input_path: str, output_path: str, trim_sec: float):
    """Trim the first *trim_sec* seconds from the video."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(trim_sec),
        "-i", input_path,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        output_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def generate_synthetic_dataset(
    originals_dir: str = ORIGINALS_DIR,
    synthetic_dir: str = SYNTHETIC_DIR,
    metadata_dir: str = METADATA_DIR,
    offsets_ms: list = None,
):
    """
    Generate synthetic offset-shifted videos and write metadata CSV.

    Parameters
    ----------
    originals_dir : str
        Directory containing original source videos.
    synthetic_dir : str
        Output directory for shifted videos.
    metadata_dir : str
        Output directory for the metadata CSV.
    offsets_ms : list[int]
        List of offsets in milliseconds (positive = prepend, negative = trim).
    """
    if offsets_ms is None:
        offsets_ms = DEFAULT_OFFSETS_MS

    os.makedirs(synthetic_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    # Discover original videos
    videos = sorted([
        f for f in os.listdir(originals_dir)
        if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
    ])
    if not videos:
        raise FileNotFoundError(f"No video files found in {originals_dir}")

    logger.info("Found %d original videos in %s", len(videos), originals_dir)
    logger.info("Offsets (ms): %s", offsets_ms)

    rows = []

    for video_fname in videos:
        video_id = os.path.splitext(video_fname)[0]
        original_path = os.path.join(originals_dir, video_fname)

        # Compute sensitivity tags once per original video
        logger.info("Computing sensitivity tags for %s ...", video_fname)
        duration_sec = _get_video_duration(original_path)
        motion_level = _compute_motion_level(original_path)
        audio_energy = _compute_audio_energy(original_path)

        logger.info(
            "  duration=%.1fs  motion=%.4f  audio_energy=%.4f",
            duration_sec, motion_level, audio_energy,
        )

        for offset_ms in offsets_ms:
            offset_sec = offset_ms / 1000.0
            suffix = f"{'+' if offset_ms >= 0 else ''}{offset_ms}ms"
            out_fname = f"{video_id}_offset_{suffix}.mp4"
            out_path = os.path.join(synthetic_dir, out_fname)

            logger.info("  Generating %s (offset=%dms) ...", out_fname, offset_ms)

            if offset_ms > 0:
                _apply_positive_offset(original_path, out_path, offset_sec)
            elif offset_ms < 0:
                _apply_negative_offset(original_path, out_path, abs(offset_sec))
            else:
                # Zero offset — just re-encode for consistency
                _apply_negative_offset(original_path, out_path, 0)

            rows.append({
                "video_id": video_id,
                "synthetic_file_path": out_path,
                "true_offset_ms": offset_ms,
                "video_length_sec": round(duration_sec, 2),
                "motion_level": round(motion_level, 6),
                "audio_energy_level": round(audio_energy, 6),
            })

    # Write metadata
    csv_path = os.path.join(metadata_dir, "synthetic_metadata.csv")
    fieldnames = [
        "video_id", "synthetic_file_path", "true_offset_ms",
        "video_length_sec", "motion_level", "audio_energy_level",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %d entries to %s", len(rows), csv_path)
    logger.info("Synthetic dataset generation complete.")
    return csv_path


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    generate_synthetic_dataset()
