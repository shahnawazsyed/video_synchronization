"""
video_sync.py
--------------
Applies time offsets to entire media files (audio+video together) using ffmpeg.

Offset semantics (Option C):
- Positive offset  (>0): delay the whole file (video+audio start later)
- Negative offset (<0): trim the first |offset| seconds (drop early content)

Uses stream copy when possible (fast, lossless).
If container/codec conflicts occur, falls back to safe re-encode.
"""

import os
import subprocess
from typing import Dict
from tqdm import tqdm

EPS = 1e-3  # ignore tiny offsets (<1 ms)


def _run(cmd):
    """Run ffmpeg command list, raise on error."""
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf8"))
    return result.stdout.decode("utf8")


def _delay_stream_copy(input_path, output_path, offset):
    """
    Positive offset: delay entire file using -itsoffset.
    This preserves both audio and video PTS if ffmpeg can stream copy.
    """
    return [
        "ffmpeg", "-y",
        "-itsoffset", str(offset),
        "-i", input_path,
        "-i", input_path,
        "-map", "0:v", "-map", "0:a",
        "-c", "copy",
        output_path
    ]


def _trim_stream_copy(input_path, output_path, trim):
    """Negative offset: fast trim using -ss before -i."""
    return [
        "ffmpeg", "-y",
        "-ss", str(trim),
        "-i", input_path,
        "-c", "copy",
        output_path
    ]


def _reencode_delay(input_path, output_path, offset):
    """
    Fallback method: re-encode and shift using tpad filter.
    This inserts actual black frames at the start, ensuring OpenCV respects the delay.
    """
    ms = int(offset * 1000)
    return [
        "ffmpeg", "-y",
        "-i", input_path,
        "-filter_complex",
        f"[0:v]tpad=start_duration={offset}:color=black[v];"
        f"[0:a]adelay={ms}|{ms}[a]",
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        output_path
    ]


def _reencode_trim(input_path, output_path, trim):
    """Fallback for negative offset: trim then re-encode."""
    return [
        "ffmpeg", "-y",
        "-ss", str(trim),
        "-i", input_path,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        output_path
    ]


def apply_video_offsets(video_dir: str, offsets: Dict[str, float], output_dir: str, verbose: bool = True):
    """
    Apply time offsets to entire video files.

    Parameters
    ----------
    video_dir : str
        Directory containing original videos.
    offsets : Dict[str, float]
        Mapping filename → offset_in_seconds. Positive = delay, Negative = trim.
    output_dir : str
        Directory where synced videos will be saved.
    verbose : bool
        Show tqdm progress.

    Returns
    -------
    True on completion.
    """

    os.makedirs(output_dir, exist_ok=True)

    pbar = tqdm(offsets.items(), disable=not verbose, desc="sync_videos", unit="file")

    for fname, off in pbar:
        in_path = os.path.join(video_dir, fname)
        
        # Add _synced suffix to output filename
        base_name, ext = os.path.splitext(fname)
        out_fname = f"{base_name}_synced{ext}"
        out_path = os.path.join(output_dir, out_fname)

        pbar.set_postfix({"file": fname, "offset": f"{off:.3f}s"})

        if not os.path.exists(in_path):
            pbar.write(f"WARNING: missing input {in_path}")
            continue

        # tiny offset → just copy
        if abs(off) < EPS:
            try:
                _run(["ffmpeg", "-y", "-i", in_path, "-c", "copy", out_path])
                continue
            except Exception:
                pass  # try full path below

        # positive offset = delay
        # MUST re-encode to burn in black frames (tpad) so OpenCV respects the delay
        if off > 0:
            try:
                # Direct re-encode with tpad
                _run(_reencode_delay(in_path, out_path, off))
            except Exception as e:
                pbar.write(f"Reencode delay failed for {fname}: {e}")

        # negative offset = trim
        else:
            trim = abs(off)
            try:
                _run(_trim_stream_copy(in_path, out_path, trim))
                continue
            except Exception as e:
                pbar.write(f"Stream-copy trim failed for {fname}: {e}")
                try:
                    _run(_reencode_trim(in_path, out_path, trim))
                except Exception as e2:
                    pbar.write(f"Reencode trim also failed for {fname}: {e2}")
    return True
