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
import logging
import subprocess
from typing import Dict
from tqdm import tqdm

from .utils import setup_logger, log_execution_time

logger = setup_logger(__name__)

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


def _get_video_duration(video_path):
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return float(result.stdout.decode("utf8").strip())
    except Exception as e:
        raise RuntimeError(f"Failed to get duration for {video_path}: {e}")


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


def _reencode_delay(input_path, output_path, offset, end_pad=0):
    """
    Fallback method: re-encode and shift using tpad filter.
    This inserts actual black frames at the start and/or end, ensuring OpenCV respects the delay.
    
    Parameters
    ----------
    input_path : str
        Path to input video
    output_path : str
        Path to output video
    offset : float
        Start padding duration in seconds (black frames at the beginning)
    end_pad : float
        End padding duration in seconds (black frames at the end)
    """
    ms = int(offset * 1000)
    
    # Build tpad filter - start_duration for beginning, stop_duration for end
    tpad_params = []
    if offset > 0:
        tpad_params.append(f"start_duration={offset}")
    if end_pad > 0:
        tpad_params.append(f"stop_duration={end_pad}")
    
    if tpad_params:
        tpad_str = f"[0:v]tpad={':'.join(tpad_params)}:color=black[v]"
    else:
        tpad_str = "[0:v]copy[v]"
    
    # Audio delay for start, and apad for end
    audio_filters = []
    if offset > 0:
        audio_filters.append(f"adelay={ms}|{ms}")
    if end_pad > 0:
        # Convert to milliseconds for apad
        end_ms = int(end_pad * 1000)
        audio_filters.append(f"apad=pad_dur={end_ms}ms")
    
    if audio_filters:
        audio_str = f"[0:a]{','.join(audio_filters)}[a]"
    else:
        audio_str = "[0:a]copy[a]"
    
    filter_complex = f"{tpad_str};{audio_str}"
    
    return [
        "ffmpeg", "-y",
        "-i", input_path,
        "-filter_complex", filter_complex,
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

    # Step 1: Get durations and calculate max final duration

    logger.info("Applying video offsets...")
    
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Get durations and calculate max final duration
    logger.info("Calculating video durations...")
    
    durations = {}
    final_durations = {}
    
    for fname in offsets.keys():
        in_path = os.path.join(video_dir, fname)
        if not os.path.exists(in_path):
            logger.warning("Missing input file: %s", in_path)
            continue
        
        try:
            duration = _get_video_duration(in_path)
            durations[fname] = duration
            
            # Final duration = original duration + offset (if positive) - trim (if negative)
            final_durations[fname] = duration + offsets[fname]
            
            logger.debug("  %s: %.2fs -> %.2fs", fname, duration, final_durations[fname])
        except Exception as e:
            logger.error("Error getting duration for %s: %s", fname, e)
            continue
    
    # Find the maximum final duration
    max_duration = max(final_durations.values()) if final_durations else 0
    
    logger.info("Max final duration: %.2fs", max_duration)
    logger.info("Synchronizing videos...")

    # Step 2: Apply offsets with end padding
    pbar = tqdm(offsets.items(), disable=not verbose, desc="sync_videos", unit="file")

    with log_execution_time(logger, "Apply Video Offsets"):
        for fname, off in pbar:
            in_path = os.path.join(video_dir, fname)
            
            # Add _synced suffix to output filename
            base_name, ext = os.path.splitext(fname)
            out_fname = f"{base_name}_synced{ext}"
            out_path = os.path.join(output_dir, out_fname)

            pbar.set_postfix({"file": fname, "offset": f"{off:.3f}s"})
            logger.info("Processing %s with offset %+.3fs", fname, off)

            if not os.path.exists(in_path) or fname not in durations:
                logger.warning("Skipping %s (missing file or duration)", fname)
                continue
            
            # Calculate end padding needed
            # end_pad = max_duration - (original_duration + offset)
            end_pad = max(0, max_duration - final_durations[fname])

            # tiny offset and no end padding → just copy
            if abs(off) < EPS and end_pad < EPS:
                try:
                    _run(["ffmpeg", "-y", "-i", in_path, "-c", "copy", out_path])
                    continue
                except Exception:
                    pass  # try full path below

            # positive offset = delay (may also need end padding)
            # MUST re-encode to burn in black frames (tpad) so OpenCV respects the delay
            if off > 0:
                try:
                    # Re-encode with start padding and end padding
                    _run(_reencode_delay(in_path, out_path, off, end_pad))
                except Exception as e:
                    logger.error("Reencode delay failed for %s: %s", fname, e, exc_info=True)

            # negative offset = trim (may also need end padding)
            elif off < 0:
                trim = abs(off)
                
                # If we need end padding after trimming, we must re-encode
                if end_pad > EPS:
                    try:
                        # We need to trim AND pad the end
                        # Use a filter that trims then pads
                        trim_ms = int(trim * 1000)
                        end_ms = int(end_pad * 1000)
                        
                        cmd = [
                            "ffmpeg", "-y",
                            "-ss", str(trim),  # Trim from start
                            "-i", in_path,
                            "-filter_complex",
                            f"[0:v]tpad=stop_duration={end_pad}:color=black[v];"
                            f"[0:a]apad=pad_dur={end_ms}ms[a]",
                            "-map", "[v]", "-map", "[a]",
                            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                            "-c:a", "aac", "-b:a", "128k",
                            out_path
                        ]
                        _run(cmd)
                    except Exception as e:
                        logger.error("Trim + end pad failed for %s: %s", fname, e, exc_info=True)
                else:
                    # Just trim, no end padding needed
                    try:
                        _run(_trim_stream_copy(in_path, out_path, trim))
                        continue
                    except Exception as e:
                        logger.warning("Stream-copy trim failed for %s: %s. Retrying with re-encode.", fname, e)
                        try:
                            _run(_reencode_trim(in_path, out_path, trim))
                        except Exception as e2:
                            logger.error("Reencode trim also failed for %s: %s", fname, e2, exc_info=True)
            
            # zero offset but needs end padding
            else:
                try:
                    _run(_reencode_delay(in_path, out_path, 0, end_pad))
                except Exception as e:
                    logger.error("End padding failed for %s: %s", fname, e, exc_info=True)
    
    return True

