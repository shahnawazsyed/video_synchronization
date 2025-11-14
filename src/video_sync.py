"""
video_sync.py
--------------
Applies time offsets to video streams to produce synchronized outputs.
"""
import os
from typing import Dict
from .utils import ensure_dir, setup_logger

logger = setup_logger(__name__)

def apply_video_offsets(video_dir: str, offsets: Dict[str, float], output_dir: str, verbose: bool = True):
    """
    Synchronize multiple video feeds using precomputed time offsets.
    Args:
        video_dir: Path to raw, unsynchronized videos.
        offsets: Dictionary of {filename.wav or filename.mp4: offset_seconds}.
                 Offsets are seconds to ADD to that file to align it with the reference.
                 Typical usage: offsets keys are the same base-names as extracted audio.
        output_dir: Directory to store synchronized video outputs.
    """
    ensure_dir(output_dir)

    try:
        from moviepy.editor import VideoFileClip, CompositeVideoClip
    except Exception as e:
        raise RuntimeError(
            f"error: {e}"
        )
    video_exts = {".mp4"}
    for fname in sorted(os.listdir(video_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in video_exts:
            continue
        in_path = os.path.join(video_dir, fname)
        base = os.path.splitext(fname)[0]
        # offsets probably keyed by audio filename like "<base>.wav" — try both
        off = None
        if fname in offsets:
            off = offsets[fname]
        elif base + ".wav" in offsets:
            off = offsets[base + ".wav"]
        else: # if a video has no offset, treat as reference (0)
            off = 0.0

        logger.info("Syncing %s with offset %.3fs", fname, off)
        clip = VideoFileClip(in_path)

        if off >= 0:
            shifted = clip.set_start(off)
            comp = CompositeVideoClip([shifted], size=clip.size).set_duration(shifted.end)
        else: #negative offset
            trim_start = min(max(0.0, -off), clip.duration)
            trimmed = clip.subclip(trim_start, clip.duration)
            comp = trimmed.set_start(0).set_duration(trimmed.duration)

        out_path = os.path.join(output_dir, fname)
        comp.write_videofile(out_path, codec="libx264", audio_codec="aac", verbose=verbose, logger=None)
        clip.close()
        comp.close()
