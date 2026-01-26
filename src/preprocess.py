"""
preprocess.py
--------------
Extracts and preprocesses audio from input video files.
"""
import os
import shlex
import subprocess
from typing import Optional
from .utils import ensure_dir, setup_logger, ffmpeg_exists

logger = setup_logger(__name__)


def extract_audio_from_videos(video_dir: str, audio_dir: str, target_sr: int = 16000):
    """
    Extract audio tracks from all video files in a directory using ffmpeg.
    Produces mono WAV files at target_sr in audio_dir with same base filenames.
    """
    ensure_dir(audio_dir)
    if not ffmpeg_exists():
        raise RuntimeError("ffmpeg not found on PATH — required to extract audio. Install ffmpeg and retry.")

    video_exts = {".mp4", ".mov"}
    files = sorted([f for f in os.listdir(video_dir) if os.path.splitext(f)[1].lower() in video_exts])
    logger.info("Found %d video files in %s", len(files), video_dir)

    for fname in files:
        in_path = os.path.join(video_dir, fname)
        out_name = os.path.splitext(fname)[0] + ".wav"
        out_path = os.path.join(audio_dir, out_name)
        # ffmpeg: convert to mono, resample
        cmd = f'ffmpeg -y -hide_banner -loglevel error -i {shlex.quote(in_path)} -ac 1 -ar {target_sr} -vn {shlex.quote(out_path)}'
        logger.debug("Extracting audio: %s -> %s", in_path, out_path)
        try:
            subprocess.check_call(shlex.split(cmd))
        except subprocess.CalledProcessError:
            logger.error("Failed to extract audio from %s", in_path, exc_info=True)
            raise

    logger.info("Audio extraction complete.")


