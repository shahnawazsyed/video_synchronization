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
    for fname in sorted(os.listdir(video_dir)):
        if os.path.splitext(fname)[1].lower() not in video_exts:
            continue
        in_path = os.path.join(video_dir, fname)
        out_name = os.path.splitext(fname)[0] + ".wav"
        out_path = os.path.join(audio_dir, out_name)
        # ffmpeg: convert to mono, resample
        cmd = f'ffmpeg -y -hide_banner -loglevel error -i {shlex.quote(in_path)} -ac 1 -ar {target_sr} -vn {shlex.quote(out_path)}'
        logger.info("Extracting audio: %s -> %s", in_path, out_path)
        subprocess.check_call(shlex.split(cmd))

def preprocess_audio(audio_path: str, target_sr: int = 16000):
    """
    Normalize and resample an audio file for synchronization.
    This function is a small helper to resave a file at target_sr, mono using ffmpeg.
    Returns path to preprocessed file (overwrites original path).
    """
    if not ffmpeg_exists():
        raise RuntimeError("ffmpeg not found on PATH — required to preprocess audio.")
    tmp = audio_path + ".tmp.wav"
    cmd = f'ffmpeg -y -hide_banner -loglevel error -i {shlex.quote(audio_path)} -ac 1 -ar {target_sr} {shlex.quote(tmp)}'
    subprocess.check_call(cmd, shell=True)
    os.replace(tmp, audio_path)
    return audio_path, target_sr
