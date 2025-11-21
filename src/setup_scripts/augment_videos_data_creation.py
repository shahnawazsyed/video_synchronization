import os
import random
import subprocess
import json
from tqdm import tqdm
import shutil

# --- CONFIG ---
# Assuming this file lives in: video_synchronization/src/setup_scripts
# and data lives in: video_synchronization/data/raw
SYNCED_DIR = "data/raw"           # Folder with synchronized videos
AUG_DIR = "data/augmented"        # Folder to save augmented videos
os.makedirs(AUG_DIR, exist_ok=True)

# Realistic range of camera clock skew or sync drift (seconds)
MIN_OFFSET = -2.0
MAX_OFFSET = 2.0

# Target clip length (to ensure enough overlap)
TARGET_CLIP_LEN = 60.0


def get_video_duration(path):
    """Use ffprobe to read video duration."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path
    ]
    out = subprocess.check_output(cmd).decode().strip()
    return float(out)


def shift_video(input_path, output_path, shift_sec):
    """
    Shift a video by shift_sec seconds.
    Positive: delayed start (pad at beginning)
    Negative: early start (trim beginning)
    Maintains approximately equal clip length.
    """
    if shift_sec >= 0:
        # Add leading offset (delay video/audio)
        # Use itsoffset for video (metadata shift)
        # Use adelay for audio (physical silence insertion) + re-encode
        shift_ms = int(shift_sec * 1000)
        cmd = [
            "ffmpeg", "-y",
            "-itsoffset", str(shift_sec),
            "-i", input_path,
            "-i", input_path,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "aac",
            "-af", f"aformat=channel_layouts=mono,adelay={shift_ms}",
            "-shortest", output_path
        ]
    else:
        # Trim start by abs(shift_sec)
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(-shift_sec),
            "-i", input_path,
            "-t", str(TARGET_CLIP_LEN),
            "-c", "copy", output_path
        ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"❌ ffmpeg failed for {output_path}: {e.stderr.decode()}")
        if os.path.exists(output_path):
            os.remove(output_path)


# --- MAIN AUGMENTATION LOOP ---
ground_truth = []

video_files = [f for f in os.listdir(SYNCED_DIR) if f.endswith(".mp4")]

for video_file in tqdm(video_files, desc="Augmenting videos"):
    input_path = os.path.join(SYNCED_DIR, video_file)

    try:
        duration = get_video_duration(input_path)
    except Exception:
        print(f"⚠️ Could not get duration for {video_file}, skipping.")
        continue

    # Random offset within realistic range
    offset = random.uniform(MIN_OFFSET, MAX_OFFSET)

    # Random start time ensuring enough remaining length
    start_time = random.uniform(0, max(0, duration - TARGET_CLIP_LEN))

    output_file = f"{os.path.splitext(video_file)[0]}_aug.mp4"
    output_path = os.path.join(AUG_DIR, output_file)

    # Extract segment and apply offset in one step
    cmd_extract = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-i", input_path,
        "-t", str(TARGET_CLIP_LEN),
        "-c", "copy", "temp_clip.mp4"
    ]
    subprocess.run(cmd_extract, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    shift_video("temp_clip.mp4", output_path, offset)
    os.remove("temp_clip.mp4")

    # Verify output
    if not os.path.exists(output_path):
        shutil.copy(input_path, output_path)

    ground_truth.append({
        "video": video_file,
        "augmented_file": output_file,
        "time_offset_sec": offset
    })

# --- SAVE GROUND TRUTH ---
with open(os.path.join(AUG_DIR, "ground_truth.json"), "w") as f:
    json.dump(ground_truth, f, indent=4)

print(f"✅ Augmentation completed. {len(ground_truth)} videos saved to {AUG_DIR}")
