"""
Configuration settings for the video synchronization project.
"""
import os



import tempfile

# Sync method: "visual" (motion-based) or "audio" (GCC-PHAT)
SYNC_METHOD = "visual"

# Use system temp directory
TEMP_BASE = os.path.join(tempfile.gettempdir(), "video_synchronization")

# Directories
VIDEO_DIR = os.path.join(TEMP_BASE, "raw")
OUTPUT_DIR = os.path.join(TEMP_BASE, "synced")
AUDIO_DIR = os.path.join(TEMP_BASE, "audio")
VISUAL_SYNC_OUTPUT_DIR = os.path.join(TEMP_BASE, "visual_sync_debug")

# Ensure directories exist
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(VISUAL_SYNC_OUTPUT_DIR, exist_ok=True)
