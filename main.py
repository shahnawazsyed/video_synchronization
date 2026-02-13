"""
main.py
---------
Entry point for the multi-camera synchronization application.
Launches the Flask Web UI.
"""

import os
import shutil
import logging
from src import config
from src.ui import run_app, configure_logging

def cleanup_temp():
    """Wipe the temporary directory to start fresh."""
    if os.path.exists(config.TEMP_BASE):
        try:
            shutil.rmtree(config.TEMP_BASE)
            logging.info("Cleaned up temporary directory: %s", config.TEMP_BASE)
        except Exception as e:
            logging.warning("Failed to clean up temp dir: %s", e)
    
    # Re-create necessary directories
    os.makedirs(config.VIDEO_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.AUDIO_DIR, exist_ok=True)
    os.makedirs(config.VISUAL_SYNC_OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    configure_logging()
    cleanup_temp()
    run_app()