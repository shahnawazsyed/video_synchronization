"""
main.py
---------
Entry point for the multi-camera synchronization application.
Launches the interactive Tkinter GUI.
"""

from src.ui import run_app

if __name__ == "__main__":
    run_app()

#TODO: fix video uploading -- allow from any directory while maintaining security (where is it stored?)
#TODO: test with muted audio
#TODO: figure out what the audio/visual approach split is