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
#TODO: it should add black screen to the end/start of inconsistencies, such that the videos are of same length
#TODO: universal seeking (since they are in theory synced and of same length)