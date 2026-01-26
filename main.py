"""
main.py
---------
Entry point for the multi-camera synchronization application.
Launches the Flask Web UI.
"""

from src.ui import run_app, configure_logging

if __name__ == "__main__":
    configure_logging()
    run_app()

#TODO: track changes, ask claude/gemini whatever to cross reference system architecture with final thing