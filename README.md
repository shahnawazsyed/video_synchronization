# Multi-Camera Video Synchronization

A Flask-based tool for aligning multiple video tracks using visual motion detection or audio cross-correlation. Designed for synchronizing multi-view recordings where start times aren't perfectly aligned.

## Prerequisites

- **Python 3.9+**
- **FFmpeg**: Required for audio extraction and video manipulation. Ensure it's in your system PATH.

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd video_synchronization

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. **Start the application**:
   ```bash
   python main.py
   ```
2. **Access the UI**: Open your browser to `http://127.0.0.1:5000`.
3. **Workflow**:
   - **Upload**: Select 2 or more videos. Files are staged in a temporary directory.
   - **Sync**: Choose between **Visual (Motion)** or **Audio (GCC-PHAT)** synchronization in `src/config.py` (Visual is the default).
   - **Review**: Use the multi-video previewer with a universal seek bar and audio master toggle to verify alignment.
   - **Export**: Download the synchronized videos as a ZIP.

## Technical Details

- **Visual Sync**: Extracts motion energy timeseries from video frames and uses cross-correlation to find temporal offsets. Robust against silent videos or noisy environments.
- **Audio Sync**: Uses GCC-PHAT (Generalized Cross-Correlation with Phase Transform) on extracted audio tracks for high-precision alignment.
- **Processing**: Synchronization is applied via `ffmpeg` re-encoding (with `tpad` and `adelay`) to ensure sub-frame accuracy and compatibility across players.

## Configuration

Settings such as `SYNC_METHOD` and directory paths can be modified in `src/config.py`.