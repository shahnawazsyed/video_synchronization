# Multi-Camera Video Synchronization

A Flask-based tool for aligning multiple video tracks using visual motion detection or audio cross-correlation. Designed for synchronizing multi-view recordings where start times aren't perfectly aligned.

## Prerequisites

- **Python 3.11+** (required by `scipy` and `pandas`; developed/tested on Python 3.12)
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
2. **Access the UI**: Open your browser to `http://127.0.0.1:5050`.
3. **Workflow**:
   - **Upload**: Select 2 or more videos (**.mp4, .mov, or .avi**). Files are staged in a temporary directory.
   - **Sync**: Choose between **Visual (Motion)** or **Audio (GCC-PHAT)** synchronization by setting `SYNC_METHOD` in `src/config.py` **before starting the application**. The synchronization method is not configurable from the UI.
   - **Review**: Use the multi-video previewer with a universal seek bar and audio master toggle to verify alignment.
   - **Export**: Download the synchronized videos as a ZIP.

## Technical Details

- **Visual Sync**: Extracts motion energy timeseries from video frames and uses cross-correlation to find temporal offsets. Robust against silent videos or noisy environments.
- **Audio Sync**: Uses GCC-PHAT (Generalized Cross-Correlation with Phase Transform) on extracted audio tracks for high-precision alignment.
- **Processing**: Synchronization is applied via `ffmpeg` re-encoding (with `tpad` and `adelay`) to ensure sub-frame accuracy and compatibility across players.

## Authentication

The app supports optional token-based authentication. Set the `VIDEO_SYNC_TOKEN` environment variable to require a token before accessing the UI:

```bash
export VIDEO_SYNC_TOKEN="your-secret-token"
python main.py
```

If `VIDEO_SYNC_TOKEN` is **not set**, the app runs in dev mode with no authentication.

You can also set `SECRET_KEY` to provide a persistent Flask session secret (otherwise a random key is generated on each restart).

## Configuration

Settings such as `SYNC_METHOD` and directory paths can be modified in `src/config.py`.

## Evaluation Suite

A fully script-driven, reproducible pipeline for assessing the accuracy, confidence reliability, and efficiency of both synchronization methods.

### Directory Structure

```
evaluation/
├── originals/          # Place your 4 source videos here
├── synthetic/          # Generated offset-shifted videos
├── metadata/
│   └── synthetic_metadata.csv
├── results/
│   └── results.csv
├── metrics/
│   └── metrics_summary.json
├── plots/              # Publication-ready PNG plots
│
├── offset_generation.py
├── run_batch.py
├── compute_metrics.py
└── visualize_results.py
```

### Running the Evaluation

1. **Place original videos** (`.mp4`, `.mov`, or `.avi`) in `evaluation/originals/`.

2. **Generate synthetic dataset** — creates offset-shifted copies of each video at six offsets (-1000, -500, -100, +100, +500, +1000 ms) along with a metadata CSV:
   ```bash
   python -m evaluation.offset_generation
   ```
   This produces **24 files** in `evaluation/synthetic/` (4 videos x 6 offsets) and a 24-row `evaluation/metadata/synthetic_metadata.csv`.

3. **Run batch synchronization** — runs audio (GCC-PHAT) and visual (motion) sync on every synthetic case:
   ```bash
   python -m evaluation.run_batch
   ```
   This produces `evaluation/results/results.csv` with **48 rows** (24 cases x 2 methods), containing estimated offsets, absolute errors, confidence scores, runtimes, and peak CPU/memory usage.

4. **Compute metrics** — aggregates accuracy, confidence validation, and efficiency statistics:
   ```bash
   python -m evaluation.compute_metrics
   ```
   This produces `evaluation/metrics/metrics_summary.json` with metric categories: `accuracy`, `cross_method_agreement`, `confidence_validation`, `efficiency`, and `resource_usage`. A human-readable summary table is also printed to the console.

5. **Generate plots** — produces publication-ready visualizations:
   ```bash
   python -m evaluation.visualize_results
   ```
   This saves **8 plot types** to `evaluation/plots/`:

   | Plot | Description |
   |------|-------------|
   | `error_vs_offset.png` | MAE per offset magnitude (audio vs visual) |
   | `confidence_vs_error.png` | Scatter plot with regression line |
   | `audio_video_diff_histogram.png` | Distribution of audio-visual estimate differences |
   | `runtime_comparison.png` | Mean runtime by method |
   | `error_distribution_boxplot.png` | Boxplot of error distribution by method and offset |
   | `resource_usage.png` | Peak CPU and memory usage by method |
   | `before_after/*.png` | Per-case motion signal overlay before & after alignment |
   | `timelines/*.png` | Per-case timeline bars with offset arrows (pad/trim) |

## Troubleshooting

- **FFmpeg not found**: If the application fails during sync or audio extraction, ensure FFmpeg is installed and accessible in your system PATH.
  - **macOS**: `brew install ffmpeg`
  - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add the `bin` folder to your System PATH.
  - **Linux**: `sudo apt install ffmpeg`