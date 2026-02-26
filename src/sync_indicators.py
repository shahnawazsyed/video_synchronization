"""
sync_indicators.py
-------------------
Generates visual sync indicator images with bounding boxes around detected
motion regions. These images help visualize WHERE motion was detected in each
video frame, providing a visual verification of the synchronization process.

Outputs saved to the project-level /results directory:
  - Per-video annotated peak-motion frames with bounding boxes
  - Side-by-side comparison composites at sync points
  - Motion timeline summary strip
"""

import os
import logging
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional

# Force non-interactive backend for matplotlib (required for background threads)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .utils import setup_logger, log_execution_time

logger = setup_logger(__name__)

# ── Visual styling constants ──────────────────────────────────────────────────
BOX_COLOR_PRIMARY = (0, 217, 255)       # Cyan  (#00D9FF) - main bounding box
BOX_COLOR_SECONDARY = (0, 204, 102)     # Green (#00CC66) - secondary boxes
TEXT_BG_COLOR = (0, 0, 0)               # Black background for labels
TEXT_COLOR = (255, 255, 255)            # White text
HEADER_COLOR = (0, 119, 255)            # Blue  (#0077FF) - header bar
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1


def _find_motion_regions(frame_curr: np.ndarray,
                         frame_prev: np.ndarray,
                         min_area: int = 500) -> List[Tuple[int, int, int, int, float]]:
    """
    Detect motion regions between two consecutive frames.

    Returns list of (x, y, w, h, intensity) tuples for each detected region,
    sorted by intensity (strongest first).
    """
    gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)
    gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    gray_curr = cv2.GaussianBlur(gray_curr, (5, 5), 0)
    gray_prev = cv2.GaussianBlur(gray_prev, (5, 5), 0)

    # Compute absolute difference
    diff = cv2.absdiff(gray_curr, gray_prev)

    # Threshold to get binary motion mask
    _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

    # Dilate to connect nearby regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        # Calculate motion intensity as mean pixel difference in the region
        region_diff = diff[y:y+h, x:x+w]
        intensity = float(np.mean(region_diff)) / 255.0
        regions.append((x, y, w, h, intensity))

    # Sort by intensity, strongest first
    regions.sort(key=lambda r: r[4], reverse=True)
    return regions


def _draw_bounding_boxes(frame: np.ndarray,
                         regions: List[Tuple[int, int, int, int, float]],
                         video_name: str = "",
                         frame_idx: int = 0,
                         timestamp: float = 0.0) -> np.ndarray:
    """
    Draw styled bounding boxes and labels on a frame.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Draw header bar
    header_h = 36
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), HEADER_COLOR, -1)
    cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

    # Header text
    header_text = f"{video_name}  |  Frame {frame_idx}  |  {timestamp:.2f}s"
    cv2.putText(annotated, header_text, (10, 24),
                FONT, 0.55, TEXT_COLOR, 1, cv2.LINE_AA)

    # Draw bounding boxes
    for i, (x, y, bw, bh, intensity) in enumerate(regions):
        color = BOX_COLOR_PRIMARY if i == 0 else BOX_COLOR_SECONDARY

        # Draw box with slight transparency
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, BOX_THICKNESS)

        # Corner accents (small L shapes at corners)
        corner_len = min(15, bw // 4, bh // 4)
        # Top-left
        cv2.line(annotated, (x, y), (x + corner_len, y), color, BOX_THICKNESS + 1)
        cv2.line(annotated, (x, y), (x, y + corner_len), color, BOX_THICKNESS + 1)
        # Top-right
        cv2.line(annotated, (x + bw, y), (x + bw - corner_len, y), color, BOX_THICKNESS + 1)
        cv2.line(annotated, (x + bw, y), (x + bw, y + corner_len), color, BOX_THICKNESS + 1)
        # Bottom-left
        cv2.line(annotated, (x, y + bh), (x + corner_len, y + bh), color, BOX_THICKNESS + 1)
        cv2.line(annotated, (x, y + bh), (x, y + bh - corner_len), color, BOX_THICKNESS + 1)
        # Bottom-right
        cv2.line(annotated, (x + bw, y + bh), (x + bw - corner_len, y + bh), color, BOX_THICKNESS + 1)
        cv2.line(annotated, (x + bw, y + bh), (x + bw, y + bh - corner_len), color, BOX_THICKNESS + 1)

        # Label with intensity
        label = f"Motion {i+1}: {intensity:.0%}"
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
        label_y = max(y - 8, th + header_h + 4)

        # Label background
        cv2.rectangle(annotated,
                      (x, label_y - th - 4),
                      (x + tw + 8, label_y + 4),
                      TEXT_BG_COLOR, -1)
        cv2.putText(annotated, label, (x + 4, label_y),
                    FONT, FONT_SCALE, color, FONT_THICKNESS, cv2.LINE_AA)

    # Region count badge (bottom-right)
    count_text = f"{len(regions)} region{'s' if len(regions) != 1 else ''}"
    (tw, th), _ = cv2.getTextSize(count_text, FONT, FONT_SCALE, FONT_THICKNESS)
    cv2.rectangle(annotated, (w - tw - 16, h - th - 16), (w, h), TEXT_BG_COLOR, -1)
    cv2.putText(annotated, count_text, (w - tw - 8, h - 8),
                FONT, FONT_SCALE, BOX_COLOR_PRIMARY, FONT_THICKNESS, cv2.LINE_AA)

    return annotated


def _find_peak_motion_frames(video_path: str,
                             num_peaks: int = 5,
                             step: int = 2,
                             min_area: int = 500) -> List[dict]:
    """
    Scan a video and return frames with the highest motion activity.

    Returns list of dicts with keys: frame_idx, timestamp, frame, prev_frame, energy, regions
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # First pass: compute motion energy per sampled frame
    frame_energies = []
    prev_frame = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0 and prev_frame is not None:
            gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray_curr, gray_prev)
            energy = float(np.mean(diff))
            frame_energies.append((frame_idx, energy))

        if frame_idx % step == 0:
            prev_frame = frame.copy()

        frame_idx += 1

    cap.release()

    if not frame_energies:
        return []

    # Sort by energy and pick top peaks (with minimum spacing)
    frame_energies.sort(key=lambda x: x[1], reverse=True)
    min_spacing = max(int(fps * 1.0), 10)  # At least 1 second apart

    selected = []
    selected_indices = set()

    for idx, energy in frame_energies:
        if len(selected) >= num_peaks:
            break
        # Check spacing from already selected
        too_close = False
        for sel_idx in selected_indices:
            if abs(idx - sel_idx) < min_spacing:
                too_close = True
                break
        if not too_close:
            selected.append((idx, energy))
            selected_indices.add(idx)

    # Sort selected by frame index
    selected.sort(key=lambda x: x[0])

    # Second pass: extract the actual frames and detect regions
    results = []
    cap = cv2.VideoCapture(video_path)

    for target_idx, energy in selected:
        # Seek to one frame before
        prev_idx = max(0, target_idx - step)
        cap.set(cv2.CAP_PROP_POS_FRAMES, prev_idx)
        ret_prev, prev_frame = cap.read()

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret_curr, curr_frame = cap.read()

        if not ret_prev or not ret_curr:
            continue

        regions = _find_motion_regions(curr_frame, prev_frame, min_area=min_area)
        timestamp = target_idx / fps if fps > 0 else 0.0

        results.append({
            "frame_idx": target_idx,
            "timestamp": timestamp,
            "frame": curr_frame,
            "prev_frame": prev_frame,
            "energy": energy,
            "regions": regions,
        })

    cap.release()
    return results


def _create_comparison_composite(frames_by_video: Dict[str, np.ndarray],
                                 title: str = "") -> np.ndarray:
    """
    Create a side-by-side comparison image from multiple annotated frames.
    """
    if not frames_by_video:
        return np.zeros((100, 400, 3), dtype=np.uint8)

    # Resize all frames to same height
    target_h = 480
    resized = []
    for name, frame in frames_by_video.items():
        h, w = frame.shape[:2]
        scale = target_h / h
        new_w = int(w * scale)
        resized_frame = cv2.resize(frame, (new_w, target_h))
        resized.append(resized_frame)

    # Add separators between frames
    separator = np.full((target_h, 3, 3), 60, dtype=np.uint8)  # Dark gray line

    parts = []
    for i, frame in enumerate(resized):
        if i > 0:
            parts.append(separator)
        parts.append(frame)

    composite = np.hstack(parts)

    # Add title bar if provided
    if title:
        title_h = 40
        title_bar = np.zeros((title_h, composite.shape[1], 3), dtype=np.uint8)
        cv2.putText(title_bar, title, (10, 28),
                    FONT, 0.7, BOX_COLOR_PRIMARY, 1, cv2.LINE_AA)
        composite = np.vstack([title_bar, composite])

    return composite


def _create_motion_timeline(video_path: str,
                            peak_frames: List[dict],
                            output_path: str,
                            video_name: str = ""):
    """
    Create a timeline strip showing peak motion moments as thumbnails.
    """
    if not peak_frames:
        return

    fig, axes = plt.subplots(1, len(peak_frames),
                             figsize=(4 * len(peak_frames), 4))
    if len(peak_frames) == 1:
        axes = [axes]

    fig.suptitle(f"Motion Timeline: {video_name}",
                 fontsize=14, color='#00D9FF', fontweight='bold', y=0.98)
    fig.patch.set_facecolor('#1a1a2e')

    for ax, peak in zip(axes, peak_frames):
        # Convert BGR to RGB for matplotlib
        frame_rgb = cv2.cvtColor(peak["frame"], cv2.COLOR_BGR2RGB)

        # Draw boxes on the frame for the thumbnail
        for x, y, w, h, intensity in peak["regions"][:3]:
            cv2.rectangle(frame_rgb, (x, y), (x + w, y + h),
                          (0, 217, 255), 2)

        ax.imshow(frame_rgb)
        ax.set_title(f"t={peak['timestamp']:.1f}s\n{len(peak['regions'])} regions",
                     fontsize=9, color='white')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#333')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()


def generate_sync_indicators(video_dir: str,
                             selected_files: List[str],
                             offsets: Dict[str, float],
                             results_dir: str,
                             num_peaks: int = 5) -> List[str]:
    """
    Generate visual sync indicator images and save to results directory.

    Parameters
    ----------
    video_dir : str
        Directory containing original video files.
    selected_files : list of str
        Filenames of the videos being synchronized.
    offsets : dict
        Mapping of filename → computed offset in seconds.
    results_dir : str
        Output directory for indicator images (project root /results).
    num_peaks : int
        Number of peak-motion frames to capture per video.

    Returns
    -------
    list of str
        Paths to all generated indicator images.
    """
    logger.info("Generating visual sync indicators...")
    os.makedirs(results_dir, exist_ok=True)

    generated_files = []

    # ── Per-video: annotated peak motion frames ──────────────────────────────
    all_peaks = {}  # video_name -> list of peak dicts

    for fname in selected_files:
        video_path = os.path.join(video_dir, fname)
        video_name = os.path.splitext(fname)[0]

        if not os.path.exists(video_path):
            logger.warning("  Skipping %s (file not found)", fname)
            continue

        logger.info("  Analyzing %s for peak motion frames...", fname)

        with log_execution_time(logger, f"Peak detection for {fname}"):
            peaks = _find_peak_motion_frames(video_path, num_peaks=num_peaks)

        if not peaks:
            logger.warning("  No motion peaks found in %s", fname)
            continue

        all_peaks[fname] = peaks
        logger.info("  Found %d peak motion frames in %s", len(peaks), fname)

        # Save individual annotated frames
        for i, peak in enumerate(peaks):
            annotated = _draw_bounding_boxes(
                peak["frame"],
                peak["regions"],
                video_name=video_name,
                frame_idx=peak["frame_idx"],
                timestamp=peak["timestamp"]
            )

            out_path = os.path.join(results_dir,
                                    f"{video_name}_motion_peak_{i+1}.png")
            cv2.imwrite(out_path, annotated)
            generated_files.append(out_path)
            logger.debug("  Saved: %s", os.path.basename(out_path))

        # Save motion timeline strip
        timeline_path = os.path.join(results_dir,
                                     f"{video_name}_motion_timeline.png")
        _create_motion_timeline(video_path, peaks, timeline_path,
                                video_name=video_name)
        generated_files.append(timeline_path)

    # ── Cross-video comparison at sync points ────────────────────────────────
    if len(all_peaks) >= 2:
        logger.info("  Creating cross-video comparison composites...")

        # Find the closest matching peak times across videos
        # Use the first video's peaks as reference timestamps
        ref_file = selected_files[0]
        if ref_file in all_peaks:
            for peak_idx, ref_peak in enumerate(all_peaks[ref_file]):
                ref_time = ref_peak["timestamp"]

                # For each other video, find the frame closest in
                # sync-adjusted time
                composite_frames = {}
                ref_name = os.path.splitext(ref_file)[0]
                annotated_ref = _draw_bounding_boxes(
                    ref_peak["frame"],
                    ref_peak["regions"],
                    video_name=ref_name,
                    frame_idx=ref_peak["frame_idx"],
                    timestamp=ref_peak["timestamp"]
                )
                composite_frames[ref_name] = annotated_ref

                for other_file in selected_files[1:]:
                    if other_file not in all_peaks:
                        continue

                    other_name = os.path.splitext(other_file)[0]
                    offset_diff = offsets.get(other_file, 0) - offsets.get(ref_file, 0)

                    # Find closest peak in the other video
                    adjusted_time = ref_time + offset_diff
                    best_peak = min(
                        all_peaks[other_file],
                        key=lambda p: abs(p["timestamp"] - adjusted_time)
                    )

                    annotated_other = _draw_bounding_boxes(
                        best_peak["frame"],
                        best_peak["regions"],
                        video_name=other_name,
                        frame_idx=best_peak["frame_idx"],
                        timestamp=best_peak["timestamp"]
                    )
                    composite_frames[other_name] = annotated_other

                # Create side-by-side comparison
                composite = _create_comparison_composite(
                    composite_frames,
                    title=f"Sync Point {peak_idx+1}  |  "
                          f"Reference t={ref_time:.2f}s  |  "
                          f"Offset-adjusted comparison"
                )

                comp_path = os.path.join(results_dir,
                                         f"sync_comparison_{peak_idx+1}.png")
                cv2.imwrite(comp_path, composite)
                generated_files.append(comp_path)

    # ── Summary: offset visualization ────────────────────────────────────────
    _create_offset_summary(selected_files, offsets, results_dir)
    generated_files.append(os.path.join(results_dir, "offset_summary.png"))

    logger.info("Sync indicators complete. Generated %d images in %s",
                len(generated_files), results_dir)
    return generated_files


def _create_offset_summary(selected_files: List[str],
                           offsets: Dict[str, float],
                           results_dir: str):
    """
    Create a visual summary chart showing the computed offsets for all videos.
    """
    fig, ax = plt.subplots(figsize=(10, max(3, len(selected_files) * 0.8)))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    names = [os.path.splitext(f)[0] for f in selected_files]
    offset_vals = [offsets.get(f, 0.0) for f in selected_files]

    colors = []
    for val in offset_vals:
        if abs(val) < 0.01:
            colors.append('#00D9FF')   # Reference (zero offset)
        elif val > 0:
            colors.append('#00CC66')   # Positive offset (delayed)
        else:
            colors.append('#FF6B6B')   # Negative offset (trimmed)

    bars = ax.barh(names, offset_vals, color=colors, height=0.5, edgecolor='white',
                   linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, offset_vals):
        label_x = bar.get_width()
        ha = 'left' if val >= 0 else 'right'
        offset_px = 5 if val >= 0 else -5
        ax.text(label_x + (0.1 if val >= 0 else -0.1), bar.get_y() + bar.get_height() / 2,
                f'{val:+.3f}s', va='center', ha=ha,
                fontsize=10, color='white', fontweight='bold')

    ax.set_xlabel('Offset (seconds)', color='#aaa', fontsize=11)
    ax.set_title('Video Synchronization Offsets', color='#00D9FF',
                 fontsize=14, fontweight='bold', pad=15)
    ax.axvline(x=0, color='#555', linewidth=1, linestyle='--')
    ax.tick_params(colors='#aaa')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#00D9FF', label='Reference'),
        Patch(facecolor='#00CC66', label='Delayed (+)'),
        Patch(facecolor='#FF6B6B', label='Trimmed (-)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right',
              facecolor='#2a2a3e', edgecolor='#444', labelcolor='white',
              fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(results_dir, "offset_summary.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    logger.info("  Saved offset summary: %s", out_path)
