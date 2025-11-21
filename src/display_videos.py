#!/usr/bin/env python3
"""
display_videos.py
-----------------
Display multiple videos in a synchronized grid layout.

This module provides visual verification of video synchronization by showing
videos in a grid with playback controls.

Usage:
    from display_videos import show_video_grid
    
    # Show desynchronized videos
    show_video_grid("data/augmented/", title="Before Sync")
    
    # Show synchronized videos
    show_video_grid("data/synced/", title="After Sync")
"""

import os
import cv2
import numpy as np
import math


def load_video_captures(video_dir):
    """
    Load all video files from a directory.
    
    Args:
        video_dir: Path to directory containing video files
        
    Returns:
        tuple: (filenames, video_captures)
    """
    files = sorted([f for f in os.listdir(video_dir) 
                   if f.lower().endswith((".mp4", ".mov", ".avi"))])
    
    if not files:
        raise FileNotFoundError(f"No video files found in {video_dir}")
    
    print(f"\nFound {len(files)} videos in {video_dir}")
    
    caps = [cv2.VideoCapture(os.path.join(video_dir, f)) for f in files]
    
    # Verify all captures opened successfully
    for i, (cap, fname) in enumerate(zip(caps, files)):
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {fname}")
    
    return files, caps


def get_video_properties(cap):
    """Get FPS, resolution, and frame count from a video capture."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps, width, height, frames


def label_frame(frame, label, font_scale=0.7, thickness=2):
    """
    Draw filename label on top-left corner of frame.
    
    Args:
        frame: Video frame (numpy array)
        label: Text to display
        font_scale: Font size
        thickness: Text thickness
        
    Returns:
        Labeled frame
    """
    text_color = (255, 255, 255)
    bg_color = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
    cv2.rectangle(frame, (5, 5), (10 + text_w, 10 + text_h), bg_color, -1)
    cv2.putText(frame, label, (8, 8 + text_h), font, font_scale, text_color, 
                thickness, cv2.LINE_AA)
    
    return frame


def arrange_in_grid(frames, grid_shape):
    """
    Arrange list of frames into a grid layout.
    
    Args:
        frames: List of video frames (numpy arrays)
        grid_shape: Tuple of (rows, cols)
        
    Returns:
        Single image with frames arranged in grid
    """
    rows, cols = grid_shape
    h, w = frames[0].shape[:2]
    
    # Pad with black frames if needed
    while len(frames) < rows * cols:
        frames.append(np.zeros((h, w, 3), dtype=np.uint8))
    
    # Create horizontal rows
    row_blocks = [np.hstack(frames[r * cols:(r + 1) * cols]) for r in range(rows)]
    
    # Stack rows vertically
    return np.vstack(row_blocks)


def read_frames_at_position(caps, labels, frame_idx, resize_width=400):
    """
    Read one frame from each video at a given frame index.
    
    Args:
        caps: List of video captures
        labels: List of filename labels
        frame_idx: Frame index to read
        resize_width: Target width for each thumbnail
        
    Returns:
        List of frames
    """
    frames = []
    
    for cap, label in zip(caps, labels):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            # Create blank frame if video ended
            if frames:
                h, w = frames[0].shape[:2]
            else:
                h, w = 240, 320
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            label = f"{label} (end)"
        
        # Resize frame
        h, w = frame.shape[:2]
        scale = resize_width / w
        frame = cv2.resize(frame, (resize_width, int(h * scale)))
        
        # Add label
        frame = label_frame(frame, label)
        frames.append(frame)
    
    return frames


def show_video_grid(video_dir, title="Video Grid", resize_width=250, grid_size=2, frame_skip=2, selected_files=None):
    """
    Display multiple videos in a grid with playback controls.
    
    Args:
        video_dir: Directory containing video files
        title: Window title
        resize_width: Width of each video thumbnail in pixels (default 250 for performance)
        grid_size: Grid dimensions (default 2 for 2x2 grid)
        frame_skip: Skip every N frames for faster playback (default 2)
        selected_files: List of specific filenames (e.g., ["F1C7LR_aug.mp4", ...]) to display. If None, defaults to first grid_size^2 videos.
        
    Controls:
        - Space: Play/Pause
        - ←/→: Seek backward/forward
        - Slider: Jump to frame
        - q: Quit
    """
    # Load videos
    files, caps = load_video_captures(video_dir)
    n = len(caps)
    
    if n == 0:
        print("No videos to display")
        return
    
    # If specific files are requested, filter to those (preserve order)
    if selected_files is not None:
        # Ensure we have the exact filenames (including extensions)
        selected_set = set(selected_files)
        filtered = [(f, cap) for f, cap in zip(files, caps) if f in selected_set]
        if not filtered:
            print(f"No matching videos found for selection: {selected_files}")
            return
        files, caps = zip(*filtered)
        files = list(files)
        caps = list(caps)
        n = len(caps)
        print(f"Displaying selected videos: {', '.join(files)}")
    
    # Use fixed grid size for better performance (2x2 = 4 videos)
    grid_shape = (grid_size, grid_size)
    
    # Limit to first grid_size^2 videos for performance if more were passed
    max_videos = grid_size * grid_size
    if n > max_videos:
        print(f"Showing first {max_videos} of {n} videos for performance")
        files = files[:max_videos]
        caps = caps[:max_videos]
        n = max_videos
    
    # Get video properties
    frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
    fps_values = [cap.get(cv2.CAP_PROP_FPS) for cap in caps]
    total_frames = min(frame_counts)
    fps = min(fps_values) if fps_values else 30
    
    # Cap display FPS for smoother performance
    display_fps = min(fps, 15)
    delay = int(1000 / display_fps)
    
    # Print info
    print(f"Grid layout: {grid_shape[0]}x{grid_shape[1]}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps:.2f}")
    print(f"\nControls:")
    print(f"  [Space] = Play/Pause")
    print(f"  [←/→]   = Seek backward/forward")
    print(f"  [Slider] = Jump to frame")
    print(f"  [q]     = Quit and continue\n")
    
    # Create window
    window_name = title
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # State
    playing = True
    frame_idx = 0
    
    def update_frame(pos):
        nonlocal frame_idx
        frame_idx = pos
    
    # Create trackbar
    cv2.createTrackbar("Frame", window_name, 0, total_frames - 1, update_frame)
    
    # Main playback loop
    while True:
        # Auto-advance if playing (with frame skipping for performance)
        if playing:
            frame_idx += frame_skip
            if frame_idx >= total_frames:
                frame_idx = 0  # Loop
        
        # Read and display frames
        frames = read_frames_at_position(caps, files, frame_idx, resize_width)
        grid_frame = arrange_in_grid(frames, grid_shape)
        cv2.imshow(window_name, grid_frame)
        cv2.setTrackbarPos("Frame", window_name, frame_idx)
        
        # Handle keyboard input
        key = cv2.waitKey(delay) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):  # Spacebar toggle
            playing = not playing
        elif key == 81:  # Left arrow
            frame_idx = max(0, frame_idx - 1)
            playing = False
        elif key == 83:  # Right arrow
            frame_idx = min(total_frames - 1, frame_idx + 1)
            playing = False
    
    # Cleanup
    cv2.destroyAllWindows()
    for cap in caps:
        cap.release()
    
    print(f"Closed {title}\n")


if __name__ == "__main__":
    # Test with augmented videos
    import sys
    
    if len(sys.argv) > 1:
        video_dir = sys.argv[1]
    else:
        video_dir = "data/augmented/"
    
    show_video_grid(video_dir, title=f"Videos from {video_dir}")
