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

def read_frames_sequentially(caps, labels, current_frame_indices, target_frame_idx, resize_width=400):
    """
    Read frames from captures, seeking only if necessary, with uniform sizing.
    """
    frames = []
    new_indices = []
    target_height = None
    
    for i, (cap, label) in enumerate(zip(caps, labels)):
        curr_idx = current_frame_indices[i]
        
        # Seek if we are not at the target index
        if curr_idx != target_frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
            curr_idx = target_frame_idx
            
        ret, frame = cap.read()
        new_indices.append(curr_idx + 1)
        
        if not ret:
            # Create blank frame if video ended
            h = target_height if target_height else 240
            w = resize_width
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            label = f"{label} (end)"
        else:
            # Determine target height from first valid frame to ensure consistency
            h_orig, w_orig = frame.shape[:2]
            if target_height is None:
                scale = resize_width / w_orig
                target_height = int(h_orig * scale)
            
            # Use fixed size resize to avoid off-by-one pixel errors between different videos
            frame = cv2.resize(frame, (resize_width, target_height))
        
        # Add label
        frame = label_frame(frame, label)
        frames.append(frame)
    
    return frames, new_indices

def show_video_grid(video_dir, title="Video Grid", resize_width=300, grid_size=2, frame_skip=1, selected_files=None):
    """
    Display multiple videos in a grid with playback controls.
    
    Args:
        video_dir: Directory containing video files
        title: Window title
        resize_width: Width of each video thumbnail
        grid_size: Grid dimensions
        frame_skip: Frames to skip per iteration (1 = every frame)
        selected_files: List of specific filenames to display.
    """
    # Load videos
    files, caps = load_video_captures(video_dir)
    n = len(caps)
    
    if n == 0:
        print("No videos to display")
        return
    
    if selected_files is not None:
        selected_set = set(selected_files)
        filtered = [(f, cap) for f, cap in zip(files, caps) if f in selected_set]
        if not filtered:
            print(f"No matching videos found for selection")
            return
        files, caps = zip(*filtered)
        files = list(files); caps = list(caps); n = len(caps)
    
    # Calculate grid size dynamically if needed
    if n > grid_size * grid_size:
        grid_size = math.ceil(math.sqrt(n))
    
    grid_shape = (grid_size, grid_size)
    max_videos = grid_size * grid_size
    # No need to truncate unless we strictly want to adhere to a passed grid_size, 
    # but we just expanded it so it should fit.

    
    # Get video properties
    frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
    fps_values = [cap.get(cv2.CAP_PROP_FPS) for cap in caps]
    total_frames = min(frame_counts) if frame_counts else 0
    avg_fps = sum(fps_values)/len(fps_values) if fps_values else 30
    
    # Target display FPS (higher for smoother playback)
    display_fps = min(avg_fps, 30)
    delay = max(1, int(1000 / display_fps))
    
    print(f"Grid: {grid_shape[0]}x{grid_shape[1]} | FPS: {avg_fps:.1f} | Frames: {total_frames}")
    print("\nControls: [Space] Play/Pause, [<-/->] Seek, [q] Quit\n")
    
    window_name = title
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    playing = True
    frame_idx = 0
    current_caps_indices = [0] * n
    
    def on_trackbar(pos):
        nonlocal frame_idx
        frame_idx = pos
    
    cv2.createTrackbar("Frame", window_name, 0, total_frames - 1, on_trackbar)
    
    while True:
        # Read frames using sequential optimization
        frames, next_indices = read_frames_sequentially(
            caps, files, current_caps_indices, frame_idx, resize_width
        )
        current_caps_indices = next_indices
        
        grid_frame = arrange_in_grid(frames, grid_shape)
        cv2.imshow(window_name, grid_frame)
        cv2.setTrackbarPos("Frame", window_name, frame_idx)
        
        # Handle timing
        key = cv2.waitKey(delay) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            playing = not playing
        elif key == 81: # Left
            frame_idx = max(0, frame_idx - 10)
            playing = False
        elif key == 83: # Right
            frame_idx = min(total_frames - 1, frame_idx + 10)
            playing = False
        
        if playing:
            frame_idx += frame_skip
            if frame_idx >= total_frames:
                frame_idx = 0
        else:
            # If paused, wait for user input without advancing
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'): break
            elif key == ord(' '): playing = True
            elif key == 81: frame_idx = max(0, frame_idx - 1)
            elif key == 83: frame_idx = min(total_frames - 1, frame_idx + 1)
    
    cv2.destroyAllWindows()
    for cap in caps: cap.release()
    
    print(f"Closed {title}\n")

if __name__ == "__main__":
    # Test with augmented videos
    import sys
    
    if len(sys.argv) > 1:
        video_dir = sys.argv[1]
    else:
        video_dir = "data/augmented/"
    
    show_video_grid(video_dir, title=f"Videos from {video_dir}")
