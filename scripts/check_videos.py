import os
import cv2
import numpy as np
import math


"""
Remove later
"""

# Automatically resolve project root if running from inside src/
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")

def load_video_captures(path=RAW_DIR):
    """Load all video files in the given directory."""
    files = sorted([f for f in os.listdir(path) if f.lower().endswith((".mp4", ".mov", ".avi"))])
    if not files:
        raise FileNotFoundError(f"No video files found in {path}")
    print(f"Found {len(files)} videos:")
    for i, f in enumerate(files):
        print(f"  [{i+1}] {f}")
    caps = [cv2.VideoCapture(os.path.join(path, f)) for f in files]
    return files, caps

def get_video_properties(cap):
    """Get FPS, resolution, and frame count from a video."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps, width, height, frames

def label_frame(frame, label, font_scale=0.7, thickness=2):
    """Draw filename label on top-left corner of frame."""
    text_color = (255, 255, 255)
    bg_color = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
    cv2.rectangle(frame, (5, 5), (10 + text_w, 10 + text_h), bg_color, -1)
    cv2.putText(frame, label, (8, 8 + text_h), font, font_scale, text_color, thickness, cv2.LINE_AA)
    return frame

def arrange_in_grid(frames, grid_shape):
    """Arrange list of frames into a grid layout."""
    rows, cols = grid_shape
    h, w = frames[0].shape[:2]
    while len(frames) < rows * cols:
        frames.append(np.zeros((h, w, 3), dtype=np.uint8))
    row_blocks = [np.hstack(frames[r * cols:(r + 1) * cols]) for r in range(rows)]
    return np.vstack(row_blocks)

def read_frames_at_position(caps, labels, frame_idx, resize_width=400):
    """Read one frame from each video at a given frame index."""
    frames = []
    for cap, label in zip(caps, labels):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            # Blank filler if this stream ended
            if frames:
                h, w = frames[0].shape[:2]
            else:
                h, w = 240, 320
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            label = f"{label} (end)"
        h, w = frame.shape[:2]
        scale = resize_width / w
        frame = cv2.resize(frame, (resize_width, int(h * scale)))
        frame = label_frame(frame, label)
        frames.append(frame)
    return frames

def show_grid_with_seekbar(caps, labels, grid_shape=None, resize_width=400):
    """Display multiple videos in a grid with playback + scrollbar + controls."""
    n = len(caps)
    if not n:
        return

    if grid_shape is None:
        side = int(math.ceil(math.sqrt(n)))
        grid_shape = (side, side)

    frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
    fps_values = [cap.get(cv2.CAP_PROP_FPS) for cap in caps]
    total_frames = min(frame_counts)
    fps = min(fps_values) if fps_values else 30
    delay = int(1000 / fps)

    window_name = "Multi-View Sync Grid (space=play/pause, ←/→ seek, q=quit)"
    cv2.namedWindow(window_name)

    playing = True
    frame_idx = 0

    def update_frame(pos):
        nonlocal frame_idx
        frame_idx = pos

    cv2.createTrackbar("Frame", window_name, 0, total_frames - 1, update_frame)

    print(f"\n▶ Loaded {n} videos ({total_frames} frames @ {fps:.2f} FPS)")
    print("Controls: [space]=Play/Pause, ←/→=Seek, slider=Jump, [q]=Quit")

    while True:
        if playing:
            frame_idx += 1
            if frame_idx >= total_frames:
                frame_idx = 0
        frames = read_frames_at_position(caps, labels, frame_idx, resize_width)
        grid_frame = arrange_in_grid(frames, grid_shape)
        cv2.imshow(window_name, grid_frame)
        cv2.setTrackbarPos("Frame", window_name, frame_idx)

        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Spacebar toggle
            playing = not playing
        elif key == 81:  # Left arrow
            frame_idx = max(0, frame_idx - 1)
        elif key == 83:  # Right arrow
            frame_idx = min(total_frames - 1, frame_idx + 1)

    cv2.destroyAllWindows()

def main():
    files, caps = load_video_captures(RAW_DIR)
    for i, cap in enumerate(caps):
        fps, w, h, fcount = get_video_properties(cap)
        print(f"✅ {files[i]}: {w}x{h} @ {fps:.2f} FPS — {fcount} frames")

    grid_shape = (3, 3)# if len(caps) == 9 else None
    show_grid_with_seekbar(caps, files, grid_shape=grid_shape)

    for cap in caps:
        cap.release()

if __name__ == "__main__":
    main()
