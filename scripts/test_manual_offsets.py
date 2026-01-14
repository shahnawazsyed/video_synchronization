"""
Manual offset testing tool for video synchronization.
Allows testing different offset values to find correct alignment.
"""
import cv2
import numpy as np
import os
import argparse

def load_video_frame(cap, frame_num):
    """Load a specific frame from video"""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    return frame if ret else None

def test_offsets(video_dir, output_dir, offsets_dict, test_frame_time=10.0):
    """
    Apply manual offsets and show frames side by side for comparison.
    
    Args:
        video_dir: Directory with source videos
        output_dir: Where to save comparison images
        offsets_dict: Dict of {video_name: offset_seconds}
        test_frame_time: Time in seconds to sample frames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    videos = list(offsets_dict.keys())
    caps = {}
    fps_dict = {}
    
    # Open all videos
    for vid in videos:
        path = os.path.join(video_dir, vid)
        if os.path.exists(path):
            cap = cv2.VideoCapture(path)
            caps[vid] = cap
            fps_dict[vid] = cap.get(cv2.CAP_PROP_FPS)
            print(f"Loaded {vid}: {cap.get(cv2.CAP_PROP_FRAME_COUNT):.0f} frames at {fps_dict[vid]:.2f} fps")
        else:
            print(f"Warning: {vid} not found")
    
    if not caps:
        print("No videos loaded!")
        return
    
    # Test multiple time points
    ref_video = videos[0]
    ref_fps = fps_dict[ref_video]
    
    print(f"\nReference video: {ref_video}")
    print(f"Testing offsets: {offsets_dict}")
    print()
    
    for test_time in [5.0, 10.0, 15.0, 20.0]:
        frames = []
        labels = []
        
        for vid in videos:
            if vid not in caps:
                continue
                
            cap = caps[vid]
            fps = fps_dict[vid]
            offset = offsets_dict[vid]
            
            # Adjusted time with offset
            adjusted_time = test_time - offset
            frame_num = int(adjusted_time * fps)
            
            if frame_num < 0 or frame_num >= cap.get(cv2.CAP_PROP_FRAME_COUNT):
                print(f"  {vid}: frame {frame_num} out of range")
                continue
            
            frame = load_video_frame(cap, frame_num)
            if frame is not None:
                # Resize for display
                h, w = frame.shape[:2]
                new_w = 400
                new_h = int(h * new_w / w)
                frame = cv2.resize(frame, (new_w, new_h))
                
                # Add label
                label = f"{vid} (offset={offset:+.2f}s, frame={frame_num})"
                cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 1)
                
                frames.append(frame)
                labels.append(vid)
        
        if len(frames) >= 2:
            # Create 2x2 grid
            while len(frames) < 4:
                frames.append(np.zeros_like(frames[0]))
            
            row1 = np.hstack(frames[:2])
            row2 = np.hstack(frames[2:4])
            grid = np.vstack([row1, row2])
            
            # Add time label  
            cv2.putText(grid, f"Ref Time: {test_time:.1f}s", (10, grid.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out_path = os.path.join(output_dir, f"comparison_t{test_time:.0f}s.jpg")
            cv2.imwrite(out_path, grid)
            print(f"Saved {out_path}")
    
    # Cleanup
    for cap in caps.values():
        cap.release()

def main():
    parser = argparse.ArgumentParser(description='Test manual video offsets')
    parser.add_argument('--offsets', type=str, default='0,0,0,0',
                       help='Comma-separated offsets for b,h,r,v (default: 0,0,0,0)')
    args = parser.parse_args()
    
    # Parse offsets
    offset_values = [float(x) for x in args.offsets.split(',')]
    
    videos = ["3_video_b.mov", "3_video_h.mp4", "3_video_r.mp4", "3_video_v.mp4"]
    
    if len(offset_values) != len(videos):
        print(f"Error: expected {len(videos)} offset values, got {len(offset_values)}")
        return
    
    offsets_dict = dict(zip(videos, offset_values))
    
    print("="*60)
    print("MANUAL OFFSET TESTING")
    print("="*60)
    
    # Test with current algorithm offsets
    print("\n1. Testing with ALGORITHM offsets:")
    algo_offsets = {
        "3_video_b.mov": 0.000,
        "3_video_h.mp4": -0.919,
        "3_video_r.mp4": -5.761,
        "3_video_v.mp4": -7.894
    }
    test_offsets("data/raw", "outputs/test_algo_offsets", algo_offsets)
    
    # Test with user-provided offsets
    if args.offsets != '0,0,0,0':
        print(f"\n2. Testing with USER offsets: {offsets_dict}")
        test_offsets("data/raw", "outputs/test_user_offsets", offsets_dict)
    
    print("\nCheck outputs/ folder for comparison images.")

if __name__ == "__main__":
    main()
