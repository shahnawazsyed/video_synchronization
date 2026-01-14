# Synchronization Issues Summary

## Current Status
- **Pipeline:** Logic updated to handle `.mov` and `.mp4` files from `data/raw/`.
- **Method:** Visual synchronization (motion energy correlation) is the default method.
- **Results:**
  - Files are essentially synchronized, but a **1-2 second lag** persists between the reference video (`3_video_b.mov`) and others.
  - Video padding issue for positive offsets (delays) has been **fixed** using re-encoding.

## Known Issues

### 1. Visual Synchronization Precision
- **Symptom:** Videos are globally aligned (same events shown), but not frame-perfect. Reference video `b` appears 1-2s "behind" others (shows events later).
- **Cause:**
  - Low resolution (240p-480p) limits motion tracking precision.
  - "Sit down" event is relatively slow (takes 2-3 seconds), creating broad correlation peaks.
  - Correlation logic finds the *best mathematical match* of motion curves, which may be offset by 1-2s from the *perceptual* event center.
- **Attempted Fixes:**
  - Increased motion extraction resolution (downsample 4 -> 2).
  - Added Center Cropping (ROI) to focus on the subject.
  - Result: Minimal improvement (<0.3s shift), confirming the limitation is likely inherent to the motion features.

### 2. Audio Synchronization Failure
- **Symptom:** Audio-based sync (GCC-PHAT) fails completely, producing random offsets (errors >10s).
- **Cause:**
  - Quiet ambient audio with no distinct, sharp events (impulses/claps).
  - High background noise correlation leads to false positives.
  - Alternative methods (Energy Envelope, Spectral) also failed to converge on consistent results.

## Recommendations for Future Work

### 1. Advanced Visual Features
Instead of simple frame-differencing (motion energy), implement more robust features:
- **Optical Flow**: Calculate consistent motion vectors to better define the "start" of a movement.
- **Pose Estimation**: Use MediaPipe/OpenPose to track specific keypoints (e.g., hips/knees) to pinpoint the exact frame a person sits. This would be much more precise than global motion energy.

### 2. Constraints & Code Quality
- **Strinct Constraint:** Manual offsets or fine-tuning are **NOT PERMISSIBLE**. The solution must be fully automated.
- **Code Cleanup:** The codebase requires refactoring to remove unused legacy modules (e.g., `test_manual_offsets.py`, old verify logic) and consolidate duplicate logic.

### 3. Alternative Motion Correlation
- Explore **Dynamic Time Warping (DTW)** on the motion signals instead of global cross-correlation. This might handle slight speed variations better, though it assumes linear time for applying a single offset.
