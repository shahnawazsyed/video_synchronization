#!/usr/bin/env python3
import numpy as np
from scipy.io import wavfile

# Load the extracted audio
sr, data = wavfile.read("/tmp/f1c12_test.wav")
if data.dtype == np.int16:
    data = data.astype(np.float32) / 32768.0

# Check for silence at the start
threshold = 1e-4
silence_samples = 0
for i, sample in enumerate(data):
    if abs(sample) > threshold:
        silence_samples = i
        break

silence_sec = silence_samples / sr
print(f"Sample rate: {sr} Hz")
print(f"Duration: {len(data) / sr:.3f}s")
print(f"Silence at start: {silence_sec:.4f}s")
print(f"Expected delay (GT): +0.875s")
print(f"First 10 samples: {data[:10]}")
