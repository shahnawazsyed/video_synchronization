import subprocess, numpy as np
from scipy.io import wavfile
import os, scipy.signal as sig


"""
Remove later
"""



RAW_DIR = "../data/raw"
TMP_AUDIO = "../data/audio_tmp"
os.makedirs(TMP_AUDIO, exist_ok=True)

def extract_audio_ffmpeg(video_path, wav_path, sr=16000):
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1", "-ar", str(sr),
        wav_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def read_wav(path):
    sr, y = wavfile.read(path)
    y = y.astype(np.float32)
    y /= np.max(np.abs(y)) + 1e-9
    return sr, y

files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith(".mp4")])
signals = {}
sr = 16000

for f in files:
    in_path = os.path.join(RAW_DIR, f)
    out_path = os.path.join(TMP_AUDIO, f.replace(".mp4", ".wav"))
    extract_audio_ffmpeg(in_path, out_path, sr)
    sr, y = read_wav(out_path)
    signals[f] = y[:sr*10]
    print(f"✅ Extracted {f}: {len(y)/sr:.1f}s audio")

# cross-correlate first vs others
ref_name = files[2]
ref_sig = signals[ref_name]
for f, y in signals.items():
    if f == ref_name: continue
    corr = sig.correlate(y, ref_sig, mode="full")
    lags = sig.correlation_lags(len(y), len(ref_sig))
    lag = lags[np.argmax(corr)] / sr
    print(f"{f} offset vs {ref_name}: {lag:.3f} s")
