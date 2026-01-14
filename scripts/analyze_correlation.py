"""
Diagnostic script to visualize GCC-PHAT correlation and find correct peaks
"""
import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate
import matplotlib.pyplot as plt
import os
import itertools

def gcc_phat_full(sig1, sig2, fs, max_offset_sec=20.0):
    """
    Compute GCC-PHAT and return full correlation function for visualization.
    """
    n = len(sig1) + len(sig2) - 1
    n_fft = 2 ** int(np.ceil(np.log2(n)))
    
    # FFT
    SIG1 = np.fft.rfft(sig1, n_fft)
    SIG2 = np.fft.rfft(sig2, n_fft)
    
    # Cross-power spectrum with PHAT weighting
    R = SIG1 * np.conj(SIG2)
    R_phat = R / (np.abs(R) + 1e-10)
    
    # IFFT to get correlation
    cc = np.fft.irfft(R_phat, n_fft)
    cc = np.real(cc)
    
    # Create lag array
    max_lag = int(max_offset_sec * fs)
    mid = len(cc) // 2
    
    # Extract relevant portion
    cc = np.concatenate([cc[-max_lag:], cc[:max_lag+1]])
    lags = np.arange(-max_lag, max_lag + 1) / fs
    
    return lags, cc

def find_peaks(cc, lags, n_peaks=5):
    """Find top N peaks in correlation"""
    # Find local maxima
    peaks = []
    for i in range(1, len(cc) - 1):
        if cc[i] > cc[i-1] and cc[i] > cc[i+1]:
            peaks.append((cc[i], lags[i]))
    
    # Sort by value and return top N
    peaks.sort(reverse=True)
    return peaks[:n_peaks]

def main():
    audio_dir = "data/audio/selected"
    files = ["3_video_b.wav", "3_video_h.wav", "3_video_r.wav", "3_video_v.wav"]
    
    # Load audio
    audio = {}
    for f in files:
        sr, sig = wavfile.read(os.path.join(audio_dir, f))
        if len(sig.shape) == 2:
            sig = sig.mean(axis=1)
        sig = sig.astype(float)
        sig = sig / (np.max(np.abs(sig)) + 1e-10)
        audio[f] = (sr, sig)
        print(f"{f}: {len(sig)/sr:.2f}s duration")
    
    print()
    
    # Create visualization for each pair
    pairs = list(itertools.combinations(files, 2))
    
    fig, axes = plt.subplots(len(pairs), 1, figsize=(14, 3*len(pairs)))
    
    for idx, (f1, f2) in enumerate(pairs):
        sr, sig1 = audio[f1]
        _, sig2 = audio[f2]
        
        lags, cc = gcc_phat_full(sig1, sig2, sr, max_offset_sec=20.0)
        peaks = find_peaks(cc, lags, n_peaks=5)
        
        ax = axes[idx]
        ax.plot(lags, cc, 'b-', alpha=0.7, linewidth=0.5)
        ax.set_xlabel('Lag (seconds)')
        ax.set_ylabel('Correlation')
        ax.set_title(f'{f1} <-> {f2}')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-20, 20)
        
        # Mark peaks
        for i, (val, lag) in enumerate(peaks):
            color = 'red' if i == 0 else 'orange'
            ax.axvline(lag, color=color, linestyle='--', alpha=0.7)
            ax.annotate(f'{lag:.2f}s', (lag, val), textcoords="offset points", 
                       xytext=(5, 5), fontsize=8)
        
        print(f"{f1} <-> {f2}:")
        print(f"  Top peaks: {[(f'{lag:.2f}s', f'{val:.3f}') for val, lag in peaks]}")
        print()
    
    plt.tight_layout()
    plt.savefig('outputs/correlation_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved correlation plot to outputs/correlation_analysis.png")
    plt.close()

if __name__ == "__main__":
    main()
