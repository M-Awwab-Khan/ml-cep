"""Clip -> fixed-length feature vector, with named feature groups for ablation.

Framing: n_fft=1024 samples (64 ms), hop=256 (16 ms) at 16 kHz.
Each per-frame feature (a matrix: n_coefficients x n_frames) is pooled over
time into [mean, std] per coefficient, so the vector length does not depend
on how long the clip is.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import librosa
import numpy as np
from joblib import Parallel, delayed

from common import CACHE_DIR, SR, load_audio, load_table, lowpass

N_FFT, HOP = 1024, 256
N_MFCC, N_MELS, N_CHROMA, N_CONTRAST_BANDS, N_FFT_BANDS = 20, 64, 12, 6, 32


def _pool(m: np.ndarray) -> np.ndarray:
    """(coeffs x frames) -> [mean per coeff..., std per coeff...]."""
    return np.concatenate([m.mean(axis=1), m.std(axis=1)])


def extract(y: np.ndarray) -> dict[str, np.ndarray]:
    """Return {group_name: 1-D vector} for one (already filtered) clip."""
    kw = dict(sr=SR, n_fft=N_FFT, hop_length=HOP)
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) ** 2  # power spectrogram

    mel = librosa.feature.melspectrogram(S=S, sr=SR, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=1.0)  # absolute dB, keeps loudness
    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=N_MFCC)
    chroma = librosa.feature.chroma_stft(S=S, sr=SR, n_chroma=N_CHROMA)
    centroid = librosa.feature.spectral_centroid(S=np.sqrt(S), sr=SR)
    bandwidth = librosa.feature.spectral_bandwidth(S=np.sqrt(S), sr=SR)
    contrast = librosa.feature.spectral_contrast(
        S=np.sqrt(S), sr=SR, n_bands=N_CONTRAST_BANDS, fmin=100.0)

    # "Fourier transform" group: long-term average spectrum (mean over time of
    # the STFT magnitude), pooled into equal-width 250 Hz bands, in dB.
    mean_mag = np.sqrt(S).mean(axis=1)                       # (513,)
    edges = np.linspace(0, len(mean_mag) - 1, N_FFT_BANDS + 1).astype(int)
    fft_bands = np.array([mean_mag[a:b + 1].mean() for a, b in zip(edges[:-1], edges[1:])])
    fft_db = 20 * np.log10(fft_bands + 1e-8)

    return {
        "mfcc": _pool(mfcc),
        "mel": _pool(mel_db),
        "chroma": _pool(chroma),
        "centroid": _pool(centroid),
        "bandwidth": _pool(bandwidth),
        "contrast": _pool(contrast),
        "fft": fft_db,
    }


def _one(path: str, cutoff: float | None) -> dict[str, np.ndarray]:
    y = load_audio(path)
    if cutoff is not None:
        y = lowpass(y, cutoff)
    return extract(y)


def build(cutoff: float | None, n_jobs: int = 10, limit: int | None = None):
    """Extract all clips in parallel and cache to cache/features_<cutoff>.npz."""
    df = load_table()
    if limit:
        df = df.iloc[:limit]
    out = Parallel(n_jobs=n_jobs, batch_size=32)(
        delayed(_one)(p, cutoff) for p in df["path"])
    groups = list(out[0])
    X = np.stack([np.concatenate([o[g] for g in groups]) for o in out])
    group_of = np.concatenate([[g] * len(out[0][g]) for g in groups])
    tag = "raw" if cutoff is None else f"lp{int(cutoff)}"
    CACHE_DIR.mkdir(exist_ok=True)
    if not limit:
        np.savez(CACHE_DIR / f"features_{tag}.npz", X=X, group_of=group_of,
                 paths=df["path"].to_numpy().astype(str))
    return X, group_of


if __name__ == "__main__":
    import time

    # Demo on one clip: show shapes at every stage.
    df = load_table()
    y = lowpass(load_audio(df["path"].iloc[0]), 4000)
    print(f"clip: {len(y)} samples = {len(y)/SR:.2f} s")
    print(f"frames: {1 + len(y)//HOP}  (frame 64 ms, hop 16 ms)")
    for g, v in extract(y).items():
        print(f"  {g:10s} -> {v.shape[0]:4d} numbers")
    print("  TOTAL      ->", sum(v.shape[0] for v in extract(y).values()))

    # Check a short vs a long clip give the same vector length.
    short = y[:SR]
    print("1 s clip total:", sum(v.shape[0] for v in extract(short).values()))

    t = time.time(); X, _ = build(4000, limit=100)
    dt = time.time() - t
    print(f"100 clips: {dt:.1f}s -> est. all 7442: {dt*74.42/60:.1f} min; "
          f"NaN/inf present: {not np.isfinite(X).all()}")
