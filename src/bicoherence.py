"""Bicoherence features: mean/var/skew/kurtosis of magnitude and phase (8 numbers).

b(f1,f2) = E[X(f1) X(f2) conj(X(f1+f2))]
           / sqrt( E|X(f1) X(f2)|^2 * E|X(f1+f2)|^2 )
The expectation E[] is an average over STFT frames of one clip.
|b| is the bicoherence (0..1); angle(b) is the biphase.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import librosa
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import kurtosis, skew

from common import CACHE_DIR, SR, load_audio, load_table, lowpass

N_FFT, HOP, KMAX = 256, 128, 64  # 62.5 Hz bins; f1,f2 in 1..64 so f1+f2 <= 128


def bicoherence_stats(y: np.ndarray) -> np.ndarray:
    X = librosa.stft(y, n_fft=N_FFT, hop_length=HOP, window="hann")  # (129, T)
    k = np.arange(1, KMAX + 1)
    f1, f2 = np.meshgrid(k, k, indexing="ij")
    A, B, C = X[f1], X[f2], np.conj(X[f1 + f2])             # each (64, 64, T)
    bis = (A * B * C).mean(axis=-1)                          # bispectrum
    denom = np.sqrt((np.abs(A * B) ** 2).mean(-1) * (np.abs(C) ** 2).mean(-1)) + 1e-12
    mag = np.abs(bis) / denom
    phase = np.angle(bis)
    out = []
    for m in (mag.ravel(), phase.ravel()):
        out += [m.mean(), m.var(), skew(m), kurtosis(m)]
    return np.array(out, dtype=np.float64)


def _one(path: str, cutoff):
    y = load_audio(path)
    if cutoff is not None:
        y = lowpass(y, cutoff)
    return bicoherence_stats(y)


def build(cutoff, n_jobs=10, limit=None):
    df = load_table()
    if limit:
        df = df.iloc[:limit]
    X = np.stack(Parallel(n_jobs=n_jobs, batch_size=16)(
        delayed(_one)(p, cutoff) for p in df["path"]))
    tag = "raw" if cutoff is None else f"lp{int(cutoff)}"
    if not limit:
        np.savez(CACHE_DIR / f"bico_{tag}.npz", X=X)
    return X


if __name__ == "__main__":
    import time
    names = [f"{c}_{s}" for c in ("mag", "phase") for s in ("mean", "var", "skew", "kurt")]
    df = load_table()
    for emo in ["ANG", "SAD"]:
        p = df[(df.actor == 1001) & (df.sentence == "DFA") & (df.emotion == emo)].path.iloc[0]
        v = bicoherence_stats(lowpass(load_audio(p), 4000))
        print(emo, dict(zip(names, v.round(3))))
    t = time.time(); X = bicoherence_stats(lowpass(load_audio(df.path.iloc[0]), 4000))
    print("one clip: %.2fs" % (time.time() - t), "finite:", np.isfinite(X).all())
