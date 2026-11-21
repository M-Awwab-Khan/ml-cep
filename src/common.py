"""Shared helpers: dataset table, speaker-independent split, low-pass filter."""
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, filtfilt

ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT / "data_src" / "SentimentAnalysisDataset" / "NoiseAudioWAV"
CACHE_DIR = ROOT / "cache"
SR = 16000  # every CREMA-D clip is 16 kHz mono
EMOTIONS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
SEED = 42
# Clips excluded from all experiments (data-quality issues found during EDA).
# 1076_MTI_SAD_XX.wav is digital silence: all 41,108 samples are exactly 0.
EXCLUDE = {"1076_MTI_SAD_XX"}


def load_table() -> pd.DataFrame:
    """One row per clip: path, actor, sentence, emotion, level."""
    rows = []
    for f in sorted(AUDIO_DIR.glob("*.wav")):
        actor, sentence, emotion, level = f.stem.split("_")
        rows.append(dict(path=str(f), actor=int(actor), sentence=sentence,
                         emotion=emotion, level=level))
    return pd.DataFrame(rows)


def speaker_split(df: pd.DataFrame, val_frac=0.15, test_frac=0.15, seed=SEED):
    """Assign each ACTOR (not each clip) to train/val/test.

    No actor appears in more than one split, so test accuracy measures
    generalisation to unseen speakers.
    """
    actors = np.array(sorted(df["actor"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(actors)
    n_test = round(len(actors) * test_frac)
    n_val = round(len(actors) * val_frac)
    test, val, train = (actors[:n_test], actors[n_test:n_test + n_val],
                        actors[n_test + n_val:])
    split = df["actor"].map(
        {**{a: "train" for a in train}, **{a: "val" for a in val},
         **{a: "test" for a in test}})
    return df.assign(split=split)


def load_audio(path: str) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32")
    assert sr == SR, f"unexpected sample rate {sr}"
    return y


def lowpass(y: np.ndarray, cutoff_hz: float = 4000.0, order: int = 5,
            sr: int = SR) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter.

    cutoff is normalised by the Nyquist frequency (sr/2). filtfilt runs the
    filter forwards then backwards, so the signal is not time-shifted.
    """
    b, a = butter(order, cutoff_hz / (sr / 2), btype="low")
    return filtfilt(b, a, y).astype(np.float32)
