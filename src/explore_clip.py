import sys; sys.path.insert(0, "src")
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, librosa, librosa.display
from common import *

df = speaker_split(load_table())
print(df.groupby("split").agg(clips=("path", "size"), actors=("actor", "nunique")))
print(pd.crosstab(df["split"], df["emotion"]))
sets = {s: set(g["actor"]) for s, g in df.groupby("split")}
assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])
print("OK: no actor appears in more than one split")

# One angry and one sad clip from the same actor + sentence
fig, ax = plt.subplots(3, 2, figsize=(12, 9))
for col, emo in enumerate(["ANG", "SAD"]):
    p = df[(df.actor == 1001) & (df.sentence == "DFA") & (df.emotion == emo)].path.iloc[0]
    y = load_audio(p); yf = lowpass(y, 4000)
    t = np.arange(len(y)) / SR
    ax[0, col].plot(t, y, lw=.5, label="original"); ax[0, col].plot(t, yf, lw=.5, label="low-pass 4 kHz")
    ax[0, col].set_title(f"{emo}: waveform (amplitude vs time)"); ax[0, col].legend()
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(S, sr=SR, x_axis="time", y_axis="hz", ax=ax[1, col])
    ax[1, col].set_title(f"{emo}: spectrogram, original")
    Sf = librosa.amplitude_to_db(np.abs(librosa.stft(yf)), ref=np.max)
    librosa.display.specshow(Sf, sr=SR, x_axis="time", y_axis="hz", ax=ax[2, col])
    ax[2, col].set_title(f"{emo}: spectrogram, after 4 kHz low-pass")
plt.tight_layout(); plt.savefig("figures/01_clip_overview.png", dpi=110)
