"""Generate all report figures into report/figures/ from cached results."""
import pickle
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import librosa
import librosa.display
import matplotlib
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfreqz

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import EMOTIONS, ROOT, SR, load_audio, load_table, lowpass

OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# validated categorical slots 1-3 (light surface) and text tokens
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CHANCE = 1 / 6

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5, "text.color": INK,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 100, "savefig.dpi": 220,
    "savefig.bbox": "tight", "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left",
})


def save(name):
    plt.savefig(OUT / name)
    plt.close()
    print("wrote", name)


def bar_labels(ax, bars, fmt="{:.3f}", dy=0.006):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy,
                fmt.format(b.get_height()), ha="center", va="bottom",
                fontsize=8, color=INK)


# ---------------------------------------------------------------- 1. noise
def fig_noise():
    df = load_table()
    p = df[(df.actor == 1001) & (df.sentence == "DFA") & (df.emotion == "ANG")].path.iloc[0]
    y = load_audio(p)
    yf = lowpass(y, 5500)
    ref = np.abs(librosa.stft(y, n_fft=1024, hop_length=256)).max()   # shared dB reference
    fig = plt.figure(figsize=(11, 3.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.3)
    for i, (sig, title) in enumerate([(y, "(a) Original clip"),
                                      (yf, "(b) After 5.5 kHz low-pass")]):
        ax = fig.add_subplot(gs[i])
        S = librosa.amplitude_to_db(np.abs(librosa.stft(sig, n_fft=1024, hop_length=256)),
                                    ref=ref, amin=1e-7)
        librosa.display.specshow(S, sr=SR, hop_length=256, x_axis="time", y_axis="hz",
                                 ax=ax, cmap="magma", vmin=-90, vmax=-15)
        ax.set_title(title); ax.grid(False)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)" if i == 0 else "")
        if i == 0:
            ax.annotate("added 7.5 kHz tone", xy=(1.9, 7500), xytext=(1.2, 6100),
                        color="white", fontsize=8.5, ha="center",
                        arrowprops=dict(arrowstyle="->", color="white"))
    ax = fig.add_subplot(gs[2])
    mag = np.abs(np.fft.rfft(y * np.hanning(len(y)))) / len(y)
    f = np.fft.rfftfreq(len(y), 1 / SR)
    spec = 20 * np.log10(mag + 1e-9); spec -= spec.max()               # 0 dB = the tone
    ax.plot(f, spec, color="#b9b8b2", lw=0.8, label="clip spectrum (0 dB = tone)")
    for c, col in [(3000, BLUE), (4000, ORANGE), (5500, AQUA)]:
        sos = butter(5, c / (SR / 2), output="sos")
        w, h = sosfreqz(sos, worN=2048, fs=SR)
        ax.plot(w, 20 * np.log10(np.abs(h) ** 2 + 1e-9), color=col, lw=1.8,
                label=f"filter, {c/1000:g} kHz cutoff")
    ax.axvline(7500, color=INK2, ls=":", lw=1)
    ax.text(7440, -22, "added tone\n7.5 kHz (0 dB)", ha="right", fontsize=8, color=INK)
    ax.set_ylim(-100, 6); ax.set_xlim(0, 8000)
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Level / filter gain (dB)")
    ax.set_title("(c) Tone level vs. filter gain")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7.5, frameon=False)
    save("fig_noise.png")


# ----------------------------------------------------------------- 2. cutoff
def fig_cutoff():
    d = pd.read_csv(ROOT / "results" / "cutoff_comparison.csv")
    order = [("raw", "Unfiltered"), ("lp3000", "3 kHz"), ("lp4000", "4 kHz"), ("lp5500", "5.5 kHz"), ("lp7000", "7 kHz")]
    fig, ax = plt.subplots(figsize=(6.8, 3.3))
    w = 0.36
    for k, (m, col) in enumerate([("LogReg", BLUE), ("SVM", ORANGE)]):
        v = [d[(d.features == t) & (d.model == m)].test_macro_f1.iloc[0] for t, _ in order]
        bars = ax.bar(np.arange(len(order)) + (k - .5) * w, v, w * 0.92, color=col,
                      label="Logistic regression" if m == "LogReg" else "SVM (RBF)")
        bar_labels(ax, bars)
    ax.axhline(CHANCE, color=INK2, ls="--", lw=1)
    ax.text(1.01, CHANCE, "chance\n(0.167)", transform=ax.get_yaxis_transform(), va="center", fontsize=8, color=INK2)
    ax.set_xticks(range(len(order)), [l for _, l in order]); ax.set_ylim(0, 0.7)
    ax.set_ylabel("Test macro-F1"); ax.set_xlabel("Low-pass cutoff")
    ax.set_title("Filter cutoff has little effect on accuracy")
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=8.5)
    ax.grid(axis="x", visible=False)
    save("fig_cutoff.png")


# --------------------------------------------------------------- 3. ablation
def fig_ablation():
    a = pd.read_csv(ROOT / "results" / "ablation_single_group.csv")
    a = a[a.model == "SVM"].set_index("group").test_macro_f1
    names = {"mfcc": "MFCC (40)", "fft": "FFT spectrum (32)", "mel": "Mel-spectrogram (128)",
             "contrast": "Spectral contrast (14)", "chroma": "Chromagram (24)",
             "centroid": "Spectral centroid (2)", "bandwidth": "Spectral bandwidth (2)",
             "bico": "Bicoherence (8)"}
    a = a.sort_values()
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    bars = ax.barh([names[g] for g in a.index], a.values, color=BLUE, height=0.62)
    for b in bars:
        inside = b.get_width() > 0.45          # long bars: label inside, clear of the reference line
        ax.text(b.get_width() + (-0.008 if inside else 0.008), b.get_y() + b.get_height() / 2,
                f"{b.get_width():.3f}", va="center", ha="right" if inside else "left",
                fontsize=8.5, color="white" if inside else INK)
    fm = pd.read_csv(ROOT / "results" / "final_models.csv")
    full = float(fm[(fm.features == "lp5500") & (fm.model == "SVM")].test_macro_f1.iloc[0])
    ax.axvline(full, color=ORANGE, lw=1.6)
    ax.text(full + 0.006, 7.75, f"all 242 features: {full:.3f}", color=INK2, fontsize=8, va="center")
    ax.axvline(CHANCE, color=INK2, ls="--", lw=1)
    ax.text(CHANCE + 0.006, 7.75, "chance", fontsize=8, color=INK2, va="center")
    ax.set_ylim(-0.6, 8.1)
    ax.set_xlim(0, 0.68); ax.set_xlabel("Test macro-F1 (SVM, one feature group alone, 5.5 kHz)")
    ax.set_title("MFCCs alone nearly match the full feature set")
    ax.grid(axis="y", visible=False)
    save("fig_ablation.png")


# ------------------------------------------------------------------ 4. models
def fig_models():
    d = pd.read_csv(ROOT / "results" / "final_models.csv")
    order = ["LogReg", "SVM", "NeuralNet", "KNN", "NaiveBayes"]
    lab = ["Logistic\nregression", "SVM\n(RBF)", "Neural\nnetwork", "KNN", "Naive\nBayes"]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    w = 0.36
    for k, (tag, col, name) in enumerate([("lp5500", BLUE, "Base features (242)"),
                                           ("lp5500+bico", ORANGE, "Base + bicoherence (250)")]):
        v = [d[(d.features == tag) & (d.model == m)].test_macro_f1.iloc[0] for m in order]
        bars = ax.bar(np.arange(5) + (k - .5) * w, v, w * 0.92, color=col, label=name)
        bar_labels(ax, bars, fmt="{:.2f}", dy=0.006)
    ax.axhline(CHANCE, color=INK2, ls="--", lw=1)
    ax.text(1.01, CHANCE, "chance\n(0.167)", transform=ax.get_yaxis_transform(), va="center", fontsize=8, color=INK2)
    ax.set_xticks(range(5), lab); ax.set_ylim(0, 0.72); ax.set_ylabel("Test macro-F1")
    ax.set_title("Simple models match complex ones; bicoherence adds nothing")
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    ax.grid(axis="x", visible=False)
    save("fig_models.png")


# ------------------------------------------------------------- 5. confusion
def fig_confusion():
    cms = pickle.load(open(ROOT / "results" / "final_confusion.pkl", "rb"))["lp5500"]
    names = {"ANG": "Anger", "DIS": "Disgust", "FEA": "Fear", "HAP": "Happy",
             "NEU": "Neutral", "SAD": "Sad"}
    lab = [names[e] for e in EMOTIONS]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3))
    for ax, key, title in zip(axes, ["LogReg", "SVM"], ["Logistic regression", "SVM (RBF)"]):
        cm = cms[key]; rec = cm / cm.sum(1, keepdims=True)
        ax.imshow(rec, cmap="Blues", vmin=0, vmax=0.85); ax.grid(False)
        ax.set_xticks(range(6), lab, rotation=35, ha="right"); ax.set_yticks(range(6), lab)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{rec[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                        color="white" if rec[i, j] > 0.5 else INK)
        for s in ax.spines.values():
            s.set_visible(False)
    plt.tight_layout(w_pad=2)
    save("fig_confusion.png")


if __name__ == "__main__":
    fig_noise(); fig_cutoff(); fig_ablation(); fig_models(); fig_confusion()
