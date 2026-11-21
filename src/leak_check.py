import sys, warnings; sys.path.insert(0, "src"); warnings.filterwarnings("ignore")
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from common import *

def tone_feats(p):
    y = load_audio(p)
    mag = np.abs(np.fft.rfft(y * np.hanning(len(y)))) / len(y)
    f = np.fft.rfftfreq(len(y), 1 / SR)
    tone = mag[(f > 7450) & (f < 7550)].max()
    speech = lowpass(y, 5500)
    return [np.log(tone), np.log(np.abs(y).max()), np.log(np.sqrt((speech ** 2).mean()) + 1e-9)]

df = speaker_split(load_table()); df = df[~df.path.map(lambda p: Path(p).stem in EXCLUDE)]
F = np.array(Parallel(n_jobs=10)(delayed(tone_feats)(p) for p in df.path))
y = df.emotion.to_numpy(); tr = (df.split == "train").to_numpy(); te = (df.split == "test").to_numpy()
for name, cols in [("tone amplitude only", [0]), ("tone amp + raw peak", [0, 1]), ("speech RMS only (filtered)", [2])]:
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(F[tr][:, cols], y[tr])
    print(f"{name:28s} test acc = {accuracy_score(y[te], m.predict(F[te][:, cols])):.3f}  (chance 0.167)")
print("corr(log tone amp, log speech RMS) = %.3f" % np.corrcoef(F[:, 0], F[:, 2])[0, 1])
print("corr(log tone amp, log raw peak)   = %.3f" % np.corrcoef(F[:, 0], F[:, 1])[0, 1])
