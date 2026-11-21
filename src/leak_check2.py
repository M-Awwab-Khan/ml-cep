import sys, warnings; sys.path.insert(0, "src"); warnings.filterwarnings("ignore")
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from common import *

def f(p):
    y = load_audio(p); mag = np.abs(np.fft.rfft(y*np.hanning(len(y))))/len(y); fr = np.fft.rfftfreq(len(y), 1/SR)
    tone = mag[(fr>7450)&(fr<7550)].max()
    sp = lowpass(y, 5500)
    return [np.log(tone), np.log(np.sqrt((sp**2).mean())+1e-9)]
df = speaker_split(load_table()); df = df[~df.path.map(lambda p: Path(p).stem in EXCLUDE)]
F = np.array(Parallel(n_jobs=10)(delayed(f)(p) for p in df.path))
F = np.column_stack([F, F[:,1]-F[:,0]])      # col2 = log(speech RMS / tone) = speech-to-tone ratio
y = df.emotion.to_numpy(); tr=(df.split=="train").to_numpy(); te=(df.split=="test").to_numpy(); va=(df.split=="val").to_numpy()
for nm, c in [("tone amplitude",[0]),("speech RMS",[1]),("speech-to-tone ratio",[2]),("tone + speech RMS (jointly)",[0,1])]:
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(F[tr][:,c], y[tr])
    print(f"{nm:30s} val acc {accuracy_score(y[va], m.predict(F[va][:,c])):.3f} | test acc {accuracy_score(y[te], m.predict(F[te][:,c])):.3f} macroF1 {f1_score(y[te], m.predict(F[te][:,c]), average='macro'):.3f}")
