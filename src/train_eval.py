"""Train the 5 required classifiers on cached features; tune on val, report on test."""
import itertools
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import CACHE_DIR, EMOTIONS, EXCLUDE, SEED, load_table, speaker_split

# name -> (estimator class, fixed kwargs, grid of kwargs to search on val)
MODELS = {
    "KNN": (KNeighborsClassifier, {}, {"n_neighbors": [3, 7, 15, 31],
                                       "weights": ["uniform", "distance"]}),
    "LogReg": (LogisticRegression, {"max_iter": 3000},
               {"C": [0.01, 0.1, 1, 10]}),
    "NaiveBayes": (GaussianNB, {}, {"var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1]}),
    "SVM": (SVC, {"kernel": "rbf"}, {"C": [1, 10, 100],
                                    "gamma": ["scale", 0.001, 0.0003]}),
    "NeuralNet": (MLPClassifier, {"max_iter": 400, "early_stopping": True,
                                  "random_state": SEED},
                  {"hidden_layer_sizes": [(128,), (256, 128)],
                   "alpha": [1e-4, 1e-2]}),
}


def load(tag: str, groups=None):
    """Return X, y, split arrays aligned to the cached feature file."""
    base, _, extra = tag.partition("+")          # e.g. "lp5500+bico"
    z = np.load(CACHE_DIR / f"features_{base}.npz", allow_pickle=False)
    # Cache rows follow load_table() order (sorted filenames).
    df = speaker_split(load_table())
    X, group_of = z["X"], z["group_of"]
    if extra == "bico":                          # append 8 bicoherence stats
        B = np.load(CACHE_DIR / f"bico_{base}.npz", allow_pickle=False)["X"]
        X = np.hstack([X, B])
        group_of = np.concatenate([group_of, ["bico"] * B.shape[1]])
    assert len(df) == len(X), "cache and dataset are out of sync"
    if groups is not None:
        X = X[:, np.isin(group_of, groups)]
    keep = ~df["path"].map(lambda p: Path(p).stem in EXCLUDE).to_numpy()
    return X[keep], df["emotion"].to_numpy()[keep], df["split"].to_numpy()[keep]


def run(tag: str, groups=None, models=None, verbose=True):
    X, y, split = load(tag, groups)
    tr, va, te = (split == s for s in ("train", "val", "test"))
    rows, cms = [], {}
    for name in models or MODELS:
        cls, fixed, grid = MODELS[name]
        best = (-1, None)
        for combo in itertools.product(*grid.values()):
            params = dict(zip(grid, combo))
            pipe = make_pipeline(StandardScaler(), cls(**fixed, **params))
            pipe.fit(X[tr], y[tr])
            f1v = f1_score(y[va], pipe.predict(X[va]), average="macro")
            if f1v > best[0]:
                best = (f1v, params, pipe)
        f1v, params, pipe = best
        pred = pipe.predict(X[te])
        rows.append(dict(features=tag, model=name, val_f1=f1v,
                         test_acc=accuracy_score(y[te], pred),
                         test_macro_f1=f1_score(y[te], pred, average="macro"),
                         best_params=str(params), n_features=X.shape[1]))
        cms[name] = confusion_matrix(y[te], pred, labels=EMOTIONS)
        if verbose:
            r = rows[-1]
            print(f"[{tag}] {name:10s} val_f1={r['val_f1']:.3f} "
                  f"test_acc={r['test_acc']:.3f} test_f1={r['test_macro_f1']:.3f} {params}",
                  flush=True)
    return pd.DataFrame(rows), cms


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "lp4000"
    df, _ = run(tag)
    print(df.drop(columns="best_params").round(3).to_string(index=False))
