"""Consistent re-run: silent clip excluded everywhere, 7 kHz added, ablation at 5.5 kHz."""
import os, sys, warnings
sys.path.insert(0, "src"); warnings.filterwarnings("ignore")
import pandas as pd
which = sys.argv[1]
if which == "cutoff":
    from features import build
    build(7000)
    from train_eval import run
    parts = [run(t, models=["LogReg", "SVM"])[0] for t in ["raw", "lp3000", "lp4000", "lp5500", "lp7000"]]
    pd.concat(parts).to_csv("results/cutoff_comparison.csv", index=False)
else:
    from train_eval import run
    rows = []
    for g in ["mfcc", "mel", "chroma", "centroid", "bandwidth", "contrast", "fft", "bico"]:
        df, _ = run("lp5500+bico", groups=[g], models=["LogReg", "SVM"], verbose=False)
        df["group"] = g; rows.append(df); print(g, df[["model", "test_acc", "test_macro_f1"]].round(3).values.tolist(), flush=True)
    pd.concat(rows).to_csv("results/ablation_single_group.csv", index=False)
print("RERUN_DONE_" + which, flush=True)
