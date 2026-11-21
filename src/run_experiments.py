import sys, warnings
sys.path.insert(0, "src"); warnings.filterwarnings("ignore")
import pandas as pd
from train_eval import run
M = ["LogReg", "SVM"]

# 1) cutoff comparison (all features)
parts = [run(t, models=M)[0] for t in ["raw", "lp3000", "lp4000", "lp5500"]]
pd.concat(parts).to_csv("results/cutoff_comparison.csv", index=False)
print("CUTOFF_DONE", flush=True)

# 2) ablation on the 4 kHz features: one group at a time
groups = ["mfcc", "mel", "chroma", "centroid", "bandwidth", "contrast", "fft"]
rows = []
for g in groups:
    df, _ = run("lp4000", groups=[g], models=M, verbose=False)
    df["group"] = g; rows.append(df)
    print(g, df[["model", "test_acc", "test_macro_f1"]].round(3).values.tolist(), flush=True)
pd.concat(rows).to_csv("results/ablation_single_group.csv", index=False)
print("ALL_DONE", flush=True)
