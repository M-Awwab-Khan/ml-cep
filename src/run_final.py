import sys, warnings, pickle
sys.path.insert(0, "src"); warnings.filterwarnings("ignore")
import pandas as pd
from train_eval import run
res, cms = [], {}
# bicoherence on its own (SVM + LogReg), then the two full feature sets with all 5 models
d, _ = run("lp5500+bico", groups=["bico"], models=["LogReg", "SVM"]); d["features"] = "bico_only"; res.append(d)
for tag in ["lp5500", "lp5500+bico"]:
    d, c = run(tag); res.append(d); cms[tag] = c
pd.concat(res).to_csv("results/final_models.csv", index=False)
pickle.dump(cms, open("results/final_confusion.pkl", "wb"))
print("FINAL_DONE", flush=True)
