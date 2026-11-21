import sys, time, warnings
sys.path.insert(0, "src"); warnings.filterwarnings("ignore")
from features import build
for c in [4000, 3000, 5500, None]:
    t = time.time(); X, _ = build(c)
    print(f"cutoff={c}: X{X.shape} in {time.time()-t:.0f}s", flush=True)
