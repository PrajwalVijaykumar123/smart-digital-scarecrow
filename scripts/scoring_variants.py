import glob, os, sys
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

ROOT = "data/nbaiot"
devices = sys.argv[1:] or ["Danmini_Doorbell", "Ecobee_Thermostat"]
K = 5  # a feature counts as "extreme" beyond K standard deviations
scorers = {
    "mean|z|": lambda z: np.abs(z).mean(axis=1),
    "max|z|": lambda z: np.abs(z).max(axis=1),
    f"n(|z|>{K})": lambda z: (np.abs(z) > K).sum(axis=1),
}

def sample(path, n=5000):
    rows = sum(1 for _ in open(path)) - 1
    return pd.read_csv(path).sample(min(n, rows), random_state=42)

for dev in devices:
    benign = pd.read_csv(f"{ROOT}/{dev}/benign_traffic.csv")
    train, test = train_test_split(benign, test_size=0.3, random_state=42)
    sc = StandardScaler().fit(train)
    ztr = sc.transform(train)
    thr = {k: np.percentile(f(ztr), 99) for k, f in scorers.items()}
    def rate(df):
        z = sc.transform(df)
        return "  ".join(f"{k}: {(f(z) > thr[k]).mean():6.1%}" for k, f in scorers.items())
    print(f"\n==== {dev}  (threshold = 99th percentile of normal training data) ====")
    print("FALSE ALARMS on normal traffic:\n  " + rate(test))
    print("ATTACKS DETECTED:")
    for family in ("mirai_attacks", "gafgyt_attacks"):
        for f in sorted(glob.glob(f"{ROOT}/{dev}/{family}/*.csv")):
            print(f"  {family[:5]}/{os.path.basename(f):<13} {rate(sample(f))}")
