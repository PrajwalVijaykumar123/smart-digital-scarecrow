import glob, os
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

D = "data/nbaiot/Danmini_Doorbell"
benign = pd.read_csv(f"{D}/benign_traffic.csv")
train, test = train_test_split(benign, test_size=0.3, random_state=42)
scaler = StandardScaler().fit(train)
model = IsolationForest(n_estimators=100, contamination=0.01,
                        random_state=42, n_jobs=-1).fit(scaler.transform(train))

def report(name, df):
    s = model.decision_function(scaler.transform(df))   # below 0 = anomaly
    uncertain = (np.abs(s) < 0.03).mean()
    print(f"{name:<24} median {np.median(s):+.3f}   p5 {np.percentile(s,5):+.3f}   p95 {np.percentile(s,95):+.3f}   uncertain {uncertain:.1%}")

report("benign (test)", test)
for family in ("mirai_attacks", "gafgyt_attacks"):
    for f in sorted(glob.glob(f"{D}/{family}/*.csv")):
        report(f"{family[:5]}/{os.path.basename(f)}", pd.read_csv(f).sample(5000, random_state=42))
