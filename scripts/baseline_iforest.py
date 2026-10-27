import glob, os, time
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

D = "data/nbaiot/Danmini_Doorbell"
benign = pd.read_csv(f"{D}/benign_traffic.csv")
train, test = train_test_split(benign, test_size=0.3, random_state=42)

scaler = StandardScaler().fit(train)
Xtr = scaler.transform(train)

t = time.time()
model = IsolationForest(n_estimators=100, contamination=0.01,
                        random_state=42, n_jobs=-1).fit(Xtr)
print(f"Training time: {time.time()-t:.1f} s")

def flagged(df):
    return (model.predict(scaler.transform(df)) == -1).mean()

print(f"\nFalse alarm rate on normal traffic: {flagged(test):.1%}")

for family in ("mirai_attacks", "gafgyt_attacks"):
    print(f"\n{family}:")
    for f in sorted(glob.glob(f"{D}/{family}/*.csv")):
        df = pd.read_csv(f).sample(5000, random_state=42)
        print(f"  {os.path.basename(f):<14} detected {flagged(df):.1%}")

joblib.dump(model, "models/iforest.joblib")
print(f"\nModel size on disk: {os.path.getsize('models/iforest.joblib')/1024:.0f} KB")
sample = scaler.transform(test.sample(1000, random_state=1))
t = time.time(); model.predict(sample)
print(f"Inference time: {(time.time()-t)/1000*1000:.3f} ms per sample (on this Mac)")
