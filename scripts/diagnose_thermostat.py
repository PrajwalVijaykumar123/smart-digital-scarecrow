import sys, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

ROOT = "data/nbaiot"
dev = sys.argv[1] if len(sys.argv) > 1 else "Ecobee_Thermostat"
benign = pd.read_csv(f"{ROOT}/{dev}/benign_traffic.csv")
train, test = train_test_split(benign, test_size=0.3, random_state=42)
scaler = StandardScaler().fit(train)
cols = benign.columns
row_score = lambda df: np.abs(scaler.transform(df)).mean(axis=1)
thr = np.percentile(row_score(train), 99)
print(f"{dev}: alarm threshold (99th percentile of normal) = {thr:.2f}")
print(f"Constant features in normal traffic (zero variance): {(train.std() == 0).sum()} of {len(cols)}")

def show(name, df):
    z = np.abs(scaler.transform(df))
    s = z.mean(axis=1)
    top = z.mean(axis=0).argsort()[::-1][:3]
    print(f"\n{name}: median row score {np.median(s):.2f}, above threshold {(s > thr).mean():.1%}, "
          f"features with mean |z| > 3: {(z.mean(axis=0) > 3).sum()}")
    print("   top features: " + ", ".join(f"{cols[i]} ({z[:, i].mean():.1f})" for i in top))

show("normal (test)", test)
for f in ("mirai_attacks/ack", "gafgyt_attacks/combo", "gafgyt_attacks/tcp", "gafgyt_attacks/udp"):
    rows = sum(1 for _ in open(f"{ROOT}/{dev}/{f}.csv")) - 1
    show(f, pd.read_csv(f"{ROOT}/{dev}/{f}.csv").sample(min(5000, rows), random_state=42))
