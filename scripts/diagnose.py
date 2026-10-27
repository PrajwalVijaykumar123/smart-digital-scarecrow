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
cols = benign.columns

sets = [("benign (test)", test),
        ("gafgyt/tcp", pd.read_csv(f"{D}/gafgyt_attacks/tcp.csv").sample(5000, random_state=42)),
        ("mirai/udp", pd.read_csv(f"{D}/mirai_attacks/udp.csv").sample(5000, random_state=42))]

for name, df in sets:
    z = scaler.transform(df)
    s = model.decision_function(z)
    mean_abs = np.abs(z).mean(axis=0)
    print(f"\n{name}: distinct scores = {len(np.unique(np.round(s, 6)))}, "
          f"features with mean |z| > 3 = {(mean_abs > 3).sum()} of {len(cols)}")
    for i in mean_abs.argsort()[::-1][:5]:
        print(f"   {cols[i]:<26} mean |z| = {mean_abs[i]:.1f}")
