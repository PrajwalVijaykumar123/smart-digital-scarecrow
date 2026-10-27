import glob, os, sys, time
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

D = sys.argv[1] if len(sys.argv) > 1 else "data/nbaiot/Danmini_Doorbell"
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
tf.keras.utils.set_random_seed(42)

benign = pd.read_csv(f"{D}/benign_traffic.csv")
train, test = train_test_split(benign, test_size=0.3, random_state=42)
scaler = StandardScaler().fit(train)
Xtr = scaler.transform(train).astype("float32")
Xte = scaler.transform(test).astype("float32")
n = Xtr.shape[1]

# Detector 1: Isolation Forest (higher score = more abnormal)
forest = IsolationForest(n_estimators=100, random_state=42, n_jobs=-1).fit(Xtr)
s_forest = lambda X: -forest.decision_function(X)

# Detector 2: mean absolute z-score (distance from normal)
s_z = lambda X: np.abs(X).mean(axis=1)

# Detector 3: small autoencoder (error = how badly it rebuilds the row)
ae = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(n,)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(n),
])
ae.compile(optimizer="adam", loss="mse")
t = time.time()
ae.fit(Xtr, Xtr, epochs=EPOCHS, batch_size=256, validation_split=0.1, verbose=0)
print(f"Autoencoder trained in {time.time()-t:.0f} s, {ae.count_params()} parameters "
      f"(about {ae.count_params()*4/1024:.0f} KB as float32)")
s_ae = lambda X: np.mean((X - ae.predict(X, batch_size=2048, verbose=0)) ** 2, axis=1)

detectors = {"iforest": s_forest, "zscore": s_z, "autoenc": s_ae}
thr = {k: np.percentile(f(Xtr), 99) for k, f in detectors.items()}

def rate(name, X):
    return "  ".join(f"{k}: {(f(X) > thr[k]).mean():6.1%}" for k, f in detectors.items())

print("\nShare of rows flagged as attack (threshold = 99th percentile of normal training data)")
print("\nFALSE ALARMS on normal test traffic:\n  " + rate("b", Xte))
for family in ("mirai_attacks", "gafgyt_attacks"):
    print(f"\n{family}:")
    for f in sorted(glob.glob(f"{D}/{family}/*.csv")):
        df = pd.read_csv(f).sample(min(5000, sum(1 for _ in open(f)) - 1), random_state=42)
        X = scaler.transform(df).astype("float32")
        print(f"  {os.path.basename(f):<14} {rate(f, X)}")

os.makedirs("models", exist_ok=True)
ae.save("models/autoencoder.keras")
joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(thr, "models/thresholds.joblib")
print("\nSaved models/autoencoder.keras, scaler.joblib, thresholds.joblib")
