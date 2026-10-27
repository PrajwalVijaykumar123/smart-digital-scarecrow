import glob, os, sys, time
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

ROOT = "data/nbaiot"
A = sys.argv[1] if len(sys.argv) > 1 else "Danmini_Doorbell"
B = sys.argv[2] if len(sys.argv) > 2 else "Ecobee_Thermostat"
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
tf.keras.utils.set_random_seed(42)

def fit(dev):
    """Train all three detectors on one device's normal traffic."""
    benign = pd.read_csv(f"{ROOT}/{dev}/benign_traffic.csv")
    train, test = train_test_split(benign, test_size=0.3, random_state=42)
    scaler = StandardScaler().fit(train)
    Xtr = scaler.transform(train).astype("float32")
    forest = IsolationForest(n_estimators=100, random_state=42, n_jobs=-1).fit(Xtr)
    n = Xtr.shape[1]
    ae = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(n)])
    ae.compile(optimizer="adam", loss="mse")
    ae.fit(Xtr, Xtr, epochs=EPOCHS, batch_size=256, validation_split=0.1, verbose=0)
    scores = {
        "iforest": lambda X: -forest.decision_function(X),
        "zscore": lambda X: np.abs(X).mean(axis=1),
        "autoenc": lambda X: np.mean((X - ae.predict(X, batch_size=2048, verbose=0)) ** 2, axis=1),
    }
    thr = {k: np.percentile(f(Xtr), 99) for k, f in scores.items()}
    return dict(scaler=scaler, scores=scores, thr=thr, test=test)

def rate(m, df):
    X = m["scaler"].transform(df).astype("float32")
    return "  ".join(f"{k}: {(f(X) > m['thr'][k]).mean():6.1%}" for k, f in m["scores"].items())

def sample(path, n=5000):
    rows = sum(1 for _ in open(path)) - 1
    return pd.read_csv(path).sample(min(n, rows), random_state=42)

def evaluate(m, dev, benign_df, title):
    print(f"\n==== {title} ====")
    print("FALSE ALARMS on normal traffic (lower is better):")
    print("  " + rate(m, benign_df))
    print("ATTACKS DETECTED (higher is better):")
    for family in ("mirai_attacks", "gafgyt_attacks"):
        for f in sorted(glob.glob(f"{ROOT}/{dev}/{family}/*.csv")):
            print(f"  {family[:5]}/{os.path.basename(f):<13} {rate(m, sample(f))}")

mA = fit(A)
mB = fit(B)
evaluate(mA, B, sample(f"{ROOT}/{B}/benign_traffic.csv"),
         f"trained on {A}, tested on {B}  (one model for every device)")
evaluate(mB, B, mB["test"],
         f"trained on {B}, tested on {B}  (a model per device)")
