import glob, os, sys, time
import numpy as np, pandas as pd, joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split

D = sys.argv[1] if len(sys.argv) > 1 else "data/nbaiot/Danmini_Doorbell"
ae = tf.keras.models.load_model("models/autoencoder.keras")
scaler = joblib.load("models/scaler.joblib")
thr_float = joblib.load("models/thresholds.joblib")["autoenc"]

benign = pd.read_csv(f"{D}/benign_traffic.csv")
train, test = train_test_split(benign, test_size=0.3, random_state=42)
Xtr = scaler.transform(train).astype("float32")

# Convert to full 8-bit integer model, calibrated on normal traffic only
def rep():
    for row in Xtr[:1000]:
        yield [row[None, :]]
conv = tf.lite.TFLiteConverter.from_keras_model(ae)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8
blob = conv.convert()
open("models/autoencoder_int8.tflite", "wb").write(blob)

interp = tf.lite.Interpreter(model_content=blob)
interp.allocate_tensors()
i_det, o_det = interp.get_input_details()[0], interp.get_output_details()[0]
s_in, z_in = i_det["quantization"]
s_out, z_out = o_det["quantization"]

def q_score(X):
    out = np.empty(len(X))
    for k, row in enumerate(X):
        xq = np.clip(np.round(row / s_in + z_in), -128, 127).astype(np.int8)
        interp.set_tensor(i_det["index"], xq[None, :])
        interp.invoke()
        y = (interp.get_tensor(o_det["index"])[0].astype(np.float32) - z_out) * s_out
        seen = (xq.astype(np.float32) - z_in) * s_in      # what the device actually sees
        out[k] = np.mean((seen - y) ** 2)
    return out

f_score = lambda X: np.mean((X - ae.predict(X, batch_size=2048, verbose=0)) ** 2, axis=1)
thr_q = np.percentile(q_score(Xtr[:10000]), 99)

print(f"Float model:  {ae.count_params()} parameters, about {ae.count_params()*4/1024:.0f} KB")
print(f"Int8 model:   {len(blob)/1024:.1f} KB as a .tflite file")
print(f"Input range the int8 model can represent: {(-128-z_in)*s_in:+.1f} to {(127-z_in)*s_in:+.1f} standard deviations")
t = time.time(); q_score(Xtr[:2000])
print(f"Int8 inference on this Mac: {(time.time()-t)/2000*1000:.2f} ms per row (the ESP32 will be slower)")

def both(df):
    X = scaler.transform(df).astype("float32")
    return f"float: {(f_score(X) > thr_float).mean():6.1%}   int8: {(q_score(X) > thr_q).mean():6.1%}"

def sample(path, n=3000):
    rows = sum(1 for _ in open(path)) - 1
    return pd.read_csv(path).sample(min(n, rows), random_state=42)

print("\nFALSE ALARMS on normal test traffic:\n  " + both(test.sample(min(3000, len(test)), random_state=1)))
for family in ("mirai_attacks", "gafgyt_attacks"):
    print(f"\n{family}:")
    for f in sorted(glob.glob(f"{D}/{family}/*.csv")):
        print(f"  {os.path.basename(f):<14} {both(sample(f))}")
