"""CPU vs GPU end-to-end analytics benchmark on NYC TLC taxi trips.

The same pipeline runs on two backends:
  cpu: pandas + scikit-learn + XGBoost (CPU)
  gpu: cuDF + cuML + XGBoost (device="cuda")

Task: predict whether a credit-card trip gets a generous tip (tip >= 20% of fare).

    python src/benchmark.py --backend cpu --months 2023-01 2023-02 2023-03
    python src/benchmark.py --backend gpu --months 2023-01 2023-02 2023-03

Each run appends one JSON line per stage timing to results/runs.jsonl.
"""
import argparse
import json
import os
import platform
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet"
ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
DATA_DIR = Path("data")
RESULTS = Path("results/runs.jsonl")

FEATURES = [
    "passenger_count", "trip_distance", "fare_amount", "duration_min", "speed_mph",
    "pickup_hour", "pickup_dow", "is_weekend", "PULocationID", "DOLocationID",
    "pu_zone_trips", "pu_zone_avg_fare", "pu_borough_code", "do_borough_code",
]


def download(months):
    DATA_DIR.mkdir(exist_ok=True)
    paths = []
    for m in months:
        p = DATA_DIR / f"yellow_{m}.parquet"
        if not p.exists():
            print(f"downloading {m} ...")
            urllib.request.urlretrieve(DATA_URL.format(month=m), p)
        paths.append(p)
    zones = DATA_DIR / "zones.csv"
    if not zones.exists():
        urllib.request.urlretrieve(ZONES_URL, zones)
    return paths, zones


class Timer:
    def __init__(self, sync=None):
        self.times = {}
        self.sync = sync or (lambda: None)

    @contextmanager
    def stage(self, name):
        self.sync()
        t0 = time.perf_counter()
        yield
        self.sync()  # GPU work is async: wait for it before stopping the clock
        self.times[name] = round(time.perf_counter() - t0, 3)
        print(f"  {name:<12} {self.times[name]:8.3f}s")


def get_backend(name):
    if name == "gpu":
        import cudf as xdf
        import cupy as cp
        from cuml.model_selection import train_test_split
        from cuml.metrics import roc_auc_score
        return xdf, train_test_split, roc_auc_score, cp.cuda.Device().synchronize, "cuda"
    import pandas as xdf
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    return xdf, train_test_split, roc_auc_score, None, "cpu"


def run(backend, months, seed=42):
    import xgboost as xgb

    xdf, split, auc, sync, device = get_backend(backend)
    paths, zones_path = download(months)
    t = Timer(sync)
    print(f"backend={backend} months={months}")

    with t.stage("load"):
        df = xdf.concat([xdf.read_parquet(p) for p in paths], ignore_index=True)
        zones = xdf.read_csv(zones_path)
    rows_raw = len(df)

    with t.stage("clean"):
        df = df[(df["payment_type"] == 1) & (df["fare_amount"] > 2.5) & (df["fare_amount"] < 250)
                & (df["trip_distance"] > 0) & (df["trip_distance"] < 100)]
        df = df.dropna(subset=["passenger_count"])
        delta = df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
        dur = delta.dt.days * 1440 + delta.dt.seconds / 60
        df = df.assign(duration_min=dur)
        df = df[(df["duration_min"] > 1) & (df["duration_min"] < 180)]

    with t.stage("features"):
        df["speed_mph"] = df["trip_distance"] / (df["duration_min"] / 60)
        df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
        df["pickup_dow"] = df["tpep_pickup_datetime"].dt.dayofweek
        df["is_weekend"] = (df["pickup_dow"] >= 5).astype("int8")
        df["label"] = (df["tip_amount"] >= 0.2 * df["fare_amount"]).astype("int8")
        # Group-by aggregates per pickup zone
        agg = df.groupby("PULocationID").agg(pu_zone_trips=("fare_amount", "size"),
                                             pu_zone_avg_fare=("fare_amount", "mean")).reset_index()
        df = df.merge(agg, on="PULocationID", how="left")

    with t.stage("join"):
        z = zones[["LocationID", "Borough"]].copy()
        z["borough_code"] = z["Borough"].astype("category").cat.codes
        df = df.merge(z[["LocationID", "borough_code"]].rename(
            columns={"LocationID": "PULocationID", "borough_code": "pu_borough_code"}), on="PULocationID", how="left")
        df = df.merge(z[["LocationID", "borough_code"]].rename(
            columns={"LocationID": "DOLocationID", "borough_code": "do_borough_code"}), on="DOLocationID", how="left")
        X = df[FEATURES].astype("float32")
        y = df["label"]
    rows_model = len(df)

    with t.stage("split"):
        X_tr, X_te, y_tr, y_te = split(X, y, test_size=0.2, random_state=seed)

    with t.stage("train"):
        model = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                                  tree_method="hist", device=device, random_state=seed)
        model.fit(X_tr, y_tr)

    with t.stage("predict"):
        proba = model.predict_proba(X_te)[:, 1]
        score = float(auc(y_te, proba))

    total = round(sum(t.times.values()), 3)
    print(f"  {'TOTAL':<12} {total:8.3f}s   ROC-AUC={score:.4f}  rows={rows_model:,}")

    record = {
        "backend": backend, "months": months, "rows_raw": rows_raw, "rows_model": rows_model,
        "positive_rate": round(float(y.mean()), 4), "roc_auc": round(score, 4),
        "stages_s": t.times, "total_s": total, "hardware": hardware(backend),
    }
    RESULTS.parent.mkdir(exist_ok=True)
    with RESULTS.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def hardware(backend):
    info = {"cpu": platform.processor() or platform.machine(), "cpu_cores": os.cpu_count()}
    if backend == "gpu":
        import cupy as cp
        props = cp.cuda.runtime.getDeviceProperties(0)
        info["gpu"] = props["name"].decode()
        info["gpu_mem_gb"] = round(props["totalGlobalMem"] / 1e9, 1)
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    ap.add_argument("--months", nargs="+", default=["2023-01"])
    args = ap.parse_args()
    run(args.backend, args.months)
