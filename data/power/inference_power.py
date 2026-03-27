#!/usr/bin/env python3
"""
Power System Attack Detection - Streaming Inference
Loads a trained power_model.pkl and simulates live PMU data streaming.

Usage:
    python inference_power.py --model power_model.pkl --data ./power_data/
    python inference_power.py --model power_model.pkl --data ./power_data/ --speed 0.5
    python inference_power.py --model power_model.pkl --data data1.csv --speed 0 --threshold 0.3
    python inference_power.py --help
"""

import argparse
import sys
import os
import time
import pickle
import warnings
from datetime import datetime, timedelta
from collections import deque

warnings.filterwarnings("ignore")

import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ─────────────────────────────────────────────
# CLI helpers
# ─────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
RED    = "\033[31m"
DIM    = "\033[2m"
CLEAR_LINE = "\033[2K\033[1G"

def header(msg):
    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {msg}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}")

def info(msg):
    print(f"  {DIM}▸{RESET} {msg}")

def success(msg):
    print(f"  {GREEN}✔{RESET}  {msg}")

def warn(msg):
    print(f"  {YELLOW}⚠{RESET}  {msg}")

def alert(msg):
    print(f"\n  {RED}{BOLD}⚠ ALERT:{RESET} {RED}{msg}{RESET}")

# ─────────────────────────────────────────────
# Load model bundle
# ─────────────────────────────────────────────

def load_model(path):
    if not os.path.exists(path):
        print(f"  {RED}✘{RESET}  Model file not found: {path}")
        sys.exit(1)
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    success(f"Loaded model from '{path}' (trained {bundle['trained_at']})")
    info(f"Expected features: {len(bundle['feature_names'])}")

    # Recover label mapping
    encoder = bundle.get("label_encoder")
    if encoder is not None:
        class_mapping = dict(zip(encoder.transform(encoder.classes_), encoder.classes_))
        info(f"Label mapping: {class_mapping}")
    else:
        class_mapping = {0: "Natural/Normal", 1: "Attack"}

    return bundle["model"], bundle["scaler"], bundle["feature_names"], class_mapping

# ─────────────────────────────────────────────
# Load & prepare raw data (same pipeline as training)
# ─────────────────────────────────────────────

def prepare_stream(path, interval_seconds=1, start="2024-01-01"):
    if not os.path.exists(path):
        print(f"  {RED}✘{RESET}  Path not found: {path}")
        sys.exit(1)

    if os.path.isdir(path):
        csv_files = sorted([
            os.path.join(path, f) for f in os.listdir(path)
            if f.lower().endswith(".csv")
        ])
        if not csv_files:
            print(f"  {RED}✘{RESET}  No CSV files found in folder: {path}")
            sys.exit(1)
        info(f"Found {len(csv_files)} CSV files in '{path}'")
        frames = []
        for csv_path in csv_files:
            part = pd.read_csv(csv_path)
            info(f"  {os.path.basename(csv_path):>12}: {len(part):>8,} rows")
            frames.append(part)
        df = pd.concat(frames, ignore_index=True)
        success(f"Combined {len(csv_files)} files → {len(df):,} rows")
    else:
        df = pd.read_csv(path)
        success(f"Loaded {len(df):,} rows from '{path}'")

    # Clean: remove NaN/Inf rows
    before = len(df)
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)].copy()
    df.reset_index(drop=True, inplace=True)
    removed = before - len(df)
    if removed > 0:
        warn(f"Removed {removed:,} rows containing NaN/Inf")

    # Simulate timestamps
    start_dt = datetime.fromisoformat(start)
    df["timestamp"] = [
        start_dt + timedelta(seconds=interval_seconds * i)
        for i in range(len(df))
    ]

    # Encode and keep ground truth
    encoder = LabelEncoder()
    ground_truth = encoder.fit_transform(df["marker"])
    ground_truth_labels = df["marker"].values  # keep original string labels

    # ── Time-based features ──
    df["hour_of_day"]      = df["timestamp"].dt.hour
    df["minute_of_hour"]   = df["timestamp"].dt.minute
    df["second_of_minute"] = df["timestamp"].dt.second

    # ── PMU signal features ──
    pmu_cols = [c for c in df.columns if c.startswith(("R1-", "R2-", "R3-", "R4-"))]
    relay_cols = [c for c in df.columns if c in ["R1:S", "R2:S", "R3:S", "R4:S"]]

    for pmu in ["R1", "R2", "R3", "R4"]:
        vh_cols = [c for c in pmu_cols if c.startswith(f"{pmu}-") and ":VH" in c]
        ih_cols = [c for c in pmu_cols if c.startswith(f"{pmu}-") and ":IH" in c]

        if len(vh_cols) >= 2:
            df[f"{pmu}_voltage_phase_spread"] = df[vh_cols].max(axis=1) - df[vh_cols].min(axis=1)

        if len(ih_cols) >= 2:
            df[f"{pmu}_current_phase_spread"] = df[ih_cols].max(axis=1) - df[ih_cols].min(axis=1)

        pmu_specific = [c for c in pmu_cols if c.startswith(f"{pmu}-")]
        if pmu_specific:
            df[f"{pmu}_signal_mean"] = df[pmu_specific].mean(axis=1)
            df[f"{pmu}_signal_std"]  = df[pmu_specific].std(axis=1)

    # ── Rolling window features ──
    window = 10
    for pmu in ["R1", "R2", "R3", "R4"]:
        repr_col = f"{pmu}-PA1:VH"
        if repr_col in df.columns:
            safe = repr_col.replace("-", "_").replace(":", "_")
            df[f"{safe}_roll_mean"] = df[repr_col].rolling(window, min_periods=1).mean()
            df[f"{safe}_roll_std"]  = df[repr_col].rolling(window, min_periods=1).std().fillna(0)

    # ── Cross-PMU features ──
    if "R1-PA1:VH" in df.columns and "R3-PA1:VH" in df.columns:
        df["line1_line2_voltage_diff"] = df["R1-PA1:VH"] - df["R3-PA1:VH"]

    if relay_cols:
        df["relay_status_sum"] = df[relay_cols].sum(axis=1)

    snort_cols = [c for c in df.columns if "snort" in c.lower()]
    if snort_cols:
        df["snort_alert_sum"] = df[snort_cols].sum(axis=1)

    control_cols = [c for c in df.columns if "control" in c.lower()]
    if control_cols:
        df["control_log_sum"] = df[control_cols].sum(axis=1)

    # Encode marker to numeric (same as training)
    df["marker"] = encoder.transform(df["marker"])

    timestamps = df["timestamp"].values
    df.drop(columns=["timestamp", "marker"], inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    return df, ground_truth, ground_truth_labels, timestamps

# ─────────────────────────────────────────────
# Streaming loop
# ─────────────────────────────────────────────

def run_stream(model, scaler, feature_names, class_mapping,
               df, ground_truth, ground_truth_labels, timestamps,
               speed, threshold, max_rows, api_url=None, node_id="node_1"):

    # ── API helper ──
    api_session = requests.Session() if api_url else None
    api_endpoint = f"{api_url}/api/power/reading" if api_url else None
    heartbeat_endpoint = f"{api_url}/api/heartbeat" if api_url else None
    api_failures = 0

    def send_to_api(payload):
        nonlocal api_failures
        if not api_session:
            return
        try:
            resp = api_session.post(api_endpoint, json=payload, timeout=2)
            if resp.status_code != 201:
                api_failures += 1
        except requests.RequestException:
            api_failures += 1

    def send_heartbeat():
        if not api_session:
            return
        try:
            api_session.post(heartbeat_endpoint, json={
                "agent_type": "power", "node_id": node_id
            }, timeout=2)
        except requests.RequestException:
            pass

    header("Live PMU Sensor Stream")
    print(f"  Threshold : {threshold:.0%}  |  Speed: {'max' if speed == 0 else f'{speed}s/reading'}")
    print(f"  Streaming : {max_rows:,} readings\n")
    print(f"  {DIM}{'Timestamp':<20} {'R1:VH':>8} {'R2:VH':>8} {'R3:VH':>8} {'R4:VH':>8} {'Prob':>7} {'Status'}{RESET}")
    print(f"  {'─'*78}")

    total = min(max_rows, len(df))
    alerts = 0
    correct = 0
    tp = fp = tn = fn = 0

    recent_probs = deque(maxlen=50)

    for i in range(total):
        row = df.iloc[i][feature_names].values.reshape(1, -1)
        row_scaled = scaler.transform(row)
        prob = model.predict_proba(row_scaled)[0][1]
        pred = int(prob >= threshold)
        actual = int(ground_truth[i])
        ts = pd.Timestamp(timestamps[i]).strftime("%Y-%m-%d %H:%M:%S")

        recent_probs.append(prob)

        # Confusion tracking
        if pred == 1 and actual == 1: tp += 1
        elif pred == 1 and actual == 0: fp += 1
        elif pred == 0 and actual == 0: tn += 1
        else: fn += 1

        if pred == actual:
            correct += 1

        # Pull key PMU voltage readings for display
        r1_vh = df.iloc[i].get("R1-PA1:VH", 0)
        r2_vh = df.iloc[i].get("R2-PA1:VH", 0)
        r3_vh = df.iloc[i].get("R3-PA1:VH", 0)
        r4_vh = df.iloc[i].get("R4-PA1:VH", 0)

        # Colour-code probability
        if prob >= threshold:
            prob_str = f"{RED}{BOLD}{prob:.2%}{RESET}"
            status   = f"{RED}{BOLD}ATTACK {RESET}"
            status_label = "ATTACK"
            alerts  += 1
        elif prob >= threshold * 0.6:
            prob_str = f"{YELLOW}{prob:.2%}{RESET}"
            status   = f"{YELLOW}WARNING{RESET}"
            status_label = "WARNING"
        else:
            prob_str = f"{GREEN}{prob:.2%}{RESET}"
            status   = f"{GREEN}normal {RESET}"
            status_label = "normal"

        print(f"  {ts:<20} {r1_vh:>8.3f} {r2_vh:>8.3f} {r3_vh:>8.3f} {r4_vh:>8.3f} {prob_str:>7}  {status}")

        # ── Send to API server ──
        relay_status = df.iloc[i].get("relay_status_sum", 0)

        send_to_api({
            "node_id":       node_id,
            "timestamp":     ts,
            "r1_vh":         round(float(r1_vh), 4),
            "r2_vh":         round(float(r2_vh), 4),
            "r3_vh":         round(float(r3_vh), 4),
            "r4_vh":         round(float(r4_vh), 4),
            "relay_status":  int(relay_status),
            "probability":   round(float(prob), 4),
            "prediction":    pred,
            "actual":        actual,
            "status":        status_label,
        })

        # Print detail on misclassifications
        if actual == 1 and pred == 0:
            print(f"  {DIM}  └─ missed attack (prob {prob:.2%} below threshold {threshold:.0%}){RESET}")
        elif actual == 0 and pred == 1:
            print(f"  {DIM}  └─ false alarm{RESET}")

        if speed > 0:
            time.sleep(speed)

    # ── Summary ──
    header("Session Summary")
    acc = correct / total if total > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    print(f"\n  {'Readings processed':<28} {total:>8,}")
    print(f"  {'Attacks in window':<28} {tp+fn:>8,}")
    print(f"  {'Alerts fired':<28} {alerts:>8,}")
    print(f"  {'─'*38}")
    print(f"  {'Accuracy':<28} {acc:>8.2%}")
    print(f"  {'Precision':<28} {prec:>8.2%}")
    print(f"  {'Recall':<28} {rec:>8.2%}")
    print(f"  {'F1 Score':<28} {f1:>8.2%}")
    print(f"\n  Confusion matrix")
    print(f"  {'─'*38}")
    print(f"  {'':>14} Pred 0   Pred 1")
    print(f"  {'Actual 0':>14}  {tn:>5}    {fp:>5}")
    print(f"  {'Actual 1':>14}  {fn:>5}    {tp:>5}")

    if api_url:
        print(f"\n  API: {api_endpoint}  ({api_failures} failed POSTs)")

    print(f"\n{GREEN}{BOLD}Done.{RESET}\n")

# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream PMU sensor readings through a trained power system attack detection model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference_power.py --model power_model.pkl --data ./power_data/
  python inference_power.py --model power_model.pkl --data ./power_data/ --speed 0.1
  python inference_power.py --model power_model.pkl --data ./power_data/ --shuffle --rows 500
  python inference_power.py --model power_model.pkl --data ./power_data/ --threshold 0.3
        """
    )
    parser.add_argument("--model",     required=True,            help="Path to power_model.pkl")
    parser.add_argument("--data",      required=True,            help="Path to a CSV file or folder containing data1.csv–data15.csv")
    parser.add_argument("--speed",     type=float, default=0.05, help="Seconds per reading (0 = instant, default: 0.05)")
    parser.add_argument("--threshold", type=float, default=0.5,  help="Attack probability threshold (default: 0.5)")
    parser.add_argument("--rows",      type=int,   default=200,  help="Number of rows to stream (default: 200)")
    parser.add_argument("--interval",  type=int,   default=1,    help="Simulated seconds per reading (default: 1)")
    parser.add_argument("--start",     default="2024-01-01",     help="Simulation start date (default: 2024-01-01)")
    parser.add_argument("--shuffle",   action="store_true",      help="Shuffle rows before streaming for a representative sample")
    parser.add_argument("--api-url",   default=None,             help="API server URL (e.g. http://localhost:5000). If set, readings are POSTed to /api/power/reading")
    parser.add_argument("--node-name", default=None,             help="Optional human-readable node label (e.g. 'pmu-bus7'). Server assigns the actual node_id on registration.")
    return parser.parse_args()


def register_node(api_url, agent_type, node_name=None):
    """Register with the API server and return the server-assigned node_id."""
    reg_url = f"{api_url}/api/register"
    payload = {"agent_type": agent_type}
    if node_name:
        payload["node_name"] = node_name
    try:
        resp = requests.post(reg_url, json=payload, timeout=5)
        if resp.status_code == 201:
            data = resp.json()
            node_id = data["node_id"]
            success(f"Registered with API → node_id={node_id}  (name={data.get('node_name', node_id)})")
            return node_id
        else:
            print(f"  {RED}✘{RESET}  Registration failed (HTTP {resp.status_code}): {resp.text}")
            sys.exit(1)
    except requests.RequestException as e:
        print(f"  {RED}✘{RESET}  Cannot reach API server at {api_url}: {e}")
        sys.exit(1)


def main():
    args = parse_args()

    print(f"\n{BOLD}Power System Attack Detection — Streaming Inference{RESET}")
    print(f"{DIM}Live PMU sensor simulation | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")

    # ── Register with API server ──
    node_id = None
    if args.api_url:
        header("Registering")
        node_id = register_node(args.api_url, "power", args.node_name)
    else:
        node_id = "local"

    header("Loading")
    model, scaler, feature_names, class_mapping = load_model(args.model)

    header("Preparing Data")
    df, ground_truth, ground_truth_labels, timestamps = prepare_stream(
        args.data,
        interval_seconds=args.interval,
        start=args.start
    )

    if args.shuffle:
        header("Shuffling Data")
        idx = np.random.default_rng(seed=42).permutation(len(df))
        df = df.iloc[idx].reset_index(drop=True)
        ground_truth = ground_truth[idx]
        ground_truth_labels = ground_truth_labels[idx]
        timestamps = timestamps[idx]
        success(f"Shuffled {len(df):,} rows (seed=42)")

    run_stream(
        model, scaler, feature_names, class_mapping,
        df, ground_truth, ground_truth_labels, timestamps,
        speed=args.speed,
        threshold=args.threshold,
        max_rows=args.rows,
        api_url=args.api_url,
        node_id=node_id
    )


if __name__ == "__main__":
    main()