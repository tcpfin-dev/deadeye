#!/usr/bin/env python3
"""
Predictive Maintenance - Streaming Inference
Loads a trained model.pkl and simulates live sensor data streaming.

Usage:
    python inference.py --model model.pkl --data ai4i2020.csv
    python inference.py --model model.pkl --data ai4i2020.csv --speed 0.5
    python inference.py --model model.pkl --data ai4i2020.csv --speed 0 --threshold 0.3
    python inference.py --help
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
from sklearn.preprocessing import MinMaxScaler

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
    return bundle["model"], bundle["scaler"], bundle["feature_names"]

# ─────────────────────────────────────────────
# Load & prepare raw data (same pipeline as training)
# ─────────────────────────────────────────────

def prepare_stream(path, interval_minutes=5, start="2024-01-01"):
    if not os.path.exists(path):
        print(f"  {RED}✘{RESET}  Data file not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path)
    success(f"Loaded {len(df):,} rows from '{path}'")

    # Simulate timestamps
    start_dt = datetime.fromisoformat(start)
    df["timestamp"] = [
        start_dt + timedelta(minutes=interval_minutes * i)
        for i in range(len(df))
    ]

    # Keep ground truth before dropping
    ground_truth = df["Machine failure"].values

    # Drop ID cols and failure mode labels
    drop_cols = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Encode type
    if "Type" in df.columns:
        df = pd.get_dummies(df, columns=["Type"], drop_first=True)

    # Time features
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["shift"] = pd.cut(
        df["timestamp"].dt.hour,
        bins=[-1, 7, 15, 23],
        labels=[0, 1, 2]
    ).astype(int)

    # Rolling window features
    window = 12
    for col in ["Torque [Nm]", "Rotational speed [rpm]", "Air temperature [K]"]:
        if col in df.columns:
            safe = col.replace(" ", "_").replace("[", "").replace("]", "")
            df[f"{safe}_roll_mean"] = df[col].rolling(window, min_periods=1).mean()
            df[f"{safe}_roll_std"]  = df[col].rolling(window, min_periods=1).std().fillna(0)

    # Physics features
    if "Torque [Nm]" in df.columns and "Rotational speed [rpm]" in df.columns:
        df["power_W"] = df["Torque [Nm]"] * (df["Rotational speed [rpm]"] * 2 * np.pi / 60)

    if "Air temperature [K]" in df.columns and "Process temperature [K]" in df.columns:
        df["temp_delta"] = df["Process temperature [K]"] - df["Air temperature [K]"]

    if "Tool wear [min]" in df.columns and "Torque [Nm]" in df.columns:
        df["wear_torque"] = df["Tool wear [min]"] * df["Torque [Nm]"]
        df["cumulative_wear"] = df["Tool wear [min]"].cumsum()

    timestamps = df["timestamp"].values
    df.drop(columns=["timestamp", "Machine failure"], inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    return df, ground_truth, timestamps

# ─────────────────────────────────────────────
# Streaming loop
# ─────────────────────────────────────────────

def run_stream(model, scaler, feature_names, df, ground_truth, timestamps,
               speed, threshold, max_rows, api_url=None, node_id="node_1"):

    # ── API helper ──
    api_session = requests.Session() if api_url else None
    api_endpoint = f"{api_url}/api/maintenance/reading" if api_url else None
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

    header("Live Sensor Stream")
    print(f"  Threshold : {threshold:.0%}  |  Speed: {'max' if speed == 0 else f'{speed}s/reading'}")
    print(f"  Streaming : {max_rows:,} readings\n")
    print(f"  {DIM}{'Timestamp':<20} {'Tool wear':>9} {'Torque':>8} {'Power W':>9} {'Prob':>7} {'Status'}{RESET}")
    print(f"  {'─'*70}")

    total = min(max_rows, len(df))
    alerts = 0
    correct = 0
    tp = fp = tn = fn = 0

    # Stats window for summary
    recent_probs = deque(maxlen=50)

    for i in range(total):
        row = df.iloc[i][feature_names].values.reshape(1, -1)
        row_scaled = scaler.transform(row)
        prob = model.predict_proba(row_scaled)[0][1]
        pred = int(prob >= threshold)
        actual = int(ground_truth[i])
        ts = pd.Timestamp(timestamps[i]).strftime("%Y-%m-%d %H:%M")

        recent_probs.append(prob)

        # Confusion tracking
        if pred == 1 and actual == 1: tp += 1
        elif pred == 1 and actual == 0: fp += 1
        elif pred == 0 and actual == 0: tn += 1
        else: fn += 1

        if pred == actual:
            correct += 1

        # Pull key sensor values for display
        tool_wear = df.iloc[i].get("Tool wear [min]", 0)
        torque    = df.iloc[i].get("Torque [Nm]", 0)
        power     = df.iloc[i].get("power_W", 0)

        # Colour-code probability
        if prob >= threshold:
            prob_str = f"{RED}{BOLD}{prob:.2%}{RESET}"
            status   = f"{RED}{BOLD}FAILURE{RESET}"
            status_label = "FAILURE"
            alerts  += 1
        elif prob >= threshold * 0.6:
            prob_str = f"{YELLOW}{prob:.2%}{RESET}"
            status   = f"{YELLOW}WARNING{RESET}"
            status_label = "WARNING"
        else:
            prob_str = f"{GREEN}{prob:.2%}{RESET}"
            status   = f"{GREEN}normal {RESET}"
            status_label = "normal"

        print(f"  {ts:<20} {tool_wear:>9.1f} {torque:>8.1f} {power:>9.0f} {prob_str:>7}  {status}")

        # ── Send to API server ──
        air_temp  = df.iloc[i].get("Air temperature [K]", 0)
        proc_temp = df.iloc[i].get("Process temperature [K]", 0)
        rot_speed = df.iloc[i].get("Rotational speed [rpm]", 0)

        send_to_api({
            "node_id":           node_id,
            "timestamp":         ts,
            "tool_wear":         round(float(tool_wear), 2),
            "torque":            round(float(torque), 2),
            "power_w":           round(float(power), 1),
            "air_temp":          round(float(air_temp), 2),
            "process_temp":      round(float(proc_temp), 2),
            "rotational_speed":  round(float(rot_speed), 1),
            "probability":       round(float(prob), 4),
            "prediction":        pred,
            "actual":            actual,
            "status":            status_label,
        })

        # Print alert detail on actual failures
        if actual == 1 and pred == 0:
            print(f"  {DIM}  └─ missed failure (prob {prob:.2%} below threshold {threshold:.0%}){RESET}")
        elif actual == 0 and pred == 1:
            print(f"  {DIM}  └─ false alarm{RESET}")

        if speed > 0:
            time.sleep(speed)

    # ── Summary ──
    header("Session Summary")
    acc = correct / total
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    print(f"\n  {'Readings processed':<28} {total:>8,}")
    print(f"  {'Failures in window':<28} {tp+fn:>8,}")
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
        description="Stream sensor readings through a trained maintenance model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py --model model.pkl --data ai4i2020.csv
  python inference.py --model model.pkl --data ai4i2020.csv --speed 0.1
  python inference.py --model model.pkl --data ai4i2020.csv --speed 0 --rows 500
  python inference.py --model model.pkl --data ai4i2020.csv --threshold 0.3
        """
    )
    parser.add_argument("--model",     required=True,            help="Path to model.pkl")
    parser.add_argument("--data",      required=True,            help="Path to ai4i2020.csv")
    parser.add_argument("--speed",     type=float, default=0.05, help="Seconds per reading (0 = instant, default: 0.05)")
    parser.add_argument("--threshold", type=float, default=0.5,  help="Failure probability threshold (default: 0.5)")
    parser.add_argument("--rows",      type=int,   default=200,  help="Number of rows to stream (default: 200)")
    parser.add_argument("--interval",  type=int,   default=5,    help="Simulated minutes per reading (default: 5)")
    parser.add_argument("--start",     default="2024-01-01",     help="Simulation start date (default: 2024-01-01)")
    parser.add_argument("--api-url",   default=None,             help="API server URL (e.g. http://localhost:5000). If set, readings are POSTed to /api/maintenance/reading")
    parser.add_argument("--node-name", default=None,             help="Optional human-readable node label (e.g. 'turbine-A3'). Server assigns the actual node_id on registration.")
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

    print(f"\n{BOLD}Predictive Maintenance — Streaming Inference{RESET}")
    print(f"{DIM}Live sensor simulation | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")

    # ── Register with API server ──
    node_id = None
    if args.api_url:
        header("Registering")
        node_id = register_node(args.api_url, "maintenance", args.node_name)
    else:
        node_id = "local"

    header("Loading")
    model, scaler, feature_names = load_model(args.model)

    header("Preparing Data")
    df, ground_truth, timestamps = prepare_stream(
        args.data,
        interval_minutes=args.interval,
        start=args.start
    )

    run_stream(
        model, scaler, feature_names,
        df, ground_truth, timestamps,
        speed=args.speed,
        threshold=args.threshold,
        max_rows=args.rows,
        api_url=args.api_url,
        node_id=node_id
    )


if __name__ == "__main__":
    main()