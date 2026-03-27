#!/usr/bin/env python3
"""
Power System Attack Detection - Training Script
CLI training script with simulated timestamps and time-based features.

Based on the Power System Attack Dataset (MSU / ORNL).
Binary classification: Natural/Normal events vs Attack events.

Usage:
    python train_power.py --data ./power_data/
    python train_power.py --data ./power_data/ --interval 1
    python train_power.py --data data1.csv --output power_model.pkl
    python train_power.py --help
"""

import argparse
import sys
import os
import time
import warnings
import pickle
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix
)

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

def error(msg):
    print(f"  {RED}✘{RESET}  {msg}")
    sys.exit(1)

def step(n, total, msg):
    print(f"\n{BOLD}[{n}/{total}]{RESET} {msg}")

# ─────────────────────────────────────────────
# Step 1 — Load data
# ─────────────────────────────────────────────

def load_data(path):
    if not os.path.exists(path):
        error(f"Path not found: {path}")

    if os.path.isdir(path):
        csv_files = sorted([
            os.path.join(path, f) for f in os.listdir(path)
            if f.lower().endswith(".csv")
        ])
        if not csv_files:
            error(f"No CSV files found in folder: {path}")

        info(f"Found {len(csv_files)} CSV files in '{path}'")
        frames = []
        for csv_path in csv_files:
            part = pd.read_csv(csv_path)
            info(f"  {os.path.basename(csv_path):>12}: {len(part):>8,} rows")
            frames.append(part)
        df = pd.concat(frames, ignore_index=True)
        success(f"Combined {len(csv_files)} files → {len(df):,} rows, {len(df.columns)} columns")
    else:
        df = pd.read_csv(path)
        success(f"Loaded {len(df):,} rows, {len(df.columns)} columns from '{path}'")

    return df

# ─────────────────────────────────────────────
# Step 2 — Clean data
# ─────────────────────────────────────────────

def clean_data(df):
    before = len(df)
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)].copy()
    removed = before - len(df)
    if removed > 0:
        warn(f"Removed {removed:,} rows containing NaN/Inf values")
    success(f"Clean dataset: {len(df):,} rows")
    return df

# ─────────────────────────────────────────────
# Step 3 — Add simulated timestamps
# ─────────────────────────────────────────────

def add_timestamps(df, interval_seconds=1, start="2024-01-01"):
    """Power grid PMU data is typically sampled at high frequency (e.g. 1s intervals)."""
    start_dt = datetime.fromisoformat(start)
    df = df.copy()
    df["timestamp"] = [
        start_dt + timedelta(seconds=interval_seconds * i)
        for i in range(len(df))
    ]
    end_dt = df["timestamp"].iloc[-1]
    duration = end_dt - start_dt
    success(
        f"Timestamps added — {interval_seconds}s intervals, "
        f"{start} → {end_dt} (duration: {duration})"
    )
    return df

# ─────────────────────────────────────────────
# Step 4 — Feature engineering
# ─────────────────────────────────────────────

def engineer_features(df):
    df = df.copy()

    # Encode target: marker column (binary: natural/normal vs attack)
    encoder = LabelEncoder()
    df["marker"] = encoder.fit_transform(df["marker"])
    class_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    info(f"Target encoding: {class_mapping}")

    # ── Time-based features ──
    df["hour_of_day"]  = df["timestamp"].dt.hour
    df["minute_of_hour"] = df["timestamp"].dt.minute
    df["second_of_minute"] = df["timestamp"].dt.second

    # ── PMU signal features ──
    # Identify PMU measurement columns (R1-*, R2-*, R3-*, R4-*)
    pmu_cols = [c for c in df.columns if c.startswith(("R1-", "R2-", "R3-", "R4-"))]
    relay_cols = [c for c in df.columns if c in ["R1:S", "R2:S", "R3:S", "R4:S"]]

    # Voltage phase angle columns per PMU
    for pmu in ["R1", "R2", "R3", "R4"]:
        vh_cols = [c for c in pmu_cols if c.startswith(f"{pmu}-") and ":VH" in c]
        ih_cols = [c for c in pmu_cols if c.startswith(f"{pmu}-") and ":IH" in c]

        # Voltage phase angle spread (max - min across phases)
        if len(vh_cols) >= 2:
            df[f"{pmu}_voltage_phase_spread"] = df[vh_cols].max(axis=1) - df[vh_cols].min(axis=1)

        # Current phase angle spread
        if len(ih_cols) >= 2:
            df[f"{pmu}_current_phase_spread"] = df[ih_cols].max(axis=1) - df[ih_cols].min(axis=1)

        # Mean and std of all PMU signals
        pmu_specific = [c for c in pmu_cols if c.startswith(f"{pmu}-")]
        if pmu_specific:
            df[f"{pmu}_signal_mean"] = df[pmu_specific].mean(axis=1)
            df[f"{pmu}_signal_std"]  = df[pmu_specific].std(axis=1)

    # ── Rolling window features (window=10 readings) ──
    window = 10
    # Pick a representative voltage column from each PMU for rolling stats
    for pmu in ["R1", "R2", "R3", "R4"]:
        repr_col = f"{pmu}-PA1:VH"
        if repr_col in df.columns:
            safe = repr_col.replace("-", "_").replace(":", "_")
            df[f"{safe}_roll_mean"] = df[repr_col].rolling(window, min_periods=1).mean()
            df[f"{safe}_roll_std"]  = df[repr_col].rolling(window, min_periods=1).std().fillna(0)

    # ── Cross-PMU features ──
    # Difference between PMU R1 and R3 voltage (Line 1 vs Line 2)
    if "R1-PA1:VH" in df.columns and "R3-PA1:VH" in df.columns:
        df["line1_line2_voltage_diff"] = df["R1-PA1:VH"] - df["R3-PA1:VH"]

    # Relay status sum (total active relays)
    if relay_cols:
        df["relay_status_sum"] = df[relay_cols].sum(axis=1)

    # Snort alert columns
    snort_cols = [c for c in df.columns if "snort" in c.lower()]
    if snort_cols:
        df["snort_alert_sum"] = df[snort_cols].sum(axis=1)

    # Control panel log columns
    control_cols = [c for c in df.columns if "control" in c.lower()]
    if control_cols:
        df["control_log_sum"] = df[control_cols].sum(axis=1)

    # Drop timestamp — not a numeric feature
    df.drop(columns=["timestamp"], inplace=True)

    # Fill any remaining NaNs
    df.fillna(df.mean(numeric_only=True), inplace=True)

    feature_count = df.shape[1] - 1  # minus target
    success(f"Engineered additional feature groups (time, PMU spreads, rolling, cross-PMU)")
    info(f"Final feature count: {feature_count}")
    return df, encoder

# ─────────────────────────────────────────────
# Step 5 — Split & scale
# ─────────────────────────────────────────────

def prepare_splits(df, test_size=0.2, random_state=21):
    target = "marker"
    X = df.drop(columns=[target])
    y = df[target]

    attack_rate = y.mean() * 100
    info(f"Class balance — attacks: {y.sum()} ({attack_rate:.2f}%), normal/natural: {(y==0).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    success(f"Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
    return X_train_s, X_test_s, y_train, y_test, scaler, list(X.columns)

# ─────────────────────────────────────────────
# Step 6 — Train
# ─────────────────────────────────────────────

def train_model(X_train, y_train):
    model = ExtraTreesClassifier(
        n_estimators=100,
        n_jobs=-1,
        random_state=42
    )
    info("Training Extra Trees Classifier (n_estimators=100)...")
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    success(f"Extra Trees trained in {elapsed:.2f}s")
    return model

# ─────────────────────────────────────────────
# Step 7 — Evaluate
# ─────────────────────────────────────────────

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    mcc  = matthews_corrcoef(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob) if y_prob is not None else None

    print(f"\n  {'Metric':<22} {'Value':>10}")
    print(f"  {'─'*34}")
    print(f"  {'Accuracy':<22} {acc:>10.2%}")
    print(f"  {'Precision':<22} {prec:>10.2%}")
    print(f"  {'Recall':<22} {rec:>10.2%}")
    print(f"  {'F1 Score':<22} {f1:>10.2%}")
    print(f"  {'MCC':<22} {mcc:>10.4f}")
    if auc is not None:
        print(f"  {'ROC-AUC':<22} {auc:>10.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Confusion matrix")
    print(f"  {'─'*34}")
    print(f"  {'':>12} Pred 0   Pred 1")
    print(f"  {'Actual 0':>12}  {tn:>5}    {fp:>5}")
    print(f"  {'Actual 1':>12}  {fn:>5}    {tp:>5}")

    return {"accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "mcc": mcc, "auc": auc}

# ─────────────────────────────────────────────
# Step 8 — Feature importance
# ─────────────────────────────────────────────

def show_feature_importance(model, feature_names, top_n=15):
    if not hasattr(model, "feature_importances_"):
        info("Model does not expose feature importances.")
        return

    importances = model.feature_importances_
    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    print(f"\n  {'Feature':<45} {'Importance':>10}")
    print(f"  {'─'*57}")
    for feat, imp in ranked[:top_n]:
        bar = "█" * int(imp * 80)
        print(f"  {feat:<45} {imp:>8.4f}  {bar}")

# ─────────────────────────────────────────────
# Step 9 — Save
# ─────────────────────────────────────────────

def save_artifacts(model, scaler, feature_names, encoder, output_path):
    bundle = {
        "model":         model,
        "scaler":        scaler,
        "feature_names": feature_names,
        "label_encoder": encoder,
        "trained_at":    datetime.now().isoformat(),
    }
    with open(output_path, "wb") as f:
        pickle.dump(bundle, f)
    size_kb = os.path.getsize(output_path) / 1024
    success(f"Saved model bundle to '{output_path}' ({size_kb:.1f} KB)")

# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an Extra Trees attack detection classifier on the Power System Attack dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_power.py --data ./power_data/
  python train_power.py --data ./power_data/ --interval 1 --output power_model.pkl
  python train_power.py --data data1.csv --test-size 0.3 --no-save
        """
    )
    parser.add_argument("--data",      required=True,            help="Path to a CSV file or folder containing data1.csv–data15.csv")
    parser.add_argument("--interval",  type=int, default=1,      help="Seconds between PMU readings (default: 1)")
    parser.add_argument("--start",     default="2024-01-01",     help="Simulation start date (default: 2024-01-01)")
    parser.add_argument("--test-size", type=float, default=0.2,  help="Test split fraction (default: 0.2)")
    parser.add_argument("--top-n",     type=int, default=15,     help="Top N features to display (default: 15)")
    parser.add_argument("--output",    default="power_model.pkl",help="Output path for saved model (default: power_model.pkl)")
    parser.add_argument("--no-save",   action="store_true",      help="Skip saving the model")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{BOLD}Power System Attack Detection — Training{RESET}")
    print(f"{DIM}MSU/ORNL Power System Dataset | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")

    total_steps = 7

    # 1. Load
    step(1, total_steps, "Loading dataset")
    df = load_data(args.data)

    # 2. Clean
    step(2, total_steps, "Cleaning data (removing NaN/Inf rows)")
    df = clean_data(df)

    # 3. Timestamps
    step(3, total_steps, "Simulating timestamps")
    df = add_timestamps(df, interval_seconds=args.interval, start=args.start)

    # 4. Features
    step(4, total_steps, "Engineering features")
    df, encoder = engineer_features(df)

    # 5. Split & scale
    step(5, total_steps, "Splitting and scaling")
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_splits(
        df, test_size=args.test_size
    )

    # 6. Train
    step(6, total_steps, "Training model: Extra Trees Classifier")
    model = train_model(X_train, y_train)

    # 7. Evaluate
    step(7, total_steps, "Evaluating on test set")
    evaluate(model, X_test, y_test)

    # Feature importance
    header("Feature Importance")
    show_feature_importance(model, feature_names, top_n=args.top_n)

    # Save
    if not args.no_save:
        header("Saving Model")
        save_artifacts(model, scaler, feature_names, encoder, args.output)

    print(f"\n{GREEN}{BOLD}Done.{RESET}\n")


if __name__ == "__main__":
    main()