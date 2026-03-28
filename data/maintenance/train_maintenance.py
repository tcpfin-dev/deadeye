"""
Predictive Maintenance - Milling Machine
CLI training script with simulated timestamps and time-based features.

Usage:
    python train_maintenance.py --data ai4i2020.csv
    python train_maintenance.py --data ai4i2020.csv --interval 5
    python train_maintenance.py --data ai4i2020.csv --output model.pkl
    python train_maintenance.py --help
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
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
        error(f"File not found: {path}")
    df = pd.read_csv(path)
    success(f"Loaded {len(df):,} rows, {len(df.columns)} columns from '{path}'")
    return df

# ─────────────────────────────────────────────
# Step 2 — Add simulated timestamps
# ─────────────────────────────────────────────

def add_timestamps(df, interval_minutes=5, start="2024-01-01"):
    start_dt = datetime.fromisoformat(start)
    df = df.copy()
    df["timestamp"] = [
        start_dt + timedelta(minutes=interval_minutes * i)
        for i in range(len(df))
    ]
    end_dt = df["timestamp"].iloc[-1]
    duration_days = (end_dt - start_dt).days
    success(
        f"Timestamps added — {interval_minutes}-min intervals, "
        f"{start} → {end_dt.date()} ({duration_days} days)"
    )
    return df

# ─────────────────────────────────────────────
# Step 3 — Feature engineering
# ─────────────────────────────────────────────

def engineer_features(df):
    df = df.copy()

    # Drop non-predictive ID columns and individual failure mode labels
    drop_cols = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Encode product type
    if "Type" in df.columns:
        df = pd.get_dummies(df, columns=["Type"], drop_first=True)

    # ── Time-based features ──
    df["hour_of_day"]  = df["timestamp"].dt.hour
    df["day_of_week"]  = df["timestamp"].dt.dayofweek
    df["shift"] = pd.cut(
        df["timestamp"].dt.hour,
        bins=[-1, 7, 15, 23],
        labels=[0, 1, 2]          # 0=night, 1=day, 2=evening
    ).astype(int)

    # ── Rolling window features (12 readings = 1 hour at 5-min intervals) ──
    window = 12
    for col in ["Torque [Nm]", "Rotational speed [rpm]", "Air temperature [K]"]:
        if col in df.columns:
            safe = col.replace(" ", "_").replace("[", "").replace("]", "")
            df[f"{safe}_roll_mean"] = df[col].rolling(window, min_periods=1).mean()
            df[f"{safe}_roll_std"]  = df[col].rolling(window, min_periods=1).std().fillna(0)

    # ── Derived physics features ──
    if "Torque [Nm]" in df.columns and "Rotational speed [rpm]" in df.columns:
        df["power_W"] = df["Torque [Nm]"] * (df["Rotational speed [rpm]"] * 2 * np.pi / 60)

    if "Air temperature [K]" in df.columns and "Process temperature [K]" in df.columns:
        df["temp_delta"] = df["Process temperature [K]"] - df["Air temperature [K]"]

    if "Tool wear [min]" in df.columns and "Torque [Nm]" in df.columns:
        df["wear_torque"] = df["Tool wear [min]"] * df["Torque [Nm]"]
        df["cumulative_wear"] = df["Tool wear [min]"].cumsum()

    # Drop timestamp — not a numeric feature
    df.drop(columns=["timestamp"], inplace=True)

    # Fill any NaNs
    df.fillna(df.mean(numeric_only=True), inplace=True)

    features_added = [
        "hour_of_day", "day_of_week", "shift",
        "rolling means/stds (torque, speed, temp)",
        "power_W", "temp_delta", "wear_torque", "cumulative_wear"
    ]
    success(f"Engineered {len(features_added)} feature groups")
    info(f"Final feature count: {df.shape[1] - 1}")  # minus target
    return df

# ─────────────────────────────────────────────
# Step 4 — Split & scale
# ─────────────────────────────────────────────

def prepare_splits(df, test_size=0.2, random_state=42):
    target = "Machine failure"
    X = df.drop(columns=[target])
    y = df[target]

    failure_rate = y.mean() * 100
    info(f"Class balance — failures: {y.sum()} ({failure_rate:.2f}%), normal: {(y==0).sum()}")

    if failure_rate < 5:
        warn("Dataset is heavily imbalanced (<5% failures). Consider --oversample flag.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    success(f"Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
    return X_train_s, X_test_s, y_train, y_test, scaler, list(X.columns)

# ─────────────────────────────────────────────
# Step 5 — Train
# ─────────────────────────────────────────────

def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )
    info("Training Random Forest (n_estimators=100, bootstrap=True)...")
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    success(f"Random Forest trained in {elapsed:.2f}s")
    return model

# ─────────────────────────────────────────────
# Step 6 — Evaluate
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
# Step 7 — Feature importance
# ─────────────────────────────────────────────

def show_feature_importance(model, feature_names, top_n=10):
    if not hasattr(model, "feature_importances_"):
        info("Model does not expose feature importances.")
        return

    importances = model.feature_importances_
    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    print(f"\n  {'Feature':<35} {'Importance':>10}")
    print(f"  {'─'*47}")
    for feat, imp in ranked[:top_n]:
        bar = "█" * int(imp * 40)
        print(f"  {feat:<35} {imp:>8.4f}  {bar}")

# ─────────────────────────────────────────────
# Step 8 — Save
# ─────────────────────────────────────────────

def save_artifacts(model, scaler, feature_names, output_path):
    bundle = {
        "model":         model,
        "scaler":        scaler,
        "feature_names": feature_names,
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
        description="Train a Random Forest predictive maintenance classifier on the AI4I 2020 milling machine dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_maintenance.py --data ai4i2020.csv
  python train_maintenance.py --data ai4i2020.csv --interval 10 --output model.pkl
  python train_maintenance.py --data ai4i2020.csv --test-size 0.3 --no-save
        """
    )
    parser.add_argument("--data",      required=True,          help="Path to ai4i2020.csv")
    parser.add_argument("--interval",  type=int, default=5,    help="Minutes between sensor readings (default: 5)")
    parser.add_argument("--start",     default="2024-01-01",   help="Simulation start date (default: 2024-01-01)")
    parser.add_argument("--test-size", type=float, default=0.2,help="Test split fraction (default: 0.2)")
    parser.add_argument("--top-n",     type=int, default=10,   help="Top N features to display (default: 10)")
    parser.add_argument("--output",    default="model.pkl",    help="Output path for saved model (default: model.pkl)")
    parser.add_argument("--no-save",   action="store_true",    help="Skip saving the model")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{BOLD}Predictive Maintenance — Milling Machine{RESET}")
    print(f"{DIM}AI4I 2020 dataset | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")

    total_steps = 6

    # 1. Load
    step(1, total_steps, "Loading dataset")
    df = load_data(args.data)

    # 2. Timestamps
    step(2, total_steps, "Simulating timestamps")
    df = add_timestamps(df, interval_minutes=args.interval, start=args.start)

    # 3. Features
    step(3, total_steps, "Engineering features")
    df = engineer_features(df)

    # 4. Split & scale
    step(4, total_steps, "Splitting and scaling")
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_splits(
        df, test_size=args.test_size
    )

    # 5. Train
    step(5, total_steps, "Training model: Random Forest")
    model = train_model(X_train, y_train)

    # 6. Evaluate
    step(6, total_steps, "Evaluating on test set")
    evaluate(model, X_test, y_test)

    # Feature importance
    header("Feature Importance")
    show_feature_importance(model, feature_names, top_n=args.top_n)

    # Save
    if not args.no_save:
        header("Saving Model")
        save_artifacts(model, scaler, feature_names, args.output)

    print(f"\n{GREEN}{BOLD}Done.{RESET}\n")


if __name__ == "__main__":
    main()