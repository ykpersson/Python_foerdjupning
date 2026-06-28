import os
import joblib
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import models_setting


# =========================================================
# 0. Paths
# =========================================================
MODEL_DIR = models_setting.MODELS_OPTIMIZED_DIR
SPLIT_DIR = os.path.join(models_setting.BASE, "splits")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)


# =========================================================
# 1. Timesplit save/load
# =========================================================
def save_timesplit(group_name, X_train, X_val, X_test, y_train, y_val, y_test, save_dir=SPLIT_DIR):
    os.makedirs(save_dir, exist_ok=True)

    joblib.dump(X_train, os.path.join(save_dir, f"{group_name}_X_train.pkl"))
    joblib.dump(X_val,   os.path.join(save_dir, f"{group_name}_X_val.pkl"))
    joblib.dump(X_test,  os.path.join(save_dir, f"{group_name}_X_test.pkl"))

    joblib.dump(y_train, os.path.join(save_dir, f"{group_name}_y_train.pkl"))
    joblib.dump(y_val,   os.path.join(save_dir, f"{group_name}_y_val.pkl"))
    joblib.dump(y_test,  os.path.join(save_dir, f"{group_name}_y_test.pkl"))

    print(f"[SPLIT] Timesplit sparad för {group_name}")


def load_timesplit(group_name, split_dir=SPLIT_DIR):
    X_train = joblib.load(os.path.join(split_dir, f"{group_name}_X_train.pkl"))
    X_val   = joblib.load(os.path.join(split_dir, f"{group_name}_X_val.pkl"))
    X_test  = joblib.load(os.path.join(split_dir, f"{group_name}_X_test.pkl"))

    y_train = joblib.load(os.path.join(split_dir, f"{group_name}_y_train.pkl"))
    y_val   = joblib.load(os.path.join(split_dir, f"{group_name}_y_val.pkl"))
    y_test  = joblib.load(os.path.join(split_dir, f"{group_name}_y_test.pkl"))

    return X_train, X_val, X_test, y_train, y_val, y_test


# =========================================================
# 2. Model save/load
# =========================================================
def get_model_path(ticker):
    return os.path.join(MODEL_DIR, f"{ticker}_optimized.pkl")


def save_model(model, ticker):
    path = get_model_path(ticker)
    joblib.dump(model, path)
    print(f"[MODEL] Saved optimized model → {path}")


def load_model(ticker):
    path = get_model_path(ticker)
    if not os.path.exists(path):
        raise FileNotFoundError(f"[MODEL] No saved model found for {ticker}")
    print(f"[MODEL] Loaded optimized model ← {path}")
    return joblib.load(path)


def model_exists(ticker):
    return os.path.exists(get_model_path(ticker))


# =========================================================
# 3. Ticker helpers
# =========================================================
def safe_ticker(ticker: str) -> str:
    return ticker.replace(".", "_").replace("-", "_")


def to_base_ticker(ticker: str) -> str:
    return ticker.replace(".ST", "")


# =========================================================
# 4. Load dataset for ONE ticker
# =========================================================
def load_dataset(ticker):
    base = safe_ticker(ticker)
    path = os.path.join(models_setting.BASE, "data", f"dataset_{base}.pkl")

    if not os.path.exists(path):
        print(f"[DATA] Dataset not found: {path}")
        return None

    return joblib.load(path)


# =========================================================
# 5. Timeseries split (corrected)
# =========================================================
def timeseries_split(df, train_size=0.70, val_size=0.15):
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    leakage_cols = [
        "Date", "Ticker", "Band",
        "target", "y_target", "y",
        "Return_h", "Index_Return", "rel_move"
    ]

    leakage_cols += [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
    leakage_cols += [c for c in df.columns if c.startswith("ticker_")]
    leakage_cols = [c for c in leakage_cols if c in df.columns]

    X = df.drop(columns=leakage_cols)
    y = df["y"]

    X_train_list, X_val_list, X_test_list = [], [], []
    y_train_list, y_val_list, y_test_list = [], [], []

    for ticker in df["Ticker"].unique():
        df_t = df[df["Ticker"] == ticker]

        n = len(df_t)
        train_end = int(n * train_size)
        val_end   = int(n * (train_size + val_size))

        X_t = X.loc[df_t.index]
        y_t = y.loc[df_t.index]

        X_train_list.append(X_t.iloc[:train_end])
        y_train_list.append(y_t.iloc[:train_end])

        X_val_list.append(X_t.iloc[train_end:val_end])
        y_val_list.append(y_t.iloc[train_end:val_end])

        X_test_list.append(X_t.iloc[val_end:])
        y_test_list.append(y_t.iloc[val_end:])

    X_train = pd.concat(X_train_list)
    X_val   = pd.concat(X_val_list)
    X_test  = pd.concat(X_test_list)

    y_train = pd.concat(y_train_list)
    y_val   = pd.concat(y_val_list)
    y_test  = pd.concat(y_test_list)

    print("\n[LEAKAGE] Droppade kolumner:")
    for c in leakage_cols:
        print("  -", c)

    print(f"\n[FEATURES] Features kvar i X: {X_train.shape[1]}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# =========================================================
# 6. Load full panel
# =========================================================
def load_panel():
    path = models_setting.DATASET_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"[DATA] Hittar inte dataset: {path}")

    panel = joblib.load(path)
    panel = panel.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    drop_cols = ["Date", "Ticker", "band", "y_target"]
    drop_cols = [c for c in drop_cols if c in panel.columns]

    X = panel.drop(columns=drop_cols)
    y = panel["y_target"]

    print("\n📊 [PANEL] Paneldata laddad:")
    print(f"   Rader: {len(panel)}")
    print(f"   Tickers: {panel['Ticker'].nunique()}")
    print(f"   Features: {X.shape[1]}")

    return panel, X, y


# =========================================================
# 7. Confusion matrix
# =========================================================
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(title)
    plt.tight_layout()
    filename = f"confusion_matrix_{title.replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()
    print(f"[PLOT] Confusion matrix sparad → {filename}")


# =========================================================
# 8. Feature correlation
# =========================================================
def plot_feature_correlation(X, y, top_n=20):
    correlations = X.corrwith(y).abs().sort_values(ascending=False)
    top_features = correlations.head(top_n)

    plt.figure(figsize=(10, 6))
    top_features.plot(kind='bar')
    plt.title(f"Top {top_n} Feature Correlations with Target")
    plt.xlabel("Features")
    plt.ylabel("Absolute Correlation")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("feature_correlations.png")
    plt.close()

    print("[PLOT] Feature correlations sparade → feature_correlations.png")
    return correlations

