import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from rmf_model import RFModel
from hgb_model import HGBModel
from cat_boost_model import CatBoostModel
from ensamble_modell import EnsembleModel


# ============================
#   TEST ONE MODEL
# ============================
def test_model(model, X_test, y_test):
    preds = model.predict(X_test)
    preds = np.asarray(preds)

    # Hantera modeller som returnerar 2D-probabilities
    if preds.ndim > 1:
        preds = preds.argmax(axis=1)

    acc = accuracy_score(y_test, preds)
    bal_acc = balanced_accuracy_score(y_test, preds)

    return acc, bal_acc


# ============================
#   TEST ALL GROUPS
# ============================
def test_all_groups():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(BASE_DIR)

    splits_dir = os.path.join(PROJECT_DIR, "splits")
    models_dir = os.path.join(PROJECT_DIR, "models_final")

    model_classes = [RFModel, HGBModel, CatBoostModel]
    groups = ["G1", "G2", "G3", "G4"]

    results = []

    for group in groups:
        print(f"\n==============================")
        print(f"   TESTING GROUP {group}")
        print(f"==============================")

        # Load test split
       
        X_test  = joblib.load(os.path.join(splits_dir, f"{group}_X_test.pkl"))
        y_test  = joblib.load(os.path.join(splits_dir, f"{group}_y_test.pkl"))


        # --- Test RF, HGB, CatBoost ---
        for model_class in model_classes:
            model_name = model_class.__name__

            print(f"\n--- Testing {model_name} on {group} ---")

            model_path = os.path.join(models_dir, f"{group}_{model_name}_final.pkl")
            model = joblib.load(model_path)

            acc, bal_acc = test_model(model, X_test, y_test)

            print(f"Accuracy: {acc:.4f}")
            print(f"Balanced Accuracy: {bal_acc:.4f}")

            results.append({
                "group": group,
                "model": model_name,
                "accuracy": acc,
                "balanced_accuracy": bal_acc
            })

        # --- Test Ensemble ---
        print(f"\n--- Testing EnsembleModel on {group} ---")

        ensemble_path = os.path.join(models_dir, f"{group}_Ensemble_final.pkl")
        ensemble = joblib.load(ensemble_path)

        preds = ensemble.predict(X_test)
        acc = accuracy_score(y_test, preds)
        bal_acc = balanced_accuracy_score(y_test, preds)

        print(f"Accuracy: {acc:.4f}")
        print(f"Balanced Accuracy: {bal_acc:.4f}")

        results.append({
            "group": group,
            "model": "EnsembleModel",
            "accuracy": acc,
            "balanced_accuracy": bal_acc
        })

    return pd.DataFrame(results)


# ============================
#   MAIN ENTRYPOINT
# ============================
if __name__ == "__main__":
    df_results = test_all_groups()

    print("\n=== TEST RESULTS (SUMMARY DF) ===")
    print(df_results.to_string(index=False))
