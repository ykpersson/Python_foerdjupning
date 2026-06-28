import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import shap


def safe_predict(model, X):
    """Robust predict som funkar för RF, HGB och CatBoost."""
    try:
        return model.predict(X)
    except:
        try:
            proba = model.predict_proba(X)
            return np.argmax(proba, axis=1)
        except:
            raise ValueError("Modellen kan inte predicera med varken predict eller predict_proba.")


def get_feature_importance(model):
    """Returnerar feature importance för RF, HGB och CatBoost – även med wrappers."""
    
    # 1. Om modellen har en .model (wrapper)
    if hasattr(model, "model"):
        inner = model.model
        
        # Sklearn RF / HGB
        if hasattr(inner, "feature_importances_"):
            return inner.feature_importances_
        
        # CatBoost
        if hasattr(inner, "get_feature_importance"):
            return inner.get_feature_importance()
    
    # 2. Om modellen inte är wrapper
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    
    if hasattr(model, "get_feature_importance"):
        return model.get_feature_importance()
    
    return None


def evaluate_group(group_name, model_dir, X_test, y_test):

    print(f"\n=== EVALUATION {group_name} ===")

    # Label distribution
    print("\nLabel distribution (y_test):")
    print(pd.Series(y_test).value_counts())

    # Majority baseline
    majority_class = pd.Series(y_test).value_counts().idxmax()
    baseline_acc = (y_test == majority_class).mean()
    print(f"\nMajority class baseline accuracy: {baseline_acc:.4f}")

    # Utvärdera modeller
    for model_name in ["RFModel", "HGBModel", "CatBoostModel"]:

        model_path = os.path.join(model_dir, f"{group_name}_{model_name}_final.pkl")

        if not os.path.exists(model_path):
            print(f" Modell saknas: {model_path}")
            continue

        print(f"\n--- {model_name} ({group_name}) ---")
        model = joblib.load(model_path)

        preds = safe_predict(model, X_test)

        print("\nClassification Report:")
        print(classification_report(y_test, preds))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, preds))

        fi = get_feature_importance(model)
        if fi is not None:
            print("\nFeature Importance:")
            print(fi)

        
if __name__ == "__main__":

    groups = ["G1", "G2", "G3", "G4"]

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    model_dir = os.path.join(base_dir, "models_final")
    splits_dir = os.path.join(base_dir, "splits")

    print("Model dir:", model_dir)
    print("Splits dir:", splits_dir)

    for group_name in groups:

        print("\n====================================")
        print(f"   RUNNING EVALUATION FOR {group_name}")
        print("====================================")

        # 1. Ladda test-split
        X_test = joblib.load(os.path.join(splits_dir, f"{group_name}_X_test.pkl"))
        y_test = joblib.load(os.path.join(splits_dir, f"{group_name}_y_test.pkl"))

        # 2. Ladda X_train för att få rätt feature-ordning
        X_train = joblib.load(os.path.join(splits_dir, f"{group_name}_X_train.pkl"))

        # 3. Align features (DETTA ÄR DET VIKTIGA)
        X_test = X_test[X_train.columns]

        # 4. Kör utvärdering
        evaluate_group(group_name, model_dir, X_test, y_test)

