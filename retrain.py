import os
import joblib
import pandas as pd
from model_support import load_timesplit
from models_setting import MODELS_OPTIMIZED_DIR, MODELS_FINAL_DIR
from ensamble_modell import EnsembleModel   

def retrain_group(model_classes, group_name):

    print(f"\n==============================")
    print(f"   RETRAINING MODELS ON {group_name}")
    print(f"==============================")

    # 1. Load timesplit
    X_train, X_val, X_test, y_train, y_val, y_test = load_timesplit(group_name)

    # 2. Combine train + val
    X_full = pd.concat([X_train, X_val], ignore_index=True)
    y_full = pd.concat([y_train, y_val], ignore_index=True)

    # Store final models for ensemble
    final_models = {}

    # 3. Retrain each model
    for model_class in model_classes:

        model_name = model_class.__name__
        print(f"\n--- Retraining {model_name} on {group_name} ---")

        # Load best params
        params_path = os.path.join(
            MODELS_OPTIMIZED_DIR,
            f"{group_name}_{model_name}_best_params.pkl"
        )

        if not os.path.exists(params_path):
            raise FileNotFoundError(f"Missing best params file: {params_path}")

        best_params = joblib.load(params_path)
        if "use_best_model" in best_params:
            best_params["use_best_model"] = False

        # Create model wrapper
        model_wrapper = model_class(ticker=f"{model_name}_{group_name}_final")
        model_wrapper.model = model_wrapper.create_model(**best_params)

        # Fit final model
        model_wrapper.fit(X_full, y_full)

        # Save final model
        save_path = os.path.join(
            MODELS_FINAL_DIR,
            f"{group_name}_{model_name}_final.pkl"
        )
        joblib.dump(model_wrapper, save_path)

        print(f"Saved {model_name} → {save_path}")

        # Add to ensemble dict
        final_models[model_name] = model_wrapper

    # 4. Build ensemble
    print(f"\n--- Building EnsembleModel for {group_name} ---")

    ensemble = EnsembleModel()
    ensemble.create_model(
        rf=final_models["RFModel"],
        hgb=final_models["HGBModel"],
        cat=final_models["CatBoostModel"]
    )

    # 5. Save ensemble
    ensemble_path = os.path.join(
        MODELS_FINAL_DIR,
        f"{group_name}_Ensemble_final.pkl"
    )
    joblib.dump(ensemble, ensemble_path)

    print(f"Saved Ensemble model → {ensemble_path}")


# ============================================================
#  LOCAL MAIN – kör ALLA grupper i en loop
# ============================================================

if __name__ == "__main__":
    from rmf_model import RFModel
    from hgb_model import HGBModel
    from cat_boost_model import CatBoostModel

    MODEL_CLASSES = [RFModel, HGBModel, CatBoostModel]
    GROUPS = ["G1", "G2", "G3", "G4"]   # ← här styr du ordningen

    print("\n====================================")
    print("   STARTING FULL RETRAIN PIPELINE")
    print("====================================")

    for group in GROUPS:
        retrain_group(MODEL_CLASSES, group)

    print("\n====================================")
    print("   RETRAIN PIPELINE COMPLETED")
    print("====================================")
