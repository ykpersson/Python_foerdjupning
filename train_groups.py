import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from model_support import timeseries_split, save_timesplit
from cat_boost_model import CatBoostModel
from hgb_model import HGBModel
from rmf_model import RFModel

from models_setting import DATASET_PATH, GROUPS


# Modellklasser som ska tränas
MODEL_CLASSES = [RFModel, HGBModel, CatBoostModel]


# =========================================================
# 1. Träna EN grupp med EN modellklass
# =========================================================
def train_group(group_name, model_class):

    print("\n======================================")
    print(f"   TRAINING {model_class.__name__} ON {group_name}")
    print("======================================")

    # 1. Ladda hela paneldatasetet
    df_all = joblib.load(DATASET_PATH)

    # 2. Filtrera på band (G1–G4)
    df = df_all[df_all["Band"] == group_name].copy()

    if df.empty:
        print(f"[WARN] Inga rader hittades för {group_name}. Hoppar över.")
        return model_class.__name__, None

    # 3. Sortera korrekt
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # 3. Sortera korrekt
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # 4. Splitta data – låt timeseries_split göra jobbet
    X_train, X_val, X_test, y_train, y_val, y_test = timeseries_split(df)

    # 5. Spara split
    save_timesplit(group_name, X_train, X_val, X_test, y_train, y_val, y_test)

    # 6. Initiera modell
    model = model_class(ticker=f"{group_name}_{model_class.__name__}")
    model.model = model.create_model()

    # 7. Träna
    model.fit(X_train, y_train, X_val, y_val)

    # 8. Validera
    preds = model.model.predict(X_val)
    score = accuracy_score(y_val, preds)

    print(f"[RESULT] Validation accuracy for {group_name} ({model_class.__name__}): {score:.3f}")

    return model_class.__name__, score


# =========================================================
# 2. Träna ALLA grupper med ALLA modeller
# =========================================================
def train_all_groups(model_classes=MODEL_CLASSES):

    results = []

    print("\n======================================")
    print("    STARTAR TRAINING FÖR ALLA GRUPPER")
    print("======================================")

    for group_name in GROUPS:
        for model_class in model_classes:

            model_name, score = train_group(group_name, model_class)

            if score is not None:
                results.append({
                    "group": group_name,
                    "model": model_name,
                    "score": score
                })

    df_results = pd.DataFrame(results)

    print("\n======================================")
    print("    TRAINING SUMMARY")
    print("======================================")
    print(df_results)

    return df_results

