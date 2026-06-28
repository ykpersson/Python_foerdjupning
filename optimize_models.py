import os
import joblib
import optuna
import pandas as pd

from model_support import load_timesplit
from rmf_model import RFModel
from hgb_model import HGBModel
from cat_boost_model import CatBoostModel
from models_setting import (
    DATASET_PATH,
    MODELS_OPTIMIZED_DIR,
    GROUPS
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUPS_PATH = os.path.join(BASE_DIR, "eda", "groups.pkl")


# -------------------------------------------------------
# Param spaces per group (G1–G4) och modell
# -------------------------------------------------------

def rf_param_space_g1(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, 800),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 6),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
        "max_features": trial.suggest_float("max_features", 0.4, 0.9),
        "bootstrap": True,
    }


def hgb_param_space_g1(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.08),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 16, 48),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 60),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 2.0),
        "max_bins": trial.suggest_int("max_bins", 128, 255),
    }

def cat_param_space_g1(trial):
    return {
        "depth": trial.suggest_int("depth", 3, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.08, 0.20),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "iterations": trial.suggest_int("iterations", 200, 600),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 20),
        "bootstrap_type": "Bayesian",
        "subsample": 1.0,
        "random_strength": trial.suggest_float("random_strength", 1.0, 3.0),
        "border_count": trial.suggest_int("border_count", 32, 128),
        "grow_policy": "SymmetricTree",
    }


def rf_param_space_g2(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 400, 900),
        "max_depth": trial.suggest_int("max_depth", 5, 12),
        "min_samples_split": trial.suggest_int("min_samples_split", 4, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 6),
        "max_features": trial.suggest_float("max_features", 0.3, 0.8),
        "bootstrap": True,
    }


def hgb_param_space_g2(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.06),
        "max_depth": trial.suggest_int("max_depth", 4, 7),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 24, 64),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 30, 80),
        "l2_regularization": trial.suggest_float("l2_regularization", 1.0, 5.0),
        "max_bins": trial.suggest_int("max_bins", 128, 255),
    }


def cat_param_space_g2(trial):
    return {
        "depth": trial.suggest_int("depth", 5, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.06),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 5.0, 12.0),
        "iterations": trial.suggest_int("iterations", 800, 1500),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 6),
        "bootstrap_type": "Bayesian",
        "subsample": 1.0,
        "random_strength": trial.suggest_float("random_strength", 0.4, 0.8),
        "border_count": trial.suggest_int("border_count", 128, 254),
        "grow_policy": "SymmetricTree",
    }


def rf_param_space_g3(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 500, 1000),
        "max_depth": trial.suggest_int("max_depth", 6, 14),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
        "max_features": trial.suggest_float("max_features", 0.4, 1.0),
        "bootstrap": True,
    }


def hgb_param_space_g3(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.05),
        "max_depth": trial.suggest_int("max_depth", 5, 8),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 32, 96),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 60),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.5, 3.0),
        "max_bins": trial.suggest_int("max_bins", 128, 255),
    }


def cat_param_space_g3(trial):
    return {
        "depth": trial.suggest_int("depth", 6, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.025, 0.05),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 3.0, 8.0),
        "iterations": trial.suggest_int("iterations", 1200, 2000),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 4),
        "bootstrap_type": "Bayesian",
        "subsample": 1.0,
        "random_strength": trial.suggest_float("random_strength", 0.5, 1.0),
        "border_count": trial.suggest_int("border_count", 128, 254),
        "grow_policy": "SymmetricTree",
    }


def rf_param_space_g4(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 400, 900),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 6),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
        "max_features": trial.suggest_float("max_features", 0.4, 0.9),
        "bootstrap": True,
    }


def hgb_param_space_g4(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.07),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 24, 64),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 60),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.5, 3.0),
        "max_bins": trial.suggest_int("max_bins", 128, 255),
    }


def cat_param_space_g4(trial):
    return {
        "depth": trial.suggest_int("depth", 4, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.06),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 6.0, 14.0),
        "iterations": trial.suggest_int("iterations", 800, 1500),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 6),
        "bootstrap_type": "Bayesian",
        "subsample": 1.0,
        "random_strength": trial.suggest_float("random_strength", 0.4, 0.9),
        "border_count": trial.suggest_int("border_count", 128, 254),
        "grow_policy": "SymmetricTree",
    }


# -------------------------------------------------------
# Param space mapping: modellklass + grupp → funktion
# -------------------------------------------------------

PARAM_SPACES = {
    "RFModel": {
        "G1": rf_param_space_g1,
        "G2": rf_param_space_g2,
        "G3": rf_param_space_g3,
        "G4": rf_param_space_g4,
    },
    "HGBModel": {
        "G1": hgb_param_space_g1,
        "G2": hgb_param_space_g2,
        "G3": hgb_param_space_g3,
        "G4": hgb_param_space_g4,
    },
    "CatBoostModel": {
        "G1": cat_param_space_g1,
        "G2": cat_param_space_g2,
        "G3": cat_param_space_g3,
        "G4": cat_param_space_g4,
    },
}


def get_param_space(model_class, group_name, trial):
    model_name = model_class.__name__
    if model_name not in PARAM_SPACES:
        raise ValueError(f"No param space defined for model {model_name}")
    if group_name not in PARAM_SPACES[model_name]:
        raise ValueError(f"No param space defined for group {group_name} and model {model_name}")
    return PARAM_SPACES[model_name][group_name](trial)


# -------------------------------------------------------
# Objective
# -------------------------------------------------------

def objective(trial, model_class, group_name, X_train, y_train, X_val, y_val):

    # Wrapper-instans (RFModel, HGBModel, CatBoostModel)
    model_wrapper = model_class(ticker=f"{model_class.__name__}_opt")

    y_train = y_train.astype(int)
    y_val = y_val.astype(int)

    # Hämta rätt param-space baserat på modell + grupp
    params = get_param_space(model_class, group_name, trial)

    # Skapa och träna modellen
    model_wrapper.model = model_wrapper.create_model(**params)
    model_wrapper.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    preds = model_wrapper.predict(X_val)

    if hasattr(preds, "ndim") and preds.ndim > 1:
        preds = preds.argmax(axis=1)

    score = (preds == y_val).mean()

    return score


# -------------------------------------------------------
# Convergence stopper (stopp-funktion)
# -------------------------------------------------------

class ConvergenceStopper:
    def __init__(self, threshold=0.002, patience=5):
        self.threshold = threshold
        self.patience = patience
        self.last_best = None
        self.bad_count = 0

    def __call__(self, study, trial):
        current = trial.value
        best = study.best_value

        if self.last_best is None:
            self.last_best = best
            return

        improvement = best - self.last_best

        if improvement < self.threshold:
            self.bad_count += 1
        else:
            self.bad_count = 0
            self.last_best = best

        if self.bad_count >= self.patience:
            print(f"\n[OPTUNA] {self.patience} dåliga trials i rad → bolt out.")
            study.stop()

# -------------------------------------------------------
# Optimize one group
# -------------------------------------------------------

def optimize_group(group_name, model_class, dataset_path, save_dir):

    print(f"\n==============================")
    print(f"   OPTIMIZING {model_class.__name__} ON {group_name}")
    print("==============================")

    X_train, X_val, X_test, y_train, y_val, y_test = load_timesplit(group_name)

    n_trials = 50 if group_name in ["G3", "G4"] else 30
    stopper = ConvergenceStopper(threshold=0.002, patience=5)

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=0,
            interval_steps=1
        )
    )

    study.optimize(
        lambda trial: objective(trial, model_class, group_name, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        callbacks=[stopper],
        show_progress_bar=True
    )

    print(f"Best score: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{group_name}_{model_class.__name__}_best_params.pkl")
    joblib.dump(study.best_params, save_path)

    return study.best_params, study.best_value


# -------------------------------------------------------
# Optimize all groups
# -------------------------------------------------------

def optimize_all_groups(model_classes):

    results = []

    for group_name in GROUPS:
        for model_class in model_classes:

            best_params, best_score = optimize_group(
                group_name,
                model_class,
                DATASET_PATH,
                MODELS_OPTIMIZED_DIR
            )

            results.append({
                "group": group_name,
                "model": model_class.__name__,
                "best_score": best_score,
                "best_params": best_params
            })

    df_results = pd.DataFrame(results)
    summary_path = os.path.join(MODELS_OPTIMIZED_DIR, "optimization_summary.pkl")
    joblib.dump(df_results, summary_path)

    print("\n=== OPTIMIZATION SUMMARY ===")
    print(df_results)

    return df_results
