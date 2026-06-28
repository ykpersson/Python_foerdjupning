import pandas as pd
from train_groups import train_all_groups
from optimize_models import optimize_all_groups
from test_groups import test_all_groups
from evaluate_groups import evaluate_group
from plot_groups import plot_all_groups
from retrain import retrain_group
from final_run import final_prediction_df
from rmf_model import RFModel
from hgb_model import HGBModel
from cat_boost_model import CatBoostModel
from models_setting import DATASET_PATH, MODELS_FINAL_DIR, MODELS_OPTIMIZED_DIR, GROUPS


def main():
    model_classes = [RFModel, HGBModel, CatBoostModel]

    print("\n=== STEG 1: TRAIN ===")
    train_all_groups()

    print("\n=== STEG 2: OPTIMIZE ===")
    optimize_all_groups(model_classes)
                        
    print("\n=== STEG 3: Retrain med best param ===")
    retrain_models = retrain_group(model_classes)

    print("\n=== STEG 4:  test")
    test_results = test_all_groups(retrain_models)

    print("\n=== STEG 5: evaluate")
    evaluation_results = evaluate_group(test_results)

    print("\n=== STEG 6: plot ===")
    plot_all_groups(evaluation_results)
    

if __name__ == "__main__":
    main()
