# plot_groups.py

import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_confusion_matrix(cm, title="Confusion Matrix", save_path=None):
    labels = ["Down (-1)", "Neutral (0)", "Up (1)"]

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"[SAVED] {save_path}")

    plt.close()


def plot_all_groups(evaluation_results, save_dir="plots"):
    """
    evaluation_results = output från evaluate_all_groups()
    Ritar confusion matrices för alla grupper och modeller.
    """

    for group_name, models in evaluation_results.items():
        print(f"\n=== PLOTS FOR {group_name} ===")

        for model_name, data in models.items():
            cm = data["confusion_matrix"]

            save_path = f"{save_dir}/{group_name}_{model_name}_cm.png"

            plot_confusion_matrix(
                cm,
                title=f"{group_name} – {model_name}",
                save_path=save_path
            )



