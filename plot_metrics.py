# Version 5 source snapshot
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_metrics():
    # Result directories are stored next to this script.
    result_root = os.path.dirname(os.path.abspath(__file__))
    models = ["geometry", "flow_geometry", "clinical_only", "clinical_age_sex", "ensemble", "gnn"]

    # Store aggregated data
    all_data = []

    for model in models:
        res_dir = f"results_{model}"
        full_dir = os.path.join(result_root, res_dir)
        if not os.path.exists(full_dir):
            continue

        # Get fold_averages or calculate it if it doesn't exist
        fold_files = glob.glob(os.path.join(full_dir, "fold_*_metrics.csv"))

        for fold_csv in fold_files:
            try:
                df = pd.read_csv(fold_csv)
                df["model"] = model
                fold_num = os.path.basename(fold_csv).split("_")[1]
                df["fold"] = int(fold_num)
                all_data.append(df)
            except Exception as e:
                print(f"Error reading {fold_csv}: {e}")

    if not all_data:
        print("No metrics matching 'fold_*_metrics.csv' found.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    # Plot metrics
    sns.set_theme(style="whitegrid")

    # Create a unified figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    plot_targets = [
        ("val_auc", "Validation ROC-AUC by Epoch"),
        ("val_pr_auc", "Validation PR-AUC by Epoch"),
        ("val_acc", "Validation Accuracy by Epoch"),
        ("val_loss", "Validation Loss by Epoch"),
    ]

    for idx, (metric, title) in enumerate(plot_targets):
        if metric in combined_df.columns:
            sns.lineplot(
                data=combined_df, x="epoch", y=metric, hue="model", ax=axes[idx], errorbar="sd"
            )
            axes[idx].set_title(title, fontsize=14)
            axes[idx].set_ylim(bottom=0.3 if "auc" in metric or "acc" in metric else None)

    plt.tight_layout()
    plot_path = os.path.join(result_root, "model_comparison_plots.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved metric graphs to {plot_path}")


if __name__ == "__main__":
    plot_metrics()
