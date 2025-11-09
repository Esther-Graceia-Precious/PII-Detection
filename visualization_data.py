"""
visualization_data.py
======================
Generates and saves visualizations from PII model evaluation results.

Creates:
- Grouped bar chart of F1 scores
- Heatmap for F1 comparisons
- Radar chart for model strengths
- Exports results to CSV

Author: Esther Graceia
Date: 2025-11-09
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from loguru import logger


# ========== 1️⃣ MAIN VISUALIZATION FUNCTION ==========
def visualize_results(summary, output_dir="results/plots", show_plots=False):
    """
    Generate visualizations and CSV summaries from evaluation metrics.

    Args:
        summary (dict): Dictionary containing model metrics (from evaluate_models_on_dataset()).
        output_dir (str): Directory to save visualization files.
        show_plots (bool): Whether to display plots live (useful in VSCode, not API).
    """
    # ---------- Convert Summary to DataFrame ----------
    data = []
    for model, entities in summary.items():
        for entity, metrics in entities.items():
            data.append({
                "Model": model,
                "Entity": entity,
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1 Score": metrics["f1"],
                "Accuracy": metrics["accuracy"]
            })

    df = pd.DataFrame(data)
    os.makedirs(output_dir, exist_ok=True)

    # ---------- Save CSV ----------
    csv_path = os.path.join(output_dir, "pii_model_summary.csv")
    df.to_csv(csv_path, index=False)
    logger.success(f"📁 Summary metrics saved to {csv_path}")

    # ---------- Visualization 1: Grouped Bar Chart ----------
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Entity", y="F1 Score", hue="Model", data=df)
    plt.title("Model-wise F1 Score Comparison Across Entities", fontsize=14)
    plt.ylim(0, 1.1)
    plt.legend(title="Model")
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    bar_path = os.path.join(output_dir, "f1_grouped_bar.png")
    plt.savefig(bar_path)
    if show_plots: plt.show()
    plt.close()
    logger.success(f"📊 Grouped Bar Chart saved to {bar_path}")

    # ---------- Visualization 2: Heatmap ----------
    plt.figure(figsize=(8, 5))
    heatmap_df = df.pivot_table(values="F1 Score", index="Model", columns="Entity")
    sns.heatmap(heatmap_df, annot=True, cmap="YlGnBu", fmt=".2f", cbar=True)
    plt.title("F1 Score Heatmap Across Models and Entities", fontsize=14)
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "f1_heatmap.png")
    plt.savefig(heatmap_path)
    if show_plots: plt.show()
    plt.close()
    logger.success(f"🔥 Heatmap saved to {heatmap_path}")

    # ---------- Visualization 3: Radar Chart ----------
    labels = list(df["Entity"].unique())
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(8, 8))
    for model in df["Model"].unique():
        subset = df[df["Model"] == model].sort_values(by="Entity")
        values = subset["F1 Score"].tolist()
        values += values[:1]
        plt.polar(angles, values, marker="o", label=model)

    plt.xticks(angles[:-1], labels)
    plt.title("Model-wise F1 Radar Chart", size=15)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    radar_path = os.path.join(output_dir, "f1_radar_chart.png")
    plt.savefig(radar_path)
    if show_plots: plt.show()
    plt.close()
    logger.success(f"🕸 Radar Chart saved to {radar_path}")

    logger.info("✅ All visualizations generated successfully!")

    return {
        "csv_path": csv_path,
        "bar_chart": bar_path,
        "heatmap": heatmap_path,
        "radar_chart": radar_path
    }


# ========== 2️⃣ OPTIONAL: LOAD RESULTS FROM JSON ==========
def visualize_from_json(json_path, output_dir="results/plots", show_plots=False):
    """
    Load results from a saved JSON file (like Postman output)
    and generate visualizations without re-running evaluation.

    Args:
        json_path (str): Path to the saved JSON results file.
        output_dir (str): Where to save visualizations.
        show_plots (bool): Whether to display plots interactively.
    """
    if not os.path.exists(json_path):
        logger.error(f"❌ JSON file not found: {json_path}")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    summary = data.get("results") or data
    logger.info(f"📖 Loaded results from {json_path}")

    return visualize_results(summary, output_dir=output_dir, show_plots=show_plots)


# ========== 3️⃣ MAIN EXECUTION (Standalone Run) ==========
if __name__ == "__main__":
    sample_json_path = r"C:\Users\A Esther Graceia\Desktop\main_pii_research\results\postman_results.json"
    paths = visualize_from_json(sample_json_path, show_plots=False)

    # 🔍 Confirm the generated files
    if paths:
        print("\n✅ Visualization complete! Files saved:")
        for name, path in paths.items():
            print(f" - {name}: {os.path.abspath(path)}")

