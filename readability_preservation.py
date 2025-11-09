"""
readability_preservation_full.py
================================
Evaluates readability preservation after PII masking using the full dataset.
Computes the Flesch Reading Ease for each text and masked text, then calculates:

Preservation = masked_readability / original_readability

Average preservation score represents how much the masking process affects readability.

Author: Esther Graceia Precious A
Date: 2025-11-09
"""

import pandas as pd
from textstat import flesch_reading_ease
from loguru import logger
import numpy as np
import os
import matplotlib.pyplot as plt

# ========== 1️⃣ Load Files ==========
original_csv = r"C:\Users\A Esther Graceia\Desktop\main_pii_research\data\pii_dataset.csv"
masked_csv   = r"C:\Users\A Esther Graceia\Desktop\main_pii_research\results\pii_masked.csv"

if not os.path.exists(original_csv) or not os.path.exists(masked_csv):
    raise FileNotFoundError("❌ One of the CSV files (original/masked) is missing!")

df_orig = pd.read_csv(original_csv)
df_mask = pd.read_csv(masked_csv)

# ========== 2️⃣ Clean & Align Data ==========
df_orig = df_orig.dropna(subset=["text"]).reset_index(drop=True)
df_mask = df_mask.dropna(subset=["masked_text"]).reset_index(drop=True)
df_mask = df_mask.head(len(df_orig))

logger.info(f"✅ Loaded datasets with {len(df_orig)} samples for readability analysis.")

# ========== 3️⃣ Compute Readability Preservation ==========
preservation_scores = []
invalid_count = 0

for idx, (o_text, m_text) in enumerate(zip(df_orig["text"], df_mask["masked_text"]), 1):
    try:
        orig_score = flesch_reading_ease(o_text)
        masked_score = flesch_reading_ease(m_text)
        preservation = masked_score / orig_score if orig_score > 0 else 0
        preservation_scores.append(preservation)
    except Exception as e:
        invalid_count += 1
        continue

    if idx % 500 == 0:
        logger.info(f"📊 Processed {idx}/{len(df_orig)} samples...")

# ========== 4️⃣ Compute Final Results ==========
avg_preservation = np.mean(preservation_scores)
std_dev = np.std(preservation_scores)

logger.success(f"🧾 Readability Preservation Score (Full Dataset): {avg_preservation:.3f}")
logger.info(f"Standard Deviation: {std_dev:.3f}")
logger.info(f"Ignored {invalid_count} invalid samples.")

# ========== 5️⃣ Save Results ==========
os.makedirs("results/readability", exist_ok=True)
output_path = "results/readability/readability_preservation_full.csv"

pd.DataFrame({
    "Average_Preservation": [avg_preservation],
    "Std_Deviation": [std_dev],
    "Samples_Used": [len(df_orig) - invalid_count]
}).to_csv(output_path, index=False)

logger.success(f"💾 Readability preservation results saved to {output_path}")


clipped_scores = np.clip(preservation_scores, 0, 2)

plt.figure(figsize=(8, 5))
plt.hist(clipped_scores, bins=40, color="#4C72B0", edgecolor="black", alpha=0.8)
plt.title("Distribution of Readability Preservation Scores (Clipped to [0, 2])")
plt.xlabel("Preservation Ratio (Masked / Original)")
plt.ylabel("Frequency")

plt.axvline(avg_preservation, color='red', linestyle='--', label=f"Avg = {avg_preservation:.3f}")
plt.legend()
plt.tight_layout()
plt.savefig("results/readability/readability_distribution_clipped.png")
plt.show()

plt.figure(figsize=(6, 2))
plt.boxplot(clipped_scores, vert=False, patch_artist=True,
            boxprops=dict(facecolor="#A1CAF1", color="black"),
            medianprops=dict(color="red", linewidth=2))
plt.title("Readability Preservation Boxplot (Clipped 0–2)")
plt.xlabel("Preservation Ratio")
plt.tight_layout()
plt.savefig("results/readability/readability_boxplot.png")
plt.show()
