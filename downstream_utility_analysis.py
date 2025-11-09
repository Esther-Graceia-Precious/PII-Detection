"""
downstream_utility_analysis.py (Full Dataset Version)
====================================================
Evaluates data utility after PII masking using the entire dataset.
Compares model performance on original vs. masked data via clustering-based pseudo-labels.

Metric:
U = Accuracy_masked / Accuracy_original

Author: Esther Graceia Precious A
Date: 2025-11-09
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.cluster import MiniBatchKMeans
from loguru import logger
import os
import warnings

warnings.filterwarnings("ignore")

# ==================== 1️⃣ Load Datasets ====================
original_path = r"C:\Users\A Esther Graceia\Desktop\main_pii_research\data\pii_dataset.csv"
masked_path   = r"C:\Users\A Esther Graceia\Desktop\main_pii_research\results\pii_masked.csv"

if not os.path.exists(original_path) or not os.path.exists(masked_path):
    raise FileNotFoundError("❌ One of the datasets (original/masked) was not found.")

df_orig = pd.read_csv(original_path)
df_masked = pd.read_csv(masked_path)

# Clean and align datasets
df_orig = df_orig.dropna(subset=["text"]).reset_index(drop=True)
df_masked = df_masked.dropna(subset=["masked_text"]).reset_index(drop=True)
df_masked = df_masked.head(len(df_orig))

logger.info(f"✅ Loaded datasets: {len(df_orig)} total samples for analysis.")

# ==================== 2️⃣ Generate Topic Labels via Clustering ====================
logger.info("🧠 Generating synthetic topic labels (unsupervised)...")

vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
X = vectorizer.fit_transform(df_orig["text"])

# Use MiniBatchKMeans (faster & memory-efficient for large data)
n_clusters = 6  # you can adjust if dataset is very large
km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
df_orig["topic_label"] = km.fit_predict(X)

logger.success(f"🧩 Generated {n_clusters} topic clusters for downstream classification task.")

# ==================== 3️⃣ Train-Test Split ====================
X_train_orig, X_test_orig, y_train, y_test = train_test_split(
    df_orig["text"], df_orig["topic_label"], test_size=0.2, random_state=42
)
X_train_masked, X_test_masked, _, _ = train_test_split(
    df_masked["masked_text"], df_orig["topic_label"], test_size=0.2, random_state=42
)

# ==================== 4️⃣ Define Model Pipeline ====================
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=3000, stop_words='english')),
    ('clf', LogisticRegression(max_iter=500, solver='lbfgs'))
])

# ==================== 5️⃣ Train on Original ====================
logger.info("⚙️ Training classifier on original text data...")
model.fit(X_train_orig, y_train)
y_pred_orig = model.predict(X_test_orig)
acc_orig = accuracy_score(y_test, y_pred_orig)
logger.success(f"🎯 Original Data Accuracy: {acc_orig:.4f}")

# ==================== 6️⃣ Train on Masked ====================
logger.info("⚙️ Training classifier on masked text data...")
model.fit(X_train_masked, y_train)
y_pred_masked = model.predict(X_test_masked)
acc_masked = accuracy_score(y_test, y_pred_masked)
logger.success(f"🎯 Masked Data Accuracy: {acc_masked:.4f}")

# ==================== 7️⃣ Compute Utility Preservation ====================
U = acc_masked / acc_orig if acc_orig > 0 else 0
logger.info(f"📊 Utility Preservation Score (U): {U:.3f}")

# ==================== 8️⃣ Save Results ====================
output_dir = "results/downstream"
os.makedirs(output_dir, exist_ok=True)

result_path = os.path.join(output_dir, "utility_evaluation_full.csv")
pd.DataFrame({
    "Accuracy_Original": [acc_orig],
    "Accuracy_Masked": [acc_masked],
    "Utility_Score": [U],
    "Samples_Used": [len(df_orig)],
    "Clusters": [n_clusters]
}).to_csv(result_path, index=False)

logger.success(f"💾 Results saved to {result_path}")
print(f"\n✅ Full-Dataset Downstream Utility Evaluation Completed!")
print(f"Original Accuracy: {acc_orig:.3f}")
print(f"Masked Accuracy: {acc_masked:.3f}")
print(f"Utility Preservation Score: {U:.3f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
plt.bar(['Original', 'Masked'], [acc_orig, acc_masked], color=['#4C72B0', '#55A868'])
plt.title("Downstream Utility Comparison")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.text(0, acc_orig+0.01, f"{acc_orig:.3f}", ha='center')
plt.text(1, acc_masked+0.01, f"{acc_masked:.3f}", ha='center')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Downstream_utility_comparison_full.png"))
plt.show()

