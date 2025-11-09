"""
downstream_utility_analysis.py
==============================
Evaluates data utility after PII masking by comparing model performance
on original vs. masked datasets.

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
from sklearn.cluster import KMeans
from loguru import logger
import os

# ==================== 1️⃣ Load Datasets ====================
original_path = r"C:\Users\A Esther Graceia\Desktop\main_pii_research\data\pii_dataset.csv"
masked_path = r"C:\Users\A Esther Graceia\Desktop\main_pii_research\results\pii_masked.csv"

if not os.path.exists(original_path) or not os.path.exists(masked_path):
    raise FileNotFoundError("❌ One of the datasets (original/masked) was not found.")

df_orig = pd.read_csv(original_path)
df_masked = pd.read_csv(masked_path)

# Ensure same length and valid text
df_orig = df_orig.dropna(subset=["text"]).reset_index(drop=True)
df_masked = df_masked.dropna(subset=["masked_text"]).reset_index(drop=True)
df_masked = df_masked.head(len(df_orig))

logger.info(f"✅ Loaded datasets: {len(df_orig)} samples.")

# ==================== 2️⃣ Create Synthetic Labels (Clustering) ====================
# We'll simulate 'topics' for classification
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(df_orig["text"])

n_clusters = 5
km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df_orig["topic_label"] = km.fit_predict(X)

logger.info("🧠 Synthetic topic labels generated for downstream task.")

# ==================== 3️⃣ Train-Test Split ====================
X_train_orig, X_test_orig, y_train, y_test = train_test_split(
    df_orig["text"], df_orig["topic_label"], test_size=0.2, random_state=42
)
X_train_masked, X_test_masked, _, _ = train_test_split(
    df_masked["masked_text"], df_orig["topic_label"], test_size=0.2, random_state=42
)

# ==================== 4️⃣ Define Model Pipeline ====================
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
    ('clf', LogisticRegression(max_iter=300))
])

# ==================== 5️⃣ Train and Evaluate ====================
logger.info("⚙️ Training model on original data...")
model.fit(X_train_orig, y_train)
y_pred_orig = model.predict(X_test_orig)
acc_orig = accuracy_score(y_test, y_pred_orig)
logger.success(f"🎯 Original Data Accuracy: {acc_orig:.4f}")

logger.info("⚙️ Training model on masked data...")
model.fit(X_train_masked, y_train)
y_pred_masked = model.predict(X_test_masked)
acc_masked = accuracy_score(y_test, y_pred_masked)
logger.success(f"🎯 Masked Data Accuracy: {acc_masked:.4f}")

# ==================== 6️⃣ Compute Utility Preservation Score ====================
U = acc_masked / acc_orig if acc_orig > 0 else 0
logger.info(f"📊 Utility Preservation Score (U): {U:.3f}")

# ==================== 7️⃣ Save Results ====================
output_dir = "results/downstream"
os.makedirs(output_dir, exist_ok=True)

result_path = os.path.join(output_dir, "utility_evaluation.csv")
pd.DataFrame({
    "Accuracy_Original": [acc_orig],
    "Accuracy_Masked": [acc_masked],
    "Utility_Score": [U]
}).to_csv(result_path, index=False)

logger.success(f"💾 Results saved to {result_path}")
print(f"\n✅ Downstream Utility Evaluation Completed!")
print(f"Original Accuracy: {acc_orig:.3f}")
print(f"Masked Accuracy: {acc_masked:.3f}")
print(f"Utility Preservation Score: {U:.3f}")
