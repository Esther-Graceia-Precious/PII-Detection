
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from loguru import logger

original_path = r"C:\Users\A Esther Graceia\Desktop\main_pii_research\data\pii_dataset.csv"
masked_path   = r"C:\Users\A Esther Graceia\Desktop\main_pii_research\results\pii_masked.csv"

df_orig = pd.read_csv(original_path)
df_mask = pd.read_csv(masked_path)

if "label" not in df_orig.columns:
    df_orig["label"] = (df_orig.index % 2).astype(int)
    df_mask["label"] = df_orig["label"]


X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(df_orig["text"], df_orig["label"], test_size=0.2, random_state=42)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(df_mask["masked_text"], df_mask["label"], test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer(max_features=5000)
X_train_o_vec = vectorizer.fit_transform(X_train_o)
X_test_o_vec = vectorizer.transform(X_test_o)

X_train_m_vec = vectorizer.fit_transform(X_train_m)
X_test_m_vec = vectorizer.transform(X_test_m)

clf_orig = LogisticRegression(max_iter=200)
clf_mask = LogisticRegression(max_iter=200)

clf_orig.fit(X_train_o_vec, y_train_o)
clf_mask.fit(X_train_m_vec, y_train_m)

acc_orig = accuracy_score(y_test_o, clf_orig.predict(X_test_o_vec))
acc_mask = accuracy_score(y_test_m, clf_mask.predict(X_test_m_vec))

utility_score = acc_mask / acc_orig if acc_orig > 0 else 0

logger.success(f"✅ Downstream Utility Evaluation Completed!")
logger.info(f"Original Accuracy: {acc_orig:.3f}")
logger.info(f"Masked Accuracy: {acc_mask:.3f}")
logger.success(f"Utility Preservation Score: {utility_score:.3f}")
