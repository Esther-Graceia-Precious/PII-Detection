"""
Enhanced Preprocessing Pipeline for PII Dataset
-----------------------------------------------
Cleans text + extracts entity-level fields for evaluation.
Creates final structure:
text, name, email, phone, address, job, hobby, label
"""

import pandas as pd
import re
from loguru import logger


# ====== STEP 1: CLEAN TEXT ======
def clean_text(text: str) -> str:
    """Clean text by removing unwanted characters and extra spaces."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\r\n\t]', ' ', text)
    return text.strip()


# ====== STEP 2: REGEX HELPERS TO EXTRACT SPECIFIC PII ======
def extract_email(text):
    match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}", text)
    return match.group(0) if match else ""

def extract_phone(text):
    match = re.search(r"\+?\d{1,3}[\s-]?\(?\d{2,4}\)?[\s-]?\d{3,5}[\s-]?\d{3,5}", text)
    return match.group(0) if match else ""

def extract_name(text):
    # basic name detection — assumes "My name is <name>" or "<name>,"
    match = re.search(r"\b(?:name\sis|I'm|I am)\s([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)", text)
    if match:
        return match.group(1)
    # fallback to capitalized full name at sentence start
    match2 = re.search(r"\b([A-Z][a-z]+\s[A-Z][a-z]+)\b", text)
    return match2.group(1) if match2 else ""

def extract_address(text):
    match = re.search(r"\d+\s+[A-Za-z]+\s+(Street|Avenue|Road|Lane|Drive|Boulevard|Terrace|Place|Way|Court|Square|Circle|Parkway|Highway)", text)
    return match.group(0) if match else ""

def extract_job(text):
    # looks for simple job role patterns
    match = re.search(r"\b(am|as|work(ed)?\s+as|I[' ]?m\s+a)\s+([A-Za-z ]+)", text)
    if match:
        role = match.group(3).strip().split(" ")[0:3]
        return " ".join(role)
    return ""

def extract_hobby(text):
    match = re.search(r"\b(enjoy|love|like)\s([a-z]+ing)", text)
    return match.group(2) if match else ""


# ====== STEP 3: DETECT ANY PII FOR LABEL ======
def detect_pii(text: str) -> int:
    """Binary flag — does the text contain any PII at all?"""
    patterns = [
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}",  # email
        r"\+?\d{1,3}[\s-]?\(?\d{2,4}\)?[\s-]?\d{3,5}[\s-]?\d{3,5}",  # phone
        r"\d+\s+[A-Za-z]+\s+(Street|Road|Avenue|Lane|Drive)",  # address
    ]
    return 1 if any(re.search(p, text) for p in patterns) else 0


# ====== STEP 4: MAIN PREPROCESSING FUNCTION ======
def preprocess_dataset(
    raw_path=r"C:\Users\A Esther Graceia\Desktop\main_pii_research\data\raw_pii_dataset.csv",
    output_path=r"C:\Users\A Esther Graceia\Desktop\main_pii_research\data\pii_dataset.csv",
):
    logger.info(f"📂 Loading dataset from {raw_path}")
    df = pd.read_csv(raw_path)
    logger.info(f"📊 Raw dataset shape: {df.shape}")

    # Find text column
    possible_text_cols = [c for c in df.columns if "text" in c.lower()]
    if not possible_text_cols:
        raise ValueError("❌ No column named like 'text' found in dataset.")
    text_col = possible_text_cols[0]

    # Keep and clean text
    df = df[[text_col]].rename(columns={text_col: "text"})
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 20]

    # Extract entity-level PII fields
    logger.info("🔍 Extracting entity-level PII...")
    df["name"] = df["text"].apply(extract_name)
    df["email"] = df["text"].apply(extract_email)
    df["phone"] = df["text"].apply(extract_phone)
    df["address"] = df["text"].apply(extract_address)
    df["job"] = df["text"].apply(extract_job)
    df["hobby"] = df["text"].apply(extract_hobby)

    # Label: PII present (1) or not (0)
    df["label"] = df["text"].apply(detect_pii)

    # Drop duplicates and reset index
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    logger.info(f"✅ Cleaned dataset shape: {df.shape}")

    # Save
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.success(f"💾 Saved processed dataset to {output_path}")
    logger.info(f"🧩 Sample row:\n{df.head(1).to_string(index=False)}")

    return df


# ====== ENTRY POINT ======
if __name__ == "__main__":
    preprocess_dataset()
