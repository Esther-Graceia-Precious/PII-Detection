"""
main_pii_research.py
====================
Entity-level PII Detection & Masking Evaluation.

Includes:
- Pre-trained model integration: Regex, spaCy, BERT, Presidio
- Model evaluation across entity types (name, email, phone, address, job, hobby)
- Sample masking demonstration
- Optional FastAPI backend for testing

Author: Esther Graceia
Date: 2025-11-08
"""

# ========== IMPORTS ==========
import os
import pandas as pd
import re
import sys
import warnings
from loguru import logger
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings("ignore")

# Mapping model labels to our entity types
ENTITY_MAP = {
    "PER": "NAME",
    "PERSON": "NAME",
    "EMAIL": "EMAIL",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "LOCATION": "ADDRESS",
    "GPE": "ADDRESS",
    "LOC": "ADDRESS",
    "ORG": "JOB",
    "TITLE": "JOB",
}

# ========== LOGGER ==========
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")

# ========== GLOBAL MODELS ==========
nlp_spacy = None
nlp_transformer = None
presidio_analyzer = None
presidio_anonymizer = None


# ========== 1️⃣ LOAD MODELS ==========
def load_spacy_model():
    global nlp_spacy
    try:
        import spacy
        nlp_spacy = spacy.load("en_core_web_sm")
        logger.success("✅ spaCy model loaded successfully")
    except Exception as e:
        logger.error(f"❌ spaCy load error: {e}")
    return nlp_spacy


def load_transformer_model():
    global nlp_transformer
    try:
        from transformers import pipeline
        nlp_transformer = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple", framework="pt")
        logger.success("✅ BERT model loaded successfully")
    except Exception as e:
        logger.error(f"❌ BERT load error: {e}")
    return nlp_transformer


def load_presidio_models():
    global presidio_analyzer, presidio_anonymizer
    try:
        from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
        from presidio_anonymizer import AnonymizerEngine
        from presidio_analyzer.predefined_recognizers import (
            EmailRecognizer, PhoneRecognizer, CreditCardRecognizer, IpRecognizer
        )

        registry = RecognizerRegistry()
        registry.load_predefined_recognizers()
        registry.add_recognizer(EmailRecognizer())
        registry.add_recognizer(PhoneRecognizer())
        registry.add_recognizer(CreditCardRecognizer())
        registry.add_recognizer(IpRecognizer())

        presidio_analyzer = AnalyzerEngine(registry=registry)
        presidio_anonymizer = AnonymizerEngine()
        logger.success("✅ Presidio models loaded successfully")
    except Exception as e:
        logger.error(f"❌ Presidio load error: {e}")
    return presidio_analyzer, presidio_anonymizer


# ========== 2️⃣ REGEX DETECTION ==========
patterns = {
    "NAME": re.compile(r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b"),
    "EMAIL": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "PHONE": re.compile(r"\+?\d{1,3}[\s\-]?\(?\d{2,4}\)?[\s\-]?\d{3,5}[\s\-]?\d{3,5}"),
    "ADDRESS": re.compile(r"\d+\s+[A-Za-z]+\s+(Street|Road|Avenue|Lane|Drive|Boulevard|Terrace)", re.IGNORECASE),
    "JOB": re.compile(r"\b(am|as|work(ed)?\s+as|I[' ]?m\s+a)\s+[A-Za-z ]+"),
    "HOBBY": re.compile(r"\b(enjoy|love|like)\s+[a-z]+ing\b", re.IGNORECASE),
}


def detect_with_regex(text):
    entities = []
    for key, pattern in patterns.items():
        if pattern.search(text):
            entities.append(key)
    return entities


# ========== 3️⃣ EVALUATION FUNCTION ==========
def evaluate_models_on_dataset(
    dataset_path=r"C:\Users\A Esther Graceia\Desktop\main_pii_research\data\pii_dataset.csv"
):
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset not found at {dataset_path}")

    logger.info(f"📊 Evaluating {len(df)} samples...")

    # Load models
    spacy_model = load_spacy_model()
    transformer_model = load_transformer_model()
    analyzer, anonymizer = load_presidio_models()

    entity_types = ["NAME", "EMAIL", "PHONE", "ADDRESS", "JOB", "HOBBY"]
    model_results = {model: {e: [] for e in entity_types} for model in ["Regex", "spaCy", "BERT", "Presidio"]}

    for _, row in df.iterrows():
        text = str(row["text"])
        ground_truth = {e: str(row.get(e.lower(), "")).strip() != "" for e in entity_types}

        # Regex
        regex_detected = detect_with_regex(text)
        for e in entity_types:
            model_results["Regex"][e].append((ground_truth[e], e in regex_detected))

        # spaCy
        if spacy_model:
            doc = spacy_model(text)
            spacy_entities = [ENTITY_MAP.get(ent.label_.upper(), ent.label_.upper()) for ent in doc.ents]
            for e in entity_types:
                model_results["spaCy"][e].append((ground_truth[e], e in spacy_entities))

        # BERT
        if transformer_model:
            bert_entities = [ENTITY_MAP.get(ent["entity_group"].upper(), ent["entity_group"].upper())
                             for ent in transformer_model(text)]
            for e in entity_types:
                model_results["BERT"][e].append((ground_truth[e], e in bert_entities))

        # Presidio
        if analyzer:
            presidio_entities = [ENTITY_MAP.get(r.entity_type.upper(), r.entity_type.upper())
                                 for r in analyzer.analyze(text=text, language="en")]
            for e in entity_types:
                model_results["Presidio"][e].append((ground_truth[e], e in presidio_entities))

    # ===== COMPUTE METRICS =====
    summary = {}
    for model, entity_dict in model_results.items():
        summary[model] = {}
        for entity, pairs in entity_dict.items():
            y_true = [t for t, _ in pairs]
            y_pred = [p for _, p in pairs]
            if any(y_true):
                precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
                accuracy = accuracy_score(y_true, y_pred)
                summary[model][entity] = {
                    "precision": round(precision, 3),
                    "recall": round(recall, 3),
                    "f1": round(f1, 3),
                    "accuracy": round(accuracy, 3),
                    "support": sum(y_true),
                }

    logger.success("✅ Evaluation complete")

    # Display summary
    for model, scores in summary.items():
        logger.info(f"\n📘 {model} Results:")
        for entity, metrics in scores.items():
            logger.info(f"  {entity}: {metrics}")

    # ===== SAMPLE MASKING =====
    results_dir = os.path.join(os.path.dirname(dataset_path), "results")
    os.makedirs(results_dir, exist_ok=True)

    masked_examples = []
    for text in df.head(3)["text"]:
        analyzed = analyzer.analyze(text=text, language="en")
        masked = anonymizer.anonymize(text=text, analyzer_results=analyzed)
        masked_examples.append({"original": text, "masked": masked.text})

    logger.info("\n🧩 Sample Masked Outputs:")
    for ex in masked_examples:
        logger.info(f"\nOriginal: {ex['original']}\nMasked:   {ex['masked']}\n")

    # Save masked dataset in results folder
    df["masked_text"] = df["text"].apply(
        lambda t: anonymizer.anonymize(text=t, analyzer_results=analyzer.analyze(text=t, language="en")).text
    )
    masked_path = os.path.join(results_dir, "pii_masked.csv")
    df.to_csv(masked_path, index=False)
    logger.success(f"💾 Masked dataset saved to {masked_path}")

    return summary


# ========== 4️⃣ FASTAPI APP ==========
app = FastAPI(title="PII Research Backend", description="Unified PII Detection and Evaluation API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "running", "message": "PII Research Backend Active"}


@app.get("/evaluate")
def evaluate():
    try:
        results = evaluate_models_on_dataset()
        return {"success": True, "results": results}
    except Exception as e:
        logger.error(f"❌ Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========== 5️⃣ MAIN EXECUTION ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_pii_research:app", host="0.0.0.0", port=8001, reload=True)
