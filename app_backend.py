"""
main_pii_research.py
====================
Entity-level PII Detection & Masking Evaluation.

Includes:
- Pre-trained model integration: Regex, spaCy, BERT, Presidio
- Model evaluation across entity types (name, email, phone, address, job, hobby)
- Sample masking demonstration
- Optional FastAPI backend for testing

Author: Esther Graceia Precious A
Date: 2025-11-04
"""

import os
import pandas as pd
import re
import sys
import warnings
from collections import defaultdict
from loguru import logger
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from presidio_analyzer import PatternRecognizer, Pattern
import json


warnings.filterwarnings("ignore")
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


logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")


nlp_spacy = None
nlp_transformer = None
presidio_analyzer = None
presidio_anonymizer = None



def load_spacy_model():
    global nlp_spacy
    try:
        import spacy
        from spacy.pipeline import EntityRuler
        nlp_spacy = spacy.load("en_core_web_sm")

        ruler = nlp_spacy.add_pipe("entity_ruler", before="ner")

        patterns = [
            {"label": "EMAIL", "pattern": [{"TEXT": {"REGEX": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"}}]},
            {"label": "PHONE", "pattern": [{"TEXT": {"REGEX": r"(\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}"}}]},
            {"label": "PHONE", "pattern": [{"TEXT": {"REGEX": r"(?:\+?\d{1,3}[\s\-]?)?\d{10,12}"}}]},
            {"label": "PHONE", "pattern": [{"TEXT": {"REGEX": r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"}}]},
            {"label": "JOB", "pattern": [{"TEXT": {"REGEX": r"\b(am|as|work(ed)?\s+as|I[' ]?m\s+(a|an))\s+[A-Za-z ]{2,30}\b"}}]},
            {"label": "HOBBY","pattern": [{"TEXT": {"REGEX": r"\b(enjoy|love|like|prefer|adore|play|practice|do|am\s+into)\s+[a-z]+(ing)?\b"}}]},
            {"label": "HOBBY","pattern": [{"TEXT": {"REGEX": r"\bmy\s+hobb(y|ies)\s+(is|are)\s+[a-z\s]+(ing)?\b"}}]},
            {"label": "HOBBY","pattern": [{"TEXT": {"REGEX": r"\bin\s+my\s+(free|leisure|spare)\s+time[,\s]+i\s+[a-z]+\b"}}]},
            {"label": "HOBBY","pattern": [{"TEXT": {"REGEX": r"\bi\s+(do|go|play)\s+[a-z]+(ing)?(\s+for\s+fun|\s+as\s+a\s+hobby)?\b"}}]}
        ]

        ruler.add_patterns(patterns)
        logger.success("spaCy model + custom EntityRuler loaded successfully")
    except Exception as e:
        logger.error(f"spaCy load error: {e}")
    return nlp_spacy


def load_transformer_model():
    global nlp_transformer
    try:
        from transformers import pipeline
        nlp_transformer = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple", framework="pt")
        logger.success("BERT model loaded successfully")
    except Exception as e:
        logger.error(f"BERT load error: {e}")
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


        job_pattern = Pattern("JobTitle", r"\b(am|as|work(ed)?\s+as|I[' ]?m\s+(a|an))\s+[A-Za-z ]{2,30}\b", 0.6)
        job_recognizer = PatternRecognizer(supported_entity="JOB", patterns=[job_pattern])

        hobby_pattern = Pattern(
            "HobbyExtended",
            r"\b(enjoy|love|like|prefer|adore|am\s+into|passionate\s+about|do|go|play|practice)\s+[a-z]+(ing)?\b",
            0.45
        )

        hobby_recognizer = PatternRecognizer(
            supported_entity="HOBBY",
            patterns=[hobby_pattern],
            context=["hobby", "free time", "leisure", "fun", "weekends", "activity", "relaxing", "interest"]
        )

        address_pattern = Pattern(
            "AddressPattern",
            r"\b\d{1,5}\s+[A-Z][a-zA-Z\s]+(Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Boulevard|Blvd|Drive|Dr|Terrace|Way|Block|Phase|Sector|Main|Cross|Colony)\b(?:[,\sA-Z0-9]*)?",
            0.65
        )

        address_recognizer = PatternRecognizer(
            supported_entity="ADDRESS",
            patterns=[address_pattern],
            context=["road", "street", "avenue", "city", "colony", "sector", "block"]
        )
        registry.add_recognizer(job_recognizer)
        registry.add_recognizer(hobby_recognizer)
        registry.add_recognizer(address_recognizer)

        presidio_analyzer = AnalyzerEngine(registry=registry)
        presidio_anonymizer = AnonymizerEngine()

        logger.success("✅ Presidio models (with custom Address recognizer) loaded successfully")

    except Exception as e:
        logger.error(f"❌ Presidio load error: {e}")

    return presidio_analyzer, presidio_anonymizer

patterns = {
    "NAME": re.compile(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b"),
    "EMAIL": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "PHONE": re.compile(r"(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}"),
    "ADDRESS": re.compile(
    r"(?:\blive at\b|\bstaying at\b|\blocated at\b|\baddress is\b)?\s*\d{1,5}[A-Za-z\s,.-]*"
    r"(Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Boulevard|Blvd|Drive|Dr|Terrace|Way|"
    r"Place|Block|Sector|Colony|Apartment|Building)\b[^\n]*",
    re.IGNORECASE,
    ),

    "ADDRESS": re.compile(
    r"\b\d{1,5}[A-Za-z\s,.-]*(Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Boulevard|Blvd|"
    r"Drive|Dr|Terrace|Way|Place|Block|Sector|Colony|Apartment|Building)\b[^\n]*",
    re.IGNORECASE
),

    "HOBBY": re.compile(r"\b(enjoy|love|like)\s+[a-z]+ing\b", re.IGNORECASE)
}
hobby_patterns = [
    r"\b(enjoy|love|like|prefer|adore|am\s+into|passionate\s+about)\s+[a-z]+(ing)?\b",
    r"\bmy\s+hobb(y|ies)\s+(is|are)\s+[a-z\s]+(ing)?\b",
    r"\bi\s*(am|'m)?\s*(really\s+)?(keen|interested|fond)\s+of\s+[a-z\s]+(ing)?\b",
    r"\bin\s+my\s+(free|leisure|spare)\s+time[,\s]+i\s+(do|go|play|enjoy|practice|love)\s+[a-z]+(ing)?\b",
    r"\bi\s+(do|go|play)\s+[a-z]+(ing)?(\s+for\s+fun|\s+as\s+a\s+hobby|\s+on\s+weekends)?\b",
    r"\b([A-Za-z]+(ing)?)\s+is\s+(fun|relaxing|my\s+hobby|enjoyable|something\s+i\s+like)\b",
    r"\b([A-Za-z]+(ing)?)\s+(relaxes|calms|interests)\s+me\b"
]

combined_hobby_pattern = re.compile("|".join(hobby_patterns), re.IGNORECASE)
patterns["HOBBY"] = combined_hobby_pattern



def detect_with_regex(text):
    entities = []
    for key, pattern in patterns.items():
        if pattern.search(text):
            entities.append(key)
    return entities


def ensemble_mask_text(text, threshold=2):

    global nlp_spacy, nlp_transformer, presidio_analyzer, presidio_anonymizer

    if nlp_spacy is None:
        nlp_spacy = load_spacy_model()
    if nlp_transformer is None:
        nlp_transformer = load_transformer_model()
    if presidio_analyzer is None:
        presidio_analyzer, presidio_anonymizer = load_presidio_models()

    regex_entities = detect_with_regex(text)

    spacy_entities = []
    if nlp_spacy:
        doc = nlp_spacy(text)
        spacy_entities = [ENTITY_MAP.get(ent.label_.upper(), ent.label_.upper()) for ent in doc.ents]

    bert_entities = []
    if nlp_transformer:
        bert_entities = [
            ENTITY_MAP.get(ent["entity_group"].upper(), ent["entity_group"].upper())
            for ent in nlp_transformer(text)
        ]

    presidio_entities = []
    if presidio_analyzer:
        presidio_entities = [
            ENTITY_MAP.get(r.entity_type.upper(), r.entity_type.upper())
            for r in presidio_analyzer.analyze(text=text, language="en")
        ]

    all_detections = {
        "Regex": regex_entities,
        "spaCy": spacy_entities,
        "BERT": bert_entities,
        "Presidio": presidio_entities
    }

    votes = defaultdict(int)
    for ents in all_detections.values():
        for e in ents:
            votes[e] += 1

    final_entities = [e for e, v in votes.items() if v >= threshold]

    spans = []
    for e in final_entities:
        pattern = patterns.get(e)
        if not pattern:
            continue
        for match in re.finditer(pattern, text):
            spans.append((match.start(), match.end(), e))

    spans.sort(key=lambda x: x[0], reverse=True)

    masked_text = text
    for start, end, label in spans:
        masked_text = masked_text[:start] + f"[{label}]" + masked_text[end:]

    masked_text = re.sub(r'\s+\[', ' [', masked_text)
    masked_text = re.sub(r'\[\s+', '[', masked_text)

    return masked_text, all_detections, final_entities


def evaluate_models_on_dataset(
    dataset_path=r"C:\Users\A Esther Graceia\Desktop\main_pii_research\data\pii_dataset.csv"
):
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset not found at {dataset_path}")

    logger.info(f"📊 Evaluating {len(df)} samples...")

    spacy_model = load_spacy_model()
    transformer_model = load_transformer_model()
    analyzer, anonymizer = load_presidio_models()

    entity_types = ["NAME", "EMAIL", "PHONE", "ADDRESS", "JOB", "HOBBY"]
    patterns["HOBBY"] = combined_hobby_pattern

    model_results = {model: {e: [] for e in entity_types} for model in ["Regex", "spaCy", "BERT", "Presidio"]}

    for _, row in df.iterrows():
        text = str(row["text"])
        ground_truth = {e: str(row.get(e.lower(), "")).strip() != "" for e in entity_types}

        regex_detected = detect_with_regex(text)
        for e in entity_types:
            model_results["Regex"][e].append((ground_truth[e], e in regex_detected))

        if spacy_model:
            doc = spacy_model(text)
            spacy_entities = [ENTITY_MAP.get(ent.label_.upper(), ent.label_.upper()) for ent in doc.ents]
            for e in entity_types:
                model_results["spaCy"][e].append((ground_truth[e], e in spacy_entities))

        if transformer_model:
            bert_entities = [
                ENTITY_MAP.get(ent["entity_group"].upper(), ent["entity_group"].upper())
                for ent in transformer_model(text)
            ]

            regex_detected = detect_with_regex(text)
            for r_entity in regex_detected:
                if r_entity not in bert_entities:
                    bert_entities.append(r_entity)

            for e in entity_types:
                model_results["BERT"][e].append((ground_truth[e], e in bert_entities))


        if analyzer:
            presidio_entities = [ENTITY_MAP.get(r.entity_type.upper(), r.entity_type.upper())
                                 for r in analyzer.analyze(text=text, language="en")]
            for e in entity_types:
                model_results["Presidio"][e].append((ground_truth[e], e in presidio_entities))

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
                    "precision": float(round(precision, 3)),
                    "recall": float(round(recall, 3)),
                    "f1": float(round(f1, 3)),
                    "accuracy": float(round(accuracy, 3)),
                    "support": int(sum(y_true)),
                }


    logger.success("Evaluation complete")

    for model, scores in summary.items():
        logger.info(f"\n📘 {model} Results:")
        for entity, metrics in scores.items():
            logger.info(f"  {entity}: {metrics}")


    results_dir = os.path.join(os.path.dirname(dataset_path), "..", "results")
    results_dir = os.path.abspath(results_dir)

    os.makedirs(results_dir, exist_ok=True)

    masked_examples = []
    for text in df.head(1)["text"]:

        analyzed = analyzer.analyze(text=text, language="en")
        masked = anonymizer.anonymize(text=text, analyzer_results=analyzed)
        masked_examples.append({"original": text, "masked": masked.text})

    logger.info("\nSample Masked Outputs:")
    for ex in masked_examples:
        logger.info(f"\nOriginal: {ex['original']}\nMasked:   {ex['masked']}\n")

    df["masked_text"] = df["text"].apply(
        lambda t: anonymizer.anonymize(text=t, analyzer_results=analyzer.analyze(text=t, language="en")).text
    )
    masked_path = os.path.join(results_dir, "pii_masked.csv")
    df.to_csv(masked_path, index=False)
    logger.success(f"Masked dataset saved to {masked_path}")

    return summary


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
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/mask_ensemble")
def mask_with_ensemble(data: dict):
    try:
        text = data.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="No text provided")

        global nlp_spacy, nlp_transformer, presidio_analyzer, presidio_anonymizer

        if not any([nlp_spacy, nlp_transformer, presidio_analyzer]):
            logger.warning("Models not loaded yet — loading now...")
            nlp_spacy = load_spacy_model()
            nlp_transformer = load_transformer_model()
            presidio_analyzer, presidio_anonymizer = load_presidio_models()

        logger.info("Starting ensemble masking...")

        import time
        start = time.time()

        masked_text, detections, final_entities = ensemble_mask_text(
            text,
            threshold=2
        )

        elapsed = time.time() - start
        logger.info(f"Ensemble detected: {final_entities}")
        logger.success(f"Masked Text: {masked_text}")
        logger.info(f"Processing time: {elapsed:.2f} seconds")

        return {
            "success": True,
            "original_text": text,
            "masked_text": masked_text,
            "detections": detections,
            "final_entities_masked": final_entities,
            "processing_time_sec": round(elapsed, 2)
        }

    except Exception as e:
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    logger.info("🚀 Loading models at startup...")
    nlp_spacy = load_spacy_model()
    nlp_transformer = load_transformer_model()
    presidio_analyzer, presidio_anonymizer = load_presidio_models()
    logger.success("✅ All models ready!")
    import uvicorn
    uvicorn.run("main_pii_research:app", host="0.0.0.0", port=8001)