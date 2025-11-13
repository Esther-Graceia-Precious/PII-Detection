# 🔐 PII Detection and Masking Research (Python + FastAPI)

This project implements a multi-approach system for detecting and masking Personally Identifiable Information (PII) from unstructured text.
It combines **rule-based**, **statistical**, and **transformer-based** methods to improve detection accuracy and consistency.

The following approaches are used:
* **Regex-based pattern matching** (rule-based extraction, not a model)
* **spaCy NER** (en_core_web_sm) with custom EntityRuler patterns
* **BERT (dslim/bert-base-NER)** transformer-based NER
* **Microsoft Presidio** (Analyzer + Anonymizer)

These detectors are integrated using an **ensemble majority-voting pipeline**, and final anonymization is performed using **Microsoft Presidio**.

---

## Features

* **Multi-approach PII detection pipeline**
* **Entity-level evaluation** (Precision, Recall, F1-score, Accuracy)
* **Ensemble voting mechanism** for robust PII identification
* **Masked output generation** using placeholders:

  * `[NAME]`
  * `[EMAIL]`
  * `[PHONE]`
  * `[ADDRESS]`
  * `[JOB]`
  * `[HOBBY]`
* **FastAPI backend** for evaluation & real-time masking
* **Loguru logging** for research-grade tracking and debugging

---

# Data Folder

This folder contains the datasets used for PII detection evaluation:
* **raw_pii_dataset.csv**
  Unprocessed input data (synthetic or collected text samples)
* **pii_dataset.csv**
  Cleaned and structured dataset used for model benchmarking
  
*Note: These datasets are not included in the repository due to privacy, licensing, and size constraints.*

Dataset reference:
[https://www.kaggle.com/datasets/alejopaullier/pii-external-dataset](https://www.kaggle.com/datasets/alejopaullier/pii-external-dataset)

---
