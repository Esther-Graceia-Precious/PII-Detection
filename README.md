# 🔐 PII Detection and Masking Research (Python + FastAPI)

This project compares multiple **pre-trained models** for detecting Personally Identifiable Information (PII) in text, using:
- **Regex patterns**
- **spaCy NER**
- **BERT (dslim/bert-base-NER)**
- **Microsoft Presidio**

The system evaluates model performance on a labeled dataset and anonymizes detected PII fields.

---

## ⚙️ Features
- 📘 Multi-model PII detection pipeline  
- 🧠 Entity-level evaluation (precision, recall, F1, accuracy)  
- 🧩 Masked output generation using Microsoft Presidio  
- ⚡ FastAPI backend for evaluation  
- 🧮 Research-ready logs using Loguru  

---

## 🧱 Folder Structure
