# 🧬 Immune Disease Risk Detection System

Early risk detection for **Autoimmune Diseases** and **Immunodeficiency Diseases** using machine learning — with a full desktop GUI, PDF clinical report ingestion, and automated retraining.

---

## 📁 Project Structure

```
immune_risk_detection/
├── ml_model.py          # ML training pipeline (3 models)
├── gui.py               # Desktop GUI (Tkinter)
├── pdf_ingestion.py     # PDF extraction → dataset → retrain
├── requirements.txt
├── data/
│   ├── Comprehensive_Autoimmune_Disorder_Dataset.csv
│   └── immunodeficiency_dataset.csv   (add when available)
├── models/              # Saved .pkl model files (auto-generated)
├── outputs/             # Feature importance CSVs (auto-generated)
└── logs/                # Training and ingestion logs (auto-generated)
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the models
```bash
python ml_model.py
```
Or use the GUI's **Train Models** tab.

### 3. Launch the GUI
```bash
python gui.py
```

### 4. Ingest a PDF clinical report
```bash
python pdf_ingestion.py path/to/report.pdf
# Ingest entire folder:
python pdf_ingestion.py path/to/folder/ --folder
# Skip retraining:
python pdf_ingestion.py report.pdf --no-retrain
```

---

## 🤖 Models

| Model | Type | Task |
|-------|------|------|
| Risk Detection | Random Forest | Binary: At Risk vs Normal |
| Disease Category | Gradient Boosting | Multiclass: Normal / Autoimmune / Immunodeficiency |
| Risk Stratification | K-Means + Gradient Boosting | Cluster: Low / Moderate / High / Very High Risk |

---

## 🖥️ GUI Tabs

| Tab | Description |
|-----|-------------|
| **Patient Assessment** | Enter lab values & symptoms → instant risk prediction |
| **Train Models** | Train / retrain models with progress tracking |
| **PDF Ingestion** | Import PDFs, extract data, update dataset, retrain |
| **Dataset Viewer** | Browse and filter the patient dataset |
| **Model Info** | View last training metrics |

---

## 🔬 Features Used

- **Demographics**: Age, Gender, Sickness Duration
- **CBC**: RBC, Hemoglobin, WBC, Neutrophils, Lymphocytes, Platelets, etc.
- **Inflammatory markers**: ESR, CRP, ANA, C3, C4, MBL Level
- **Antibodies**: Anti-dsDNA, Rheumatoid Factor, ACPA, Anti-TPO, ANCA, and 30+ more
- **Symptoms**: Fatigue, Joint pain, Rashes, Hair loss, Dry eyes, etc.
- **Engineered features**: Symptom Burden Score, Antibody Score, Inflammation Index

---

## 📄 PDF Ingestion

The `pdf_ingestion.py` script:
1. Extracts text from clinical PDF reports using regex patterns
2. Parses lab values, antibody markers, demographics, and symptoms
3. Routes the patient record to the correct dataset (autoimmune vs immunodeficiency)
4. Appends the new record and triggers model retraining

Supports structured text PDFs. For scanned PDFs, OCR support can be added via `pytesseract`.

---

## ➕ Adding Immunodeficiency Data

Place your immunodeficiency dataset CSV at:
```
data/immunodeficiency_dataset.csv
```
It should have the same column structure as the autoimmune dataset, with the key column being `Diagnosis`.

---

## 📊 Dataset

- **Autoimmune**: 10,088 records × 79 features, 115 disease classes
- Covers: Lupus, Rheumatoid Arthritis, Multiple Sclerosis, Hashimoto's, Sjögren's, and 110+ more

---

## 🔧 Requirements

- Python 3.9+
- pandas, scikit-learn, numpy, joblib
- pdfplumber (PDF extraction)
- tkinter (built into Python — GUI)

---

## 📝 License

MIT License
