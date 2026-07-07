"""
PDF Ingestion & Dataset Updater
--------------------------------
• Reads clinical report PDFs — both text-based AND scanned/image PDFs
• Auto-detects whether OCR is needed (scanned reports)
• Extracts structured patient data (lab values, symptoms, diagnosis)
• Appends to the dataset CSV
• Triggers model retraining

OCR Pipeline (for scanned PDFs):
  PDF → rasterize pages (pdf2image / pdftoppm) → pytesseract OCR → text → parse fields

Text Pipeline (for digital PDFs):
  PDF → pdfplumber / pypdf text extraction → parse fields
"""

import os
import re
import json
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

# ── PDF text extraction backends ──────────────────────────────
try:
    import pdfplumber
    PDF_BACKEND = 'pdfplumber'
except ImportError:
    try:
        from pypdf import PdfReader
        PDF_BACKEND = 'pypdf'
    except ImportError:
        PDF_BACKEND = None

# ── OCR backends ──────────────────────────────────────────────
try:
    import pytesseract
    from PIL import Image
    TESSERACT_OK = True
except ImportError:
    TESSERACT_OK = False

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_OK = True
except ImportError:
    PDF2IMAGE_OK = False

# Minimum extractable characters before we decide OCR is needed
MIN_TEXT_CHARS = 80

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Primary autoimmune dataset path
AUTOIMMUNE_DATASET = DATA_DIR / "Comprehensive_Autoimmune_Disorder_Dataset.csv"
IMMUNODEF_DATASET = DATA_DIR / "immunodeficiency_dataset.csv"
INGESTION_LOG = LOG_DIR / "pdf_ingestion_log.csv"


# ──────────────────────────────────────────────
# FIELD EXTRACTION PATTERNS
# ──────────────────────────────────────────────
FIELD_PATTERNS = {
    # Demographics
    'Patient_ID':              r'(?:patient\s*id|patient\s*#|id)[:\s]+([A-Za-z0-9\-]+)',
    'Age':                     r'age[:\s]+(\d{1,3})',
    'Gender':                  r'(?:gender|sex)[:\s]+(male|female|m|f)',
    'Sickness_Duration_Months': r'(?:duration|illness\s*duration|sick\s*for)[:\s]+(\d+)\s*(?:months?|mo\.?)',

    # CBC
    'RBC_Count':               r'rbc(?:\s*count)?[:\s]+([\d.]+)',
    'Hemoglobin':              r'h(?:ae?|e)moglobin[:\s]+([\d.]+)',
    'Hematocrit':              r'hematocrit[:\s]+([\d.]+)',
    'MCV':                     r'\bmcv\b[:\s]+([\d.]+)',
    'MCH':                     r'\bmch\b[:\s]+([\d.]+)',
    'MCHC':                    r'\bmchc\b[:\s]+([\d.]+)',
    'RDW':                     r'\brdw\b[:\s]+([\d.]+)',
    'Reticulocyte_Count':      r'reticulocyte[:\s]+([\d.]+)',
    'WBC_Count':               r'wbc(?:\s*count)?[:\s]+([\d.]+)',
    'Neutrophils':             r'neutrophils?[:\s]+([\d.]+)',
    'Lymphocytes':             r'lymphocytes?[:\s]+([\d.]+)',
    'Monocytes':               r'monocytes?[:\s]+([\d.]+)',
    'Eosinophils':             r'eosinophils?[:\s]+([\d.]+)',
    'Basophils':               r'basophils?[:\s]+([\d.]+)',
    'PLT_Count':               r'platelet(?:\s*count)?[:\s]+([\d.]+)',
    'MPV':                     r'\bmpv\b[:\s]+([\d.]+)',

    # Inflammatory markers
    'ESR':                     r'\besr\b[:\s]+([\d.]+)',
    'CRP':                     r'\bcrp\b[:\s]+([\d.]+)',
    'ANA':                     r'\bana\b[:\s]+([\d.]+|positive|negative)',
    'C3':                      r'\bc3\b(?:\s*complement)?[:\s]+([\d.]+)',
    'C4':                      r'\bc4\b(?:\s*complement)?[:\s]+([\d.]+)',
    'MBL_Level':               r'mbl(?:\s*level)?[:\s]+([\d.]+)',
    'Esbach':                  r'esbach[:\s]+([\d.]+)',

    # Common antibodies
    'Anti-dsDNA':              r'anti[- ]?ds\s*dna[:\s]+([\d.]+|positive|negative)',
    'Rheumatoid factor':       r'rheumatoid\s*factor[:\s]+([\d.]+|positive|negative)',
    'ACPA':                    r'\bacpa\b[:\s]+([\d.]+|positive|negative)',
    'Anti-TPO':                r'anti[- ]?tpo[:\s]+([\d.]+|positive|negative)',
    'ANCA':                    r'\banca\b[:\s]+([\d.]+|positive|negative)',
    'ANA':                     r'\bana\b[:\s]+([\d.]+|positive|negative)',

    # Diagnosis
    'Diagnosis':               r'(?:diagnosis|impression|assessment)[:\s]+([A-Za-z\s\-\'()]+?)(?:\n|$|;)',
}

SYMPTOM_PATTERNS = {
    'Low-grade fever':                r'(?:low[- ]?grade\s*fever|mild\s*fever)',
    'Fatigue or chronic tiredness':   r'(?:fatigue|chronic\s*tiredness|exhaustion)',
    'Dizziness':                      r'dizziness|vertigo',
    'Weight loss':                    r'weight\s*loss',
    'Rashes and skin lesions':        r'(?:rash|skin\s*lesion)',
    'Stiffness in the joints':        r'(?:joint\s*stiffness|morning\s*stiffness)',
    'Brittle hair or hair loss':      r'(?:hair\s*loss|alopecia|brittle\s*hair)',
    'Dry eyes and/or mouth':          r'(?:dry\s*eyes?|dry\s*mouth|xerostomia)',
    "General 'unwell' feeling":       r'(?:malaise|generally\s*unwell|unwellness)',
    'Joint pain':                     r'(?:joint\s*pain|arthralgia|arthritis)',
}

POSITIVE_WORDS = {'positive', 'yes', 'present', 'detected', 'reactive', '+', '1'}
NEGATIVE_WORDS = {'negative', 'no', 'absent', 'not detected', 'non-reactive', '-', '0'}


# ──────────────────────────────────────────────
# PDF TEXT EXTRACTION  (text + OCR fallback)
# ──────────────────────────────────────────────

def _extract_text_digital(pdf_path: str) -> str:
    """Extract text from a digitally-generated (non-scanned) PDF."""
    text = ""
    if PDF_BACKEND == 'pdfplumber':
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
                # Also grab tables — lab reports often use tables
                for table in page.extract_tables():
                    for row in table:
                        if row:
                            text += " | ".join(str(c) for c in row if c) + "\n"
    elif PDF_BACKEND == 'pypdf':
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text


def _extract_text_ocr(pdf_path: str, progress_callback=None) -> str:
    """
    OCR pipeline for scanned / image-based PDFs.
    Steps: PDF → page images (pdf2image / pdftoppm) → pytesseract → text
    """
    if not TESSERACT_OK:
        raise RuntimeError(
            "pytesseract not installed. Run: pip install pytesseract pillow"
        )

    text_parts = []

    # ── Rasterize pages ──────────────────────────────────────
    if PDF2IMAGE_OK:
        # pdf2image is cleanest — 200 DPI is good for lab reports
        if progress_callback:
            progress_callback("Rasterizing PDF pages for OCR (this may take a moment)...")
        log.info("  OCR: rasterizing with pdf2image at 200 DPI")
        pages = convert_from_path(pdf_path, dpi=200)
    else:
        # Fallback: use pdftoppm CLI tool (part of poppler-utils)
        import subprocess, tempfile, glob
        if progress_callback:
            progress_callback("Rasterizing pages with pdftoppm...")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_prefix = os.path.join(tmpdir, "page")
            result = subprocess.run(
                ["pdftoppm", "-jpeg", "-r", "200", pdf_path, out_prefix],
                capture_output=True
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"pdftoppm failed: {result.stderr.decode()}\n"
                    "Install poppler-utils: sudo apt install poppler-utils  "
                    "or: brew install poppler"
                )
            image_files = sorted(glob.glob(f"{out_prefix}-*.jpg"))
            pages = [Image.open(f) for f in image_files]

    total = len(pages)
    log.info(f"  OCR: processing {total} page(s)")

    for i, page_img in enumerate(pages, 1):
        if progress_callback:
            progress_callback(f"OCR page {i}/{total}...")

        # Preprocess image for better OCR accuracy:
        # Convert to grayscale, then run tesseract with lab-report config
        if hasattr(page_img, 'convert'):
            gray = page_img.convert('L')
        else:
            from PIL import Image as PILImage
            gray = PILImage.fromarray(page_img).convert('L')

        # psm 6 = "assume a single uniform block of text" — good for reports
        # oem 3 = default OCR engine
        custom_config = r'--oem 3 --psm 6'
        ocr_text = pytesseract.image_to_string(gray, config=custom_config)
        text_parts.append(ocr_text)
        log.info(f"  OCR page {i}: extracted {len(ocr_text)} chars")

    return "\n".join(text_parts)


def _is_scanned_pdf(pdf_path: str) -> bool:
    """
    Heuristic: try digital extraction first.
    If we get fewer than MIN_TEXT_CHARS of real text, treat it as scanned.
    """
    try:
        text = _extract_text_digital(pdf_path)
        # Strip whitespace-only content
        meaningful = re.sub(r'\s+', ' ', text).strip()
        return len(meaningful) < MIN_TEXT_CHARS
    except Exception:
        return True  # if extraction fails, assume scanned


def extract_text_from_pdf(pdf_path: str, progress_callback=None) -> tuple[str, str]:
    """
    Smart extraction: auto-detects scanned vs digital PDFs.

    Returns:
        (text, method) where method is 'digital' or 'ocr'
    """
    pdf_path = str(pdf_path)

    if PDF_BACKEND is None:
        raise RuntimeError(
            "No PDF library found. Run: pip install pdfplumber"
        )

    if progress_callback:
        progress_callback("Checking PDF type (digital vs scanned)...")

    scanned = _is_scanned_pdf(pdf_path)

    if scanned:
        log.info("  Detected scanned/image PDF — switching to OCR pipeline")
        if progress_callback:
            progress_callback("Scanned PDF detected — running OCR...")
        text = _extract_text_ocr(pdf_path, progress_callback)
        return text, 'ocr'
    else:
        log.info("  Detected digital PDF — using direct text extraction")
        if progress_callback:
            progress_callback("Digital PDF detected — extracting text...")
        text = _extract_text_digital(pdf_path)
        return text, 'digital'


# ──────────────────────────────────────────────
# FIELD PARSING
# ──────────────────────────────────────────────
def parse_binary(value: str) -> int:
    """Convert positive/negative string to 0/1."""
    v = str(value).strip().lower()
    if v in POSITIVE_WORDS:
        return 1
    if v in NEGATIVE_WORDS:
        return 0
    try:
        return int(float(v) > 0)
    except ValueError:
        return np.nan


def parse_numeric(value: str) -> float:
    """Parse numeric value, handling commas."""
    try:
        return float(str(value).replace(',', '').strip())
    except ValueError:
        return np.nan


def extract_patient_data(text: str) -> dict:
    """Parse structured patient fields from raw PDF text."""
    text_lower = text.lower()
    patient = {}

    # Extract lab values and demographics
    for field, pattern in FIELD_PATTERNS.items():
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            raw = match.group(1).strip()

            if field == 'Gender':
                patient[field] = 'Male' if raw.lower() in ('m', 'male') else 'Female'

            elif field in ('ANA', 'Anti-dsDNA', 'Rheumatoid factor', 'ACPA',
                           'Anti-TPO', 'ANCA'):
                patient[field] = parse_binary(raw)

            elif field == 'Diagnosis':
                patient[field] = raw.title().strip()

            else:
                patient[field] = parse_numeric(raw)

    # Extract symptoms (binary presence)
    for symptom, pattern in SYMPTOM_PATTERNS.items():
        patient[symptom] = int(bool(re.search(pattern, text_lower, re.IGNORECASE)))

    # Ensure Patient_ID
    if 'Patient_ID' not in patient:
        patient['Patient_ID'] = f"PDF_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # Default Diagnosis if missing
    if 'Diagnosis' not in patient or not patient['Diagnosis']:
        patient['Diagnosis'] = 'Unknown'

    return patient


# ──────────────────────────────────────────────
# DATASET UPDATE
# ──────────────────────────────────────────────
def get_target_dataset(diagnosis: str) -> Path:
    """Choose which dataset to append to based on diagnosis."""
    diagnosis_lower = str(diagnosis).lower()
    immuno_keywords = [
        'immunodeficiency', 'agammaglobulinemia', 'scid',
        'digeorge', 'wiskott', 'granulomatous', 'complement deficiency'
    ]
    if any(kw in diagnosis_lower for kw in immuno_keywords):
        return IMMUNODEF_DATASET
    return AUTOIMMUNE_DATASET


def append_to_dataset(patient_data: dict, dataset_path: Path) -> bool:
    """Append extracted patient data to the CSV dataset."""
    new_row = pd.DataFrame([patient_data])

    if dataset_path.exists():
        existing = pd.read_csv(dataset_path)

        # Check duplicate
        if 'Patient_ID' in existing.columns:
            if str(patient_data.get('Patient_ID')) in existing['Patient_ID'].astype(str).values:
                log.warning(f"Patient {patient_data.get('Patient_ID')} already exists. Skipping.")
                return False

        # Align columns — add missing cols with NaN
        for col in existing.columns:
            if col not in new_row.columns:
                new_row[col] = np.nan
        new_row = new_row[existing.columns]  # match column order

        updated = pd.concat([existing, new_row], ignore_index=True)
    else:
        updated = new_row

    updated.to_csv(dataset_path, index=False)
    log.info(f"  Appended to: {dataset_path.name} (now {len(updated)} rows)")
    return True


def log_ingestion(pdf_path: str, patient_id: str, diagnosis: str,
                  dataset: str, status: str, error: str = ""):
    """Log each ingestion event."""
    entry = pd.DataFrame([{
        'timestamp': datetime.now().isoformat(),
        'pdf_file': Path(pdf_path).name,
        'patient_id': patient_id,
        'diagnosis': diagnosis,
        'dataset': dataset,
        'status': status,
        'error': error
    }])
    if INGESTION_LOG.exists():
        log_df = pd.read_csv(INGESTION_LOG)
        log_df = pd.concat([log_df, entry], ignore_index=True)
    else:
        log_df = entry
    log_df.to_csv(INGESTION_LOG, index=False)


# ──────────────────────────────────────────────
# MAIN INGESTION FUNCTION
# ──────────────────────────────────────────────
def ingest_pdf(
    pdf_path: str,
    retrain: bool = True,
    progress_callback=None
) -> dict:
    """
    Full ingestion pipeline for one PDF.
    Returns result dict with status and extracted data.
    """
    log.info(f"Ingesting PDF: {pdf_path}")
    result = {
        'pdf': pdf_path,
        'status': 'pending',
        'patient_data': {},
        'dataset_updated': False,
        'retrained': False,
        'errors': []
    }

    # Step 1: Extract text (auto-detects scanned vs digital)
    try:
        text, extract_method = extract_text_from_pdf(pdf_path, progress_callback)
        result['extraction_method'] = extract_method
        if not text.strip():
            raise ValueError(
                "No text could be extracted from this PDF.\n"
                "If it is a scanned report, ensure tesseract-ocr is installed:\n"
                "  sudo apt install tesseract-ocr   (Linux)\n"
                "  brew install tesseract            (macOS)"
            )
        log.info(f"  Extracted {len(text)} characters via {extract_method.upper()}")
    except Exception as e:
        result['status'] = 'failed'
        result['errors'].append(f"Text extraction: {e}")
        log.error(f"  Text extraction failed: {e}")
        return result

    # Step 2: Parse fields
    if progress_callback:
        progress_callback("Parsing patient fields...")
    patient_data = extract_patient_data(text)
    result['patient_data'] = patient_data
    log.info(f"  Extracted {len(patient_data)} fields")
    log.info(f"  Patient ID : {patient_data.get('Patient_ID')}")
    log.info(f"  Diagnosis  : {patient_data.get('Diagnosis')}")
    log.info(f"  Age        : {patient_data.get('Age')}")

    # Step 3: Append to dataset
    target_ds = get_target_dataset(patient_data.get('Diagnosis', ''))
    try:
        if progress_callback:
            progress_callback("Updating dataset...")
        appended = append_to_dataset(patient_data, target_ds)
        result['dataset_updated'] = appended
        log_ingestion(
            pdf_path,
            patient_data.get('Patient_ID', 'unknown'),
            patient_data.get('Diagnosis', 'unknown'),
            target_ds.name,
            'success' if appended else 'duplicate'
        )
    except Exception as e:
        result['errors'].append(f"Dataset update: {e}")
        log.error(f"  Dataset update failed: {e}")
        log_ingestion(pdf_path, patient_data.get('Patient_ID', '?'),
                      patient_data.get('Diagnosis', '?'), target_ds.name, 'error', str(e))

    # Step 4: Retrain
    if retrain and result['dataset_updated']:
        try:
            if progress_callback:
                progress_callback("Retraining models...")
            from ml_model import train_all
            metadata = train_all()
            result['retrained'] = True
            result['training_metadata'] = metadata
            log.info("  Model retrained successfully")
        except Exception as e:
            result['errors'].append(f"Retraining: {e}")
            log.error(f"  Retraining failed: {e}")

    result['status'] = 'success' if not result['errors'] else 'partial'
    return result


def ingest_folder(folder_path: str, retrain_after_all: bool = True) -> list:
    """Ingest all PDFs in a folder. Retrain once at the end."""
    folder = Path(folder_path)
    pdf_files = list(folder.glob("*.pdf"))
    log.info(f"Found {len(pdf_files)} PDFs in {folder}")

    results = []
    for i, pdf in enumerate(pdf_files):
        log.info(f"\n[{i+1}/{len(pdf_files)}] Processing: {pdf.name}")
        result = ingest_pdf(str(pdf), retrain=False)
        results.append(result)

    if retrain_after_all:
        log.info("\nRetraining models on all new data...")
        try:
            from ml_model import train_all
            train_all()
            log.info("Retraining complete.")
        except Exception as e:
            log.error(f"Retraining failed: {e}")

    return results


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Ingest clinical PDF reports into the immune disease dataset'
    )
    parser.add_argument('input', help='PDF file or folder path')
    parser.add_argument(
        '--no-retrain', action='store_true',
        help='Skip model retraining after ingestion'
    )
    parser.add_argument(
        '--folder', action='store_true',
        help='Treat input as a folder (batch ingestion)'
    )
    args = parser.parse_args()

    if args.folder:
        results = ingest_folder(args.input, retrain_after_all=not args.no_retrain)
        success = sum(1 for r in results if r['status'] in ('success', 'partial'))
        print(f"\nIngested {success}/{len(results)} PDFs successfully.")
    else:
        result = ingest_pdf(args.input, retrain=not args.no_retrain)
        print(json.dumps({
            'status': result['status'],
            'patient_id': result['patient_data'].get('Patient_ID'),
            'diagnosis': result['patient_data'].get('Diagnosis'),
            'dataset_updated': result['dataset_updated'],
            'retrained': result['retrained'],
            'errors': result['errors']
        }, indent=2))
