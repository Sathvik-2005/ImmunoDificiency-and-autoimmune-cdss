#!/bin/bash
# ══════════════════════════════════════════════════════════════════
# Immune Disease Risk Detection System — Ubuntu Setup & Launch
# Run this once after unzipping the project.
# Usage: bash setup.sh
# ══════════════════════════════════════════════════════════════════

set -e  # Exit on any error

# ── Colors for output ─────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ── Helpers ───────────────────────────────────────────────────
info()    { echo -e "${CYAN}[INFO]${NC}  $1"; }
success() { echo -e "${GREEN}[OK]${NC}    $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
step()    { echo -e "\n${BOLD}${BLUE}▶  $1${NC}"; }

# ── Banner ────────────────────────────────────────────────────
echo -e "${BOLD}${CYAN}"
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║   🧬  Immune Disease Risk Detection System           ║"
echo "  ║       Ubuntu Setup Script                            ║"
echo "  ╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── Check we are inside the project folder ────────────────────
if [ ! -f "ml_model.py" ] || [ ! -f "gui.py" ]; then
    error "Please run this script from inside the immune_risk_detection/ folder.\n  cd immune_risk_detection && bash setup.sh"
fi

PROJECT_DIR="$(pwd)"
info "Project directory: $PROJECT_DIR"

# ══════════════════════════════════════════════════════════════
# STEP 1 — System packages
# ══════════════════════════════════════════════════════════════
step "Step 1/6 — Installing system packages"

info "Updating apt package list..."
sudo apt-get update -qq

info "Installing Python, Tkinter, Tesseract OCR, Poppler..."
sudo apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-tk \
    tk-dev \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    gcc \
    g++ \
    libx11-6 \
    > /dev/null 2>&1

success "System packages installed"

# ══════════════════════════════════════════════════════════════
# STEP 2 — Python virtual environment
# ══════════════════════════════════════════════════════════════
step "Step 2/6 — Creating Python virtual environment"

if [ -d "venv" ]; then
    warn "Virtual environment already exists — skipping creation"
else
    python3 -m venv venv
    success "Virtual environment created at ./venv"
fi

# Activate venv
source venv/bin/activate
success "Virtual environment activated"

# ══════════════════════════════════════════════════════════════
# STEP 3 — Python dependencies
# ══════════════════════════════════════════════════════════════
step "Step 3/6 — Installing Python dependencies"

pip install --upgrade pip -q
pip install -r requirements.txt -q

success "All Python packages installed"

# Verify key imports
python3 -c "
import sklearn, pandas, numpy, joblib, pdfplumber, pytesseract, PIL, pdf2image
print('  All imports verified OK')
" && success "Import check passed" || error "Import check failed — check errors above"

# ══════════════════════════════════════════════════════════════
# STEP 4 — Create required directories
# ══════════════════════════════════════════════════════════════
step "Step 4/6 — Setting up project directories"

mkdir -p data models outputs logs sample_pdfs
success "Directories ready: data/ models/ outputs/ logs/ sample_pdfs/"

# Check dataset
if [ -f "data/Comprehensive_Autoimmune_Disorder_Dataset.csv" ]; then
    ROWS=$(python3 -c "import pandas as pd; print(len(pd.read_csv('data/Comprehensive_Autoimmune_Disorder_Dataset.csv')))")
    success "Autoimmune dataset found: $ROWS records"
else
    warn "Autoimmune dataset not found in data/"
    warn "Place your CSV at: data/Comprehensive_Autoimmune_Disorder_Dataset.csv"
fi

if [ -f "data/immunodeficiency_dataset.csv" ]; then
    ROWS=$(python3 -c "import pandas as pd; print(len(pd.read_csv('data/immunodeficiency_dataset.csv')))")
    success "Immunodeficiency dataset found: $ROWS records"
else
    warn "Immunodeficiency dataset not yet added (optional — add later)"
fi

# ══════════════════════════════════════════════════════════════
# STEP 5 — Train models (70/30 split)
# ══════════════════════════════════════════════════════════════
step "Step 5/6 — Training ML models (70/30 train-test split)"

if [ -f "models/risk_model.pkl" ] && [ -f "models/category_model.pkl" ]; then
    echo ""
    read -p "  Models already exist. Retrain? (y/N): " RETRAIN
    RETRAIN=${RETRAIN:-N}
    if [[ "$RETRAIN" =~ ^[Yy]$ ]]; then
        info "Retraining models..."
        python3 ml_model.py
        success "Models retrained and saved to models/"
    else
        success "Skipping retraining — using existing models"
    fi
else
    info "Training models — this takes 1-2 minutes..."
    python3 ml_model.py
    success "Models trained and saved to models/"
fi

# Verify test results
if [ -f "outputs/test_results.json" ]; then
    python3 -c "
import json
with open('outputs/test_results.json') as f:
    d = json.load(f)
rm = d['risk_model']
cm = d['category_model']
sm = d['strat_model']
print(f'  Risk Model  → AUC: {rm[\"roc_auc\"]}  Acc: {rm[\"accuracy\"]}  (Train: {rm[\"train_size\"]} | Test: {rm[\"test_size\"]})')
print(f'  Category    → Acc: {cm[\"accuracy\"]}  F1: {cm[\"f1_weighted\"]}')
print(f'  Stratif.    → Acc: {sm[\"accuracy\"]}  F1: {sm[\"f1_weighted\"]}  (Train: {sm[\"train_size\"]} | Test: {sm[\"test_size\"]})')
print()
risk_dist = sm.get('risk_distribution', {})
for level in ['No Risk', 'Low Risk', 'Moderate Risk', 'Severe Risk']:
    count = risk_dist.get(level, 0)
    if count > 0:
        print(f'    {level:<16}: {count:,} patients')
"
    success "Test results saved to outputs/test_results.json"
fi

# ══════════════════════════════════════════════════════════════
# STEP 6 — Create launch script
# ══════════════════════════════════════════════════════════════
step "Step 6/6 — Creating launch shortcut"

cat > run.sh << 'EOF'
#!/bin/bash
# Quick launch script — run this anytime to start the app
cd "$(dirname "$0")"
source venv/bin/activate
python3 gui.py
EOF

chmod +x run.sh
success "Created run.sh — use this to launch the app next time"

# ══════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}${GREEN}"
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║   ✅  Setup complete!                                ║"
echo "  ╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo -e "  ${BOLD}To launch the app:${NC}"
echo -e "  ${CYAN}bash run.sh${NC}"
echo ""
echo -e "  ${BOLD}Or manually:${NC}"
echo -e "  ${CYAN}source venv/bin/activate && python3 gui.py${NC}"
echo ""
echo -e "  ${BOLD}To ingest a PDF report:${NC}"
echo -e "  ${CYAN}source venv/bin/activate && python3 pdf_ingestion.py path/to/report.pdf${NC}"
echo ""
echo -e "  ${BOLD}To retrain models:${NC}"
echo -e "  ${CYAN}source venv/bin/activate && python3 ml_model.py${NC}"
echo ""

# Launch immediately?
read -p "  Launch the GUI now? (Y/n): " LAUNCH
LAUNCH=${LAUNCH:-Y}
if [[ "$LAUNCH" =~ ^[Yy]$ ]]; then
    info "Launching GUI..."
    python3 gui.py
fi
