# ──────────────────────────────────────────────────────────────
# Immune Disease Risk Detection System
# Supports: Linux, macOS, Windows (via X11 / VcXsrv / XQuartz)
# ──────────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="immune-risk-detection"
LABEL description="Autoimmune & Immunodeficiency Disease Risk Detection System"

# ── System dependencies ───────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Tkinter GUI
    python3-tk \
    tk-dev \
    # X11 forwarding (for GUI on host display)
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxtst6 \
    libxi6 \
    xauth \
    # Tesseract OCR
    tesseract-ocr \
    tesseract-ocr-eng \
    # Poppler (pdf2image / pdftotext)
    poppler-utils \
    # Build tools
    gcc \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy project files ────────────────────────────────────────
COPY . .

# ── Create required directories ───────────────────────────────
RUN mkdir -p data models outputs logs sample_pdfs

# ── Environment ───────────────────────────────────────────────
# DISPLAY is passed at runtime for GUI (e.g. -e DISPLAY=$DISPLAY)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── Default: launch GUI ───────────────────────────────────────
CMD ["python", "gui.py"]
