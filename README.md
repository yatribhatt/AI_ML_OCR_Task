# Shipping Label OCR - `_1_` Target Line Extractor

## 1. Project Overview

This project is an **OCR-based text extraction system** for shipping labels / waybills.

The goal is to:
1. Perform OCR on shipping label images.
2. Extract the **complete text line** containing the pattern `_1_`
   (e.g. `163233702292313922_1_lWV`).
3. Achieve **≥ 75% accuracy** on this specific target text.

The solution uses **open-source OCR (EasyOCR)** and a custom text extraction pipeline.

---

## 2. Tech Stack

- **Language:** Python
- **OCR Engine:** EasyOCR (open-source, no commercial API)
- **Image Processing:** OpenCV, NumPy
- **Frontend:** Streamlit
- **Others:** scikit-image, Pillow, tqdm

---

## 3. Usage Guide

**To start the application:**
- streamlit run app.py

Using the App:-
1. Upload a shipping label image
2. Click “Run OCR & Extract Target Line”
3. The app displays:
Original Image
Preprocessed Image
All OCR Results (raw text segments)
Reconstructed Full Lines
🎯 Final Extracted Target Line
4. A JSON output is saved in the results/ folder automatically.

---

## 4. Project Structure

```txt
project-root/
├── README.md
├── requirements.txt
├── src/
│   ├── ocr_engine.py
│   ├── preprocessing.py
│   ├── text_extraction.py
│   └── utils.py
├── app.py
├── tests/
│   └── test_text_extraction.py
├── notebooks/
│   └── README.md
└── results/
    └── README.md
