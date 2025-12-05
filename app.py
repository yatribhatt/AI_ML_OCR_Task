import streamlit as st
import numpy as np
from PIL import Image

from src.preprocessing import load_image_from_bytes, preprocess_for_ocr
from src.ocr_engine import OCREngine
from src.text_extraction import find_target_line
from src.utils import save_json_result


st.set_page_config(page_title="Shipping Label OCR - _1_ Extractor", layout="wide")

st.title("📦 Shipping Label OCR - Target Line Extractor (`_1_` pattern)")

st.markdown(
    """
This app:
1. Takes a **shipping label / waybill image**
2. Applies **OCR** using EasyOCR
3. Extracts the **full text line containing `_1_`** (e.g. `163233702292313922_1_lWV`)
"""
)

uploaded_file = st.file_uploader("Upload a shipping label image", type=["jpg", "jpeg", "png", "tif", "tiff"])

col1, col2 = st.columns(2)

if uploaded_file is not None:
    # Show original image
    with col1:
        st.subheader("Original Image")
        img_pil = Image.open(uploaded_file).convert("RGB")
        st.image(img_pil, use_container_width=True)

    if st.button("Run OCR & Extract Target Line"):
        with st.spinner("Running OCR..."):
            # Convert to OpenCV BGR
            img_bgr = load_image_from_bytes(uploaded_file.getvalue())
            preprocessed = preprocess_for_ocr(img_bgr)

            # Show preprocessed image
            with col1:
                st.subheader("Preprocessed for OCR")
                st.image(preprocessed, clamp=True, use_container_width=True)

            # OCR
            engine = OCREngine(languages=["en"], gpu=False)
            ocr_results = engine.run_ocr(preprocessed)

            # Display raw OCR results
            with col2:
                st.subheader("OCR Results (All)")
                for item in ocr_results:
                    st.write(
                        f"**Text:** `{item['text']}` | "
                        f"Confidence: `{item['confidence']:.3f}`"
                    )

            # Extract target line
            best_match, candidates = find_target_line(ocr_results)

            st.markdown("---")
            st.subheader("🎯 Target Line Extraction")

            if best_match is None:
                st.error("No line containing `_1_` (or close variant) was found in OCR output.")
            else:
                st.success("Target line detected!")

                st.markdown(
                    f"""
**Extracted Target Line:**

> `{best_match['text']}`  

**Confidence:** `{best_match['confidence']:.3f}`
"""
                )

                if len(candidates) > 1:
                    st.markdown("**Other candidate lines:**")
                    for c in candidates:
                        if c is best_match:
                            continue
                        st.write(f"- `{c['text']}` (conf: {c['confidence']:.3f})")

                # Save JSON result
                image_name = uploaded_file.name
                out_path = save_json_result(
                    image_name=image_name,
                    target_text=best_match["text"],
                    confidence=best_match["confidence"],
                )

                st.info(f"Result saved to `{out_path}` in the `results/` folder (for local runs).")
else:
    st.info("Please upload a shipping label image to start.")
