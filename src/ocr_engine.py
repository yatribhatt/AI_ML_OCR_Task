from typing import List, Dict, Any
import numpy as np
import easyocr


class OCREngine:
    """
    EasyOCR-based engine wrapper.
    Returns a list of small text segments (chars / words) with bbox + confidence.
    """

    def __init__(self, languages=None, gpu: bool = False):
        if languages is None:
            languages = ["en"]
        # EasyOCR: gpu=False is fine on your laptop
        self.reader = easyocr.Reader(languages, gpu=gpu)

    def run_ocr(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run OCR on preprocessed image.

        Returns: list like:
        [
          { "text": "D", "confidence": 0.65, "bbox": [...] },
          ...
        ]
        """
        results = self.reader.readtext(img)
        parsed: List[Dict[str, Any]] = []
        for bbox, text, conf in results:
            parsed.append(
                {
                    "text": text.strip(),
                    "confidence": float(conf),
                    "bbox": bbox,
                }
            )
        return parsed
