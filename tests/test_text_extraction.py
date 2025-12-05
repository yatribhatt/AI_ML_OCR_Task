import pytest
from src.text_extraction import find_target_line


def test_find_target_line_primary_match():
    ocr_results = [
        {"text": "HELLO WORLD", "confidence": 0.90, "bbox": []},
        {"text": "163233702292313922_1_lWV", "confidence": 0.95, "bbox": []},
        {"text": "OTHER LINE", "confidence": 0.80, "bbox": []},
    ]

    best, candidates = find_target_line(ocr_results)

    assert best is not None
    assert best["text"] == "163233702292313922_1_lWV"
    assert len(candidates) == 1


def test_find_target_line_fuzzy_match():
    ocr_results = [
        {"text": "TEXT 1", "confidence": 0.50, "bbox": []},
        {"text": "163233702292313922-1-lWV", "confidence": 0.88, "bbox": []},
    ]

    best, candidates = find_target_line(ocr_results)

    assert best is not None
    assert "163233702292313922-1-lWV" in best["text"]
    assert len(candidates) == 1
