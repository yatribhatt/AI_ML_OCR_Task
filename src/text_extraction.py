import re
from typing import List, Dict, Any, Optional, Tuple

# target pattern like 163233702292313922_1_lWV
TARGET_PATTERN = re.compile(r".*_1_.*")


def fuzzy_target_candidate(text: str) -> bool:
    """
    Backup pattern when underscores / separators are slightly wrong.
    """
    pattern = re.compile(r"[0-9]{6,}[-_ ]?1[-_ ]?[A-Za-z0-9]+")
    return bool(pattern.search(text))


def _group_segments_into_lines(
    ocr_results: List[Dict[str, Any]],
    y_threshold: float = 20.0,
):
    """
    Group small OCR pieces (often one character each) into horizontal lines.

    Returns list of "line" dicts:
    { "text": "160390797970200578_1_gsm", "confidence": 0.9, "segments": [...] }
    """
    lines: List[Dict[str, Any]] = []

    for item in ocr_results:
        bbox = item["bbox"]
        # bbox: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x_min = min(xs)
        y_center = (ys[0] + ys[2]) / 2.0

        # find best matching existing line based on y_center
        best_line = None
        best_dist = None
        for line in lines:
            dist = abs(y_center - line["y_center"])
            if dist <= y_threshold and (best_dist is None or dist < best_dist):
                best_line = line
                best_dist = dist

        if best_line is None:
            # make new line
            lines.append(
                {
                    "y_center": y_center,
                    "segments": [(x_min, item)],
                }
            )
        else:
            best_line["segments"].append((x_min, item))
            # update mean y
            n = len(best_line["segments"])
            best_line["y_center"] = (best_line["y_center"] * (n - 1) + y_center) / n

    # build merged text per line
    merged_lines: List[Dict[str, Any]] = []
    for line in lines:
        segs = sorted(line["segments"], key=lambda t: t[0])
        texts = [seg[1]["text"] for seg in segs]
        confs = [seg[1]["confidence"] for seg in segs]

        full_text = "".join(texts)  # join without spaces – good for codes
        avg_conf = sum(confs) / len(confs) if confs else 0.0

        merged_lines.append(
            {
                "text": full_text,
                "confidence": avg_conf,
                "segments": [seg[1] for seg in segs],
            }
        )

    return merged_lines


def find_target_line(
    ocr_results: List[Dict[str, Any]]
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    From raw OCR segments, reconstruct lines, then find the best matching target line.
    Returns:
      best_match: line dict or None
      candidates: all lines that matched primary or fuzzy pattern
    """
    line_results = _group_segments_into_lines(ocr_results)

    primary_matches = []
    fuzzy_matches = []

    for line in line_results:
        text = line["text"]
        if TARGET_PATTERN.search(text):
            primary_matches.append(line)
        elif fuzzy_target_candidate(text):
            fuzzy_matches.append(line)

    candidates = primary_matches if primary_matches else fuzzy_matches

    if not candidates:
        return None, []

    best_match = max(candidates, key=lambda x: x["confidence"])
    return best_match, candidates
