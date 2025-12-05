import json
import os
from typing import Optional, Dict, Any, List


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_json_result(
    image_name: str,
    target_text: Optional[str],
    confidence: Optional[float],
    out_dir: str = "results",
    extra: Optional[Dict[str, Any]] = None,
):
    """
    Save extraction result for an image as JSON.
    """
    ensure_dir(out_dir)
    base_name = os.path.splitext(os.path.basename(image_name))[0]
    out_path = os.path.join(out_dir, f"{base_name}.json")

    data = {
        "image_name": image_name,
        "target_text": target_text,
        "confidence": confidence,
    }
    if extra:
        data.update(extra)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out_path


def load_ground_truth(gt_path: str) -> Dict[str, str]:
    """
    Expected CSV format:
    image_name,target_text
    img1.jpg,163233702292313922_1_lWV
    ...
    """
    gt = {}
    import csv

    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt[row["image_name"]] = row["target_text"]
    return gt


def compute_accuracy(predictions: Dict[str, str], ground_truth: Dict[str, str]) -> float:
    """
    Simple exact match accuracy on target text.
    """
    total = 0
    correct = 0
    for img_name, gt_text in ground_truth.items():
        total += 1
        pred = predictions.get(img_name)
        if pred is not None and pred == gt_text:
            correct += 1
    if total == 0:
        return 0.0
    return correct / total
