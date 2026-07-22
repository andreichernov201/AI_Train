from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .ocr.preprocessing import quality_metrics


def score_frame(
    image: np.ndarray,
    train_detection: dict[str, Any],
    number_count: int,
    completeness: float = 0.0,
) -> tuple[float, dict[str, float]]:
    sharpness, _contrast, quality = quality_metrics(image)
    height, width = image.shape[:2]
    x1, y1, x2, y2 = map(float, train_detection.get("box", [0, 0, 0, 0]))
    area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / max(1.0, width * height)
    clipped = float(x1 <= 2 or y1 <= 2 or x2 >= width - 2 or y2 >= height - 2)
    confidence = float(train_detection.get("conf", 0.0))
    score = (
        confidence * 0.30
        + min(1.0, area_ratio * 2.5) * 0.18
        + quality * 0.20
        + min(1.0, number_count / 2.0) * 0.17
        + completeness * 0.15
        - clipped * 0.10
    )
    return round(max(0.0, min(1.0, score)), 6), {
        "train_confidence": confidence,
        "train_area_ratio": round(area_ratio, 6),
        "sharpness": round(sharpness, 3),
        "quality": quality,
        "number_count": float(number_count),
        "clipped": clipped,
    }


def encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Failed to encode evidence frame")
    return encoded.tobytes()
