from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class CropVariant:
    name: str
    image: np.ndarray
    sharpness: float
    contrast: float
    quality_score: float


def crop_with_padding(
    image: np.ndarray,
    box: list[float] | tuple[float, float, float, float],
    padding_ratio: float = 0.08,
) -> tuple[np.ndarray, list[int], bool]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    pad_x = max(2.0, (x2 - x1) * padding_ratio)
    pad_y = max(2.0, (y2 - y1) * padding_ratio)
    requested = [x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y]
    clipped = requested[0] < 0 or requested[1] < 0 or requested[2] > width or requested[3] > height
    px1 = max(0, min(width - 1, int(round(requested[0]))))
    py1 = max(0, min(height - 1, int(round(requested[1]))))
    px2 = max(px1 + 1, min(width, int(round(requested[2]))))
    py2 = max(py1 + 1, min(height, int(round(requested[3]))))
    return image[py1:py2, px1:px2].copy(), [px1, py1, px2, py2], clipped


def quality_metrics(image: np.ndarray) -> tuple[float, float, float]:
    if image.size == 0:
        return 0.0, 0.0, 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    area_factor = min(1.0, float(gray.shape[0] * gray.shape[1]) / 12000.0)
    quality = min(1.0, sharpness / 400.0) * 0.45 + min(1.0, contrast / 64.0) * 0.35 + area_factor * 0.20
    return sharpness, contrast, round(quality, 6)


def _variant(name: str, image: np.ndarray) -> CropVariant:
    sharpness, contrast, score = quality_metrics(image)
    return CropVariant(name, image, sharpness, contrast, score)


def primary_variants(crop: np.ndarray) -> list[CropVariant]:
    variants = [_variant("original", crop)]
    height, width = crop.shape[:2]
    scale = 4 if min(height, width) < 24 else 3 if min(height, width) < 48 else 2
    upscaled = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    variants.append(_variant(f"upscale_{scale}x", upscaled))
    return variants


def fallback_variants(crop: np.ndarray) -> list[CropVariant]:
    height, width = crop.shape[:2]
    scale = 4 if min(height, width) < 24 else 3 if min(height, width) < 48 else 2
    upscaled = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    clahe_bgr = cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)
    blurred = cv2.GaussianBlur(clahe_bgr, (0, 0), 1.0)
    sharpened = cv2.addWeighted(clahe_bgr, 1.45, blurred, -0.45, 0)
    return [_variant("clahe_gray", clahe_bgr), _variant("mild_sharpen", sharpened)]
