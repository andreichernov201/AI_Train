from __future__ import annotations

from math import hypot
from typing import Any, Callable

from PIL import Image


class AnalysisDetector:
    def __init__(
        self,
        model_provider: Callable[[], Any],
        result_converter: Callable[..., list[dict[str, Any]]],
        allowed_classes: frozenset[str],
    ) -> None:
        self._model_provider = model_provider
        self._result_converter = result_converter
        self._allowed_classes = allowed_classes

    def detect(self, image: Image.Image, confidence: float = 0.25) -> list[dict[str, Any]]:
        result = self._model_provider()(image, verbose=False, conf=confidence)[0]
        return self._result_converter(result, allowed_class_names=self._allowed_classes)


def _intersection_area(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def suppress_duplicate_trains(
    trains: list[dict[str, Any]],
    iou_threshold: float = 0.62,
    containment_threshold: float = 0.86,
) -> list[dict[str, Any]]:
    """Suppress overlapping train boxes while keeping distinct locomotives."""
    ordered = sorted(trains, key=lambda row: float(row.get("conf", 0.0)), reverse=True)
    kept: list[dict[str, Any]] = []
    for candidate in ordered:
        box = list(map(float, candidate.get("box", [])))
        if len(box) != 4:
            continue
        area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        duplicate = False
        for existing in kept:
            other = list(map(float, existing.get("box", [])))
            other_area = max(0.0, other[2] - other[0]) * max(0.0, other[3] - other[1])
            intersection = _intersection_area(box, other)
            union = area + other_area - intersection
            iou = intersection / union if union else 0.0
            containment = intersection / max(1e-6, min(area, other_area))
            if iou >= iou_threshold or containment >= containment_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def relative_box(box: list[float], train_box: list[float]) -> list[float]:
    width = max(1e-6, train_box[2] - train_box[0])
    height = max(1e-6, train_box[3] - train_box[1])
    return [
        (box[0] - train_box[0]) / width,
        (box[1] - train_box[1]) / height,
        (box[2] - train_box[0]) / width,
        (box[3] - train_box[1]) / height,
    ]


def link_numbers_to_trains(
    trains: list[dict[str, Any]], numbers: list[dict[str, Any]]
) -> tuple[dict[int, list[dict[str, Any]]], list[str]]:
    linked = {index: [] for index in range(len(trains))}
    warnings: list[str] = []
    for number in numbers:
        number_box = list(map(float, number.get("box", [])))
        if len(number_box) != 4:
            continue
        cx = (number_box[0] + number_box[2]) / 2.0
        cy = (number_box[1] + number_box[3]) / 2.0
        number_area = max(1e-6, (number_box[2] - number_box[0]) * (number_box[3] - number_box[1]))
        candidates: list[tuple[int, float, float, int]] = []
        for index, train in enumerate(trains):
            train_box = list(map(float, train.get("box", [])))
            if len(train_box) != 4:
                continue
            contains_center = int(train_box[0] <= cx <= train_box[2] and train_box[1] <= cy <= train_box[3])
            containment = _intersection_area(number_box, train_box) / number_area
            tx = (train_box[0] + train_box[2]) / 2.0
            ty = (train_box[1] + train_box[3]) / 2.0
            distance = hypot(cx - tx, cy - ty)
            if contains_center or containment >= 0.5:
                candidates.append((contains_center, containment, -distance, index))
        if not candidates:
            warnings.append("Номерной фрагмент не связан с составом")
            continue
        candidates.sort(reverse=True)
        chosen = candidates[0][3]
        enriched = dict(number)
        enriched["relative_box"] = relative_box(number_box, list(map(float, trains[chosen]["box"])))
        linked[chosen].append(enriched)
        if len(candidates) > 1 and candidates[0][:2] == candidates[1][:2]:
            warnings.append("Неоднозначная связь номерного фрагмента с составом")
    return linked, list(dict.fromkeys(warnings))
