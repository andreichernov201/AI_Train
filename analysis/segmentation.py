from __future__ import annotations

from math import hypot
from typing import Any, Callable

from PIL import Image


class AnalysisSegmenter:
    def __init__(
        self,
        predictor: Callable[[Image.Image], list[dict[str, Any]]],
        model_name: str = "active segmentation model",
    ) -> None:
        self._predictor = predictor
        self.model_name = model_name

    def segment(self, image: Image.Image) -> list[dict[str, Any]]:
        return self._predictor(image)

    def status(self) -> dict[str, Any]:
        return {
            "provider": "ultralytics",
            "model": self.model_name,
            "classes": ["body", "autocoupler", "axlebox", "bogie", "hose"],
        }


def _intersection_area(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def link_segments_to_trains(
    trains: list[dict[str, Any]],
    segments: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    linked = {index: [] for index in range(len(trains))}
    for segment in segments:
        box = list(map(float, segment.get("box", [])))
        polygon = segment.get("segment")
        if len(box) != 4 or not isinstance(polygon, list) or len(polygon) < 3:
            continue
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        area = max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))
        candidates: list[tuple[int, float, float, int]] = []
        for index, train in enumerate(trains):
            train_box = list(map(float, train.get("box", [])))
            if len(train_box) != 4:
                continue
            contains_center = int(train_box[0] <= cx <= train_box[2] and train_box[1] <= cy <= train_box[3])
            containment = _intersection_area(box, train_box) / area
            tx, ty = (train_box[0] + train_box[2]) / 2.0, (train_box[1] + train_box[3]) / 2.0
            distance = hypot(cx - tx, cy - ty)
            if contains_center or containment >= 0.45:
                candidates.append((contains_center, containment, -distance, index))
        if candidates:
            candidates.sort(reverse=True)
            linked[candidates[0][3]].append(dict(segment))
    return linked
