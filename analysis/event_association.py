from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Any

from .models import new_id


def box_iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


@dataclass(slots=True)
class _ActiveEvent:
    id: str
    box: list[float]
    last_timestamp_ms: int
    last_frame_index: int


class EventAssociator:
    def __init__(self, timeout_ms: int = 2500) -> None:
        self.timeout_ms = timeout_ms
        self._active: dict[str, _ActiveEvent] = {}

    @staticmethod
    def _match_score(previous: list[float], current: list[float]) -> float:
        iou = box_iou(previous, current)
        pcx, pcy = (previous[0] + previous[2]) / 2.0, (previous[1] + previous[3]) / 2.0
        ccx, ccy = (current[0] + current[2]) / 2.0, (current[1] + current[3]) / 2.0
        norm = max(1.0, previous[2] - previous[0], previous[3] - previous[1])
        motion = hypot(pcx - ccx, pcy - ccy) / norm
        area_previous = max(1.0, (previous[2] - previous[0]) * (previous[3] - previous[1]))
        area_current = max(1.0, (current[2] - current[0]) * (current[3] - current[1]))
        area_similarity = min(area_previous, area_current) / max(area_previous, area_current)
        return iou * 0.55 + max(0.0, 1.0 - motion) * 0.25 + area_similarity * 0.20

    def associate(self, detections: list[dict[str, Any]], timestamp_ms: int, frame_index: int) -> list[str]:
        assignments: list[str] = []
        claimed: set[str] = set()
        for detection in detections:
            box = list(map(float, detection.get("box", [0, 0, 0, 0])))
            candidates = [
                (self._match_score(active.box, box), event_id)
                for event_id, active in self._active.items()
                if event_id not in claimed and timestamp_ms - active.last_timestamp_ms <= self.timeout_ms
            ]
            candidates = [row for row in candidates if row[0] >= 0.30]
            if candidates:
                event_id = max(candidates)[1]
                active = self._active[event_id]
                active.box = box
                active.last_timestamp_ms = timestamp_ms
                active.last_frame_index = frame_index
            else:
                event_id = new_id("event")
                self._active[event_id] = _ActiveEvent(event_id, box, timestamp_ms, frame_index)
            claimed.add(event_id)
            assignments.append(event_id)
        return assignments

    def expire(self, timestamp_ms: int) -> list[str]:
        expired = [
            event_id
            for event_id, active in self._active.items()
            if timestamp_ms - active.last_timestamp_ms > self.timeout_ms
        ]
        for event_id in expired:
            self._active.pop(event_id, None)
        return expired

    def finish_all(self) -> list[str]:
        event_ids = list(self._active)
        self._active.clear()
        return event_ids
