from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Any

from ..models import new_id
from ..ocr.fusion import OCRFusionService


def _center(box: list[float]) -> tuple[float, float]:
    return (float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0


@dataclass(slots=True)
class _Track:
    id: str
    event_id: str
    observations: list[dict[str, Any]] = field(default_factory=list)


class NumberFragmentTracker:
    def __init__(self, event_id: str, max_center_distance: float = 0.22) -> None:
        self.event_id = event_id
        self.max_center_distance = max_center_distance
        self._tracks: list[_Track] = []
        self._fusion = OCRFusionService()

    @staticmethod
    def _compatible(track: _Track, observation: dict[str, Any]) -> float | None:
        if not track.observations:
            return None
        previous = track.observations[-1]
        px, py = _center(previous.get("relative_bbox", [0, 0, 0, 0]))
        ox, oy = _center(observation.get("relative_bbox", [0, 0, 0, 0]))
        distance = hypot(px - ox, py - oy)
        previous_text = previous.get("normalized_text")
        current_text = observation.get("normalized_text")
        same_text = bool(previous_text and current_text and previous_text == current_text)
        same_type = previous.get("fragment_type") == observation.get("fragment_type")
        if not same_text and not same_type:
            return None
        return distance * (0.45 if same_text else 1.0)

    def update(self, observation: dict[str, Any]) -> str:
        candidates: list[tuple[float, _Track]] = []
        for track in self._tracks:
            score = self._compatible(track, observation)
            if score is not None and score <= self.max_center_distance:
                candidates.append((score, track))
        if candidates:
            track = min(candidates, key=lambda row: row[0])[1]
        else:
            track = _Track(new_id("track"), self.event_id)
            self._tracks.append(track)
        observation["fragment_track_id"] = track.id
        track.observations.append(observation)
        return track.id

    def to_dicts(self) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for track in self._tracks:
            candidates = self._fusion.fuse(track.observations)
            best = candidates[0] if candidates else None
            best_observation = max(
                track.observations,
                key=lambda row: (
                    float(row.get("quality_score", 0.0)) * 0.35
                    + float(row.get("ocr_confidence", 0.0)) * 0.65
                ),
                default={},
            )
            payload.append(
                {
                    "id": track.id,
                    "event_id": self.event_id,
                    "observations": [str(row.get("id")) for row in track.observations],
                    "fragment_type": str(best_observation.get("fragment_type", "unknown_fragment")),
                    "best_text": best.get("text") if best else None,
                    "candidates": candidates,
                    "relative_position": list(best_observation.get("relative_bbox", [])),
                    "best_crop_asset_id": best_observation.get("crop_asset_id"),
                    "confidence": float(best.get("score", 0.0)) if best else 0.0,
                    "observation_count": len(track.observations),
                    "context_applied": bool(best_observation.get("context_applied", False)),
                    "context_score": float(best_observation.get("context_score", 0.0)),
                    "context_alternatives": list(best_observation.get("context_alternatives", [])),
                }
            )
        return payload
