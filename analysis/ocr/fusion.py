from __future__ import annotations

from collections import defaultdict
from typing import Any


class OCRFusionService:
    def fuse(self, observations: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for observation in observations:
            text = observation.get("normalized_text")
            if text:
                grouped[str(text)].append(observation)
        candidates: list[dict[str, Any]] = []
        for text, rows in grouped.items():
            frames = {
                (row.get("source_file_id"), row.get("frame_index"), row.get("timestamp_ms"))
                for row in rows
            }
            confidences = [float(row.get("ocr_confidence", 0.0)) for row in rows]
            qualities = [float(row.get("quality_score", 0.0)) for row in rows]
            independent = max(1, len(frames))
            avg_conf = sum(confidences) / max(1, len(confidences))
            max_conf = max(confidences, default=0.0)
            avg_quality = sum(qualities) / max(1, len(qualities))
            stability = min(1.0, independent / 3.0)
            score = min(1.0, avg_conf * 0.42 + max_conf * 0.18 + avg_quality * 0.15 + stability * 0.25)
            candidates.append(
                {
                    "text": text,
                    "score": round(score, 6),
                    "frame_count": independent,
                    "observation_count": len(rows),
                    "average_confidence": round(avg_conf, 6),
                    "max_confidence": round(max_conf, 6),
                    "observation_ids": [str(row.get("id")) for row in rows],
                }
            )
        return sorted(candidates, key=lambda row: (row["score"], row["frame_count"]), reverse=True)[:limit]

    def choose(self, observations: list[dict[str, Any]], locked_text: str | None = None) -> dict[str, Any] | None:
        if locked_text:
            return {"text": locked_text, "score": 1.0, "source": "manual", "locked": True}
        candidates = self.fuse(observations)
        return candidates[0] if candidates else None
