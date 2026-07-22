from __future__ import annotations

from math import hypot
from typing import Any

from ..models import LocomotiveIdentity, new_id
from ..series_reference import SeriesReference


def _center(box: list[float]) -> tuple[float, float]:
    return (float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0


def _geometry_score(a: dict[str, Any], b: dict[str, Any]) -> float | None:
    box_a = list(a.get("relative_bbox") or a.get("relative_position") or [])
    box_b = list(b.get("relative_bbox") or b.get("relative_position") or [])
    if len(box_a) != 4 or len(box_b) != 4:
        return None
    ax, ay = _center(box_a)
    bx, by = _center(box_b)
    distance = hypot(ax - bx, ay - by)
    width_a, height_a = max(1e-6, box_a[2] - box_a[0]), max(1e-6, box_a[3] - box_a[1])
    width_b, height_b = max(1e-6, box_b[2] - box_b[0]), max(1e-6, box_b[3] - box_b[1])
    scale_ratio = min(height_a, height_b) / max(height_a, height_b)
    horizontal = abs(ay - by) <= max(height_a, height_b) * 2.2
    vertical = abs(ax - bx) <= max(width_a, width_b) * 1.7
    if distance > 0.58 or scale_ratio < 0.28 or not (horizontal or vertical):
        return None
    alignment = 1.0 if horizontal and vertical else 0.78
    return max(0.0, min(1.0, (1.0 - distance / 0.58) * 0.7 + scale_ratio * 0.2 + alignment * 0.1))


class NumberFragmentAssembler:
    def __init__(self, reference: SeriesReference) -> None:
        self.reference = reference

    @staticmethod
    def _track_type(track: dict[str, Any]) -> str:
        return str(track.get("fragment_type", "unknown_fragment"))

    @staticmethod
    def _track_text(track: dict[str, Any]) -> str | None:
        value = track.get("best_text") or track.get("normalized_text")
        return str(value) if value else None

    def _identity(
        self,
        event_id: str,
        tracks: list[dict[str, Any]],
        recognized_number: str | None,
        assembly_confidence: float,
        warnings: list[str] | None = None,
    ) -> LocomotiveIdentity:
        series = self.reference.match_series(recognized_number)
        serial_match = None
        if recognized_number and "-" in recognized_number:
            serial_match = recognized_number.rsplit("-", 1)[1]
            if serial_match == "—":
                serial_match = None
        section_count, section_source = self.reference.section_estimate(series)
        ocr_confidence = sum(float(track.get("confidence", 0.0)) for track in tracks) / max(1, len(tracks))
        partial = not recognized_number or "—" in recognized_number or not series or not serial_match
        status = "partial_number" if partial else "confirmed" if assembly_confidence >= 0.72 else "low_confidence"
        return LocomotiveIdentity(
            id=new_id("identity"),
            event_id=event_id,
            fragment_track_ids=[str(track.get("id")) for track in tracks if track.get("id")],
            recognized_number=recognized_number,
            recognized_series=series,
            serial_number=serial_match,
            number_candidates=[{"text": recognized_number, "score": round(assembly_confidence, 6)}]
            if recognized_number
            else [],
            assembly_confidence=round(assembly_confidence, 6),
            ocr_confidence=round(ocr_confidence, 6),
            section_count=section_count,
            section_source=section_source,
            observation_count=sum(int(track.get("observation_count", 1)) for track in tracks),
            best_crop_asset_ids=[
                str(track.get("best_crop_asset_id"))
                for track in tracks
                if track.get("best_crop_asset_id")
            ],
            status=status,
            warnings=list(warnings or []),
        )

    def assemble(
        self,
        event_id: str,
        tracks: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
        identities: list[LocomotiveIdentity] = []
        hypotheses: list[dict[str, Any]] = []
        warnings: list[str] = []
        used: set[str] = set()

        full_tracks = [track for track in tracks if self._track_type(track) == "full_identifier"]
        seen_full: dict[str, LocomotiveIdentity] = {}
        for track in full_tracks:
            text = self._track_text(track)
            if not text:
                continue
            if text in seen_full:
                seen_full[text].fragment_track_ids.append(str(track.get("id")))
                seen_full[text].observation_count += int(track.get("observation_count", 1))
            else:
                identity = self._identity(event_id, [track], text, float(track.get("confidence", 0.0)))
                seen_full[text] = identity
                identities.append(identity)
            used.add(str(track.get("id")))

        series_tracks = [
            track for track in tracks if self._track_type(track) == "series_fragment" and str(track.get("id")) not in used
        ]
        serial_tracks = [
            track for track in tracks if self._track_type(track) == "serial_fragment" and str(track.get("id")) not in used
        ]
        pairs: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        for series in series_tracks:
            for serial in serial_tracks:
                geometry = _geometry_score(series, serial)
                if geometry is None:
                    continue
                score = geometry * 0.55 + min(
                    float(series.get("confidence", 0.0)), float(serial.get("confidence", 0.0))
                ) * 0.45
                pairs.append((score, series, serial))
        pairs.sort(key=lambda row: row[0], reverse=True)

        for score, series, serial in pairs:
            series_id, serial_id = str(series.get("id")), str(serial.get("id"))
            if series_id in used or serial_id in used:
                continue
            series_text, serial_text = self._track_text(series), self._track_text(serial)
            if not series_text or not serial_text:
                continue
            joined = self.reference.join(series_text, serial_text)
            competing = [
                row for row in pairs if row[1] is series or row[2] is serial
            ]
            pair_warnings: list[str] = []
            if len(competing) > 2 or score < 0.62:
                pair_warnings.append("Неоднозначное объединение номерных фрагментов")
                warnings.extend(pair_warnings)
            hypotheses.append(
                {
                    "fragment_track_ids": [series_id, serial_id],
                    "assembled_text": joined,
                    "score": round(score, 6),
                    "geometry": "horizontal_or_vertical",
                    "selected": True,
                }
            )
            identities.append(self._identity(event_id, [series, serial], joined, score, pair_warnings))
            used.update({series_id, serial_id})

        for track in tracks:
            track_id = str(track.get("id"))
            if track_id in used:
                continue
            text = self._track_text(track)
            fragment_type = self._track_type(track)
            if fragment_type == "series_fragment" and text:
                identities.append(self._identity(event_id, [track], f"{text}-—", float(track.get("confidence", 0.0))))
            elif fragment_type == "serial_fragment" and text:
                identities.append(self._identity(event_id, [track], f"—-{text}", float(track.get("confidence", 0.0))))
            else:
                identities.append(self._identity(event_id, [track], "—", float(track.get("confidence", 0.0))))
            used.add(track_id)

        if not identities:
            identities.append(
                self._identity(
                    event_id,
                    [],
                    "—",
                    0.0,
                    ["Состав обнаружен, но номерные зоны не найдены"],
                )
            )
            identities[-1].status = "number_not_detected"
        return [identity.to_dict() for identity in identities], hypotheses, list(dict.fromkeys(warnings))
