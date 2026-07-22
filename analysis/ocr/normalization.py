from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..series_reference import SeriesReference
from .context_decoder import decode_locomotive_number


@dataclass(slots=True)
class NormalizationResult:
    raw_text: str
    normalized_text: str | None
    series_part: str | None = None
    serial_part: str | None = None
    alternatives: list[str] = field(default_factory=list)
    context_applied: bool = False
    context_score: float = 0.0


def normalize_ocr_text(raw_text: str, reference: SeriesReference) -> NormalizationResult:
    original = str(raw_text or "")
    decoded = decode_locomotive_number(original, reference)
    return NormalizationResult(
        raw_text=original,
        normalized_text=decoded.normalized_text,
        series_part=decoded.series_part,
        serial_part=decoded.serial_part,
        alternatives=list(decoded.alternatives),
        context_applied=decoded.context_applied,
        context_score=decoded.context_score,
    )
