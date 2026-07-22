from __future__ import annotations

import re

from ..series_reference import SeriesReference


FRAGMENT_TYPES = frozenset(
    {
        "full_identifier",
        "series_fragment",
        "serial_fragment",
        "mixed_fragment",
        "unknown_fragment",
        "unreadable_fragment",
    }
)


def classify_fragment(text: str | None, reference: SeriesReference) -> str:
    if not text or text in {"—", "-"}:
        return "unreadable_fragment"
    value = str(text).upper().replace(" ", "")
    if (
        re.match(r"^.+-\d{2,5}$", value)
        and re.search(r"[A-ZА-Я]", value)
        and reference.match_known_series(value)
    ):
        return "full_identifier"
    if reference.match_known_series(value) and re.search(r"[A-ZА-Я]", value):
        if re.search(r"-\d{2,5}$", value):
            return "full_identifier"
        return "series_fragment"
    if re.fullmatch(r"\d{2,5}", value):
        return "serial_fragment"
    if re.search(r"[A-ZА-Я]", value) and re.search(r"\d", value):
        return "mixed_fragment"
    return "unknown_fragment"
