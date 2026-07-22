from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..series_reference import SeriesReference


_LATIN_TO_RUSSIAN = str.maketrans(
    {
        "A": "А", "B": "В", "C": "С", "D": "Д", "E": "Е", "F": "Г",
        "G": "Г", "H": "Н", "I": "И", "J": "У", "K": "К", "L": "Л",
        "M": "М", "N": "Н", "O": "О", "P": "Р", "Q": "О", "R": "Р",
        "S": "С", "T": "Т", "U": "И", "V": "В", "W": "Ш", "X": "Х",
        "Y": "У", "Z": "З",
    }
)
_DIGITISH = str.maketrans(
    {"О": "0", "З": "3", "Ч": "4", "С": "5", "Б": "6", "Т": "7", "В": "8", "И": "1", "Л": "1"}
)
_LOW_COST_PAIRS = {
    frozenset(pair)
    for pair in (
        ("В", "8"), ("О", "0"), ("З", "3"), ("Ч", "4"), ("С", "5"), ("С", "0"),
        ("Б", "6"), ("Т", "7"), ("Л", "1"), ("И", "1"), ("Е", "Э"), ("Э", "З"),
        ("П", "Р"), ("Н", "П"), ("К", "Х"), ("М", "Н"),
    )
}


@dataclass(frozen=True, slots=True)
class ContextCandidate:
    series: str
    score: float


@dataclass(frozen=True, slots=True)
class ContextDecodeResult:
    normalized_text: str | None
    series_part: str | None
    serial_part: str | None
    alternatives: tuple[str, ...]
    context_applied: bool
    context_score: float


def russian_alnum(value: str) -> str:
    value = str(value or "").upper().replace("Ё", "Е").translate(_LATIN_TO_RUSSIAN)
    return "".join(character for character in value if character.isdigit() or "А" <= character <= "Я")


def digitish_to_digits(value: str) -> str | None:
    converted = russian_alnum(value).translate(_DIGITISH)
    return converted if converted.isdigit() else None


def _substitution_cost(left: str, right: str) -> float:
    if left == right:
        return 0.0
    if frozenset((left, right)) in _LOW_COST_PAIRS:
        return 0.24
    return 1.0


def _distance(left: str, right: str) -> float:
    previous = [float(index) for index in range(len(right) + 1)]
    for left_index, left_character in enumerate(left, start=1):
        current = [float(left_index)]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1.0,
                    previous[right_index] + 1.0,
                    previous[right_index - 1] + _substitution_cost(left_character, right_character),
                )
            )
        previous = current
    return previous[-1]


def series_candidates(
    observed: str,
    reference: "SeriesReference",
    limit: int = 4,
) -> list[ContextCandidate]:
    cleaned = russian_alnum(observed)
    if not cleaned:
        return []
    candidates: list[ContextCandidate] = []
    for series in reference.known_series_names:
        distance = _distance(cleaned, series)
        score = max(0.0, 1.0 - distance / max(len(cleaned), len(series), 1))
        threshold = 0.76 if len(series) <= 3 else 0.70 if len(series) <= 5 else 0.66
        if abs(len(cleaned) - len(series)) <= 2 and score >= threshold:
            candidates.append(ContextCandidate(series, round(score, 6)))
    candidates.sort(key=lambda item: (item.score, len(item.series)), reverse=True)
    return candidates[: max(1, limit)]


def decode_locomotive_number(raw_text: str, reference: "SeriesReference") -> ContextDecodeResult:
    compact_source = str(raw_text or "").upper().replace("Ё", "Е").replace(" ", "")
    canonical_source = reference.canonicalize_prefix(compact_source)
    compact = russian_alnum(canonical_source)
    if not compact:
        return ContextDecodeResult(None, None, None, (), False, 0.0)

    minimum, maximum = reference.serial_digit_range
    hypotheses: list[tuple[float, int, str, str]] = []
    exact_alias = reference.ocr_exact_aliases.get(compact)
    if exact_alias in reference.known_series_names:
        return ContextDecodeResult(
            exact_alias,
            exact_alias,
            None,
            (exact_alias,),
            True,
            1.0,
        )

    for serial_length in range(minimum, min(maximum, len(compact) - 1) + 1):
        serial_source = compact[-serial_length:]
        serial = digitish_to_digits(serial_source)
        if not serial:
            continue
        converted_serial_characters = sum(not character.isdigit() for character in serial_source)
        prefix = compact[:-serial_length]
        for candidate in series_candidates(prefix, reference):
            effective_score = candidate.score - converted_serial_characters * 0.18
            hypotheses.append((effective_score, len(candidate.series), candidate.series, serial))
    hypotheses.sort(reverse=True)
    series_only = series_candidates(compact, reference)
    if hypotheses:
        score, _length, series, serial = hypotheses[0]
        if compact.isdigit() and series_only and series_only[0].score >= score + 0.05:
            best = series_only[0]
            return ContextDecodeResult(
                best.series,
                best.series,
                None,
                tuple(candidate.series for candidate in series_only),
                True,
                best.score,
            )

        alternatives = tuple(
            dict.fromkeys(reference.join(candidate_series, candidate_serial) for _score, _size, candidate_series, candidate_serial in hypotheses[:4])
        )
        normalized = reference.join(series, serial)
        exact_compact = russian_alnum(normalized) == compact
        return ContextDecodeResult(normalized, series, serial, alternatives, not exact_compact, score)

    if series_only:
        best = series_only[0]
        exact = compact == best.series
        return ContextDecodeResult(
            best.series,
            best.series,
            None,
            tuple(candidate.series for candidate in series_only),
            not exact,
            best.score,
        )

    serial = digitish_to_digits(compact)
    if serial and minimum <= len(serial) <= maximum:
        return ContextDecodeResult(serial, None, serial, (), False, 1.0)

    return ContextDecodeResult(compact, None, None, (), False, 0.0)
