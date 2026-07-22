from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class SeriesReference:
    def __init__(self, path: str | Path | None = None) -> None:
        source = Path(path) if path else Path(__file__).with_name("locomotive_series.json")
        payload = json.loads(source.read_text(encoding="utf-8"))
        self.version = str(payload.get("version", "unknown"))
        self.rules: dict[str, dict[str, Any]] = payload.get("series", {})
        context_source = source.with_name("locomotive_number_context.json")
        context_payload = json.loads(context_source.read_text(encoding="utf-8"))
        self.context_version = str(context_payload.get("version", "unknown"))
        self.ocr_exact_aliases = {
            str(observed).upper().replace(" ", ""): str(series).upper().replace(" ", "")
            for observed, series in context_payload.get("ocr_exact_aliases", {}).items()
            if observed and series
        }
        self.context_rules: dict[str, dict[str, Any]] = context_payload.get("series", {})
        self.known_series_names = tuple(
            sorted(self.context_rules, key=lambda value: (len(value), value), reverse=True)
        )
        self.serial_digit_range = (
            int(context_payload.get("serial_digits_min", 2)),
            int(context_payload.get("serial_digits_max", 5)),
        )
        self._aliases: list[tuple[str, str]] = []
        for canonical, rule in self.rules.items():
            for alias in rule.get("aliases", [canonical]):
                self._aliases.append((str(alias).upper().replace(" ", ""), canonical))
        self._aliases.sort(key=lambda item: len(item[0]), reverse=True)

    def canonicalize_prefix(self, text: str) -> str:
        compact = text.upper().replace(" ", "")
        for alias, canonical in self._aliases:
            if compact.startswith(alias):
                return canonical + compact[len(alias) :]
        return compact

    def match_known_series(self, text: str | None) -> str | None:
        if not text:
            return None
        compact = re.sub(r"[^0-9А-Я]", "", self.canonicalize_prefix(text).replace("Ё", "Е"))
        minimum, maximum = self.serial_digit_range
        for series in self.known_series_names:
            if compact == series:
                return series
            if compact.startswith(series):
                remainder = compact[len(series) :]
                if remainder.isdigit() and minimum <= len(remainder) <= maximum:
                    return series
        return None

    def match_series(self, text: str | None) -> str | None:
        if not text:
            return None
        canonicalized = self.canonicalize_prefix(text)
        known = self.match_known_series(canonicalized)
        if known:
            return known
        ordered = sorted(
            self.rules.items(),
            key=lambda item: (int(item[1].get("priority", 0)), len(item[0])),
            reverse=True,
        )
        for canonical, rule in ordered:
            patterns = list(rule.get("series_patterns", [])) + list(
                rule.get("full_patterns", [])
            )
            if any(re.match(pattern, canonicalized) for pattern in patterns):
                return canonical
            prefix = rule.get("family_prefix")
            if prefix and canonicalized.startswith(str(prefix)):
                return canonicalized.split("-", 1)[0]
        return None

    def rule_for(self, series: str | None) -> tuple[str | None, dict[str, Any] | None]:
        if not series:
            return None, None
        canonicalized = self.canonicalize_prefix(series).split("-", 1)[0]
        context_rule = self.context_rules.get(canonicalized)
        if context_rule is not None:
            return canonicalized, context_rule
        if canonicalized in self.rules:
            return canonicalized, self.rules[canonicalized]
        ordered = sorted(
            self.rules.items(),
            key=lambda item: (int(item[1].get("priority", 0)), len(item[0])),
            reverse=True,
        )
        for canonical, rule in ordered:
            prefix = rule.get("family_prefix")
            if prefix and canonicalized.startswith(str(prefix)):
                return canonicalized, rule
        return canonicalized, None

    def section_estimate(self, series: str | None) -> tuple[int | None, str]:
        canonical, rule = self.rule_for(series)
        if rule and rule.get("default_section_count") is not None:
            source = "exact_series" if canonical in self.rules or canonical in self.context_rules else "series_family"
            return int(rule["default_section_count"]), source
        if rule and "sections" in rule:
            return (int(rule["sections"]), "context_catalog") if rule["sections"] is not None else (None, "context_catalog")
        if canonical and canonical.startswith("ВЛ"):
            return 2, "vl_family_fallback"
        match = re.match(r"^([23])ЭС", canonical or "")
        if match:
            return int(match.group(1)), "name_prefix_fallback"
        return None, "unknown"

    def join(self, series: str, serial: str) -> str:
        canonical, rule = self.rule_for(series)
        template = str((rule or {}).get("join_template", "{series}-{serial}"))
        return template.format(series=canonical or series, serial=serial)
