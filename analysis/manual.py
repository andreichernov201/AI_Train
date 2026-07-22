from __future__ import annotations

from copy import deepcopy
from typing import Any

from .models import new_id, utc_now
from .series_reference import SeriesReference


EDITABLE_IDENTITY_FIELDS = frozenset(
    {
        "recognized_number",
        "recognized_series",
        "serial_number",
        "section_count",
        "status",
        "manually_confirmed",
    }
)


def patch_event_payload(
    event: dict[str, Any], patch: dict[str, Any], reference: SeriesReference
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = deepcopy(event)
    action = str(patch.get("action", "update"))
    change = {"action": action, "at": utc_now(), "payload": deepcopy(patch)}
    identities = list(updated.get("locomotive_identities", []))

    if action == "confirm_event":
        updated["status"] = "confirmed"
        for identity in identities:
            identity["manually_confirmed"] = True
            identity["locked"] = True
            identity["source"] = "manual"
    elif action == "add_identity":
        text = str(patch.get("recognized_number") or "—")
        series = str(patch.get("recognized_series") or reference.match_series(text) or "") or None
        sections, source = reference.section_estimate(series)
        identities.append(
            {
                "id": new_id("identity"),
                "event_id": updated["id"],
                "fragment_track_ids": [],
                "recognized_number": text,
                "recognized_series": series,
                "serial_number": patch.get("serial_number"),
                "number_candidates": [],
                "assembly_confidence": 1.0,
                "ocr_confidence": 0.0,
                "section_count": patch.get("section_count", sections),
                "section_source": "manual" if "section_count" in patch else source,
                "observation_count": 0,
                "best_frame_asset_id": updated.get("best_frame_asset_id"),
                "best_crop_asset_ids": [],
                "status": "manual",
                "warnings": [],
                "manually_confirmed": True,
                "locked": True,
                "source": "manual",
            }
        )
    elif action == "delete_identity":
        identity_id = str(patch.get("identity_id", ""))
        identities = [identity for identity in identities if str(identity.get("id")) != identity_id]
    else:
        identity_id = str(patch.get("identity_id", ""))
        for identity in identities:
            if str(identity.get("id")) != identity_id:
                continue
            values = patch.get("values", patch)
            for key in EDITABLE_IDENTITY_FIELDS:
                if key in values:
                    identity[key] = values[key]
            if "recognized_number" in values and "recognized_series" not in values:
                identity["recognized_series"] = reference.match_series(identity.get("recognized_number"))
            if "section_count" in values:
                identity["section_source"] = "manual"
            identity["source"] = "manual"
            identity["locked"] = True
            identity["manually_confirmed"] = bool(values.get("manually_confirmed", True))
            identity["status"] = str(values.get("status", "manual"))
            break

    updated["locomotive_identities"] = identities
    updated.setdefault("manual_changes", []).append(change)
    updated["metadata"] = dict(updated.get("metadata", {})) | {"last_manual_change_at": change["at"]}
    return updated, change


def merge_fragments_payload(
    event: dict[str, Any], fragment_track_ids: list[str], reference: SeriesReference, text: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = deepcopy(event)
    track_ids = list(dict.fromkeys(str(value) for value in fragment_track_ids if value))
    tracks = [track for track in updated.get("fragment_tracks", []) if str(track.get("id")) in track_ids]
    values = [str(track.get("best_text")) for track in tracks if track.get("best_text")]
    if not text:
        series = next((value for value in values if reference.match_series(value) and not value.isdigit()), None)
        serial = next((value for value in values if value.isdigit()), None)
        text = reference.join(series, serial) if series and serial else "-".join(values) or "—"
    series = reference.match_series(text)
    serial = text.rsplit("-", 1)[1] if "-" in text and text.rsplit("-", 1)[1].isdigit() else None
    sections, section_source = reference.section_estimate(series)
    identity = {
        "id": new_id("identity"),
        "event_id": updated["id"],
        "fragment_track_ids": track_ids,
        "recognized_number": text,
        "recognized_series": series,
        "serial_number": serial,
        "number_candidates": [{"text": text, "score": 1.0, "source": "manual"}],
        "assembly_confidence": 1.0,
        "ocr_confidence": sum(float(track.get("confidence", 0.0)) for track in tracks) / max(1, len(tracks)),
        "section_count": sections,
        "section_source": section_source,
        "observation_count": sum(int(track.get("observation_count", 1)) for track in tracks),
        "best_frame_asset_id": updated.get("best_frame_asset_id"),
        "best_crop_asset_ids": [track.get("best_crop_asset_id") for track in tracks if track.get("best_crop_asset_id")],
        "status": "manual",
        "warnings": [],
        "manually_confirmed": True,
        "locked": True,
        "source": "manual",
    }
    existing = [
        item for item in updated.get("locomotive_identities", [])
        if not set(map(str, item.get("fragment_track_ids", []))) & set(track_ids)
    ]
    existing.append(identity)
    updated["locomotive_identities"] = existing
    change = {"action": "merge_fragments", "at": utc_now(), "payload": {"fragment_track_ids": track_ids, "identity_id": identity["id"], "text": text}}
    updated.setdefault("manual_changes", []).append(change)
    return updated, change


def split_identity_payload(
    event: dict[str, Any], identity_id: str, groups: list[list[str]], reference: SeriesReference
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = deepcopy(event)
    identities = list(updated.get("locomotive_identities", []))
    original = next((item for item in identities if str(item.get("id")) == str(identity_id)), None)
    if not original:
        raise KeyError(identity_id)
    identities = [item for item in identities if str(item.get("id")) != str(identity_id)]
    for group in groups:
        partial, _change = merge_fragments_payload(
            {**updated, "locomotive_identities": identities}, group, reference
        )
        identities = partial["locomotive_identities"]
    updated["locomotive_identities"] = identities
    change = {"action": "split_identity", "at": utc_now(), "payload": {"identity_id": identity_id, "groups": groups}}
    updated.setdefault("manual_changes", []).append(change)
    return updated, change
