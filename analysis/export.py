from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path
from typing import Any

from .storage import AnalysisStorage


def session_payload(storage: AnalysisStorage, session_id: str) -> dict[str, Any]:
    session = storage.get_session(session_id)
    if not session:
        raise KeyError(session_id)
    events = storage.list_events(session_id)
    for event in events:
        event["manual_changes"] = storage.list_manual_changes(event["id"])
    return {
        "schema_version": 1,
        "session": session,
        "files": storage.list_files(session_id),
        "events": events,
    }


def export_json(storage: AnalysisStorage, session_id: str) -> bytes:
    return json.dumps(session_payload(storage, session_id), ensure_ascii=False, indent=2).encode("utf-8")


def export_csv(storage: AnalysisStorage, session_id: str) -> bytes:
    payload = session_payload(storage, session_id)
    files = {item["id"]: item for item in payload["files"]}
    output = io.StringIO(newline="")
    fieldnames = [
        "event_id",
        "locomotive_id",
        "date",
        "file",
        "start_time",
        "end_time",
        "recognized_number",
        "series",
        "serial_number",
        "source_fragments",
        "section_count",
        "ocr_confidence",
        "assembly_confidence",
        "frame_count",
        "manually_confirmed",
        "status",
        "warnings",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\r\n")
    writer.writeheader()
    for event in payload["events"]:
        file_row = files.get(event.get("source_file_id"), {})
        for identity in event.get("locomotive_identities", []):
            writer.writerow(
                {
                    "event_id": event.get("id"),
                    "locomotive_id": identity.get("id"),
                    "date": event.get("started_at"),
                    "file": file_row.get("original_name", ""),
                    "start_time": event.get("metadata", {}).get("start_timestamp_ms", ""),
                    "end_time": event.get("metadata", {}).get("end_timestamp_ms", ""),
                    "recognized_number": identity.get("recognized_number") or "",
                    "series": identity.get("recognized_series") or "",
                    "serial_number": identity.get("serial_number") or "",
                    "source_fragments": " + ".join(
                        track.get("best_text") or "—"
                        for track in event.get("fragment_tracks", [])
                        if str(track.get("id")) in set(map(str, identity.get("fragment_track_ids", [])))
                    ),
                    "section_count": identity.get("section_count") if identity.get("section_count") is not None else "",
                    "ocr_confidence": identity.get("ocr_confidence", 0),
                    "assembly_confidence": identity.get("assembly_confidence", 0),
                    "frame_count": event.get("metadata", {}).get("frame_count", 1),
                    "manually_confirmed": identity.get("manually_confirmed", False),
                    "status": identity.get("status", ""),
                    "warnings": " | ".join(identity.get("warnings", [])),
                }
            )
    return b"\xef\xbb\xbf" + output.getvalue().encode("utf-8")


def export_zip(storage: AnalysisStorage, session_id: str) -> bytes:
    buffer = io.BytesIO()
    manifest = session_payload(storage, session_id)
    manifest["asset_errors"] = []
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("results.csv", export_csv(storage, session_id))
        for asset, path in storage.list_assets_for_session(session_id):
            if not path.is_file():
                manifest["asset_errors"].append({"asset_id": asset["id"], "error": "Asset file is missing"})
                continue
            suffix = Path(path.name).suffix.lower()
            archive_name = f"evidence/{asset['kind']}/{asset['id']}{suffix}"
            try:
                archive.write(path, archive_name)
            except OSError as exc:
                manifest["asset_errors"].append({"asset_id": asset["id"], "error": str(exc)})
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return buffer.getvalue()
