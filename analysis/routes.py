from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response

from .export import export_csv, export_json, export_zip
from .jobs import AnalysisJobManager
from .manual import merge_fragments_payload, patch_event_payload, split_identity_payload
from .ocr.provider import OCRProvider
from .series_reference import SeriesReference
from .storage import AnalysisStorage


IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"})


def _max_bytes(kind: str) -> int:
    variable = "AI_TRAIN_ANALYSIS_MAX_VIDEO_BYTES" if kind == "video" else "AI_TRAIN_ANALYSIS_MAX_IMAGE_BYTES"
    default = 2 * 1024 * 1024 * 1024 if kind == "video" else 50 * 1024 * 1024
    try:
        return max(1, int(os.environ.get(variable, default)))
    except (TypeError, ValueError):
        return default


def create_analysis_router(
    storage: AnalysisStorage,
    jobs: AnalysisJobManager,
    ocr: OCRProvider,
    reference: SeriesReference,
) -> APIRouter:
    router = APIRouter(prefix="/api/analysis", tags=["analysis"])

    @router.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "ocr": ocr.status(), "series_reference": reference.version}

    @router.get("/sessions")
    def list_sessions(limit: int = Query(30, ge=1, le=200)) -> dict[str, Any]:
        return {"sessions": storage.list_sessions(limit)}

    @router.post("/sessions", status_code=201)
    def create_session(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
        return storage.create_session(payload.get("settings") if isinstance(payload, dict) else None)

    @router.get("/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        session = storage.get_session(session_id)
        if not session:
            raise HTTPException(404, "Analysis session not found")
        return {**session, "files": storage.list_files(session_id), "event_count": storage.count_events(session_id)}

    @router.delete("/sessions/{session_id}/contents")
    def clear_session(session_id: str) -> dict[str, Any]:
        try:
            session = storage.clear_session(session_id)
        except KeyError:
            raise HTTPException(404, "Analysis session not found") from None
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {**session, "files": [], "event_count": 0}

    @router.post("/sessions/{session_id}/files", status_code=201)
    async def upload_files(session_id: str, files: list[UploadFile] = File(...)) -> dict[str, Any]:
        if not storage.get_session(session_id):
            raise HTTPException(404, "Analysis session not found")
        if not files:
            raise HTTPException(400, "No files were uploaded")
        uploaded: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for upload in files:
            name = Path(upload.filename or "upload").name
            extension = Path(name).suffix.lower()
            mime = (upload.content_type or "").lower()
            if extension in IMAGE_EXTENSIONS and mime.startswith("image/"):
                kind = "image"
            elif extension in VIDEO_EXTENSIONS and mime.startswith("video/"):
                kind = "video"
            else:
                errors.append({"file": name, "error": "Unsupported or mismatched MIME type and extension"})
                await upload.close()
                continue
            record, target = storage.create_file_record(session_id, name, mime, extension, kind)
            total = 0
            try:
                with target.open("wb") as output:
                    while chunk := await upload.read(1024 * 1024):
                        total += len(chunk)
                        if total > _max_bytes(kind):
                            raise ValueError(f"File exceeds the configured {kind} size limit")
                        output.write(chunk)
                if total <= 0:
                    raise ValueError("Uploaded file is empty")
                storage.finalize_file(record["id"], total)
                uploaded.append(storage.get_file(record["id"]) or record)
            except Exception as exc:
                storage.delete_file_record(record["id"])
                errors.append({"file": name, "error": str(exc)})
            finally:
                await upload.close()
        if not uploaded and errors:
            raise HTTPException(400, detail={"message": "No valid files were uploaded", "errors": errors})
        return {"files": uploaded, "errors": errors}

    @router.get("/files/{file_id}/content")
    def file_content(file_id: str) -> FileResponse:
        record = storage.get_file(file_id)
        if not record:
            raise HTTPException(404, "Analysis file not found")
        path = storage.file_path(file_id)
        if not path.is_file():
            raise HTTPException(404, "Stored analysis file is missing")
        return FileResponse(path, media_type=record["mime_type"], filename=record["original_name"])

    @router.post("/sessions/{session_id}/start", status_code=202)
    def start_analysis(session_id: str, payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
        try:
            return jobs.start(session_id, payload.get("settings") if isinstance(payload, dict) else None)
        except KeyError:
            raise HTTPException(404, "Analysis session not found") from None
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        job = storage.get_job(job_id)
        if not job:
            raise HTTPException(404, "Analysis job not found")
        return job

    @router.post("/jobs/{job_id}/stop", status_code=202)
    def stop_job(job_id: str) -> dict[str, Any]:
        try:
            return jobs.stop(job_id)
        except KeyError:
            raise HTTPException(404, "Analysis job not found") from None

    @router.get("/sessions/{session_id}/events")
    def list_events(session_id: str, file_id: str | None = None) -> dict[str, Any]:
        if not storage.get_session(session_id):
            raise HTTPException(404, "Analysis session not found")
        return {"events": storage.list_events(session_id, file_id)}

    @router.get("/events/{event_id}")
    def get_event(event_id: str) -> dict[str, Any]:
        event = storage.get_event(event_id)
        if not event:
            raise HTTPException(404, "Analysis event not found")
        event["manual_changes_history"] = storage.list_manual_changes(event_id)
        return event

    @router.patch("/events/{event_id}")
    def patch_event(event_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        event = storage.get_event(event_id)
        if not event:
            raise HTTPException(404, "Analysis event not found")
        updated, change = patch_event_payload(event, payload, reference)
        storage.record_manual_change(event_id, change["action"], change["payload"])
        return storage.save_event(updated)

    @router.post("/events/{event_id}/reanalyze", status_code=202)
    def reanalyze_event(event_id: str, payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
        event = storage.get_event(event_id)
        if not event:
            raise HTTPException(404, "Analysis event not found")
        try:
            return jobs.start(event["session_id"], payload.get("settings") if isinstance(payload, dict) else None, reanalyze_event_id=event_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.post("/events/{event_id}/merge-fragments")
    def merge_fragments(event_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        event = storage.get_event(event_id)
        if not event:
            raise HTTPException(404, "Analysis event not found")
        track_ids = payload.get("fragment_track_ids")
        if not isinstance(track_ids, list) or len(track_ids) < 1:
            raise HTTPException(400, "fragment_track_ids must contain at least one track")
        updated, change = merge_fragments_payload(event, track_ids, reference, payload.get("text"))
        storage.record_manual_change(event_id, change["action"], change["payload"])
        return storage.save_event(updated)

    @router.post("/events/{event_id}/split-identity")
    def split_identity(event_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        event = storage.get_event(event_id)
        if not event:
            raise HTTPException(404, "Analysis event not found")
        identity_id = str(payload.get("identity_id", ""))
        groups = payload.get("groups")
        if not identity_id or not isinstance(groups, list) or len(groups) < 2:
            raise HTTPException(400, "identity_id and at least two track groups are required")
        try:
            updated, change = split_identity_payload(event, identity_id, groups, reference)
        except KeyError:
            raise HTTPException(404, "Locomotive identity not found") from None
        storage.record_manual_change(event_id, change["action"], change["payload"])
        return storage.save_event(updated)

    @router.get("/assets/{asset_id}")
    def asset_content(asset_id: str) -> FileResponse:
        stored = storage.get_asset(asset_id)
        if not stored:
            raise HTTPException(404, "Analysis asset not found")
        asset, path = stored
        if not path.is_file():
            raise HTTPException(404, "Stored analysis asset is missing")
        return FileResponse(path, media_type=asset["mime_type"], filename=path.name)

    @router.get("/sessions/{session_id}/export")
    def export_session(session_id: str, format: str = Query("json", pattern="^(json|csv|zip)$")) -> Response:
        try:
            if format == "csv":
                data, media_type, suffix = export_csv(storage, session_id), "text/csv; charset=utf-8", "csv"
            elif format == "zip":
                data, media_type, suffix = export_zip(storage, session_id), "application/zip", "zip"
            else:
                data, media_type, suffix = export_json(storage, session_id), "application/json; charset=utf-8", "json"
        except KeyError:
            raise HTTPException(404, "Analysis session not found") from None
        return Response(data, media_type=media_type, headers={"Content-Disposition": f'attachment; filename="analysis-{session_id}.{suffix}"'})

    return router
