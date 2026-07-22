from __future__ import annotations

import json
import mimetypes
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .models import new_id, utc_now


class _ClosingConnection(sqlite3.Connection):
    """Make the connection context manager close handles as well as commit."""

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool | None:
        try:
            return super().__exit__(exc_type, exc, traceback)
        finally:
            self.close()


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_load(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


class AnalysisStorage:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.uploads_dir = self.root / "uploads"
        self.assets_dir = self.root / "assets"
        self.exports_dir = self.root / "exports"
        for directory in (self.root, self.uploads_dir, self.assets_dir, self.exports_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "analysis.sqlite3"
        self._write_lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30.0, factory=_ClosingConnection)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            status TEXT NOT NULL,
            settings_json TEXT NOT NULL,
            model_versions_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS files (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            original_name TEXT NOT NULL,
            stored_name TEXT NOT NULL,
            mime_type TEXT NOT NULL,
            size_bytes INTEGER NOT NULL DEFAULT 0,
            kind TEXT NOT NULL,
            status TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            file_id TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
            status TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            revision INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            status TEXT NOT NULL,
            progress_json TEXT NOT NULL,
            error TEXT,
            stop_requested INTEGER NOT NULL DEFAULT 0,
            reanalyze_event_id TEXT,
            created_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT
        );
        CREATE TABLE IF NOT EXISTS assets (
            id TEXT PRIMARY KEY,
            event_id TEXT REFERENCES events(id) ON DELETE CASCADE,
            file_id TEXT REFERENCES files(id) ON DELETE CASCADE,
            kind TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            mime_type TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS manual_changes (
            id TEXT PRIMARY KEY,
            event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
            created_at TEXT NOT NULL,
            action TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_files_session ON files(session_id);
        CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
        CREATE INDEX IF NOT EXISTS idx_events_file ON events(file_id);
        CREATE INDEX IF NOT EXISTS idx_jobs_session ON jobs(session_id);
        """
        with self._connect() as connection:
            connection.executescript(schema)
            connection.execute(
                "UPDATE jobs SET status='failed', error=COALESCE(error, 'Backend restarted during analysis'), "
                "finished_at=COALESCE(finished_at, ?) WHERE status IN ('queued','processing')",
                (utc_now(),),
            )

    @staticmethod
    def _session(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "status": row["status"],
            "settings": _json_load(row["settings_json"], {}),
            "model_versions": _json_load(row["model_versions_json"], {}),
        }

    @staticmethod
    def _file(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "session_id": row["session_id"],
            "original_name": row["original_name"],
            "mime_type": row["mime_type"],
            "size_bytes": int(row["size_bytes"]),
            "kind": row["kind"],
            "status": row["status"],
            "metadata": _json_load(row["metadata_json"], {}),
            "created_at": row["created_at"],
            "content_url": f"/api/analysis/files/{row['id']}/content",
        }

    def create_session(self, settings: dict[str, Any] | None = None) -> dict[str, Any]:
        session_id = new_id("session")
        now = utc_now()
        defaults = {
            "frame_interval_sec": 1.0,
            "detect_confidence": 0.25,
            "ocr_confidence": 0.55,
            "event_timeout_sec": 2.5,
            "max_ocr_candidates": 3,
            "best_frame_count": 3,
            "preprocessing": "adaptive",
        }
        defaults.update(settings or {})
        with self._write_lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, now, now, "ready", _json_dump(defaults), _json_dump({})),
            )
        return self.get_session(session_id) or {}

    def list_sessions(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (max(1, min(200, limit)),)
            ).fetchall()
        return [self._session(row) | {"file_count": self.count_files(row["id"]), "event_count": self.count_events(row["id"])} for row in rows]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        return self._session(row) if row else None

    def update_session(
        self,
        session_id: str,
        *,
        status: str | None = None,
        settings: dict[str, Any] | None = None,
        model_versions: dict[str, Any] | None = None,
    ) -> None:
        current = self.get_session(session_id)
        if not current:
            raise KeyError(session_id)
        merged_settings = dict(current["settings"])
        merged_settings.update(settings or {})
        merged_versions = dict(current["model_versions"])
        merged_versions.update(model_versions or {})
        with self._write_lock, self._connect() as connection:
            connection.execute(
                "UPDATE sessions SET updated_at=?, status=?, settings_json=?, model_versions_json=? WHERE id=?",
                (
                    utc_now(),
                    status or current["status"],
                    _json_dump(merged_settings),
                    _json_dump(merged_versions),
                    session_id,
                ),
            )

    def create_file_record(
        self, session_id: str, original_name: str, mime_type: str, extension: str, kind: str
    ) -> tuple[dict[str, Any], Path]:
        if not self.get_session(session_id):
            raise KeyError(session_id)
        file_id = new_id("file")
        safe_extension = extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        relative = Path("uploads") / session_id / f"{file_id}{safe_extension}"
        target = (self.root / relative).resolve()
        if self.root not in target.parents:
            raise ValueError("Unsafe upload path")
        target.parent.mkdir(parents=True, exist_ok=True)
        now = utc_now()
        display_name = Path(original_name or f"upload{safe_extension}").name[:240]
        with self._write_lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO files VALUES (?, ?, ?, ?, ?, 0, ?, 'uploading', '{}', ?)",
                (file_id, session_id, display_name, relative.as_posix(), mime_type, kind, now),
            )
        return self.get_file(file_id) or {}, target

    def finalize_file(self, file_id: str, size_bytes: int, metadata: dict[str, Any] | None = None) -> None:
        with self._write_lock, self._connect() as connection:
            connection.execute(
                "UPDATE files SET size_bytes=?, status='ready', metadata_json=? WHERE id=?",
                (int(size_bytes), _json_dump(metadata or {}), file_id),
            )
            row = connection.execute("SELECT session_id FROM files WHERE id=?", (file_id,)).fetchone()
        if row:
            self.update_session(row["session_id"], status="ready")

    def delete_file_record(self, file_id: str) -> None:
        record = self._get_file_row(file_id)
        if record:
            path = self._safe_path(record["stored_name"])
            if path.is_file():
                path.unlink()
        with self._write_lock, self._connect() as connection:
            connection.execute("DELETE FROM files WHERE id=?", (file_id,))

    def _get_file_row(self, file_id: str) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute("SELECT * FROM files WHERE id=?", (file_id,)).fetchone()

    def get_file(self, file_id: str) -> dict[str, Any] | None:
        row = self._get_file_row(file_id)
        return self._file(row) if row else None

    def list_files(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM files WHERE session_id=? ORDER BY created_at, rowid", (session_id,)
            ).fetchall()
        return [self._file(row) for row in rows]

    def count_files(self, session_id: str) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM files WHERE session_id=?", (session_id,)).fetchone()[0])

    def update_file_status(self, file_id: str, status: str, metadata: dict[str, Any] | None = None) -> None:
        with self._write_lock, self._connect() as connection:
            if metadata is None:
                connection.execute("UPDATE files SET status=? WHERE id=?", (status, file_id))
            else:
                connection.execute(
                    "UPDATE files SET status=?, metadata_json=? WHERE id=?",
                    (status, _json_dump(metadata), file_id),
                )

    def _safe_path(self, relative_path: str) -> Path:
        target = (self.root / relative_path).resolve()
        if target != self.root and self.root not in target.parents:
            raise ValueError("Unsafe stored path")
        return target

    def file_path(self, file_id: str) -> Path:
        row = self._get_file_row(file_id)
        if not row:
            raise KeyError(file_id)
        return self._safe_path(str(row["stored_name"]))

    def save_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        event_id = str(payload["id"])
        now = utc_now()
        existing = self.get_event(event_id)
        revision = int(existing.get("revision", 0)) + 1 if existing else 1
        with self._write_lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO events(id,session_id,file_id,status,payload_json,revision,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET status=excluded.status,payload_json=excluded.payload_json,
                    revision=excluded.revision,updated_at=excluded.updated_at
                """,
                (
                    event_id,
                    payload["session_id"],
                    payload["source_file_id"],
                    payload.get("status", "low_confidence"),
                    _json_dump(payload),
                    revision,
                    existing.get("created_at", now) if existing else now,
                    now,
                ),
            )
        return self.get_event(event_id) or {}

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        if not row:
            return None
        payload = _json_load(row["payload_json"], {})
        payload.update({"revision": int(row["revision"]), "created_at": row["created_at"], "updated_at": row["updated_at"]})
        return payload

    def list_events(self, session_id: str, file_id: str | None = None) -> list[dict[str, Any]]:
        query = (
            "SELECT e.* FROM events e JOIN files f ON f.id=e.file_id "
            "WHERE e.session_id=?"
        )
        params: list[Any] = [session_id]
        if file_id:
            query += " AND e.file_id=?"
            params.append(file_id)
        query += " ORDER BY f.created_at, f.rowid, e.created_at, e.rowid"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        events = []
        for row in rows:
            payload = _json_load(row["payload_json"], {})
            payload.update({"revision": int(row["revision"]), "created_at": row["created_at"], "updated_at": row["updated_at"]})
            events.append(payload)
        return events

    def count_events(self, session_id: str) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM events WHERE session_id=?", (session_id,)).fetchone()[0])

    def save_asset(
        self,
        *,
        event_id: str | None,
        file_id: str | None,
        kind: str,
        data: bytes,
        suffix: str,
        mime_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        asset_id = new_id("asset")
        safe_suffix = suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
        owner = event_id or file_id or "unassigned"
        relative = Path("assets") / owner / f"{asset_id}{safe_suffix}"
        target = self._safe_path(relative.as_posix())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        mime = mime_type or mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        with self._write_lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO assets VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (asset_id, event_id, file_id, kind, relative.as_posix(), mime, _json_dump(metadata or {}), utc_now()),
            )
        return asset_id

    def get_asset(self, asset_id: str) -> tuple[dict[str, Any], Path] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM assets WHERE id=?", (asset_id,)).fetchone()
        if not row:
            return None
        payload = {
            "id": row["id"],
            "event_id": row["event_id"],
            "file_id": row["file_id"],
            "kind": row["kind"],
            "mime_type": row["mime_type"],
            "metadata": _json_load(row["metadata_json"], {}),
            "created_at": row["created_at"],
            "url": f"/api/analysis/assets/{row['id']}",
        }
        return payload, self._safe_path(str(row["relative_path"]))

    def list_assets_for_session(self, session_id: str) -> list[tuple[dict[str, Any], Path]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT a.* FROM assets a
                LEFT JOIN events e ON e.id=a.event_id
                LEFT JOIN files f ON f.id=a.file_id
                WHERE e.session_id=? OR f.session_id=?
                ORDER BY a.created_at
                """,
                (session_id, session_id),
            ).fetchall()
        results = []
        for row in rows:
            payload = {
                "id": row["id"],
                "event_id": row["event_id"],
                "file_id": row["file_id"],
                "kind": row["kind"],
                "mime_type": row["mime_type"],
                "metadata": _json_load(row["metadata_json"], {}),
            }
            results.append((payload, self._safe_path(str(row["relative_path"]))))
        return results
    def has_active_job(self, session_id: str) -> bool:
        with self._connect() as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM jobs WHERE session_id=? AND status IN ('queued','processing')",
                (session_id,),
            ).fetchone()[0]
        return bool(count)

    def clear_analysis_results(self, session_id: str, file_id: str | None = None) -> int:
        if not self.get_session(session_id):
            raise KeyError(session_id)
        if file_id:
            record = self.get_file(file_id)
            if not record or record["session_id"] != session_id:
                raise KeyError(file_id)

        event_where = "session_id=?"
        event_params: list[Any] = [session_id]
        asset_where = "e.session_id=?"
        asset_params: list[Any] = [session_id]
        file_where = "session_id=?"
        file_params: list[Any] = [session_id]
        if file_id:
            event_where += " AND file_id=?"
            event_params.append(file_id)
            asset_where += " AND e.file_id=?"
            asset_params.append(file_id)
            file_where += " AND id=?"
            file_params.append(file_id)

        stored_paths: set[str] = set()
        with self._write_lock, self._connect() as connection:
            stored_paths.update(
                str(row[0])
                for row in connection.execute(
                    f"SELECT a.relative_path FROM assets a JOIN events e ON e.id=a.event_id WHERE {asset_where}",
                    asset_params,
                ).fetchall()
            )
            deleted = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM events WHERE {event_where}", event_params
                ).fetchone()[0]
            )
            connection.execute(f"DELETE FROM events WHERE {event_where}", event_params)
            connection.execute(
                f"UPDATE files SET status='ready', metadata_json='{{}}' WHERE {file_where}",
                file_params,
            )
            connection.execute(
                "UPDATE sessions SET updated_at=?, status='ready', model_versions_json='{}' WHERE id=?",
                (utc_now(), session_id),
            )

        for relative_path in stored_paths:
            path = self._safe_path(relative_path)
            if path.is_file():
                path.unlink()
        return deleted

    def clear_session(self, session_id: str) -> dict[str, Any]:
        current = self.get_session(session_id)
        if not current:
            raise KeyError(session_id)
        stored_paths: set[str] = set()
        with self._write_lock, self._connect() as connection:
            active = connection.execute(
                "SELECT COUNT(*) FROM jobs WHERE session_id=? AND status IN ('queued','processing')",
                (session_id,),
            ).fetchone()[0]
            if active:
                raise RuntimeError("Stop the active analysis before clearing the session")
            stored_paths.update(
                str(row[0])
                for row in connection.execute(
                    "SELECT stored_name FROM files WHERE session_id=?", (session_id,)
                ).fetchall()
            )
            stored_paths.update(
                str(row[0])
                for row in connection.execute(
                    """
                    SELECT DISTINCT a.relative_path FROM assets a
                    LEFT JOIN events e ON e.id=a.event_id
                    LEFT JOIN files f ON f.id=a.file_id
                    WHERE e.session_id=? OR f.session_id=?
                    """,
                    (session_id, session_id),
                ).fetchall()
            )
            connection.execute("DELETE FROM jobs WHERE session_id=?", (session_id,))
            connection.execute("DELETE FROM files WHERE session_id=?", (session_id,))
            connection.execute(
                "UPDATE sessions SET updated_at=?, status='ready', model_versions_json='{}' WHERE id=?",
                (utc_now(), session_id),
            )
        for relative_path in stored_paths:
            path = self._safe_path(relative_path)
            if path.is_file():
                path.unlink()
        return self.get_session(session_id) or current

    def create_job(self, session_id: str, reanalyze_event_id: str | None = None) -> dict[str, Any]:
        job_id = new_id("job")
        progress = {"stage": "queued", "percent": 0.0, "processed_files": 0, "processed_frames": 0}
        with self._write_lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO jobs VALUES (?, ?, 'queued', ?, NULL, 0, ?, ?, NULL, NULL)",
                (job_id, session_id, _json_dump(progress), reanalyze_event_id, utc_now()),
            )
        return self.get_job(job_id) or {}

    def update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        progress: dict[str, Any] | None = None,
        error: str | None = None,
        stop_requested: bool | None = None,
    ) -> None:
        current = self.get_job(job_id)
        if not current:
            raise KeyError(job_id)
        merged_progress = dict(current["progress"])
        merged_progress.update(progress or {})
        next_status = status or current["status"]
        started_at = current.get("started_at") or (utc_now() if next_status == "processing" else None)
        finished_at = current.get("finished_at")
        if next_status in {"completed", "failed", "stopped"} and not finished_at:
            finished_at = utc_now()
        with self._write_lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs SET status=?, progress_json=?, error=?, stop_requested=?,
                    started_at=?, finished_at=? WHERE id=?
                """,
                (
                    next_status,
                    _json_dump(merged_progress),
                    error if error is not None else current.get("error"),
                    int(stop_requested if stop_requested is not None else current.get("stop_requested", False)),
                    started_at,
                    finished_at,
                    job_id,
                ),
            )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "session_id": row["session_id"],
            "status": row["status"],
            "progress": _json_load(row["progress_json"], {}),
            "error": row["error"],
            "stop_requested": bool(row["stop_requested"]),
            "reanalyze_event_id": row["reanalyze_event_id"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
        }

    def record_manual_change(self, event_id: str, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        change = {"id": new_id("manual"), "event_id": event_id, "created_at": utc_now(), "action": action, "payload": payload}
        with self._write_lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO manual_changes VALUES (?, ?, ?, ?, ?)",
                (change["id"], event_id, change["created_at"], action, _json_dump(payload)),
            )
        return change

    def list_manual_changes(self, event_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM manual_changes WHERE event_id=? ORDER BY created_at", (event_id,)
            ).fetchall()
        return [
            {"id": row["id"], "event_id": row["event_id"], "created_at": row["created_at"], "action": row["action"], "payload": _json_load(row["payload_json"], {})}
            for row in rows
        ]
