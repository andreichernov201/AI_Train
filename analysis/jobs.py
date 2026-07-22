from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .storage import AnalysisStorage


class AnalysisJobManager:
    def __init__(self, storage: AnalysisStorage, pipeline: Any, max_workers: int = 1) -> None:
        self.storage = storage
        self.pipeline = pipeline
        self._executor = ThreadPoolExecutor(max_workers=max(1, max_workers), thread_name_prefix="analysis")
        self._stop_events: dict[str, threading.Event] = {}
        self._lock = threading.RLock()

    def start(
        self,
        session_id: str,
        settings: dict[str, Any] | None = None,
        reanalyze_event_id: str | None = None,
    ) -> dict[str, Any]:
        session = self.storage.get_session(session_id)
        if not session:
            raise KeyError(session_id)
        files = self.storage.list_files(session_id)
        if not files:
            raise ValueError("Upload at least one photo or video before analysis")

        target_file_id: str | None = None
        if reanalyze_event_id:
            previous = self.storage.get_event(reanalyze_event_id)
            if not previous or previous.get("session_id") != session_id:
                raise KeyError(reanalyze_event_id)
            target_file_id = str(previous["source_file_id"])

        merged_settings = dict(session.get("settings", {}))
        merged_settings.update(settings or {})
        stop_event = threading.Event()
        with self._lock:
            if self.storage.has_active_job(session_id):
                raise RuntimeError("Analysis is already running")
            self.storage.clear_analysis_results(session_id, target_file_id)
            self.storage.update_session(session_id, status="queued", settings=merged_settings)
            job = self.storage.create_job(session_id, reanalyze_event_id)
            self._stop_events[job["id"]] = stop_event

        self._executor.submit(
            self._run,
            job["id"],
            session_id,
            merged_settings,
            reanalyze_event_id,
            target_file_id,
            stop_event,
        )
        return job

    def _run(
        self,
        job_id: str,
        session_id: str,
        settings: dict[str, Any],
        reanalyze_event_id: str | None,
        target_file_id: str | None,
        stop_event: threading.Event,
    ) -> None:
        try:
            self.storage.update_job(job_id, status="processing", progress={"stage": "starting"})
            self.storage.update_session(session_id, status="processing")
            self.pipeline.run_session(
                session_id=session_id,
                job_id=job_id,
                settings=settings,
                stop_check=stop_event.is_set,
                reanalyze_event_id=reanalyze_event_id,
                target_file_id=target_file_id,
            )
            if stop_event.is_set():
                self.storage.update_job(job_id, status="stopped", progress={"stage": "stopped"})
                self.storage.update_session(session_id, status="stopped")
            else:
                self.storage.update_job(job_id, status="completed", progress={"stage": "completed", "percent": 100.0})
                self.storage.update_session(session_id, status="completed")
        except Exception as exc:
            self.storage.update_job(job_id, status="failed", progress={"stage": "failed"}, error=str(exc))
            self.storage.update_session(session_id, status="failed")
        finally:
            with self._lock:
                self._stop_events.pop(job_id, None)

    def stop(self, job_id: str) -> dict[str, Any]:
        job = self.storage.get_job(job_id)
        if not job:
            raise KeyError(job_id)
        with self._lock:
            stop_event = self._stop_events.get(job_id)
            if stop_event:
                stop_event.set()
        self.storage.update_job(job_id, stop_requested=True, progress={"stage": "stopping"})
        return self.storage.get_job(job_id) or job
