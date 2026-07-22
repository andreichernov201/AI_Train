from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageOps

from .detection import AnalysisDetector, link_numbers_to_trains, suppress_duplicate_trains
from .event_association import EventAssociator
from .frame_quality import encode_jpeg, score_frame
from .models import AnalysisEvent, OCRObservation, new_id, utc_now
from .numbers.fragment_assembler import NumberFragmentAssembler
from .numbers.fragment_classifier import classify_fragment
from .numbers.fragment_tracker import NumberFragmentTracker
from .ocr.normalization import normalize_ocr_text
from .ocr.preprocessing import crop_with_padding, fallback_variants, primary_variants, quality_metrics
from .ocr.provider import OCRProvider
from .segmentation import AnalysisSegmenter, link_segments_to_trains
from .series_reference import SeriesReference
from .storage import AnalysisStorage


@dataclass(slots=True)
class _VideoEventState:
    event: AnalysisEvent
    tracker: NumberFragmentTracker
    best_frame_score: float = -1.0
    best_frame_reason: dict[str, float] = field(default_factory=dict)
    frame_count: int = 0
    start_timestamp_ms: int = 0
    end_timestamp_ms: int = 0


class AnalysisPipeline:
    def __init__(
        self,
        storage: AnalysisStorage,
        detector: AnalysisDetector,
        ocr: OCRProvider,
        reference: SeriesReference,
        segmenter: AnalysisSegmenter | None = None,
    ) -> None:
        self.storage = storage
        self.detector = detector
        self.ocr = ocr
        self.reference = reference
        self.segmenter = segmenter
        self.assembler = NumberFragmentAssembler(reference)

    @staticmethod
    def _event_status(event: AnalysisEvent, stopped: bool = False) -> str:
        if stopped:
            return "stopped"
        if event.errors and not event.locomotive_identities:
            return "analysis_error"
        statuses = {str(item.get("status")) for item in event.locomotive_identities}
        if statuses and statuses <= {"confirmed", "manual"} and not event.errors:
            return "confirmed"
        if "number_not_detected" in statuses or not event.number_observations:
            return "number_not_recognized"
        if event.errors:
            return "partial_result"
        return "low_confidence"

    def _save_event(self, event: AnalysisEvent) -> dict[str, Any]:
        event.status = self._event_status(event, event.status == "stopped")
        return self.storage.save_event(event.to_dict())

    def _progress(self, job_id: str, **values: Any) -> None:
        self.storage.update_job(job_id, progress=values)


    def _segment_frame(
        self,
        image: Image.Image,
        trains: list[dict[str, Any]],
        frame_index: int,
        timestamp_ms: int,
    ) -> tuple[dict[int, list[dict[str, Any]]], float, str | None]:
        if not self.segmenter or not trains:
            return {index: [] for index in range(len(trains))}, 0.0, None
        started = time.perf_counter()
        try:
            segments = self.segmenter.segment(image)
            linked = link_segments_to_trains(trains, segments)
            for train_index, rows in linked.items():
                for segment_index, row in enumerate(rows):
                    row["id"] = f"segment_{frame_index}_{train_index}_{segment_index}"
                    row["frame_index"] = frame_index
                    row["timestamp_ms"] = timestamp_ms
                    row["train_index"] = train_index
            return linked, (time.perf_counter() - started) * 1000.0, None
        except Exception as exc:
            return (
                {index: [] for index in range(len(trains))},
                (time.perf_counter() - started) * 1000.0,
                f"Сегментация не выполнена: {exc}",
            )
    def run_session(
        self,
        *,
        session_id: str,
        job_id: str,
        settings: dict[str, Any],
        stop_check: Callable[[], bool],
        reanalyze_event_id: str | None = None,
        target_file_id: str | None = None,
    ) -> None:
        files = self.storage.list_files(session_id)
        if target_file_id:
            files = [item for item in files if item["id"] == target_file_id]
            if not files:
                raise KeyError(target_file_id)
        elif reanalyze_event_id:
            previous = self.storage.get_event(reanalyze_event_id)
            if not previous:
                raise KeyError(reanalyze_event_id)
            files = [item for item in files if item["id"] == previous.get("source_file_id")]
        self.storage.update_session(
            session_id,
            model_versions={
                "detection": {"provider": "ultralytics", "classes": ["train", "number"]},
                "segmentation": self.segmenter.status() if self.segmenter else {"enabled": False},
                "ocr": self.ocr.status(),
                "series_reference": self.reference.version,
            },
        )
        started = time.perf_counter()
        processed_files = 0
        for index, file_record in enumerate(files):
            if stop_check():
                break
            self._progress(
                job_id,
                stage="loading_file",
                current_file=file_record["original_name"],
                current_file_id=file_record["id"],
                processed_files=processed_files,
                total_files=len(files),
                percent=round(index / max(1, len(files)) * 100.0, 2),
            )
            self.storage.update_file_status(file_record["id"], "processing")
            if file_record["kind"] == "video":
                self._analyze_video(
                    file_record,
                    session_id,
                    job_id,
                    settings,
                    stop_check,
                    reanalyze_event_id,
                )
            else:
                self._analyze_photo(
                    file_record,
                    session_id,
                    job_id,
                    settings,
                    reanalyze_event_id,
                )
            processed_files += 1
            elapsed = max(1e-6, time.perf_counter() - started)
            self._progress(
                job_id,
                stage="file_completed",
                processed_files=processed_files,
                total_files=len(files),
                percent=round(processed_files / max(1, len(files)) * 100.0, 2),
                files_per_second=round(processed_files / elapsed, 3),
                events_found=self.storage.count_events(session_id),
            )

    def _analyze_photo(
        self,
        file_record: dict[str, Any],
        session_id: str,
        job_id: str,
        settings: dict[str, Any],
        reanalyze_event_id: str | None,
    ) -> None:
        path = self.storage.file_path(file_record["id"])
        try:
            with Image.open(path) as opened:
                pil_image = ImageOps.exif_transpose(opened).convert("RGB")
        except Exception as exc:
            self.storage.update_file_status(file_record["id"], "failed", {"error": str(exc)})
            return
        bgr = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
        detect_started = time.perf_counter()
        detections = self.detector.detect(pil_image, float(settings.get("detect_confidence", 0.25)))
        detect_ms = (time.perf_counter() - detect_started) * 1000.0
        trains = suppress_duplicate_trains([row for row in detections if row.get("cls_name") == "train"])
        numbers = [row for row in detections if row.get("cls_name") == "number"]
        if not trains:
            self.storage.update_file_status(
                file_record["id"],
                "no_train",
                {"message": "Состав не обнаружен", "detection_count": len(detections)},
            )
            return
        linked, linking_warnings = link_numbers_to_trains(trains, numbers)
        segmented, segment_ms, segment_error = self._segment_frame(pil_image, trains, 0, 0)
        for train_index, train in enumerate(trains):
            event_id = new_id("event")
            event = AnalysisEvent(
                id=event_id,
                session_id=session_id,
                source_file_id=file_record["id"],
                started_at=utc_now(),
                warnings=list(linking_warnings),
                metadata={
                    "source_type": "photo",
                    "file_name": file_record["original_name"],
                    "frame_width": pil_image.width,
                    "frame_height": pil_image.height,
                    "frame_count": 1,
                    "reanalyze_of": reanalyze_event_id,
                },
                pipeline_stages=[
                    {
                        "name": "detect",
                        "status": "completed",
                        "started_at": None,
                        "finished_at": utc_now(),
                        "duration_ms": round(detect_ms, 3),
                        "provider": "ultralytics",
                        "model": "active detection model",
                        "version": None,
                        "data": {"train_count": len(trains), "number_count": len(numbers)},
                        "warnings": [],
                        "errors": [],
                    }
                ],
            )
            if segment_error:
                event.warnings.append(segment_error)
            event.pipeline_stages.append(
                {
                    "name": "segment",
                    "status": "warning" if segment_error else "completed",
                    "started_at": None,
                    "finished_at": utc_now(),
                    "duration_ms": round(segment_ms, 3),
                    "provider": "ultralytics",
                    "model": self.segmenter.model_name if self.segmenter else None,
                    "data": {"segment_count": len(segmented.get(train_index, []))},
                    "warnings": [segment_error] if segment_error else [],
                    "errors": [],
                }
            )
            train_row = dict(train)
            train_row["id"] = f"train_0_{train_index}"
            train_row["frame_index"] = 0
            train_row["timestamp_ms"] = 0
            event.train_observations.append(train_row)
            self.storage.save_event(event.to_dict())
            event.segmentation_observations.extend(segmented.get(train_index, []))
            event.metadata["segmentation_count"] = len(event.segmentation_observations)
            frame_score, reason = score_frame(bgr, train, len(linked.get(train_index, [])))
            event.best_frame_asset_id = self.storage.save_asset(
                event_id=event_id,
                file_id=file_record["id"],
                kind="best_frame",
                data=encode_jpeg(bgr),
                suffix=".jpg",
                mime_type="image/jpeg",
                metadata={"score": frame_score, "reason": reason, "frame_index": 0},
            )
            tracker = NumberFragmentTracker(event_id)
            for number_index, number in enumerate(linked.get(train_index, [])):
                observation = self._recognize_number(
                    event_id=event_id,
                    file_record=file_record,
                    frame=bgr,
                    train=train_row,
                    number=number,
                    frame_index=0,
                    timestamp_ms=0,
                    number_index=number_index,
                    settings=settings,
                )
                tracker.update(observation)
                event.number_observations.append(observation)
                if observation.get("error"):
                    event.errors.append(str(observation["error"]))
            self._finalize_event(event, tracker, stopped=False)
        self.storage.update_file_status(
            file_record["id"],
            "completed",
            {"train_count": len(trains), "number_count": len(numbers), "event_count": len(trains)},
        )

    def _recognize_number(
        self,
        *,
        event_id: str,
        file_record: dict[str, Any],
        frame: np.ndarray,
        train: dict[str, Any],
        number: dict[str, Any],
        frame_index: int,
        timestamp_ms: int,
        number_index: int,
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        crop, crop_box, clipped = crop_with_padding(frame, list(map(float, number["box"])))
        sharpness, contrast, quality = quality_metrics(crop)
        crop_asset_id = None
        try:
            crop_asset_id = self.storage.save_asset(
                event_id=event_id,
                file_id=file_record["id"],
                kind="number_crop",
                data=encode_jpeg(crop, 95),
                suffix=".jpg",
                mime_type="image/jpeg",
                metadata={"frame_index": frame_index, "timestamp_ms": timestamp_ms, "bbox": crop_box},
            )
        except Exception:
            crop_asset_id = None

        threshold = float(settings.get("ocr_confidence", 0.55))
        results: list[tuple[Any, Any]] = []
        for variant in primary_variants(crop):
            results.append((self.ocr.recognize(variant.image, variant.name), variant))
        best_result, best_variant = max(results, key=lambda pair: pair[0].confidence)
        if best_result.confidence < threshold:
            for variant in fallback_variants(crop):
                results.append((self.ocr.recognize(variant.image, variant.name), variant))
            best_result, best_variant = max(results, key=lambda pair: pair[0].confidence)
        normalized = normalize_ocr_text(best_result.raw_text, self.reference)
        fragment_type = classify_fragment(normalized.normalized_text, self.reference)
        effective_ocr_confidence = best_result.confidence
        if normalized.context_applied:
            effective_ocr_confidence *= normalized.context_score
        effective_ocr_confidence = max(0.0, min(1.0, effective_ocr_confidence))
        relative = list(number.get("relative_box", [0, 0, 0, 0]))
        warnings = []
        if normalized.context_applied:
            warnings.append(f"Серия восстановлена по контексту: {normalized.series_part}")
        if crop.shape[0] * crop.shape[1] < 900:
            warnings.append("Номерная зона слишком мала")
        if sharpness < 35:
            warnings.append("Кроп размыт")
        if contrast < 18:
            warnings.append("Низкий контраст")
        if clipped:
            warnings.append("Номер обрезан")
        observation = OCRObservation(
            id=new_id("observation"),
            event_id=event_id,
            train_detection_id=str(train["id"]),
            number_detection_id=f"number_{frame_index}_{number_index}",
            fragment_track_id=None,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            bbox=list(map(float, number["box"])),
            relative_bbox=relative,
            raw_text=best_result.raw_text,
            normalized_text=normalized.normalized_text,
            fragment_type=fragment_type,
            ocr_confidence=effective_ocr_confidence,
            detection_confidence=float(number.get("conf", 0.0)),
            quality_score=float(best_variant.quality_score),
            preprocessing=best_result.preprocessing,
            crop_asset_id=crop_asset_id,
            error=best_result.error,
            file_name=file_record["original_name"],
            crop_width=int(crop.shape[1]),
            crop_height=int(crop.shape[0]),
            crop_area=int(crop.shape[0] * crop.shape[1]),
            sharpness=sharpness,
            contrast=contrast,
            clipped=clipped,
            provider=best_result.provider,
            model_name=best_result.model_name,
            inference_ms=sum(float(result.inference_ms) for result, _variant in results),
        ).to_dict()
        observation["source_file_id"] = file_record["id"]
        observation["warnings"] = warnings
        observation["ocr_candidates"] = [
            {
                "raw_text": result.raw_text,
                "confidence": result.confidence,
                "preprocessing": result.preprocessing,
                "error": result.error,
            }
            for result, _variant in sorted(
                results, key=lambda pair: pair[0].confidence, reverse=True
            )[: max(1, int(settings.get("max_ocr_candidates", 3)))]
        ]
        observation["context_applied"] = normalized.context_applied
        observation["context_score"] = normalized.context_score
        observation["context_alternatives"] = normalized.alternatives
        return observation

    def _finalize_event(
        self, event: AnalysisEvent, tracker: NumberFragmentTracker, stopped: bool
    ) -> None:
        event.fragment_tracks = tracker.to_dicts()
        identities, hypotheses, assembly_warnings = self.assembler.assemble(
            event.id, event.fragment_tracks
        )
        for identity in identities:
            identity["best_frame_asset_id"] = event.best_frame_asset_id
        event.locomotive_identities = identities
        event.assembly_hypotheses = hypotheses
        event.warnings = list(dict.fromkeys(event.warnings + assembly_warnings))
        event.ended_at = utc_now()
        event.status = "stopped" if stopped else self._event_status(event)
        event.pipeline_stages.append(
            {
                "name": "assemble",
                "status": "completed",
                "started_at": None,
                "finished_at": utc_now(),
                "duration_ms": None,
                "provider": "internal",
                "model": "NumberFragmentAssembler",
                "version": 1,
                "data": {"track_count": len(event.fragment_tracks), "identity_count": len(identities)},
                "warnings": assembly_warnings,
                "errors": [],
            }
        )
        self._save_event(event)

    def _analyze_video(
        self,
        file_record: dict[str, Any],
        session_id: str,
        job_id: str,
        settings: dict[str, Any],
        stop_check: Callable[[], bool],
        reanalyze_event_id: str | None,
    ) -> None:
        path = self.storage.file_path(file_record["id"])
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            self.storage.update_file_status(file_record["id"], "failed", {"error": "Failed to open video"})
            return
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0:
            fps = 25.0
        interval_sec = max(0.1, float(settings.get("frame_interval_sec", 1.0)))
        frame_step = max(1, int(round(fps * interval_sec)))
        expected_samples = max(1, (total_frames + frame_step - 1) // frame_step) if total_frames else 1
        timeout_ms = int(max(0.1, float(settings.get("event_timeout_sec", 2.5))) * 1000)
        associator = EventAssociator(timeout_ms=timeout_ms)
        states: dict[str, _VideoEventState] = {}
        frame_index = 0
        processed = 0
        started = time.perf_counter()
        found_any_train = False
        try:
            while True:
                if stop_check():
                    break
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                timestamp_ms = int(round(frame_index / fps * 1000.0))
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                detections = self.detector.detect(pil_image, float(settings.get("detect_confidence", 0.25)))
                trains = suppress_duplicate_trains([row for row in detections if row.get("cls_name") == "train"])
                numbers = [row for row in detections if row.get("cls_name") == "number"]
                found_any_train = found_any_train or bool(trains)
                linked, linking_warnings = link_numbers_to_trains(trains, numbers)
                segmented, segment_ms, segment_error = self._segment_frame(pil_image, trains, frame_index, timestamp_ms)
                assignments = associator.associate(trains, timestamp_ms, frame_index)
                for train_index, event_id in enumerate(assignments):
                    train = trains[train_index]
                    if event_id not in states:
                        event = AnalysisEvent(
                            id=event_id,
                            session_id=session_id,
                            source_file_id=file_record["id"],
                            started_at=utc_now(),
                            warnings=list(linking_warnings),
                            metadata={
                                "source_type": "video",
                                "file_name": file_record["original_name"],
                                "frame_width": int(frame.shape[1]),
                                "frame_height": int(frame.shape[0]),
                                "frame_count": 0,
                                "start_timestamp_ms": timestamp_ms,
                                "end_timestamp_ms": timestamp_ms,
                                "reanalyze_of": reanalyze_event_id,
                            },
                        )
                        if segment_error:
                            event.warnings.append(segment_error)
                        event.pipeline_stages.append(
                            {
                                "name": "segment",
                                "status": "warning" if segment_error else "completed",
                                "started_at": None,
                                "finished_at": utc_now(),
                                "duration_ms": round(segment_ms, 3),
                                "provider": "ultralytics",
                                "model": self.segmenter.model_name if self.segmenter else None,
                                "data": {"segment_count": len(segmented.get(train_index, []))},
                                "warnings": [segment_error] if segment_error else [],
                                "errors": [],
                            }
                        )
                        states[event_id] = _VideoEventState(
                            event=event,
                            tracker=NumberFragmentTracker(event_id),
                            start_timestamp_ms=timestamp_ms,
                            end_timestamp_ms=timestamp_ms,
                        )
                        self.storage.save_event(event.to_dict())
                    state = states[event_id]
                    state.frame_count += 1
                    if segment_error and segment_error not in state.event.warnings:
                        state.event.warnings.append(segment_error)
                    state.end_timestamp_ms = timestamp_ms
                    train_row = dict(train)
                    train_row["id"] = f"train_{frame_index}_{train_index}"
                    train_row["frame_index"] = frame_index
                    train_row["timestamp_ms"] = timestamp_ms
                    state.event.train_observations.append(train_row)
                    state.event.segmentation_observations.extend(segmented.get(train_index, []))
                    for number_index, number in enumerate(linked.get(train_index, [])):
                        observation = self._recognize_number(
                            event_id=event_id,
                            file_record=file_record,
                            frame=frame,
                            train=train_row,
                            number=number,
                            frame_index=frame_index,
                            timestamp_ms=timestamp_ms,
                            number_index=number_index,
                            settings=settings,
                        )
                        state.tracker.update(observation)
                        state.event.number_observations.append(observation)
                        if observation.get("error"):
                            state.event.errors.append(str(observation["error"]))
                    frame_score, reason = score_frame(frame, train, len(linked.get(train_index, [])))
                    if frame_score > state.best_frame_score:
                        if state.event.best_frame_asset_id:
                            state.event.backup_frame_asset_ids.insert(0, state.event.best_frame_asset_id)
                            state.event.backup_frame_asset_ids = state.event.backup_frame_asset_ids[: max(0, int(settings.get("best_frame_count", 3)) - 1)]
                        state.event.best_frame_asset_id = self.storage.save_asset(
                            event_id=event_id,
                            file_id=file_record["id"],
                            kind="best_frame",
                            data=encode_jpeg(frame),
                            suffix=".jpg",
                            mime_type="image/jpeg",
                            metadata={"score": frame_score, "reason": reason, "frame_index": frame_index, "timestamp_ms": timestamp_ms},
                        )
                        state.best_frame_score = frame_score
                        state.best_frame_reason = {**reason, "frame_index": frame_index}
                    state.event.fragment_tracks = state.tracker.to_dicts()
                    state.event.metadata.update(
                        {
                            "frame_count": state.frame_count,
                            "end_timestamp_ms": timestamp_ms,
                            "best_frame_score": state.best_frame_score,
                            "best_frame_reason": state.best_frame_reason,
                            "segmentation_count": len(state.event.segmentation_observations),
                        }
                    )
                    self.storage.save_event(state.event.to_dict())

                for expired_id in associator.expire(timestamp_ms):
                    state = states.get(expired_id)
                    if state and not state.event.ended_at:
                        self._finalize_video_state(state, stopped=False)

                processed += 1
                elapsed = max(1e-6, time.perf_counter() - started)
                percent = min(99.0, processed / expected_samples * 100.0) if total_frames else 0.0
                self._progress(
                    job_id,
                    stage="tracking_fragments" if numbers else "detect",
                    current_file=file_record["original_name"],
                    current_file_id=file_record["id"],
                    current_frame=frame_index,
                    total_frames=total_frames,
                    timestamp_ms=timestamp_ms,
                    processed_frames=processed,
                    percent=round(percent, 2),
                    events_found=len(states),
                    identifiers_assembled=sum(len(state.event.locomotive_identities) for state in states.values()),
                    frames_per_second=round(processed / elapsed, 3),
                )
                frame_index += frame_step
                if total_frames and frame_index >= total_frames:
                    break
        finally:
            capture.release()

        stopped = stop_check()
        for event_id in associator.finish_all():
            state = states.get(event_id)
            if state and not state.event.ended_at:
                self._finalize_video_state(state, stopped=stopped)
        for state in states.values():
            if not state.event.ended_at:
                self._finalize_video_state(state, stopped=stopped)
        if not found_any_train:
            self.storage.update_file_status(file_record["id"], "no_train", {"message": "Состав не обнаружен", "processed_frames": processed})
        else:
            self.storage.update_file_status(
                file_record["id"],
                "stopped" if stopped else "completed",
                {"processed_frames": processed, "event_count": len(states), "fps": fps, "frame_step": frame_step},
            )

    def _finalize_video_state(self, state: _VideoEventState, stopped: bool) -> None:
        state.event.metadata.update(
            {
                "frame_count": state.frame_count,
                "start_timestamp_ms": state.start_timestamp_ms,
                "end_timestamp_ms": state.end_timestamp_ms,
                "best_frame_score": state.best_frame_score,
                "best_frame_reason": state.best_frame_reason,
            }
        )
        self._finalize_event(state.event, state.tracker, stopped=stopped)
