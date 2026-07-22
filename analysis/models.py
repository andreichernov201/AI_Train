from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


@dataclass(slots=True)
class OCRResult:
    raw_text: str = ""
    normalized_text: str | None = None
    confidence: float = 0.0
    provider: str = "paddleocr"
    model_name: str = "eslav_PP-OCRv5_mobile_rec"
    preprocessing: str = "original"
    inference_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OCRObservation:
    id: str
    event_id: str
    train_detection_id: str
    number_detection_id: str
    fragment_track_id: str | None
    frame_index: int | None
    timestamp_ms: int | None
    bbox: list[float]
    relative_bbox: list[float]
    raw_text: str
    normalized_text: str | None
    fragment_type: str
    ocr_confidence: float
    detection_confidence: float
    quality_score: float
    preprocessing: str
    crop_asset_id: str | None
    error: str | None = None
    file_name: str = ""
    crop_width: int = 0
    crop_height: int = 0
    crop_area: int = 0
    sharpness: float = 0.0
    contrast: float = 0.0
    clipped: bool = False
    provider: str = "paddleocr"
    model_name: str = "eslav_PP-OCRv5_mobile_rec"
    inference_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class NumberFragmentTrack:
    id: str
    event_id: str
    observations: list[str] = field(default_factory=list)
    fragment_type: str = "unknown_fragment"
    best_text: str | None = None
    candidates: list[dict[str, Any]] = field(default_factory=list)
    relative_position: list[float] = field(default_factory=list)
    best_crop_asset_id: str | None = None
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LocomotiveIdentity:
    id: str
    event_id: str
    fragment_track_ids: list[str] = field(default_factory=list)
    recognized_number: str | None = None
    recognized_series: str | None = None
    serial_number: str | None = None
    number_candidates: list[dict[str, Any]] = field(default_factory=list)
    assembly_confidence: float = 0.0
    ocr_confidence: float = 0.0
    section_count: int | None = None
    section_source: str = "unknown"
    observation_count: int = 0
    best_frame_asset_id: str | None = None
    best_crop_asset_ids: list[str] = field(default_factory=list)
    status: str = "low_confidence"
    warnings: list[str] = field(default_factory=list)
    manually_confirmed: bool = False
    locked: bool = False
    source: str = "model"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AnalysisEvent:
    id: str
    session_id: str
    source_file_id: str
    started_at: str
    ended_at: str | None = None
    status: str = "low_confidence"
    train_observations: list[dict[str, Any]] = field(default_factory=list)
    number_observations: list[dict[str, Any]] = field(default_factory=list)
    segmentation_observations: list[dict[str, Any]] = field(default_factory=list)
    fragment_tracks: list[dict[str, Any]] = field(default_factory=list)
    assembly_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    locomotive_identities: list[dict[str, Any]] = field(default_factory=list)
    best_frame_asset_id: str | None = None
    backup_frame_asset_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    manual_changes: list[dict[str, Any]] = field(default_factory=list)
    pipeline_stages: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
