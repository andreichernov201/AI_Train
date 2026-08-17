from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

import numpy as np

from ..models import OCRResult
from .provider import OCRProvider


class PaddleOCRProvider(OCRProvider):
    """Ленивая потокобезопасная обёртка PaddleOCR для распознавания."""

    model_name = "eslav_PP-OCRv5_mobile_rec"

    def __init__(self) -> None:
        self._model: Any = None
        self._load_error: str | None = None
        self._lock = threading.RLock()
        self._device = os.environ.get("AI_TRAIN_OCR_DEVICE", "cpu").strip() or "cpu"

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from paddleocr import TextRecognition

                self._model = TextRecognition(
                    model_name=self.model_name,
                    device=self._device,
                )
                self._load_error = None
                return self._model
            except Exception as exc:
                self._load_error = str(exc)
                raise RuntimeError(f"PaddleOCR model initialization failed: {exc}") from exc

    @staticmethod
    def _payload(result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            payload = result
        else:
            payload = None
            for attribute in ("json", "res"):
                value = getattr(result, attribute, None)
                if callable(value):
                    try:
                        value = value()
                    except TypeError:
                        value = None
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        value = None
                if isinstance(value, dict):
                    payload = value
                    break
            if payload is None:
                try:
                    payload = dict(result)
                except Exception:
                    payload = {}
        nested = payload.get("res")
        return nested if isinstance(nested, dict) else payload

    def recognize(self, crop: np.ndarray, preprocessing: str = "original") -> OCRResult:
        if not isinstance(crop, np.ndarray) or crop.size == 0:
            return OCRResult(preprocessing=preprocessing, error="Empty OCR crop")
        started = time.perf_counter()
        try:
            model = self._get_model()
            with self._lock:
                output = model.predict(input=crop, batch_size=1)
                first = next(iter(output), None)
            if first is None:
                raise RuntimeError("PaddleOCR returned no recognition result")
            payload = self._payload(first)
            raw_text = str(payload.get("rec_text") or "").strip()
            confidence = float(payload.get("rec_score") or 0.0)
            return OCRResult(
                raw_text=raw_text,
                confidence=max(0.0, min(1.0, confidence)),
                preprocessing=preprocessing,
                inference_ms=round((time.perf_counter() - started) * 1000.0, 3),
            )
        except Exception as exc:
            return OCRResult(
                preprocessing=preprocessing,
                inference_ms=round((time.perf_counter() - started) * 1000.0, 3),
                error=str(exc),
            )

    def status(self) -> dict[str, object]:
        return {
            "provider": "paddleocr",
            "model_name": self.model_name,
            "device": self._device,
            "loaded": self._model is not None,
            "error": self._load_error,
        }
