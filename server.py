import io
import os
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from ultralytics import YOLO


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
RUNS_DETECT_DIR = os.path.join(ROOT_DIR, "runs", "detect")
RUNS_SEGMENT_DIR = os.path.join(ROOT_DIR, "runs", "segment")
LEGACY_RUNS_SEGMENT_DIR = os.path.join(ROOT_DIR, "runs", "segmentation")
YOLO_SEG_DIR = os.path.join(ROOT_DIR, "yolo-seg")
DEFAULT_DETECT_MODEL_PATH = os.path.join(RUNS_DETECT_DIR, "weights", "best.pt")
DEFAULT_SEGMENT_MODEL_PATH = os.path.join(RUNS_SEGMENT_DIR, "weights", "best.pt")
DETECTION_CLASS_NAMES = frozenset({"train", "number"})
SEGMENTATION_CLASS_NAMES = frozenset(
    {"body", "autocoupler", "axlebox", "bogie", "hose"}
)
DEFAULT_SEGMENTATION_CONFIDENCE = 0.39
DEFAULT_SEGMENTATION_IMAGE_SIZE = 1280


def _latest_best_pt(search_roots: tuple[str, ...]) -> str | None:
    """Возвращает самый свежий best.pt независимо от имени папки запуска."""
    candidates: list[tuple[int, str]] = []
    for search_root in search_roots:
        if not os.path.isdir(search_root):
            continue
        for root, _dirs, files in os.walk(search_root):
            if os.path.basename(root).lower() != "weights" or "best.pt" not in files:
                continue
            best_pt = os.path.abspath(os.path.join(root, "best.pt"))
            try:
                candidates.append((os.stat(best_pt).st_mtime_ns, best_pt))
            except OSError:
                continue
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[1]


def resolve_detection_model_path() -> str:
    """Переменная окружения имеет приоритет, иначе берётся самый свежий best.pt."""
    override = os.environ.get("AI_TRAIN_MODEL_PATH", "").strip()
    if override and os.path.isfile(override):
        return os.path.abspath(override)
    return _latest_best_pt((RUNS_DETECT_DIR,)) or DEFAULT_DETECT_MODEL_PATH


def resolve_segmentation_model_path() -> str:
    """Ищет свежий best.pt в новой структуре и в старых папках для совместимости."""
    override = os.environ.get("AI_TRAIN_SEG_MODEL_PATH", "").strip()
    if override and os.path.isfile(override):
        return os.path.abspath(override)
    return (
        _latest_best_pt(
            (
                RUNS_SEGMENT_DIR,
                LEGACY_RUNS_SEGMENT_DIR,
                os.path.join(YOLO_SEG_DIR, "runs", "segment"),
            )
        )
        or DEFAULT_SEGMENT_MODEL_PATH
    )


def _model_signature(path: str) -> tuple[str, int, int]:
    stat = os.stat(path)
    return os.path.abspath(path), stat.st_mtime_ns, stat.st_size


_model_cache: tuple[str, int, int, YOLO] | None = None
_segmentation_model_cache: tuple[str, int, int, YOLO] | None = None


def get_model() -> YOLO:
    global _model_cache
    path = resolve_detection_model_path()
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Detection model not found: {path}. "
            f"Train a model under {RUNS_DETECT_DIR} or set AI_TRAIN_MODEL_PATH."
        )
    signature = _model_signature(path)
    if _model_cache is None or _model_cache[:3] != signature:
        _model_cache = (*signature, YOLO(path))
    return _model_cache[3]


def get_segmentation_model() -> YOLO:
    global _segmentation_model_cache
    path = resolve_segmentation_model_path()
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Segmentation model not found: {path}. "
            f"Train a model under {RUNS_SEGMENT_DIR} or set AI_TRAIN_SEG_MODEL_PATH."
        )
    signature = _model_signature(path)
    if _segmentation_model_cache is None or _segmentation_model_cache[:3] != signature:
        _segmentation_model_cache = (*signature, YOLO(path))
    return _segmentation_model_cache[3]


def _segmentation_confidence() -> float:
    raw = os.environ.get("AI_TRAIN_SEG_CONF", "").strip()
    try:
        value = float(raw) if raw else DEFAULT_SEGMENTATION_CONFIDENCE
    except ValueError:
        return DEFAULT_SEGMENTATION_CONFIDENCE
    return min(1.0, max(0.0, value))


def _segmentation_image_size() -> int:
    raw = os.environ.get("AI_TRAIN_SEG_IMGSZ", "").strip()
    try:
        value = int(raw) if raw else DEFAULT_SEGMENTATION_IMAGE_SIZE
    except ValueError:
        return DEFAULT_SEGMENTATION_IMAGE_SIZE
    return max(32, value)


def _largest_mask_polygons(masks: Any) -> list[np.ndarray]:
    """Оставляет главный связный компонент маски без склейки удалённых островков."""
    data = getattr(masks, "data", None)
    if data is None:
        return []
    if hasattr(data, "detach"):
        data = data.detach()
    if hasattr(data, "cpu"):
        data = data.cpu()
    if hasattr(data, "numpy"):
        data = data.numpy()
    mask_data = np.asarray(data)
    if mask_data.ndim == 2:
        mask_data = mask_data[None, ...]
    if mask_data.ndim != 3 or not mask_data.shape[1] or not mask_data.shape[2]:
        return []

    mask_height, mask_width = mask_data.shape[1:]
    orig_shape = getattr(masks, "orig_shape", None)
    if orig_shape is None:
        orig_shape = (mask_height, mask_width)
    target_height, target_width = int(orig_shape[0]), int(orig_shape[1])
    scale_x = target_width / mask_width
    scale_y = target_height / mask_height

    polygons: list[np.ndarray] = []
    for mask in mask_data:
        binary = np.ascontiguousarray(mask > 0.5, dtype=np.uint8)
        contours = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )[-2]
        contours = [
            contour for contour in contours if cv2.contourArea(contour) > 0
        ]
        if not contours:
            polygons.append(np.empty((0, 2), dtype=np.float32))
            continue

        contour = max(contours, key=cv2.contourArea)
        polygon = contour.reshape(-1, 2).astype(np.float32)
        if len(polygon) < 3:
            polygons.append(np.empty((0, 2), dtype=np.float32))
            continue
        polygon[:, 0] = np.clip(
            polygon[:, 0] * scale_x,
            0,
            target_width - 1,
        )
        polygon[:, 1] = np.clip(
            polygon[:, 1] * scale_y,
            0,
            target_height - 1,
        )
        polygons.append(polygon)
    return polygons


def yolo_result_to_detections(
    res: Any,
    include_segments: bool = False,
    allowed_class_names: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    names: dict[int, str] = getattr(res, "names", {}) or {}
    boxes = getattr(res, "boxes", None)
    if boxes is None:
        return []

    xyxy = boxes.xyxy.tolist()
    conf = boxes.conf.tolist()
    cls = boxes.cls.tolist()
    masks = getattr(res, "masks", None)
    polygons = (
        _largest_mask_polygons(masks)
        if include_segments and masks is not None
        else []
    )

    detections: list[dict[str, Any]] = []
    for i, (b, c, k) in enumerate(zip(xyxy, conf, cls)):
        cls_id = int(k)
        cls_name = str(names.get(cls_id, str(cls_id))).strip().lower()
        if allowed_class_names is not None and cls_name not in allowed_class_names:
            continue
        row: dict[str, Any] = {
            "id": len(detections),
            "cls_id": cls_id,
            "cls_name": cls_name,
            "conf": float(c),
            "box": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
        }
        if include_segments:
            if i >= len(polygons) or len(polygons[i]) < 3:
                continue
            row["segment"] = [
                [float(x), float(y)] for x, y in polygons[i].tolist()
            ]
        detections.append(row)
    return detections


def segment_full_image(img: Image.Image) -> list[dict[str, Any]]:
    """Сегментация на полном кадре; полигоны уже в координатах исходного изображения."""
    segment_model = get_segmentation_model()
    seg_res = segment_model(
        img,
        verbose=False,
        retina_masks=True,
        imgsz=_segmentation_image_size(),
        conf=_segmentation_confidence(),
    )[0]
    return yolo_result_to_detections(
        seg_res,
        include_segments=True,
        allowed_class_names=SEGMENTATION_CLASS_NAMES,
    )


def read_upload_image(file: UploadFile, raw: bytes) -> Image.Image | JSONResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        return JSONResponse({"error": "Only image uploads are supported"}, status_code=400)
    try:
        img = Image.open(io.BytesIO(raw))
        return ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        return JSONResponse({"error": "Failed to read image"}, status_code=400)


app = FastAPI()

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.middleware("http")
async def _disable_frontend_cache(request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.on_event("startup")
def _log_detection_model_path() -> None:
    path = resolve_detection_model_path()
    if os.path.isfile(path):
        print(f"[ai-train] detection model: {path}")
    else:
        print(
            f"[ai-train] WARNING: model not found: {path}\n"
            f"  Place weights at {DEFAULT_DETECT_MODEL_PATH}\n"
            f"  or set AI_TRAIN_MODEL_PATH to an existing best.pt"
        )
    seg_path = resolve_segmentation_model_path()
    if os.path.isfile(seg_path):
        print(f"[ai-train] segmentation model: {seg_path}")
    else:
        print(
            f"[ai-train] WARNING: segmentation model not found: {seg_path}\n"
            f"  Place weights at {DEFAULT_SEGMENT_MODEL_PATH}\n"
            f"  or set AI_TRAIN_SEG_MODEL_PATH to an existing best.pt"
        )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(
        os.path.join(FRONTEND_DIR, "index.html"),
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)) -> JSONResponse:
    raw = await file.read()
    img = read_upload_image(file, raw)
    if isinstance(img, JSONResponse):
        return img

    model = get_model()
    res = model(img, verbose=False)[0]

    return JSONResponse(
        {
            "image": {"width": img.width, "height": img.height},
            "mode": "detection",
            "detections": yolo_result_to_detections(
                res,
                allowed_class_names=DETECTION_CLASS_NAMES,
            ),
        }
    )


@app.post("/api/segment")
async def segment(file: UploadFile = File(...)) -> JSONResponse:
    raw = await file.read()
    img = read_upload_image(file, raw)
    if isinstance(img, JSONResponse):
        return img

    return JSONResponse(
        {
            "image": {"width": img.width, "height": img.height},
            "mode": "segmentation",
            "pipeline": "full-image-segment",
            "detections": segment_full_image(img),
        }
    )
