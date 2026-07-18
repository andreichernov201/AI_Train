import io
import os
import re
from typing import Any

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from ultralytics import YOLO


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
RUNS_DETECT_DIR = os.path.join(ROOT_DIR, "runs", "detect")
RUNS_SEGMENT_DIR = os.path.join(ROOT_DIR, "runs", "segmentation")
YOLO_SEG_DIR = os.path.join(ROOT_DIR, "yolo-seg")
# Базовый путь; при его отсутствии выбирается последний доступный прогон.
DEFAULT_DETECT_MODEL_PATH = os.path.join(
    RUNS_DETECT_DIR, "weights", "best.pt"
)
DETECTION_CLASS_NAMES = frozenset({"train", "number"})
SEGMENTATION_CLASS_NAMES = frozenset(
    {"body", "autocoupler", "axlebox", "bogie", "hose"}
)
DEFAULT_SEGMENT_MODEL_PATH = os.path.join(RUNS_SEGMENT_DIR, "weights", "best.pt")

# Имена средних (m) предобученных весов Ultralytics, с которых обычно стартует train.
_MEDIUM_BACKBONE_RE = re.compile(
    r"^yolo(?:v(?P<v>\d+))?(?P<rest>\d*)m\.pt$", re.IGNORECASE
)


def _parse_model_field_from_args(text: str) -> str | None:
    for line in text.replace("\r\n", "\n").split("\n"):
        line = line.strip()
        if line.lower().startswith("model:"):
            return line.split(":", 1)[1].strip().strip("'\"")
    return None


def _basename_lower(p: str) -> str:
    return os.path.basename(str(p).replace("\\", "/")).lower()


def _is_medium_backbone_file(name: str) -> bool:
    base = _basename_lower(name)
    if not base.endswith(".pt"):
        return False
    return bool(_MEDIUM_BACKBONE_RE.match(base))


def _resolve_initial_backbone_from_args_yaml(
    args_path: str, visited: set[str] | None = None, depth: int = 0
) -> str | None:
    """Ищет в args.yaml поле model; если там чужой чекпоинт — пробуем его args (resume)."""
    if visited is None:
        visited = set()
    if depth > 8 or args_path in visited or not os.path.isfile(args_path):
        return None
    visited.add(args_path)
    try:
        with open(args_path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return None
    raw = _parse_model_field_from_args(text)
    if not raw:
        return None
    base = _basename_lower(raw)
    if _is_medium_backbone_file(base):
        return base
    if base.endswith(".pt"):
        cand = os.path.normpath(os.path.join(os.path.dirname(args_path), raw))
        if not os.path.isfile(cand):
            cand = os.path.normpath(os.path.join(ROOT_DIR, raw))
        if os.path.isfile(cand):
            parent_args = os.path.join(os.path.dirname(cand), "..", "args.yaml")
            parent_args = os.path.normpath(parent_args)
            return _resolve_initial_backbone_from_args_yaml(parent_args, visited, depth + 1)
    return None


def _train_run_dir_from_best_pt(path: str) -> str:
    # .../run_name/weights/best.pt -> .../run_name
    return os.path.dirname(os.path.dirname(path))


def resolve_detection_model_path() -> str:
    """
    Путь к весам для /api/detect:
    - переменная AI_TRAIN_MODEL_PATH, если файл существует;
    - иначе последний по времени best.pt под runs/detect, у которого в args.yaml
      начальный model — medium (yolo11m.pt, yolov8m.pt, …);
    - если таких нет — последний best.pt вообще;
    - иначе DEFAULT_DETECT_MODEL_PATH.
    """
    override = os.environ.get("AI_TRAIN_MODEL_PATH", "").strip()
    if override and os.path.isfile(override):
        return os.path.abspath(override)

    if os.path.isfile(DEFAULT_DETECT_MODEL_PATH):
        return DEFAULT_DETECT_MODEL_PATH

    if not os.path.isdir(RUNS_DETECT_DIR):
        return DEFAULT_DETECT_MODEL_PATH

    candidates: list[tuple[float, str, bool]] = []
    for root, _dirs, files in os.walk(RUNS_DETECT_DIR):
        if os.path.basename(root).lower() != "weights":
            continue
        if "best.pt" not in files:
            continue
        best_pt = os.path.join(root, "best.pt")
        try:
            mtime = os.path.getmtime(best_pt)
        except OSError:
            continue
        run_dir = _train_run_dir_from_best_pt(best_pt)
        args_yaml = os.path.join(run_dir, "args.yaml")
        initial = _resolve_initial_backbone_from_args_yaml(args_yaml)
        is_medium = bool(initial and _is_medium_backbone_file(initial))
        candidates.append((mtime, best_pt, is_medium))

    if not candidates:
        return DEFAULT_DETECT_MODEL_PATH

    medium = [(t, p) for t, p, m in candidates if m]
    if medium:
        _t, path = max(medium, key=lambda x: x[0])
        return path
    _t, path, _ = max(candidates, key=lambda x: x[0])
    return path


_model_cache: tuple[str, YOLO] | None = None
_segmentation_model_cache: tuple[str, YOLO] | None = None


def get_model() -> YOLO:
    global _model_cache
    path = resolve_detection_model_path()
    if _model_cache is None or _model_cache[0] != path:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model not found: {path}. "
                f"Expected default: {DEFAULT_DETECT_MODEL_PATH} "
                f"or set AI_TRAIN_MODEL_PATH."
            )
        _model_cache = (path, YOLO(path))
    return _model_cache[1]


def resolve_segmentation_model_path() -> str:
    """
    Путь к весам для /api/segment:
    - переменная AI_TRAIN_SEG_MODEL_PATH, если файл существует;
    - иначе runs/segmentation/weights/best.pt;
    - иначе последний best.pt под runs/segmentation или yolo-seg/runs/segment.
    """
    override = os.environ.get("AI_TRAIN_SEG_MODEL_PATH", "").strip()
    if override and os.path.isfile(override):
        return os.path.abspath(override)

    if os.path.isfile(DEFAULT_SEGMENT_MODEL_PATH):
        return DEFAULT_SEGMENT_MODEL_PATH

    candidates: list[tuple[float, str]] = []
    for search_root in (RUNS_SEGMENT_DIR, os.path.join(YOLO_SEG_DIR, "runs", "segment")):
        if not os.path.isdir(search_root):
            continue
        for root, _dirs, files in os.walk(search_root):
            if os.path.basename(root).lower() != "weights":
                continue
            if "best.pt" not in files:
                continue
            best_pt = os.path.join(root, "best.pt")
            try:
                candidates.append((os.path.getmtime(best_pt), best_pt))
            except OSError:
                continue
    if candidates:
        _t, path = max(candidates, key=lambda x: x[0])
        return path
    return DEFAULT_SEGMENT_MODEL_PATH


def get_segmentation_model() -> YOLO:
    global _segmentation_model_cache
    path = resolve_segmentation_model_path()
    if _segmentation_model_cache is None or _segmentation_model_cache[0] != path:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Segmentation model not found: {path}. "
                f"Expected default: {DEFAULT_SEGMENT_MODEL_PATH} "
                f"or set AI_TRAIN_SEG_MODEL_PATH."
            )
        _segmentation_model_cache = (path, YOLO(path))
    return _segmentation_model_cache[1]


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
    polygons = []
    if include_segments and masks is not None and getattr(masks, "xy", None) is not None:
        polygons = masks.xy

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
        if include_segments and i < len(polygons):
            poly = polygons[i]
            row["segment"] = [[float(x), float(y)] for x, y in poly.tolist()]
        detections.append(row)
    return detections


def segment_full_image(img: Image.Image) -> list[dict[str, Any]]:
    """Сегментация на полном кадре; полигоны уже в координатах исходного изображения."""
    segment_model = get_segmentation_model()
    seg_res = segment_model(img, verbose=False, retina_masks=True)[0]
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
