import io
import os
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from ultralytics import YOLO


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
MODEL_PATH = os.path.join(ROOT_DIR, "runs", "detect", "train", "weights", "best.pt")


@lru_cache(maxsize=1)
def get_model() -> YOLO:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return YOLO(MODEL_PATH)


app = FastAPI()

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)) -> JSONResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        return JSONResponse({"error": "Only image uploads are supported"}, status_code=400)

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        return JSONResponse({"error": "Failed to read image"}, status_code=400)

    model = get_model()
    res = model(img, verbose=False)[0]

    names: dict[int, str] = getattr(res, "names", {}) or {}
    boxes = getattr(res, "boxes", None)
    if boxes is None:
        return JSONResponse({"image": {"width": img.width, "height": img.height}, "detections": []})

    xyxy = boxes.xyxy.tolist()
    conf = boxes.conf.tolist()
    cls = boxes.cls.tolist()

    detections: list[dict[str, Any]] = []
    for i, (b, c, k) in enumerate(zip(xyxy, conf, cls)):
        cls_id = int(k)
        detections.append(
            {
                "id": i,
                "cls_id": cls_id,
                "cls_name": names.get(cls_id, str(cls_id)),
                "conf": float(c),
                "box": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
            }
        )

    return JSONResponse(
        {
            "image": {"width": img.width, "height": img.height},
            "detections": detections,
        }
    )

