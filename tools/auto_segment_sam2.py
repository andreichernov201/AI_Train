"""Automatic annotation: YOLO detector -> SAM 2 -> PNG masks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DETECTOR = (
    ROOT
    / "runs"
    / "detect"
    / "20260604_131014_yolo11l_1280_anti_overfit"
    / "weights"
    / "best.pt"
)
DEFAULT_SAM_ROOT = Path(
    os.environ.get("AI_TRAIN_SAM2_ROOT", Path.home() / "Desktop" / "sam2")
)
DEFAULT_SAM_CHECKPOINT = Path(
    os.environ.get(
        "AI_TRAIN_SAM2_CHECKPOINT",
        DEFAULT_SAM_ROOT / "checkpoints" / "sam2.1_hiera_large.pt",
    )
)
DEFAULT_SAM_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create SAM 2 masks from YOLO detector boxes."
    )
    parser.add_argument("source", type=Path, help="Image or image directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs" / "sam2_auto_masks",
        help="Output directory",
    )
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--sam2-root", type=Path, default=DEFAULT_SAM_ROOT)
    parser.add_argument("--sam2-checkpoint", type=Path, default=DEFAULT_SAM_CHECKPOINT)
    parser.add_argument("--sam2-config", default=DEFAULT_SAM_CONFIG)
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["body"],
        help="Detector class names or IDs (default: body)",
    )
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument(
        "--box-padding",
        type=float,
        default=0.0,
        help="Expand detector box before SAM 2 by this fraction",
    )
    parser.add_argument("--alpha", type=float, default=0.42)
    parser.add_argument(
        "--no-clip-to-box",
        action="store_true",
        help="Allow a SAM mask to extend outside its detector box",
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument("--recursive", action="store_true")
    return parser.parse_args()


def checked_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def collect_images(source: Path, recursive: bool) -> tuple[list[Path], Path]:
    source = source.expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {source.suffix}")
        return [source], source.parent
    if not source.is_dir():
        raise FileNotFoundError(f"Source not found: {source}")
    iterator = source.rglob("*") if recursive else source.glob("*")
    images = sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No images found in: {source}")
    return images, source


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix.lower(), image)
    if not ok:
        raise ValueError(f"Cannot save image: {path}")
    encoded.tofile(path)


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested


def load_sam2(root: Path, checkpoint: Path, config: str, device: str):
    root = root.expanduser().resolve()
    if not (root / "sam2" / "__init__.py").is_file():
        raise FileNotFoundError(f"SAM 2 sources not found: {root}")
    sys.path.insert(0, str(root))
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            r"Run this script with .venv-sam\Scripts\python.exe"
        ) from exc
    return SAM2ImagePredictor(build_sam2(config, str(checkpoint), device=device))


def selected_class_ids(values: list[str], names: dict[int, str]) -> set[int]:
    by_name = {name.casefold(): class_id for class_id, name in names.items()}
    selected: set[int] = set()
    for value in values:
        try:
            class_id = int(value)
        except ValueError:
            if value.casefold() not in by_name:
                available = ", ".join(f"{i}:{n}" for i, n in names.items())
                raise ValueError(f"Class '{value}' not found. Available: {available}")
            class_id = by_name[value.casefold()]
        if class_id not in names:
            raise ValueError(f"Class ID {class_id} is not present in the detector")
        selected.add(class_id)
    return selected


def pad_box(
    box: np.ndarray, width: int, height: int, padding: float
) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(np.float32)
    px, py = (x2 - x1) * padding, (y2 - y1) * padding
    return np.array(
        [
            max(0.0, x1 - px),
            max(0.0, y1 - py),
            min(float(width - 1), x2 + px),
            min(float(height - 1), y2 + py),
        ],
        dtype=np.float32,
    )


def make_overlay(image: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    preview = image.copy()
    green = np.zeros_like(image)
    green[:, :, 1] = 255
    blended = cv2.addWeighted(image, 1.0 - alpha, green, alpha, 0)
    preview[mask] = blended[mask]
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(preview, contours, -1, (0, 255, 0), 2)
    return preview


def main() -> int:
    args = parse_args()
    if not 0 <= args.alpha <= 1:
        raise ValueError("--alpha must be between 0 and 1")
    if args.box_padding < 0:
        raise ValueError("--box-padding cannot be negative")

    detector_path = checked_file(args.detector, "Detector weights")
    sam_checkpoint = checked_file(args.sam2_checkpoint, "SAM 2 checkpoint")
    images, source_root = collect_images(args.source, args.recursive)
    output = args.output.expanduser().resolve()
    device = resolve_device(args.device)

    print(f"Device: {device}")
    if device.startswith("cuda"):
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"Loading detector: {detector_path}")
    detector = YOLO(str(detector_path))
    names = {int(i): str(name) for i, name in detector.names.items()}
    class_ids = selected_class_ids(args.classes, names)
    print("Classes:", ", ".join(f"{i}:{names[i]}" for i in sorted(class_ids)))

    print(f"Loading SAM 2: {sam_checkpoint}")
    predictor = load_sam2(
        args.sam2_root, sam_checkpoint, args.sam2_config, device
    )
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.startswith("cuda")
        else nullcontext()
    )
    summary: list[dict[str, object]] = []

    with torch.inference_mode(), autocast:
        for image_index, image_path in enumerate(images, 1):
            print(f"[{image_index}/{len(images)}] {image_path}")
            image = read_image(image_path)
            height, width = image.shape[:2]
            result = detector.predict(
                image,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                max_det=args.max_det,
                device=device,
                verbose=False,
            )[0]

            detections: list[tuple[np.ndarray, int, float]] = []
            if result.boxes is not None:
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                classes = result.boxes.cls.detach().cpu().numpy().astype(int)
                confidences = result.boxes.conf.detach().cpu().numpy()
                detections = [
                    (box, int(class_id), float(confidence))
                    for box, class_id, confidence in zip(boxes, classes, confidences)
                    if class_id in class_ids
                ]

            relative = image_path.relative_to(source_root).with_suffix("")
            union_mask = np.zeros((height, width), dtype=bool)
            instances: list[dict[str, object]] = []

            if detections:
                predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                for instance_index, (box, class_id, confidence) in enumerate(
                    detections, 1
                ):
                    prompt_box = pad_box(
                        box, width, height, args.box_padding
                    )
                    masks, scores, _ = predictor.predict(
                        box=prompt_box, multimask_output=False
                    )
                    mask = np.asarray(masks[0], dtype=bool)
                    if not args.no_clip_to_box:
                        x1, y1, x2, y2 = prompt_box
                        x1, y1 = int(np.floor(x1)), int(np.floor(y1))
                        x2, y2 = int(np.ceil(x2)), int(np.ceil(y2))
                        clipped = np.zeros_like(mask)
                        clipped[y1 : y2 + 1, x1 : x2 + 1] = mask[
                            y1 : y2 + 1, x1 : x2 + 1
                        ]
                        mask = clipped
                    union_mask |= mask
                    instance_path = (
                        output
                        / "instances"
                        / relative.parent
                        / f"{relative.name}__{instance_index:03d}_{names[class_id]}.png"
                    )
                    write_image(instance_path, mask.astype(np.uint8) * 255)
                    instances.append(
                        {
                            "class_id": class_id,
                            "class_name": names[class_id],
                            "detector_confidence": confidence,
                            "sam_score": float(np.asarray(scores).reshape(-1)[0]),
                            "box_xyxy": [float(value) for value in prompt_box],
                            "mask": str(instance_path.relative_to(output)),
                        }
                    )
                predictor.reset_predictor()

            mask_path = output / "masks" / relative.with_suffix(".png")
            overlay_path = output / "overlays" / relative.with_suffix(".jpg")
            metadata_path = output / "metadata" / relative.with_suffix(".json")
            write_image(mask_path, union_mask.astype(np.uint8) * 255)
            write_image(
                overlay_path, make_overlay(image, union_mask, args.alpha)
            )
            metadata = {
                "source": str(image_path),
                "width": width,
                "height": height,
                "instances": instances,
                "mask": str(mask_path.relative_to(output)),
                "overlay": str(overlay_path.relative_to(output)),
            }
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            summary.append(metadata)
            print(f"  objects found: {len(instances)}")

    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Done. Results: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
