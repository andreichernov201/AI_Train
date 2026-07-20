from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.training_common import (  # noqa: E402
    build_run_name,
    print_training_paths,
    resolve_dataset_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение YOLO Segmentation")
    parser.add_argument(
        "--data",
        help="ZIP с сайта, папка датасета или data.yaml. Без параметра берётся самый свежий dataset/segmentation/**/data.yaml.",
    )
    parser.add_argument(
        "--model",
        default="yolov8m-seg.pt",
        help="Стартовая модель или checkpoint",
    )
    parser.add_argument("--name", help="Понятная подпись запуска; дата и время добавятся автоматически")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0", help="Например: 0, 0,1 или cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache", choices=("ram", "disk", "false"), default="false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = resolve_dataset_yaml(ROOT_DIR, "segment", args.data)
    project_dir = ROOT_DIR / "runs" / "segment"
    run_name = build_run_name(args.model, args.name)

    print(f"Используется датасет: {data_path}")
    print(f"Результаты будут сохранены в: {project_dir / run_name}")

    model = YOLO(args.model)
    train_results = model.train(
        task="segment",
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache=False if args.cache == "false" else args.cache,
        retina_masks=True,
        overlap_mask=False,
        val=True,
        project=str(project_dir),
        name=run_name,
        exist_ok=False,
        patience=20,
        cos_lr=True,
        label_smoothing=0.1,
        dropout=0.1,
        weight_decay=0.001,
    )

    save_dir = Path(train_results.save_dir).resolve()
    best_path = save_dir / "weights" / "best.pt"
    print_training_paths(data_path, save_dir)
    if not best_path.is_file():
        raise FileNotFoundError(f"Обучение завершилось без best.pt: {best_path}")


if __name__ == "__main__":
    main()
