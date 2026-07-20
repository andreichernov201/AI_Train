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
    parser = argparse.ArgumentParser(description="Обучение YOLO Detect")
    parser.add_argument(
        "--data",
        help="ZIP с сайта, папка датасета или data.yaml. Без параметра берётся самый свежий dataset/detection/**/data.yaml.",
    )
    parser.add_argument("--model", default="yolo11m.pt", help="Стартовая модель или checkpoint")
    parser.add_argument("--name", help="Понятная подпись запуска; дата и время добавятся автоматически")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0", help="Например: 0, 0,1 или cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache", choices=("ram", "disk", "false"), default="ram")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = resolve_dataset_yaml(ROOT_DIR, "detect", args.data)
    project_dir = ROOT_DIR / "runs" / "detect"
    run_name = build_run_name(args.model, args.name)

    print(f"Используется датасет: {data_path}")
    print(f"Результаты будут сохранены в: {project_dir / run_name}")

    model = YOLO(args.model)
    train_results = model.train(
        task="detect",
        data=str(data_path),
        imgsz=args.imgsz,
        epochs=args.epochs,
        patience=100,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache=False if args.cache == "false" else args.cache,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.01,
        cos_lr=True,
        weight_decay=0.001,
        warmup_epochs=5,
        mosaic=0.75,
        close_mosaic=40,
        mixup=0.05,
        copy_paste=0.0,
        degrees=3.0,
        translate=0.08,
        scale=0.40,
        shear=0.5,
        perspective=0.0001,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.01,
        hsv_s=0.35,
        hsv_v=0.25,
        erasing=0.05,
        amp=True,
        seed=42,
        deterministic=True,
        val=True,
        plots=True,
        save=True,
        save_period=10,
        project=str(project_dir),
        name=run_name,
        exist_ok=False,
    )

    save_dir = Path(train_results.save_dir).resolve()
    best_path = save_dir / "weights" / "best.pt"
    print_training_paths(data_path, save_dir)
    if not best_path.is_file():
        raise FileNotFoundError(f"Обучение завершилось без best.pt: {best_path}")

    best_model = YOLO(str(best_path))
    metrics = best_model.val(
        data=str(data_path),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=True,
        split="val",
        project=str(save_dir),
        name="best_validation",
        exist_ok=True,
    )

    print("\n" + "=" * 70)
    print("Итоговые метрики best.pt")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
