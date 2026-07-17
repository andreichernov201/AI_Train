from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    # Все относительные пути считаются от папки, где находится train.py.
    base_dir = Path(__file__).resolve().parent

    data_path = base_dir / "dataset" / "detection" / "data.yaml"
    project_dir = base_dir / "runs" / "detect"
    run_name = "locomotive_consist_yolo11m_1280"

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Файл конфигурации датасета не найден:\n{data_path}"
        )

    # Загружаем предобученную YOLO11m.
    model = YOLO("yolo11m.pt")

    # Обучение.
    train_results = model.train(
        task="detect",
        data=str(data_path),

        # Разрешение и обучение
        imgsz=1280,
        epochs=400,
        patience=100,
        batch=8,
        device=0,
        workers=8,
        cache="ram",

        # Предобучение и оптимизатор
        pretrained=True,
        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.01,
        cos_lr=True,
        weight_decay=0.001,
        warmup_epochs=5,

        # Композиционные аугментации
        mosaic=0.75,
        close_mosaic=40,
        mixup=0.05,
        copy_paste=0.0,

        # Геометрические аугментации
        degrees=3.0,
        translate=0.08,
        scale=0.40,
        shear=0.5,
        perspective=0.0001,

        # Отражения
        fliplr=0.5,
        flipud=0.0,

        # Цвет и освещение
        hsv_h=0.01,
        hsv_s=0.35,
        hsv_v=0.25,

        # Дополнительная регуляризация
        erasing=0.05,

        # Технические параметры
        amp=True,
        seed=42,
        deterministic=True,
        val=True,
        plots=True,
        save=True,
        save_period=10,

        # Сохранение результатов
        project=str(project_dir),
        name=run_name,

        # False сохраняет старые запуски и при необходимости
        # создаёт папки с суффиксами -2, -3 и т. д.
        exist_ok=False,
    )

    # Получаем фактическую папку текущего запуска от Ultralytics.
    # Это работает даже при добавлении суффиксов -2, -3 и т. д.
    save_dir = Path(train_results.save_dir).resolve()
    best_path = save_dir / "weights" / "best.pt"
    last_path = save_dir / "weights" / "last.pt"

    print("\n" + "=" * 70)
    print("Обучение завершено")
    print(f"Папка текущего запуска: {save_dir}")
    print(f"Лучшая модель:          {best_path}")
    print(f"Последняя модель:       {last_path}")
    print("=" * 70)

    if not best_path.is_file():
        raise FileNotFoundError(
            "Ultralytics завершил обучение, но файл best.pt не найден:\n"
            f"{best_path}"
        )

    # Загружаем лучший checkpoint именно текущего запуска.
    best_model = YOLO(str(best_path))

    # Итоговая проверка best.pt на validation-выборке.
    metrics = best_model.val(
        data=str(data_path),
        imgsz=1280,
        batch=8,
        device=0,
        workers=8,
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