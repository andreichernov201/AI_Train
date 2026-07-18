import os

from ultralytics import YOLO

if __name__ == '__main__':
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model = YOLO("yolov8m-seg.pt")

    results = model.train(
        data=os.path.join(root_dir, "yolo-seg", "data.yaml"),
        epochs=300,
        imgsz=1280,
        batch=8,
        device=0,
        workers=8,
        retina_masks=True,
        overlap_mask=False,
        val=True,
        project=os.path.join(root_dir, "runs", "segmentation"),
        name="train",

        # --- ПАРАМЕТРЫ ПРОТИВ ПЕРЕОБУЧЕНИЯ ---
        patience=20,            # Остановка, если модель зашла в тупик и начинает зубрить
        cos_lr=True,            # Плавное синусоидальное падение скорости обучения
        label_smoothing=0.1,    # Защита от излишней самоуверенности модели
        dropout=0.1,            # Отключение части нейронов для генерализации
        weight_decay=0.001,     # Штраф за слишком сложные веса
    )
