import os

from ultralytics import YOLO

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.environ.get("AI_TRAIN_SEG_MODEL_PATH", "")
SOURCE_DIR = os.environ.get(
    "AI_TRAIN_SEG_SOURCE",
    os.path.join(ROOT_DIR, "dataset", "segmentation", "images", "val"),
)
OUTPUT_DIR = os.path.join(ROOT_DIR, "runs", "segment", "tools", "predictions")

if not MODEL_PATH or not os.path.exists(MODEL_PATH):
    print(f"❌ ОШИБКА: Файл модели не найден по пути {MODEL_PATH}")
    raise SystemExit(1)

if not os.path.exists(SOURCE_DIR):
    print(f"❌ ОШИБКА: Папка с изображениями не найдена: {SOURCE_DIR}")
    raise SystemExit(1)

model = YOLO(MODEL_PATH)
model.predict(
    source=SOURCE_DIR,
    imgsz=1280,
    conf=0.25,
    retina_masks=True,
    save=True,
    project=OUTPUT_DIR,
    name="result",
    exist_ok=True,
)

print("\n✅ ОБРАБОТКА ЗАВЕРШЕНА!")
print(f"Результаты с наложенными масками сохранены в: {OUTPUT_DIR}/result")
