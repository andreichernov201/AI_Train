import os
from ultralytics import YOLO

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.environ.get("AI_TRAIN_SEG_MODEL_PATH", "")
TEST_DIR = os.path.join(ROOT_DIR, "yolo-seg", "test", "images")
OUTPUT_DIR = os.path.join(ROOT_DIR, "runs", "segmentation", "test")

if not MODEL_PATH or not os.path.exists(MODEL_PATH):
    print(f"❌ ОШИБКА: Файл модели не найден по пути {MODEL_PATH}")
    exit()

if not os.path.exists(TEST_DIR):
    print(f"❌ ОШИБКА: Папка с тестами не найдена по пути {TEST_DIR}")
    exit()

model = YOLO(MODEL_PATH)

results = model.predict(
    source=TEST_DIR,
    imgsz=1280,
    conf=0.25,
    retina_masks=True,
    save=True,
    project=OUTPUT_DIR,
    name="predictions",
    exist_ok=True  # Исправленный параметр
)

print(f"\n✅ ОБРАБОТКА ЗАВЕРШЕНА!")
print(f"Результаты с наложенными масками сохранены в папку: {OUTPUT_DIR}/predictions")
