import os
import cv2
import numpy as np
from ultralytics import YOLO

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.environ.get("AI_TRAIN_SEG_MODEL_PATH", "")
INPUT_DIR = os.path.join(ROOT_DIR, "yolo-seg", "test", "images")
OUTPUT_DIR = os.path.join(ROOT_DIR, "runs", "segmentation", "isolated")

if not MODEL_PATH or not os.path.exists(MODEL_PATH):
    print(f"❌ ОШИБКА: Модель не найдена по пути {MODEL_PATH}")
    exit()

if not os.path.exists(INPUT_DIR):
    print(f"❌ ОШИБКА: Входная папка не найдена по пути {INPUT_DIR}")
    exit()

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)
all_files = os.listdir(INPUT_DIR)
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG')
image_files = [f for f in all_files if f.endswith(valid_extensions)]

processed_count = 0
skipped_count = 0

for img_name in image_files:
    img_path = os.path.join(INPUT_DIR, img_name)
    results = model.predict(source=img_path, imgsz=1280, retina_masks=True, conf=0.25, verbose=False)

    res = results[0]
    if res.masks is None:
        skipped_count += 1
        continue

    original_img = res.orig_img
    h, w, _ = original_img.shape
    black_bg = np.zeros_like(original_img)

    polygon = res.masks.xy[0].astype(np.int32)
    mask_img = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_img, [polygon], 255)

    isolated_body = np.where(mask_img[:, :, None] == 255, original_img, black_bg)
    bbox = res.boxes.xyxy[0].cpu().numpy().astype(int)
    cropped_body = isolated_body[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    output_path = os.path.join(OUTPUT_DIR, img_name)
    cv2.imwrite(output_path, cropped_body)
    processed_count += 1

print("\n--- СТАТИСТИКА МАССОВОЙ ВЫРЕЗКИ ---")
print(f"Успешно изолировано и сохранено корпусов: {processed_count}")
print(f"Пропущено изображений (поезд не найден): {skipped_count}")
print(f"Чистые корпуса сохранены в: {OUTPUT_DIR}")
