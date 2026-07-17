# AI Train

Веб-приложение на **FastAPI** с фронтендом из папки `frontend/` для загрузки фото и запуска распознавания.

## Быстрый старт

Установить зависимости:

```bash
python -m pip install -r requirements.txt
```

Запуск сервера:

```bash
python -m uvicorn server:app --reload --host 127.0.0.1 --port 8000
```

Открыть в браузере:
- `http://127.0.0.1:8000/`

## Примечания

- Фронтенд: `frontend/index.html` (стили `frontend/style.css`, скрипты `frontend/scripts/app.js`)
- Модель detect: `runs/detect/yolo11l_1280_anti_overfit/weights/best.pt` (папка `runs/` в `.gitignore`)
- На сервере без `runs/` можно указать путь: `$env:AI_TRAIN_MODEL_PATH="D:\path\to\best.pt"` (PowerShell) перед запуском uvicorn
- Модель segmentation: `runs/segmentation/weights/best.pt`
- Для segmentation можно переопределить путь: `$env:AI_TRAIN_SEG_MODEL_PATH="D:\path\to\best.pt"`
- В шапке сайта доступны режимы `Разметка / Анализ`. В режиме разметки `/api/detect` создаёт bbox классов `train` и `number`, а `/api/segment` — цветные полигональные маски классов `body`, `autocoupler`, `axlebox`, `bogie`, `hose`; оба результата могут одновременно находиться на одном изображении.

