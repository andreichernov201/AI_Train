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
- Модель ожидается по пути: `runs/detect/train/weights/best.pt` (папка `runs/` в `.gitignore`, чтобы не тащить большие артефакты в git)

