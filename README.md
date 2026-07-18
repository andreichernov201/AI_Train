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

- Фронтенд: `frontend/index.html`, стили `frontend/style.css`, скрипты `frontend/scripts/app/`.
- Служебные скрипты находятся только в `tools/`: `detect/`, `segmentation/` и `frontend/`.
- Датасеты, веса, результаты запусков, архивы и логи являются локальными и не входят в Git.
- Пути к локальным весам задаются через `AI_TRAIN_MODEL_PATH` и `AI_TRAIN_SEG_MODEL_PATH`.
- В шапке сайта доступны режимы `Разметка / Анализ`. В режиме разметки `/api/detect` создаёт bbox классов `train` и `number`, а `/api/segment` — цветные полигональные маски классов `body`, `autocoupler`, `axlebox`, `bogie`, `hose`; оба результата могут одновременно находиться на одном изображении.

