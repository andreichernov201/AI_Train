export const MAX_DIRECT_IMAGES = 500;
export const MAX_ZIP_IMAGES = 2000;
export const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
/** Один файл видео в браузере (извлечение кадров в батч). */
export const MAX_VIDEO_BYTES = 2 * 1024 * 1024 * 1024;
/** Максимум кадров из видео в одном проекте. */
export const MAX_VIDEO_FRAMES = 2000;
/** Интервал между кадрами при извлечении из видео (сек). */
export const DEFAULT_VIDEO_FRAME_INTERVAL_SEC = 1;
export const ZIP_IMAGE_EXT_RE = /\.(jpe?g|png|webp)$/i;
export const DATA_YAML_RE = /(^|\/)data\.ya?ml$/i;
export const WORKSPACE_DB_NAME = "ai-train-workspace";
export const WORKSPACE_DB_VERSION = 1;
export const WORKSPACE_STORE_NAME = "workspaces";
/** Дебаунс автосохранения: реже снимает нагрузку на память/IndexedDB при движении слайдеров и т.п. */
export const WORKSPACE_AUTOSAVE_DELAY_MS = 1200;
export const PROJECT_EXPORT_JSON_VERSION = "2.0";
export const HANDLE_HIT_PX = 12;
/** Минимальная сторона bbox в координатах изображения (resize / новый box). */
export const MIN_BOX_SIDE = 5;
/** Отступ при экспорте кропа bbox: доля ширины/высоты bbox с каждой стороны. */
export const CROP_BBOX_PADDING_RATIO = 0.1;

export const DETECT_ALL_CONCURRENCY = 2;
/** Ожидание ответа /api/detect; без этого зависший сервер держит кадр в «processing» бесконечно */
export const DETECT_API_FETCH_TIMEOUT_MS = 180000;
/** Мин. интервал между полными перерисовками списка батча во время пакета */
export const DETECT_ALL_BATCH_NAV_THROTTLE_MS = 150;

export const IMPORT_SUMMARY_NAME_PREVIEW = 14;
