export const MAX_DIRECT_IMAGES = 100;
export const MAX_ZIP_IMAGES = 1000;
export const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
export const ZIP_IMAGE_EXT_RE = /\.(jpe?g|png|webp)$/i;
export const DATA_YAML_RE = /(^|\/)data\.ya?ml$/i;
export const WORKSPACE_DB_NAME = "ai-train-workspace";
export const WORKSPACE_DB_VERSION = 1;
export const WORKSPACE_STORE_NAME = "workspaces";
export const WORKSPACE_AUTOSAVE_DELAY_MS = 300;
export const PROJECT_EXPORT_JSON_VERSION = "1.0";
export const HANDLE_HIT_PX = 12;
/** Минимальная сторона bbox в координатах изображения (resize / новый box). */
export const MIN_BOX_SIDE = 5;

export const DETECT_ALL_CONCURRENCY = 2;
/** Ожидание ответа /api/detect; без этого зависший сервер держит кадр в «processing» бесконечно */
export const DETECT_API_FETCH_TIMEOUT_MS = 180000;
/** Мин. интервал между полными перерисовками списка батча во время пакета */
export const DETECT_ALL_BATCH_NAV_THROTTLE_MS = 150;

export const IMPORT_SUMMARY_NAME_PREVIEW = 14;
