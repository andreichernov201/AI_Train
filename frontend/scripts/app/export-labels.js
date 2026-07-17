export const EXPORT_ACTION_LABELS = {
  "current-jpg": "Текущий кадр: фото с разметкой (.jpg)",
  "current-yolo-detect": "Текущий кадр: YOLO Detect (.txt)",
  "current-yolo-seg": "Текущий кадр: YOLO Seg (.txt)",
  "current-annotation-json": "Текущий кадр: аннотация (.json)",
  "batch-png-clean": "Весь батч: PNG ZIP без разметки",
  "batch-png-marked": "Весь батч: PNG ZIP с разметкой",
  "batch-yolo-detect-zip": "Весь батч: YOLO Detect (ZIP)",
  "batch-yolo-seg-zip": "Весь батч: YOLO Seg (ZIP)",
  "batch-annotations-zip": "Весь батч: только аннотации JSON (ZIP)",
  "batch-project-zip": "Весь батч: ZIP полного проекта",
  "batch-crops-zip": "Весь батч: кропы объектов (ZIP)",
};

/**
 * Краткое описание формата для всплывающих окон экспорта (меню и модалка).
 * @type {Record<string, { icon: string, short: string, desc: string, tag: string, scope: "current"|"batch" }>}
 */
export const EXPORT_ACTION_META = {
  "current-jpg": {
    icon: "🖼️",
    short: "Фото с разметкой",
    desc: "JPG текущего кадра с боксами/масками и подписями классов",
    tag: ".jpg",
    scope: "current",
  },
  "current-yolo-detect": {
    icon: "📄",
    short: "YOLO Detect",
    desc: "BBox train и number текущего кадра в формате YOLO Detect",
    tag: ".txt",
    scope: "current",
  },
  "current-yolo-seg": {
    icon: "🎨",
    short: "YOLO Seg",
    desc: "Полигоны пяти классов текущего кадра в формате YOLO Seg",
    tag: ".txt",
    scope: "current",
  },
  "current-annotation-json": {
    icon: "🧾",
    short: "Аннотация JSON",
    desc: "Подробное описание объектов текущего кадра в JSON",
    tag: ".json",
    scope: "current",
  },
  "batch-png-clean": {
    icon: "🖼️",
    short: "PNG без разметки",
    desc: "Исходные изображения батча без боксов и масок",
    tag: ".zip",
    scope: "batch",
  },
  "batch-png-marked": {
    icon: "🖍️",
    short: "PNG с разметкой",
    desc: "Изображения батча с нарисованными боксами/масками",
    tag: ".zip",
    scope: "batch",
  },
  "batch-yolo-detect-zip": {
    icon: "📄",
    short: "YOLO Detect",
    desc: "BBox train и number, classes.txt и data.yaml",
    tag: ".zip",
    scope: "batch",
  },
  "batch-yolo-seg-zip": {
    icon: "🎨",
    short: "YOLO Seg",
    desc: "Полигоны пяти классов, classes.txt и data.yaml",
    tag: ".zip",
    scope: "batch",
  },
  "batch-annotations-zip": {
    icon: "🧾",
    short: "Только аннотации JSON",
    desc: "Подробные JSON-аннотации без изображений — для готового набора фото",
    tag: ".zip",
    scope: "batch",
  },
  "batch-project-zip": {
    icon: "🗂️",
    short: "Полный проект",
    desc: "Изображения + YOLO-разметка + JSON-аннотации + метаданные проекта",
    tag: ".zip",
    scope: "batch",
  },
  "batch-crops-zip": {
    icon: "✂️",
    short: "Кропы объектов",
    desc: "Вырезанные по bbox объекты, разложенные по папкам классов",
    tag: ".zip",
    scope: "batch",
  },
};
