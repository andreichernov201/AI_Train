/** Фиксированные классы для каждого режима разметки. */
export const DETECTION_CLASS_NAMES = Object.freeze(["train", "number"]);
export const SEGMENTATION_CLASS_NAMES = Object.freeze([
  "body",
  "autocoupler",
  "axlebox",
  "bogie",
  "hose",
]);

/** @type {string[]} */
let detectionClassOrder = [...DETECTION_CLASS_NAMES];
/** @type {string[]} */
let segmentationClassOrder = [...SEGMENTATION_CLASS_NAMES];

/** @param {unknown} mode */
function isSegmentationMode(mode) {
  return mode === "segmentation" || mode === "segment";
}

/** @param {unknown} mode */
function fixedNamesForMode(mode) {
  return isSegmentationMode(mode)
    ? SEGMENTATION_CLASS_NAMES
    : DETECTION_CLASS_NAMES;
}

/** @param {unknown} mode */
function classOrderForMode(mode) {
  return isSegmentationMode(mode)
    ? segmentationClassOrder
    : detectionClassOrder;
}

/** @param {unknown} mode @returns {Array<{ name: string; id: number }>} */
export function trainClassesForMode(mode = "detection") {
  return classOrderForMode(mode).map((name, id) => ({ name, id }));
}

/** @param {"detect"|"segment"|"detection"|"segmentation"} [task] */
export function yoloClassOrderForTask(task = "detect") {
  return [...classOrderForMode(task)];
}

/**
 * Сброс порядка классов. Без режима сбрасываются оба набора.
 * @param {unknown} [mode]
 */
export function resetTrainClassesToDefault(mode) {
  if (mode === undefined || !isSegmentationMode(mode)) {
    detectionClassOrder = [...DETECTION_CLASS_NAMES];
  }
  if (mode === undefined || isSegmentationMode(mode)) {
    segmentationClassOrder = [...SEGMENTATION_CLASS_NAMES];
  }
}

/** @param {string[]} names @param {unknown} [mode] */
export function trainClassNamesMatchFixedSet(names, mode) {
  if (!Array.isArray(names)) return false;
  const actual = names.map((name) => String(name).trim().toLowerCase()).sort();
  const expected = [...fixedNamesForMode(mode)].sort();
  return (
    actual.length === expected.length &&
    actual.every((name, i) => name === expected[i])
  );
}

/**
 * Применяет порядок из импортированного YAML/проекта, если набор классов
 * совпадает с одним из двух фиксированных наборов сайта.
 * @param {string[]} names
 * @param {unknown} [mode]
 */
export function applyTrainClassOrderFromNames(names, mode) {
  const normalized = Array.isArray(names)
    ? names.map((name) => String(name).trim().toLowerCase())
    : [];
  const resolvedMode =
    mode === undefined
      ? trainClassNamesMatchFixedSet(normalized, "segmentation")
        ? "segmentation"
        : "detection"
      : mode;

  if (!trainClassNamesMatchFixedSet(normalized, resolvedMode)) {
    resetTrainClassesToDefault(resolvedMode);
    return false;
  }

  if (isSegmentationMode(resolvedMode)) {
    segmentationClassOrder = normalized;
  } else {
    detectionClassOrder = normalized;
  }
  return true;
}
