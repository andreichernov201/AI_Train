/** Фиксированные классы сайта: редактор не становится универсальным редактором любых классов. */
export const TRAIN_PART_CLASS_NAMES = ["body", "autocoupler", "axlebox", "bogie"];
/** Индексы классов в batch-экспорте YOLO TXT (0–3), фиксированный порядок классов сайта. */
export const YOLO_TXT_EXPORT_CLASS_ORDER = Object.freeze([...TRAIN_PART_CLASS_NAMES]);
export const DEFAULT_TRAIN_CLASSES = TRAIN_PART_CLASS_NAMES.map((name, id) => ({
  name,
  id,
}));

/** @type {Array<{ name: string; id: number }>} */
export let TRAIN_CLASSES = DEFAULT_TRAIN_CLASSES.map((tc) => ({ ...tc }));

export function resetTrainClassesToDefault() {
  TRAIN_CLASSES = DEFAULT_TRAIN_CLASSES.map((tc) => ({ ...tc }));
}

/** @param {string[]} names */
export function trainClassNamesMatchFixedSet(names) {
  const actual = names.map((name) => String(name).trim().toLowerCase()).sort();
  const expected = [...TRAIN_PART_CLASS_NAMES].sort();
  return (
    actual.length === expected.length &&
    actual.every((name, i) => name === expected[i])
  );
}

/**
 * Если переданный набор имён совпадает с фиксированными классами сайта — порядок берётся из источника, иначе классы сбрасываются по умолчанию.
 * @param {string[]} names
 */
export function applyTrainClassOrderFromNames(names) {
  if (!trainClassNamesMatchFixedSet(names)) {
    resetTrainClassesToDefault();
    return false;
  }
  TRAIN_CLASSES = names.map((name, id) => ({
    name: String(name).trim().toLowerCase(),
    id,
  }));
  return true;
}
