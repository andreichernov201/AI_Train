import { trainClassesForMode } from "./train-classes.js";

export const ANNOTATION_TYPE_DETECT = "detect";
export const ANNOTATION_TYPE_SEG = "seg";
export const MODEL_STATE_STATUSES = Object.freeze([
  "not_run",
  "running",
  "ready",
  "error",
  "reviewed",
]);

/** @param {any} value */
export function annotationTypeOf(value) {
  const raw = String(
    value?.annotation_type ?? value?.annotationType ?? value?.type ?? ""
  ).toLowerCase();
  if (["seg", "segment", "segmentation", "mask", "polygon"].includes(raw)) {
    return ANNOTATION_TYPE_SEG;
  }
  if (["detect", "detection", "bbox", "box"].includes(raw)) {
    return ANNOTATION_TYPE_DETECT;
  }
  return Array.isArray(value?.segment) && value.segment.length >= 3
    ? ANNOTATION_TYPE_SEG
    : ANNOTATION_TYPE_DETECT;
}

/** @param {any} value */
export function annotationSourceOf(value) {
  if (value?.source === "human") return "human";
  if (value?.source === "detect" || value?.source === "seg") {
    return value.source;
  }
  if (value?.source === "model") return annotationTypeOf(value);
  /** Неизвестный legacy-источник считаем ручным, чтобы повторный запуск его не удалил. */
  return "human";
}

/** @param {any} value */
export function isDetectAnnotation(value) {
  return annotationTypeOf(value) === ANNOTATION_TYPE_DETECT;
}

/** @param {any} value */
export function isSegAnnotation(value) {
  return annotationTypeOf(value) === ANNOTATION_TYPE_SEG;
}

/** @param {"detect"|"seg"} type */
export function classesForAnnotationType(type) {
  return trainClassesForMode(
    type === ANNOTATION_TYPE_SEG ? "segmentation" : "detection"
  );
}

/** @returns {{status:"not_run",error:null,updatedAt:null,revision:number}} */
export function createEmptyModelState() {
  return { status: "not_run", error: null, updatedAt: null, revision: 0 };
}

export function createEmptyModelStates() {
  return {
    detect: createEmptyModelState(),
    seg: createEmptyModelState(),
  };
}

/** @param {any} raw */
export function normalizeSingleModelState(raw) {
  const status = MODEL_STATE_STATUSES.includes(raw?.status)
    ? raw.status
    : "not_run";
  return {
    status,
    error: typeof raw?.error === "string" ? raw.error : null,
    updatedAt: typeof raw?.updatedAt === "string" ? raw.updatedAt : null,
    revision:
      Number.isInteger(raw?.revision) && raw.revision >= 0 ? raw.revision : 0,
  };
}

/**
 * Миграция старого одномодельного состояния. Тип старых аннотаций определяется
 * по полигону, а source="model" — по соответствующему типу.
 * @param {any} rawStates
 * @param {{annotations?:any[],legacyStatus?:string,legacyReviewed?:boolean,legacyError?:string|null,legacyMode?:string}} [legacy]
 */
export function normalizeModelStates(rawStates, legacy = {}) {
  if (rawStates && typeof rawStates === "object") {
    return {
      detect: normalizeSingleModelState(rawStates.detect),
      seg: normalizeSingleModelState(rawStates.seg),
    };
  }

  const states = createEmptyModelStates();
  const annotations = Array.isArray(legacy.annotations)
    ? legacy.annotations
    : [];
  for (const type of [ANNOTATION_TYPE_DETECT, ANNOTATION_TYPE_SEG]) {
    const rows = annotations.filter((d) => annotationTypeOf(d) === type);
    if (!rows.length) continue;
    states[type].status =
      legacy.legacyReviewed || rows.some((d) => d?.source === "human")
        ? "reviewed"
        : "ready";
  }

  const legacyType =
    legacy.legacyMode === "segmentation"
      ? ANNOTATION_TYPE_SEG
      : ANNOTATION_TYPE_DETECT;
  if (
    legacy.legacyStatus === "empty" &&
    states[legacyType].status === "not_run"
  ) {
    states[legacyType].status = legacy.legacyReviewed ? "reviewed" : "ready";
  } else if (legacy.legacyStatus === "failed") {
    states[legacyType].status = "error";
    states[legacyType].error =
      typeof legacy.legacyError === "string"
        ? legacy.legacyError
        : "Ошибка модели";
  }
  return states;
}

/** @param {string} status */
export function modelStateLabel(status) {
  return (
    {
      not_run: "не запускалась",
      running: "выполняется",
      ready: "готова",
      error: "ошибка",
      reviewed: "проверена вручную",
    }[status] || status
  );
}
