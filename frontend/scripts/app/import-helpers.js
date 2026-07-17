import {
  trainClassesForMode,
  yoloClassOrderForTask,
} from "./train-classes.js";
import { formatDisplayName } from "./format-display.js";

import {
  annotationSourceOf,
  annotationTypeOf,
} from "./annotation-model.js";
/** @param {string} s */
export function coerceImportedImageStatus(s) {
  const allowed = new Set([
    "idle",
    "queued",
    "processing",
    "detected",
    "empty",
    "failed",
    "skipped",
  ]);
  return allowed.has(s) ? /** @type {any} */ (s) : "detected";
}

/** @param {any} im */
export function reconcileImageStatusWithDetections(im) {
  const n = Array.isArray(im.detections) ? im.detections.length : 0;
  if (n > 0 && im.status === "empty") im.status = "detected";
  if (n === 0 && im.status === "detected") im.status = "empty";
}

/** @param {any} im */
export function inferImageStatusFromDetections(im) {
  if (im?.error) return "failed";
  const n = Array.isArray(im?.detections) ? im.detections.length : 0;
  return n > 0 ? "detected" : "empty";
}

/** Кадр не может быть одновременно пропущенным и проверенным. */
export function reconcileReviewedAndSkippedStatus(im) {
  if (!im) return;
  if (im.reviewed === true && im.status === "skipped") {
    im.status = inferImageStatusFromDetections(im);
    reconcileImageStatusWithDetections(im);
  }
  if (im.status === "skipped") {
    im.reviewed = false;
  }
}

/**
 * Восстановление детекций из annotations/*.json (формат экспорта).
 * @param {any} annot
 */
export function rawDetectionsFromAnnotationExport(annot) {
  const explicitBboxes = Array.isArray(annot?.annotations?.bboxes)
    ? annot.annotations.bboxes.map((d) => ({ ...d, annotation_type: "detect" }))
    : [];
  const explicitMasks = Array.isArray(annot?.annotations?.masks)
    ? annot.annotations.masks.map((d) => ({ ...d, annotation_type: "seg" }))
    : [];
  const list =
    explicitBboxes.length || explicitMasks.length
      ? [...explicitBboxes, ...explicitMasks]
      : Array.isArray(annot?.detections)
        ? annot.detections
        : [];
  return list.map((d, i) => {
    const bbox = d?.bbox && typeof d.bbox === "object" ? d.bbox : {};
    const rawBox = Array.isArray(d?.box) ? d.box : [];
    const x1 = Number(bbox.x1 ?? rawBox[0]);
    const y1 = Number(bbox.y1 ?? rawBox[1]);
    const x2 = Number(bbox.x2 ?? rawBox[2]);
    const y2 = Number(bbox.y2 ?? rawBox[3]);
    const cn = String(d?.class_name ?? "")
      .trim()
      .toLowerCase();
    const annotationType = annotationTypeOf(d);
    const mode = annotationType === "seg" ? "segmentation" : "detection";
    const trainClasses = trainClassesForMode(mode);
    const yoloOrder = yoloClassOrderForTask(mode);
    const fallbackClass = trainClasses[0];
    let tc = trainClasses.find((t) => t.name === cn);
    if (!tc && typeof d?.class_id === "number") {
      const byYolo = yoloOrder[d.class_id];
      if (byYolo) tc = trainClasses.find((t) => t.name === byYolo);
    }
    const clsId =
      tc?.id ??
      (Number.isInteger(d?.class_id) ? d.class_id : fallbackClass.id);
    const clsName = tc?.name || cn || fallbackClass.name;
    const detId = typeof d?.id === "number" ? d.id : i;
    const conf = typeof d?.confidence === "number" ? d.confidence : 0;
    const src = annotationSourceOf(d);
    const segment = Array.isArray(d?.segment)
      ? d.segment
          .map((p) => {
            if (Array.isArray(p) && p.length >= 2) {
              const x = Number(p[0]);
              const y = Number(p[1]);
              return Number.isFinite(x) && Number.isFinite(y) ? [x, y] : null;
            }
            if (p && typeof p === "object") {
              const x = Number(p.x);
              const y = Number(p.y);
              return Number.isFinite(x) && Number.isFinite(y) ? [x, y] : null;
            }
            return null;
          })
          .filter(Boolean)
      : [];
    const row = {
      id: detId,
      cls_id: clsId,
      cls_name: clsName,
      conf,
      box: [x1, y1, x2, y2],
      source: src,
      annotation_type: annotationType,
    };
    if (segment.length >= 3) row.segment = segment;
    return row;
  });
}

/**
 * Слот в UI после импорта — имя файла после экспорта (001.png), как в архиве.
 * @param {any} row
 * @param {string} stem exportStem
 * @param {number} ordinal1Based
 */
export function displayNameAfterProjectImport(row, stem, ordinal1Based) {
  const nested =
    row?.imageNames && typeof row.imageNames.afterExport === "string"
      ? row.imageNames.afterExport.trim()
      : "";
  if (nested) return nested;

  const exported =
    typeof row?.exportedImageFileName === "string"
      ? row.exportedImageFileName.trim()
      : "";
  if (exported) return exported;

  const st = typeof stem === "string" ? stem.trim() : "";
  if (st) return `${st}.png`;

  if (typeof row?.displayName === "string" && row.displayName.trim()) {
    return row.displayName.trim();
  }

  return formatDisplayName(ordinal1Based);
}
