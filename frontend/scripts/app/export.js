import { COLORS } from "./colors.js";
import {
  CROP_BBOX_PADDING_RATIO,
  PROJECT_EXPORT_JSON_VERSION,
} from "./constants.js";
import {
  exportNumberPadWidth,
  exportNumberedFileStem,
  exportNumberedPngName,
} from "./format-display.js";
import { yoloClassOrderForTask } from "./train-classes.js";
import { serializePanelState } from "./workspace.js";
import {
  annotationSourceOf,
  annotationTypeOf,
  isDetectAnnotation,
  isSegAnnotation,
} from "./annotation-model.js";

function getColorByClass(clsId) {
  return COLORS[Math.abs(clsId) % COLORS.length];
}

const UNCATEGORIZED_EXPORT_DIR = "uncategorized";

/** @param {number} value @param {number} min @param {number} max */
function clamp(value, min, max) {
  if (!Number.isFinite(value)) return min;
  return Math.max(min, Math.min(max, value));
}

/** @param {any} d */
function validSegmentPoints(d) {
  if (!Array.isArray(d?.segment) || d.segment.length < 3) return [];
  return d.segment
    .map((p) => {
      if (!Array.isArray(p) || p.length < 2) return null;
      const x = Number(p[0]);
      const y = Number(p[1]);
      return Number.isFinite(x) && Number.isFinite(y) ? [x, y] : null;
    })
    .filter(Boolean);
}

/** @param {any} d */
function hasValidSegment(d) {
  return validSegmentPoints(d).length >= 3;
}

/**
 * @param {CanvasRenderingContext2D} ctx
 * @param {Array<[number, number]>} segment
 * @param {number} sx
 * @param {number} sy
 */
function traceSegmentPath(ctx, segment, sx, sy) {
  segment.forEach(([px, py], i) => {
    const x = px * sx;
    const y = py * sy;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();
}

/** @param {string|null|undefined} raw */
function exportCategoryDirName(raw) {
  const base = String(raw || "").trim();
  if (!base) return UNCATEGORIZED_EXPORT_DIR;
  const safe = base.replace(/[\\/:*?"<>|]/g, "_").replace(/\s+/g, "_");
  return safe || UNCATEGORIZED_EXPORT_DIR;
}
export function normalizeExportCategoryDir(raw) {
  return exportCategoryDirName(raw);
}

/** Ключ для счётчиков нумерации: одна группа на категорию. */
function exportCategoryPlanKey(raw) {
  return exportCategoryDirName(raw);
}

/**
 * @param {any} parent
 * @param {string} categoryDir
 */
function exportFolderForCategory(parent, categoryDir) {
  const sub = parent.folder(categoryDir);
  if (!sub) throw new Error("Не удалось создать папку категории в архиве.");
  return sub;
}

/**
 * @param {any[]} items
 * @param {number} startNumber
 */
function buildPerCategoryNumberingPlan(items, startNumbersByCategory = {}) {
  /** @type {Map<string, number>} */
  const counts = new Map();
  for (const im of items) {
    const key = exportCategoryPlanKey(im?.category);
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  /** @type {Map<string, number>} */
  const starts = new Map();
  /** @type {Map<string, number>} */
  const pads = new Map();
  for (const [key, count] of counts.entries()) {
    const start = normalizeExportStartNumber(startNumbersByCategory[key] ?? 1, count);
    starts.set(key, start);
    pads.set(key, exportNumberPadWidth(start, count));
  }
  return { counts, pads, starts };
}

/** @param {any} im */
export function isImageAwaitingDetection(im) {
  return (
    im?.status === "idle" ||
    im?.status === "queued" ||
    im?.status === "processing"
  );
}

/**
 * Кадр с результатом нейросети: обработан, есть bbox (или пустой после распознавания), не пропущен без review.
 * @param {any} im
 */
export function isImageExportableAsDetected(im) {
  if (!im?.blob || im.blob.size === 0) return false;
  if (im.status === "failed") return false;
  if (im.status === "skipped") return false;
  if (im.reviewed) return true;
  if (isImageAwaitingDetection(im)) return false;
  const hasDetections = Array.isArray(im.detections) && im.detections.length > 0;
  return (
    hasDetections ||
    im.status === "detected" ||
    im.status === "empty"
  );
}

/** @param {any} im */
export function isImageExportableAsManuallyReviewed(im) {
  return (
    im?.reviewed === true &&
    im.status !== "skipped" &&
    !!im.blob &&
    im.blob.size > 0 &&
    im.status !== "failed"
  );
}

/**
 * @typedef {{ manualOnly: boolean, includeUnreviewed: boolean, includeEmptyYoloLabels?: boolean, includeAnnotations?: boolean }} ExportReviewFilter
 */

/** @param {any} im @param {ExportReviewFilter} filter @param {string} [actionKind] */
export function isImageEligibleForEmptyYoloLabelExport(
  im,
  filter,
  actionKind = ""
) {
  const task = yoloTaskForAction(actionKind);
  if (task && !imageHasYoloExportDetections(im, actionKind)) {
    const stateKey = task === "seg" ? "seg" : "detect";
    const status = im?.modelStates?.[stateKey]?.status;
    if (status !== "ready" && status !== "reviewed") return false;
    if (filter.manualOnly) return status === "reviewed";
    if (!filter.includeUnreviewed) return status === "reviewed";
    return true;
  }
  if (filter.manualOnly) {
    return isImageExportableAsManuallyReviewed(im);
  }
  if (!filter.includeUnreviewed && !im.reviewed) return false;
  if (im.reviewed) return true;
  return isImageExportableAsDetected(im);
}

function yoloTaskForAction(actionKind) {
  if (actionKind === "batch-yolo-detect-zip") return "detect";
  if (actionKind === "batch-yolo-seg-zip") return "seg";
  return null;
}

/** @param {any} im @param {string} [actionKind] */
export function imageHasYoloExportDetections(im, actionKind = "") {
  const task = yoloTaskForAction(actionKind);
  if (!task) return isImageValidForPngZipExport(im);
  return (
    Array.isArray(im?.detections) &&
    im.detections.some(task === "seg" ? isSegAnnotation : isDetectAnnotation)
  );
}

/**
 * @param {any} im
 * @param {ExportReviewFilter} filter
 * @param {string} [actionKind]
 */
export function matchesBatchExportReviewFilter(im, filter, actionKind = "") {
  if (!im?.blob || im.blob.size === 0) return false;
  if (im.status === "failed") return false;
  if (im.status === "skipped") return false;
  if (isImageAwaitingDetection(im) && !im.reviewed) return false;

  if (
    filter.includeEmptyYoloLabels &&
    yoloTaskForAction(actionKind) &&
    !imageHasYoloExportDetections(im, actionKind)
  ) {
    return isImageEligibleForEmptyYoloLabelExport(im, filter, actionKind);
  }

  if (filter.manualOnly) {
    return isImageExportableAsManuallyReviewed(im);
  }

  const clean = actionKind === "batch-png-clean";
  if (clean) {
    if (im.reviewed) return isImageEligibleForReviewedCleanPngExport(im);
    if (!filter.includeUnreviewed) return false;
    return isImageExportableAsDetected(im);
  }

  const yoloExport =
    actionKind === "batch-yolo-zip" ||
    actionKind === "batch-yolo-detect-zip" ||
    actionKind === "batch-yolo-seg-zip" ||
    actionKind === "batch-project-zip" ||
    actionKind === "batch-annotations-zip";
  if (yoloExport && filter.includeEmptyYoloLabels) {
    return isImageEligibleForEmptyYoloLabelExport(im, filter, actionKind);
  }

  if (yoloTaskForAction(actionKind) && !imageHasYoloExportDetections(im, actionKind)) {
    return false;
  }
  if (!isImageValidForPngZipExport(im)) return false;
  if (!filter.includeUnreviewed && !im.reviewed) return false;
  return true;
}

/** @param {any[]} images @param {string} actionKind @param {ExportReviewFilter} [exportFilter] */
export function computeBatchExportEligibilitySummary(
  images,
  actionKind,
  exportFilter = { manualOnly: false, includeUnreviewed: true }
) {
  const reasons = {
    empty: 0,
    skipped: 0,
    failed: 0,
    noBbox: 0,
    pending: 0,
    noBlob: 0,
    notReviewed: 0,
  };
  let exportable = 0;
  let exportableEmptyLabels = 0;
  const countEmptyLabels =
    actionKind === "batch-yolo-zip" ||
    actionKind === "batch-yolo-detect-zip" ||
    actionKind === "batch-yolo-seg-zip" ||
    actionKind === "batch-project-zip" ||
    actionKind === "batch-annotations-zip";

  for (const im of images) {
    if (matchesBatchExportReviewFilter(im, exportFilter, actionKind)) {
      exportable++;
      if (countEmptyLabels && !imageHasYoloExportDetections(im, actionKind)) {
        exportableEmptyLabels++;
      }
      continue;
    }
    if (!im.blob || im.blob.size === 0) {
      reasons.noBlob++;
      continue;
    }
    if (im.status === "skipped") {
      reasons.skipped++;
      continue;
    }
    if (im.status === "failed") {
      reasons.failed++;
      continue;
    }
    if (im.status === "empty") {
      reasons.empty++;
      continue;
    }
    if (isImageAwaitingDetection(im) && !im.reviewed) {
      reasons.pending++;
      continue;
    }
    if (!im.reviewed && !exportFilter.includeUnreviewed) {
      reasons.notReviewed++;
      continue;
    }
    if (!Array.isArray(im.detections) || im.detections.length === 0) {
      reasons.noBbox++;
      continue;
    }
    reasons.noBbox++;
  }

  const total = images.length;
  return {
    total,
    exportable,
    exportableEmptyLabels,
    skipped: total - exportable,
    reasons,
  };
}

/** @param {string|null|undefined} raw */
export function exportCropClassDirName(raw) {
  const base = String(raw || "").trim().toLowerCase();
  if (!base) return "unknown";
  const safe = base.replace(/[\\/:*?"<>|]/g, "_").replace(/\s+/g, "_");
  return safe || "unknown";
}

/** @param {any} d */
export function isDetectionValidForCropExport(d) {
  if (!isDetectAnnotation(d)) return false;
  if (!d?.box || !Array.isArray(d.box) || d.box.length < 4) return false;
  const [x1, y1, x2, y2] = d.box;
  const w = x2 - x1;
  const h = y2 - y1;
  return Number.isFinite(w) && Number.isFinite(h) && w > 0 && h > 0;
}

/** @param {any} im */
export function collectExportableCropsFromImage(im) {
  if (!Array.isArray(im?.detections)) return [];
  return im.detections.filter(isDetectionValidForCropExport);
}

/** @param {any} im @param {ExportReviewFilter} exportFilter */
function appendImageSkipReasonsForExport(im, exportFilter, reasons) {
  if (!im.blob || im.blob.size === 0) {
    reasons.noBlob++;
    return;
  }
  if (im.status === "skipped") {
    reasons.skipped++;
    return;
  }
  if (im.status === "failed") {
    reasons.failed++;
    return;
  }
  if (im.status === "empty") {
    reasons.empty++;
    return;
  }
  if (isImageAwaitingDetection(im) && !im.reviewed) {
    reasons.pending++;
    return;
  }
  if (exportFilter.manualOnly && im.reviewed !== true) {
    reasons.notReviewed++;
    return;
  }
  if (!im.reviewed && !exportFilter.includeUnreviewed) {
    reasons.notReviewed++;
    return;
  }
  if (!Array.isArray(im.detections) || im.detections.length === 0) {
    reasons.noBbox++;
    return;
  }
  reasons.noCrops++;
}

/**
 * @param {any[]} images
 * @param {ExportReviewFilter} [exportFilter]
 * @param {Set<string>|null} [includedClassDirs] null — все классы
 */
export function computeBatchCropExportSummary(
  images,
  exportFilter = { manualOnly: false, includeUnreviewed: true },
  includedClassDirs = null
) {
  const reasons = {
    empty: 0,
    skipped: 0,
    failed: 0,
    noBbox: 0,
    pending: 0,
    noBlob: 0,
    notReviewed: 0,
    noCrops: 0,
    excludedClass: 0,
  };
  let exportable = 0;
  let exportableImages = 0;
  /** @type {Map<string, number>} */
  const classCounts = new Map();

  for (const im of images) {
    if (!matchesBatchExportReviewFilter(im, exportFilter, "batch-crops-zip")) {
      appendImageSkipReasonsForExport(im, exportFilter, reasons);
      continue;
    }
    const crops = collectExportableCropsFromImage(im);
    if (!crops.length) {
      reasons.noCrops++;
      continue;
    }

    let imageIncluded = false;
    let imageHasSelectedClass = false;
    for (const d of crops) {
      const key = exportCropClassDirName(d.cls_name);
      classCounts.set(key, (classCounts.get(key) || 0) + 1);
      if (includedClassDirs !== null && !includedClassDirs.has(key)) continue;
      imageHasSelectedClass = true;
      exportable++;
    }
    if (imageHasSelectedClass) {
      exportableImages++;
      imageIncluded = true;
    }
    if (!imageIncluded) {
      reasons.excludedClass++;
    }
  }

  const total = images.length;
  return {
    total,
    exportable,
    exportableImages,
    skipped: total - exportableImages,
    classDirs: Array.from(classCounts.keys()).sort((a, b) => a.localeCompare(b, "ru")),
    classCounts,
    reasons,
    isCropSummary: true,
  };
}

/** @param {any[]} images @param {ExportReviewFilter} exportFilter */
export function exportCropClassDirectoriesForAction(images, exportFilter) {
  return computeBatchCropExportSummary(images, exportFilter).classDirs;
}

/** @param {any[]} images @param {string} actionKind @param {ExportReviewFilter} exportFilter */
export function exportCategoryDirectoriesForAction(images, actionKind, exportFilter) {
  /** @type {Set<string>} */
  const dirs = new Set();
  for (const im of images) {
    if (!matchesBatchExportReviewFilter(im, exportFilter, actionKind)) continue;
    dirs.add(exportCategoryPlanKey(im?.category));
  }
  return Array.from(dirs).sort((a, b) => a.localeCompare(b, "ru"));
}

/** @param {any} im */
export function isImageEligibleForReviewedCleanPngExport(im) {
  if (!im.blob || im.blob.size === 0) return false;
  if (im.status === "failed") return false;
  return im.reviewed === true;
}

/** @param {any} im */
export function isImageValidForPngZipExport(im) {
  if (!im.blob || im.blob.size === 0) return false;
  if (!Array.isArray(im.detections) || im.detections.length === 0) return false;
  if (im.status === "failed") return false;
  if (im.status === "skipped") return false;
  if (isImageAwaitingDetection(im) && !im.reviewed) return false;
  return true;
}

/** @param {number} ordinal1Based @param {number} [startNumber=1] @param {number} [padWidth] */
export function pngZipEntryBaseName(ordinal1Based, startNumber = 1, padWidth) {
  return exportNumberedPngName(ordinal1Based, startNumber, padWidth);
}

/** @param {number} startNumber @param {number} count */
export function normalizeExportStartNumber(startNumber, count = 1) {
  const n = Math.floor(Number(startNumber));
  if (!Number.isFinite(n) || n < 1) return 1;
  return n;
}

/** @param {number} startNumber @param {number} count */
export function buildExportNamingPreview(startNumber, count) {
  const start = normalizeExportStartNumber(startNumber, count);
  const pad = exportNumberPadWidth(start, count);
  if (count <= 1) {
    return `${exportNumberedPngName(1, start, pad)}`;
  }
  const first = exportNumberedPngName(1, start, pad);
  const last = exportNumberedPngName(count, start, pad);
  return `${first} … ${last}`;
}

/** @param {Blob} blob @param {string} filename */
export function triggerBlobDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.setTimeout(() => URL.revokeObjectURL(url), 400);
}

/** @param {string} clsName @param {"detect"|"segment"|"detection"|"segmentation"} [task] */
export function yoloTxtExportClassId(clsName, task = "detect") {
  const n = String(clsName ?? "").trim().toLowerCase();
  const order = yoloClassOrderForTask(task);
  const idx = order.indexOf(n);
  return idx;
}

/**
 * @param {any} im
 * @returns {Promise<{width:number,height:number}|null>}
 */
export async function labelNormalizationSize(im) {
  if (im.width > 0 && im.height > 0) {
    return { width: im.width, height: im.height };
  }
  let bmp = null;
  try {
    bmp = await createImageBitmap(im.blob);
    return { width: bmp.width, height: bmp.height };
  } catch {
    return null;
  } finally {
    if (bmp && "close" in bmp && typeof bmp.close === "function") {
      bmp.close();
    }
  }
}

/** @param {any} im */
export async function ensurePositiveExportDimensions(im) {
  const s = await labelNormalizationSize(im);
  if (s && s.width > 0 && s.height > 0) return s;
  const w = typeof im.width === "number" && im.width > 0 ? im.width : 1;
  const h = typeof im.height === "number" && im.height > 0 ? im.height : 1;
  return { width: w, height: h };
}

/**
 * @param {any} im
 * @param {number} iw
 * @param {number} ih
 */
export function buildYoloTxtFileBody(im, iw, ih, task = "detect") {
  if (!(iw > 0 && ih > 0)) return "";
  const segTask = task === "segment" || task === "segmentation" || task === "seg";
  const lines = [];
  for (const d of im.detections) {
    if (segTask ? !isSegAnnotation(d) : !isDetectAnnotation(d)) continue;
    const segment = validSegmentPoints(d);
    const clsId = yoloTxtExportClassId(d.cls_name, segTask ? "segment" : "detect");
    if (clsId < 0) continue;
    if (segTask) {
      if (segment.length < 3) continue;
      const coords = [];
      for (const [x, y] of segment) {
        coords.push((clamp(x / iw, 0, 1)).toFixed(6));
        coords.push((clamp(y / ih, 0, 1)).toFixed(6));
      }
      lines.push([clsId, ...coords].join(" "));
      continue;
    }

    const [x1, y1, x2, y2] = d.box;
    const w = x2 - x1;
    const h = y2 - y1;
    if (!(w > 0 && h > 0)) continue;
    const cx = (x1 + x2) / 2;
    const cy = (y1 + y2) / 2;
    const nx = cx / iw;
    const ny = cy / ih;
    const nw = w / iw;
    const nh = h / ih;
    lines.push(
      [
        clsId,
        clamp(nx, 0, 1).toFixed(6),
        clamp(ny, 0, 1).toFixed(6),
        clamp(nw, 0, 1).toFixed(6),
        clamp(nh, 0, 1).toFixed(6),
      ].join(" ")
    );
  }
  if (!lines.length) return "";
  return `${lines.join("\n")}\n`;
}

/** @param {"detect"|"segment"} [task] @param {{includeDatasetPaths?:boolean}} [opts] */
export function buildProjectExportDataYaml(task = "detect", opts = {}) {
  const order = yoloClassOrderForTask(task);
  const lines = [];
  if (opts.includeDatasetPaths) {
    lines.push("path: .", "train: images", "val: images");
  }
  lines.push(`task: ${task}`, "names:");
  for (let i = 0; i < order.length; i++) {
    lines.push(`  ${i}: ${order[i]}`);
  }
  return `${lines.join("\n")}\n`;
}

/** Список классов для YOLO: одна строка — один class id по порядку. */
export function buildProjectExportClassesTxt(task = "detect") {
  return `${yoloClassOrderForTask(task).join("\n")}\n`;
}

/**
 * @param {any} im
 * @param {{width:number,height:number}} size
 * @param {string} exportStem
 */
export function buildAnnotationExportJsonObject(im, size, exportStem) {
  const serializeAnnotation = (d) => {
    const [bx1, by1, bx2, by2] = d.box;
    const annotation_type = annotationTypeOf(d);
    const segment = validSegmentPoints(d);
    const row = {
      id: d.id,
      annotation_type,
      class_id: yoloTxtExportClassId(
        d.cls_name,
        annotation_type === "seg" ? "segment" : "detect"
      ),
      class_name: d.cls_name,
      confidence: d.conf,
      source: annotationSourceOf(d),
      bbox: { x1: bx1, y1: by1, x2: bx2, y2: by2 },
    };
    if (annotation_type === "seg") {
      row.segment = segment.map(([x, y]) => ({ x, y }));
    }
    return row;
  };
  return {
    schema: "ai-train.annotation.v2",
    exportStem,
    image: {
      id: im.id,
      displayName: im.displayName,
      originalName: im.originalName,
      width: size.width,
      height: size.height,
      imageNames: {
        beforeUpload: im.originalName,
        afterUploadOnSite: im.displayName,
        afterExport: `${exportStem}.png`,
      },
    },
    modelStates: {
      detect: { ...(im.modelStates?.detect || {}) },
      seg: { ...(im.modelStates?.seg || {}) },
    },
    annotations: {
      bboxes: im.detections.filter(isDetectAnnotation).map(serializeAnnotation),
      masks: im.detections.filter(isSegAnnotation).map(serializeAnnotation),
    },
  };
}

/**
 * @param {{
 *   getBatchState: () => any,
 *   showToast: (msg: string, opts?: object) => void,
 *   convertImageBlobToPng: (blob: Blob) => Promise<Blob>,
 *   yieldToMain: () => Promise<void>,
 *   syncExportMenuBusyWithFlags: () => void,
 *   exportMenuToggle: HTMLButtonElement,
 *   exportMenuPanel: HTMLElement,
 *   fmtConf: (x: number) => string,
 * }} deps
 */
export function createZipExportHandlers(deps) {
  const {
    getBatchState,
    showToast,
    convertImageBlobToPng,
    yieldToMain,
    syncExportMenuBusyWithFlags,
    exportMenuToggle,
    exportMenuPanel,
    fmtConf,
  } = deps;

  let pngZipExportInFlight = false;
  let yoloTxtZipExportInFlight = false;
  let annotationsZipExportInFlight = false;
  let fullProjectZipExportInFlight = false;
  let cropsZipExportInFlight = false;

  function busy() {
    return (
      pngZipExportInFlight ||
      yoloTxtZipExportInFlight ||
      annotationsZipExportInFlight ||
      fullProjectZipExportInFlight ||
      cropsZipExportInFlight
    );
  }

  function syncLocalExportMenu() {
    const b = busy();
    exportMenuToggle.disabled = !!b;
    exportMenuPanel.querySelectorAll(".export-menu-item").forEach((btn) => {
      btn.disabled = !!b;
    });
  }

  /**
   * @param {any} im
   * @returns {Promise<Blob>}
   */
  async function renderAnnotatedPngBlobFromItem(im) {
    const bitmap = await createImageBitmap(im.blob);
    try {
      const imgW = bitmap.width;
      const imgH = bitmap.height;
      const ow = im.width > 0 ? im.width : imgW;
      const oh = im.height > 0 ? im.height : imgH;
      const sx = imgW / ow;
      const sy = imgH / oh;

      const canvas = document.createElement("canvas");
      canvas.width = imgW;
      canvas.height = imgH;
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("2d context недоступен");

      ctx.drawImage(bitmap, 0, 0);

      ctx.lineWidth = 2;
      ctx.font = "16px system-ui, sans-serif";

      for (const d of im.detections) {
        const [x1, y1, x2, y2] = d.box;
        const x = x1 * sx;
        const y = y1 * sy;
        const w = (x2 - x1) * sx;
        const h = (y2 - y1) * sy;

        const color = getColorByClass(d.cls_id);
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        const segment = validSegmentPoints(d);
        if (segment.length >= 3) {
          ctx.save();
          ctx.beginPath();
          traceSegmentPath(ctx, segment, sx, sy);
          ctx.globalAlpha = 0.22;
          ctx.fillStyle = color;
          ctx.fill();
          ctx.globalAlpha = 0.95;
          ctx.lineWidth = 2;
          ctx.strokeStyle = color;
          ctx.stroke();
          ctx.restore();
        }
        if (isDetectAnnotation(d)) ctx.strokeRect(x, y, w, h);

        const label = `${d.cls_name} #${d.id + 1} (${fmtConf(d.conf)})`;
        const padX = 8;
        const padY = 6;
        const textW = ctx.measureText(label).width;
        const boxW = textW + padX * 2;
        const boxH = 24;
        const bx = Math.max(0, Math.min(x, imgW - boxW));
        const by = Math.max(0, y - boxH - 2);

        ctx.globalAlpha = 0.85;
        ctx.fillRect(bx, by, boxW, boxH);
        ctx.globalAlpha = 1;
        ctx.fillStyle = "#0b0b0b";
        ctx.fillText(label, bx + padX, by + (boxH - padY));
      }

      return await new Promise((resolve, reject) => {
        canvas.toBlob(
          (b) => {
            if (b) resolve(b);
            else reject(new Error("toBlob не вернул PNG"));
          },
          "image/png"
        );
      });
    } finally {
      if ("close" in bitmap && typeof bitmap.close === "function") {
        bitmap.close();
      }
    }
  }

  /**
   * @param {boolean} withMarkup
   * @param {{ startNumber?: number, startNumbersByCategory?: Record<string, number>, exportFilter?: import("./export.js").ExportReviewFilter, actionKind?: string }} [opts]
   */
  async function exportBatchPngZip(withMarkup, opts = {}) {
    if (pngZipExportInFlight) return;
    if (typeof JSZip !== "function") {
      showToast("JSZip не загружен. Обновите страницу и попробуйте снова.", {
        type: "error",
        durationMs: 6000,
      });
      return;
    }

    const batchState = getBatchState();
    const actionKind = withMarkup ? "batch-png-marked" : "batch-png-clean";
    const exportFilter = opts.exportFilter ?? {
      manualOnly: false,
      includeUnreviewed: true,
    };
    const items = batchState.images.filter((im) =>
      matchesBatchExportReviewFilter(im, exportFilter, actionKind)
    );
    if (!items.length) {
      showToast(
        withMarkup
          ? "Нет кадров по выбранным фильтрам: нужна разметка после распознавания или отметка «проверено»."
          : "Нет кадров по выбранным фильтрам проверки.",
        { type: "warning", durationMs: 4600 }
      );
      return;
    }

    pngZipExportInFlight = true;
    syncLocalExportMenu();
    syncExportMenuBusyWithFlags();

    try {
      const zip = new JSZip();
      const folder = zip.folder("images");
      if (!folder) throw new Error("Не удалось создать папку images в архиве.");

      const plan = buildPerCategoryNumberingPlan(items, opts.startNumbersByCategory);
      /** @type {Map<string, number>} */
      const categoryOrdinals = new Map();

      for (const im of items) {
        const planKey = exportCategoryPlanKey(im.category);
        const catDirName = exportCategoryDirName(im.category);
        const next = (categoryOrdinals.get(planKey) || 0) + 1;
        categoryOrdinals.set(planKey, next);
        const name = pngZipEntryBaseName(
          next,
          plan.starts.get(planKey),
          plan.pads.get(planKey)
        );
        const targetFolder = exportFolderForCategory(folder, catDirName);
        const pngBlob = withMarkup
          ? await renderAnnotatedPngBlobFromItem(im)
          : await convertImageBlobToPng(im.blob);
        targetFolder.file(name, pngBlob);
        await yieldToMain();
      }

      const zipBlob = await zip.generateAsync({ type: "blob" });
      const stamp = new Date().toISOString().replace(/[:.]/g, "-");
      const fname = withMarkup
        ? `images_png_marked_${stamp}.zip`
        : `images_png_clean_${stamp}.zip`;
      triggerBlobDownload(zipBlob, fname);
      showToast(
        withMarkup
          ? `Экспорт завершён: архив с разметкой, ${items.length} файлов в images/.`
          : `Экспорт завершён: архив без разметки, ${items.length} файлов в images/.`,
        { type: "success" }
      );
    } catch (err) {
      console.warn("[export png zip]", err);
      showToast(
        `Ошибка экспорта ZIP: ${err && err.message ? err.message : String(err)}`,
        { type: "error", durationMs: 6500 }
      );
    } finally {
      pngZipExportInFlight = false;
      syncLocalExportMenu();
      syncExportMenuBusyWithFlags();
    }
  }

  /** @param {{ startNumber?: number, startNumbersByCategory?: Record<string, number>, exportFilter?: import("./export.js").ExportReviewFilter }} [opts] */
  async function exportBatchYoloTxtZip(opts = {}) {
    if (yoloTxtZipExportInFlight) return;
    if (typeof JSZip !== "function") {
      showToast("JSZip не загружен. Обновите страницу и попробуйте снова.", {
        type: "error",
        durationMs: 6000,
      });
      return;
    }

    const batchState = getBatchState();
    const task =
      opts.task === "segment" || opts.task === "seg" ? "segment" : "detect";
    const actionKind =
      task === "segment" ? "batch-yolo-seg-zip" : "batch-yolo-detect-zip";
    const exportFilter = opts.exportFilter ?? {
      manualOnly: false,
      includeUnreviewed: true,
    };
    const items = batchState.images.filter((im) =>
      matchesBatchExportReviewFilter(im, exportFilter, actionKind)
    );
    if (!items.length) {
      showToast(
        "Нет кадров для экспорта YOLO по выбранным фильтрам проверки.",
        { type: "warning", durationMs: 4200 }
      );
      return;
    }

    yoloTxtZipExportInFlight = true;
    syncLocalExportMenu();
    syncExportMenuBusyWithFlags();

    try {
      const zip = new JSZip();
      const folderLabels = zip.folder("labels");
      const folderImages = zip.folder("images");
      if (!folderLabels || !folderImages) {
        throw new Error("Не удалось создать папки images/labels в архиве.");
      }

      const plan = buildPerCategoryNumberingPlan(items, opts.startNumbersByCategory);
      /** @type {Map<string, number>} */
      const categoryOrdinals = new Map();

      for (const im of items) {
        const planKey = exportCategoryPlanKey(im.category);
        const catDirName = exportCategoryDirName(im.category);
        const next = (categoryOrdinals.get(planKey) || 0) + 1;
        categoryOrdinals.set(planKey, next);
        const stem = exportNumberedFileStem(
          next,
          plan.starts.get(planKey),
          plan.pads.get(planKey)
        );
        const targetLabelFolder = exportFolderForCategory(folderLabels, catDirName);
        const targetImageFolder = exportFolderForCategory(folderImages, catDirName);
        const size = await labelNormalizationSize(im);
        const body =
          size && size.width > 0 && size.height > 0
            ? buildYoloTxtFileBody(im, size.width, size.height, task)
            : "";
        targetLabelFolder.file(`${stem}.txt`, body || "");
        targetImageFolder.file(`${stem}.png`, await convertImageBlobToPng(im.blob));
        await yieldToMain();
      }

      zip.file("data.yaml", buildProjectExportDataYaml(task, { includeDatasetPaths: true }));
      zip.file("classes.txt", buildProjectExportClassesTxt(task));
      const zipBlob = await zip.generateAsync({ type: "blob" });
      const stamp = new Date().toISOString().replace(/[:.]/g, "-");
      triggerBlobDownload(zipBlob, `yolo_${task}_${stamp}.zip`);
      showToast(
        `Экспорт завершён: YOLO ${task === "segment" ? "Seg" : "Detect"}, ${items.length} пар images/labels.`,
        { type: "success" }
      );
    } catch (err) {
      console.warn("[export yolo txt zip]", err);
      showToast(
        `Ошибка экспорта ZIP: ${err && err.message ? err.message : String(err)}`,
        { type: "error", durationMs: 6500 }
      );
    } finally {
      yoloTxtZipExportInFlight = false;
      syncLocalExportMenu();
      syncExportMenuBusyWithFlags();
    }
  }

  /** @param {{ startNumber?: number, startNumbersByCategory?: Record<string, number>, exportFilter?: import("./export.js").ExportReviewFilter }} [opts] */
  async function exportBatchAnnotationsZip(opts = {}) {
    if (annotationsZipExportInFlight) return;
    if (typeof JSZip !== "function") {
      showToast("JSZip не загружен. Обновите страницу и попробуйте снова.", {
        type: "error",
        durationMs: 6000,
      });
      return;
    }

    const batchState = getBatchState();
    const exportFilter = opts.exportFilter ?? {
      manualOnly: false,
      includeUnreviewed: true,
    };
    const items = batchState.images.filter((im) =>
      matchesBatchExportReviewFilter(im, exportFilter, "batch-annotations-zip")
    );
    if (!items.length) {
      showToast(
        "Нет кадров для экспорта аннотаций по выбранным фильтрам проверки.",
        { type: "warning", durationMs: 4200 }
      );
      return;
    }

    annotationsZipExportInFlight = true;
    syncLocalExportMenu();
    syncExportMenuBusyWithFlags();

    try {
      const zip = new JSZip();
      const folder = zip.folder("annotations");
      if (!folder) throw new Error("Не удалось создать папку annotations в архиве.");

      const plan = buildPerCategoryNumberingPlan(items, opts.startNumbersByCategory);
      /** @type {Map<string, number>} */
      const categoryOrdinals = new Map();
      let written = 0;

      for (const im of items) {
        const planKey = exportCategoryPlanKey(im.category);
        const catDirName = exportCategoryDirName(im.category);
        const next = (categoryOrdinals.get(planKey) || 0) + 1;
        categoryOrdinals.set(planKey, next);
        const stem = exportNumberedFileStem(
          next,
          plan.starts.get(planKey),
          plan.pads.get(planKey)
        );
        const targetFolder = exportFolderForCategory(folder, catDirName);
        const size = await ensurePositiveExportDimensions(im);
        const annotObj = buildAnnotationExportJsonObject(im, size, stem);
        targetFolder.file(`${stem}.json`, `${JSON.stringify(annotObj, null, 2)}\n`);
        written++;
        await yieldToMain();
      }

      zip.file("classes.detect.txt", buildProjectExportClassesTxt("detect"));
      zip.file("classes.segment.txt", buildProjectExportClassesTxt("segment"));

      const zipBlob = await zip.generateAsync({ type: "blob" });
      const stamp = new Date().toISOString().replace(/[:.]/g, "-");
      triggerBlobDownload(zipBlob, `annotations_json_${stamp}.zip`);
      showToast(
        `Экспорт завершён: ${written} JSON-аннотаций в annotations/.`,
        { type: "success" }
      );
    } catch (err) {
      console.warn("[export annotations zip]", err);
      showToast(
        `Ошибка экспорта ZIP: ${err && err.message ? err.message : String(err)}`,
        { type: "error", durationMs: 6500 }
      );
    } finally {
      annotationsZipExportInFlight = false;
      syncLocalExportMenu();
      syncExportMenuBusyWithFlags();
    }
  }

  /** @param {{ startNumber?: number, startNumbersByCategory?: Record<string, number>, exportFilter?: import("./export.js").ExportReviewFilter }} [opts] */
  async function exportFullProjectZip(opts = {}) {
    if (fullProjectZipExportInFlight) return;
    if (typeof JSZip !== "function") {
      showToast("JSZip не загружен. Обновите страницу и попробуйте снова.", {
        type: "error",
        durationMs: 6000,
      });
      return;
    }

    const batchState = getBatchState();
    const exportFilter = opts.exportFilter ?? {
      manualOnly: false,
      includeUnreviewed: true,
    };
    const items = batchState.images.filter((im) =>
      matchesBatchExportReviewFilter(im, exportFilter, "batch-project-zip")
    );
    if (!items.length) {
      showToast(
        "Нет кадров для экспорта проекта по выбранным фильтрам проверки.",
        { type: "warning", durationMs: 4200 }
      );
      return;
    }

    fullProjectZipExportInFlight = true;
    syncLocalExportMenu();
    syncExportMenuBusyWithFlags();

    const includeAnnotations = opts.exportFilter?.includeAnnotations !== false;

    try {
      const zip = new JSZip();
      const folderImages = zip.folder("images");
      const folderLabelsRoot = zip.folder("labels");
      const folderLabelsDetect = folderLabelsRoot?.folder("detect");
      const folderLabelsSeg = folderLabelsRoot?.folder("segment");
      const folderAnnot = includeAnnotations ? zip.folder("annotations") : null;
      if (
        !folderImages ||
        !folderLabelsDetect ||
        !folderLabelsSeg ||
        (includeAnnotations && !folderAnnot)
      ) {
        throw new Error("Не удалось создать папки images/labels/annotations в архиве.");
      }

      /** @type {Array<Record<string, unknown>>} */
      const imagesMeta = [];

      const plan = buildPerCategoryNumberingPlan(items, opts.startNumbersByCategory);
      /** @type {Map<string, number>} */
      const categoryOrdinals = new Map();

      for (const im of items) {
        const planKey = exportCategoryPlanKey(im.category);
        const catDirName = exportCategoryDirName(im.category);
        const next = (categoryOrdinals.get(planKey) || 0) + 1;
        categoryOrdinals.set(planKey, next);
        const stem = exportNumberedFileStem(
          next,
          plan.starts.get(planKey),
          plan.pads.get(planKey)
        );
        const imageCatFolder = exportFolderForCategory(folderImages, catDirName);
        const detectLabelCatFolder = exportFolderForCategory(
          folderLabelsDetect,
          catDirName
        );
        const segLabelCatFolder = exportFolderForCategory(folderLabelsSeg, catDirName);

        const size = await ensurePositiveExportDimensions(im);

        const pngBlob = await convertImageBlobToPng(im.blob);
        imageCatFolder.file(`${stem}.png`, pngBlob);

        const detectLabelBody = buildYoloTxtFileBody(
          im,
          size.width,
          size.height,
          "detect"
        );
        const segLabelBody = buildYoloTxtFileBody(
          im,
          size.width,
          size.height,
          "segment"
        );
        const detectStatus = im.modelStates?.detect?.status;
        const segStatus = im.modelStates?.seg?.status;
        if (
          detectLabelBody ||
          detectStatus === "ready" ||
          detectStatus === "reviewed"
        ) {
          detectLabelCatFolder.file(`${stem}.txt`, detectLabelBody || "");
        }
        if (segLabelBody || segStatus === "ready" || segStatus === "reviewed") {
          segLabelCatFolder.file(`${stem}.txt`, segLabelBody || "");
        }

        if (includeAnnotations && folderAnnot) {
          const annotCatFolder = exportFolderForCategory(folderAnnot, catDirName);
          const annotObj = buildAnnotationExportJsonObject(im, size, stem);
          annotCatFolder.file(`${stem}.json`, `${JSON.stringify(annotObj, null, 2)}\n`);
        }

        imagesMeta.push({
          exportStem: stem,
          exportedImageFileName: `${stem}.png`,
          exportCategoryDir: catDirName,
          id: im.id,
          displayName: im.displayName,
          originalName: im.originalName,
          category: im.category ?? null,
          imageNames: {
            beforeUpload: im.originalName,
            afterUploadOnSite: im.displayName,
            afterExport: `${catDirName}/${stem}.png`,
          },
          status: im.status,
          reviewed: im.reviewed,
          edited: im.edited,
          width: size.width,
          height: size.height,
          detectionsCount: im.detections.length,
          bboxCount: im.detections.filter(isDetectAnnotation).length,
          maskCount: im.detections.filter(isSegAnnotation).length,
          modelStates: {
            detect: { ...(im.modelStates?.detect || {}) },
            seg: { ...(im.modelStates?.seg || {}) },
          },
          fileType: im.fileType,
          fileSize: im.fileSize,
          panel: serializePanelState(im.panel),
        });

        await yieldToMain();
      }

      if (!imagesMeta.length) {
        showToast(
          "Не удалось экспортировать ни одного кадра (нет размеров изображения).",
          { type: "warning", durationMs: 4200 }
        );
        return;
      }

      const curIm = batchState.images[batchState.currentIndex];
      const expCurIdx = curIm ? items.indexOf(curIm) : -1;
      const exportedCurrentIndex = expCurIdx >= 0 ? expCurIdx : 0;

      const exportedAt = new Date().toISOString();
      const projectPayload = {
        schema: "ai-train.project-export.v2",
        version: PROJECT_EXPORT_JSON_VERSION,
        batchId: batchState.batchId,
        createdAt: batchState.createdAt,
        updatedAt: batchState.updatedAt,
        exportedAt,
        currentIndex: exportedCurrentIndex,
        includesAnnotationsFolder: includeAnnotations,
        yoloTasks: ["detect", "segment"],
        classes: {
          detect: yoloClassOrderForTask("detect").map((name, id) => ({ id, name })),
          segment: yoloClassOrderForTask("segment").map((name, id) => ({ id, name })),
        },
        settings: {
          ...batchState.settings,
          classVisibility: { ...batchState.settings.classVisibility },
        },
        images: imagesMeta.map((row) => ({
          exportStem: row.exportStem,
          exportCategoryDir: row.exportCategoryDir,
          exportedImageFileName: row.exportedImageFileName,
          id: row.id,
          displayName: row.displayName,
          originalName: row.originalName,
          category: row.category,
          imageNames: row.imageNames,
          status: row.status,
          reviewed: row.reviewed,
          edited: row.edited,
          width: row.width,
          bboxCount: row.bboxCount,
          maskCount: row.maskCount,
          modelStates: row.modelStates,
          height: row.height,
          detectionsCount: row.detectionsCount,
          fileType: row.fileType,
          fileSize: row.fileSize,
          panel: row.panel,
        })),
      };

      zip.file("project.json", `${JSON.stringify(projectPayload, null, 2)}\n`);
      zip.file("data.detect.yaml", buildProjectExportDataYaml("detect"));
      zip.file("data.segment.yaml", buildProjectExportDataYaml("segment"));
      folderLabelsDetect.file("classes.txt", buildProjectExportClassesTxt("detect"));
      folderLabelsSeg.file(
        "classes.txt",
        buildProjectExportClassesTxt("segment")
      );

      const zipBlob = await zip.generateAsync({ type: "blob" });
      const stamp = new Date().toISOString().replace(/[:.]/g, "-");
      triggerBlobDownload(zipBlob, `project_${stamp}.zip`);
      showToast(
        includeAnnotations
          ? `Экспорт завершён: полный проект, ${imagesMeta.length} кадров.`
          : `Экспорт завершён: полный проект без annotations/, ${imagesMeta.length} кадров.`,
        { type: "success", durationMs: 3800 }
      );
    } catch (err) {
      console.warn("[export full project zip]", err);
      showToast(
        `Ошибка экспорта ZIP: ${err && err.message ? err.message : String(err)}`,
        { type: "error", durationMs: 6500 }
      );
    } finally {
      fullProjectZipExportInFlight = false;
      syncLocalExportMenu();
      syncExportMenuBusyWithFlags();
    }
  }

  /**
   * Прямоугольник кропа в пикселях bitmap: bbox + отступ CROP_BBOX_PADDING_RATIO от его размера с каждой стороны.
   * @param {ImageBitmap} bitmap
   * @param {any} im
   * @param {readonly [number, number, number, number]} box
   */
  function detectionCropRectPx(bitmap, im, box) {
    const imgW = bitmap.width;
    const imgH = bitmap.height;
    const ow = im.width > 0 ? im.width : imgW;
    const oh = im.height > 0 ? im.height : imgH;
    const sx = imgW / ow;
    const sy = imgH / oh;

    const [x1, y1, x2, y2] = box;
    const px = Math.max(0, Math.min(imgW - 1, Math.floor(x1 * sx)));
    const py = Math.max(0, Math.min(imgH - 1, Math.floor(y1 * sy)));
    const px2 = Math.max(px + 1, Math.min(imgW, Math.ceil(x2 * sx)));
    const py2 = Math.max(py + 1, Math.min(imgH, Math.ceil(y2 * sy)));
    const pw = px2 - px;
    const ph = py2 - py;

    const padX = Math.round(pw * CROP_BBOX_PADDING_RATIO);
    const padY = Math.round(ph * CROP_BBOX_PADDING_RATIO);
    const outPx = Math.max(0, px - padX);
    const outPy = Math.max(0, py - padY);
    const outPx2 = Math.min(imgW, px2 + padX);
    const outPy2 = Math.min(imgH, py2 + padY);
    return {
      px: outPx,
      py: outPy,
      pw: outPx2 - outPx,
      ph: outPy2 - outPy,
    };
  }

  /**
   * @param {ImageBitmap} bitmap
   * @param {any} im
   * @param {any} detection
   * @returns {Promise<Blob>}
   */
  async function renderCropPngBlobFromDetection(bitmap, im, detection) {
    const { px, py, pw, ph } = detectionCropRectPx(bitmap, im, detection.box);

    const canvas = document.createElement("canvas");
    canvas.width = pw;
    canvas.height = ph;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("2d context недоступен");
    ctx.drawImage(bitmap, px, py, pw, ph, 0, 0, pw, ph);

    return await new Promise((resolve, reject) => {
      canvas.toBlob(
        (b) => {
          if (b) resolve(b);
          else reject(new Error("toBlob не вернул PNG"));
        },
        "image/png"
      );
    });
  }

  /**
   * @param {{ startNumber?: number, startNumbersByCategory?: Record<string, number>, exportFilter?: ExportReviewFilter, includedCropClasses?: Set<string>|null }} [opts]
   */
  async function exportBatchCropsZip(opts = {}) {
    if (cropsZipExportInFlight) return;
    if (typeof JSZip !== "function") {
      showToast("JSZip не загружен. Обновите страницу и попробуйте снова.", {
        type: "error",
        durationMs: 6000,
      });
      return;
    }

    const batchState = getBatchState();
    const exportFilter = opts.exportFilter ?? {
      manualOnly: false,
      includeUnreviewed: true,
    };
    const includedCropClasses = opts.includedCropClasses ?? null;
    const items = batchState.images.filter((im) =>
      matchesBatchExportReviewFilter(im, exportFilter, "batch-crops-zip")
    );

    /** @type {Array<{ im: any, detection: any }>} */
    const cropJobs = [];
    for (const im of items) {
      for (const detection of collectExportableCropsFromImage(im)) {
        const classKey = exportCropClassDirName(detection.cls_name);
        if (includedCropClasses !== null && !includedCropClasses.has(classKey)) continue;
        cropJobs.push({ im, detection });
      }
    }

    if (!cropJobs.length) {
      showToast(
        "Нет кропов для экспорта: выберите классы или добавьте кадры с bbox.",
        { type: "warning", durationMs: 4600 }
      );
      return;
    }

    cropsZipExportInFlight = true;
    syncLocalExportMenu();
    syncExportMenuBusyWithFlags();

    /** @type {Map<string, ImageBitmap>} */
    const bitmapCache = new Map();

    try {
      const zip = new JSZip();
      const folder = zip.folder("crops");
      if (!folder) throw new Error("Не удалось создать папку crops в архиве.");

      const plan = buildPerCategoryNumberingPlan(
        cropJobs.map((job) => ({
          category: exportCropClassDirName(job.detection.cls_name),
        })),
        opts.startNumbersByCategory
      );
      /** @type {Map<string, number>} */
      const classOrdinals = new Map();

      for (const { im, detection } of cropJobs) {
        const planKey = exportCropClassDirName(detection.cls_name);
        const classDirName = planKey;
        const next = (classOrdinals.get(planKey) || 0) + 1;
        classOrdinals.set(planKey, next);
        const name = pngZipEntryBaseName(
          next,
          plan.starts.get(planKey),
          plan.pads.get(planKey)
        );

        let bitmap = bitmapCache.get(im.id);
        if (!bitmap) {
          bitmap = await createImageBitmap(im.blob);
          bitmapCache.set(im.id, bitmap);
        }

        const targetFolder = exportFolderForCategory(folder, classDirName);
        const pngBlob = await renderCropPngBlobFromDetection(bitmap, im, detection);
        targetFolder.file(name, pngBlob);
        await yieldToMain();
      }

      const zipBlob = await zip.generateAsync({ type: "blob" });
      const stamp = new Date().toISOString().replace(/[:.]/g, "-");
      triggerBlobDownload(zipBlob, `crops_bbox_${stamp}.zip`);
      showToast(
        `Экспорт завершён: ${cropJobs.length} кропов в crops/<класс>/.`,
        { type: "success", durationMs: 4200 }
      );
    } catch (err) {
      console.warn("[export crops zip]", err);
      showToast(
        `Ошибка экспорта кропов: ${err && err.message ? err.message : String(err)}`,
        { type: "error", durationMs: 6500 }
      );
    } finally {
      for (const bitmap of bitmapCache.values()) {
        if ("close" in bitmap && typeof bitmap.close === "function") {
          bitmap.close();
        }
      }
      cropsZipExportInFlight = false;
      syncLocalExportMenu();
      syncExportMenuBusyWithFlags();
    }
  }

  return {
    exportBatchPngZip,
    exportBatchYoloTxtZip,
    exportBatchAnnotationsZip,
    exportFullProjectZip,
    exportBatchCropsZip,
    isZipExportBusy: () => busy(),
    syncLocalExportMenu,
  };
}

/**
 * @param {{
 *   showToast: (msg: string, opts?: object) => void,
 *   getCurrentImage: () => any | null,
 *   previewImage: HTMLImageElement,
 *   effectiveOriginalSize: () => { width: number; height: number } | null,
 *   currentDetections: () => any[],
 *   confThreshold: () => number,
 *   classHidden: (name: string) => boolean,
 *   detectionSourceLabel: (d: any) => string,
 *   fmtConf: (x: number) => string,
 * }} deps
 */
export function createFrameExportHandlers(deps) {
  const {
    showToast,
    getCurrentImage,
    previewImage,
    effectiveOriginalSize,
    currentDetections,
    confThreshold,
    classHidden,
    fmtConf,
  } = deps;

  /** @param {{ startNumber?: number }} [opts] */
  function downloadAnnotatedImage(opts = {}) {
    const originalSize = effectiveOriginalSize();
    const imCur = getCurrentImage();
    if (!originalSize || !previewImage.src) {
      showToast("Нет изображения или размеров для сохранения.", {
        type: "warning",
        durationMs: 4200,
      });
      return;
    }
    if (imCur?.status === "skipped") {
      showToast("Пропущенные изображения не экспортируются.", {
        type: "warning",
        durationMs: 4200,
      });
      return;
    }
    if (!imCur?.detections?.length) {
      showToast(
        "На этом кадре нет разметки. Пустое изображение без разметки не экспортируется.",
        { type: "warning", durationMs: 4500 }
      );
      return;
    }
    const detections = currentDetections();

    const imgW = previewImage.naturalWidth || originalSize.width;
    const imgH = previewImage.naturalHeight || originalSize.height;

    const canvas = document.createElement("canvas");
    canvas.width = imgW;
    canvas.height = imgH;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(previewImage, 0, 0, imgW, imgH);

    const sx = imgW / originalSize.width;
    const sy = imgH / originalSize.height;

    ctx.lineWidth = 2;
    ctx.font = "16px system-ui, sans-serif";

    const categoryState = imCur?.panel.categoryState ?? new Map();
    const detEnabled = imCur?.panel.detEnabled ?? new Map();

    for (const d of detections) {
      if (d.conf < confThreshold()) continue;
      if (classHidden(d.cls_name)) continue;
      const cat = d.cls_name;
      const catEnabled = categoryState.get(cat)?.enabled ?? true;
      if (!catEnabled) continue;
      if (detEnabled.get(d.id) === false) continue;

      const [x1, y1, x2, y2] = d.box;
      const x = x1 * sx;
      const y = y1 * sy;
      const w = (x2 - x1) * sx;
      const h = (y2 - y1) * sy;

      const color = getColorByClass(d.cls_id);
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      const segment = validSegmentPoints(d);
      if (segment.length >= 3) {
        ctx.save();
        ctx.beginPath();
        traceSegmentPath(ctx, segment, sx, sy);
        ctx.globalAlpha = 0.22;
        ctx.fillStyle = color;
        ctx.fill();
        ctx.globalAlpha = 0.95;
        ctx.lineWidth = 2;
        ctx.strokeStyle = color;
        ctx.stroke();
        ctx.restore();
      }
      if (isDetectAnnotation(d)) ctx.strokeRect(x, y, w, h);

      const label = `${d.cls_name} #${d.id + 1} (${fmtConf(d.conf)})`;
      const padX = 8;
      const padY = 6;
      const textW = ctx.measureText(label).width;
      const boxW = textW + padX * 2;
      const boxH = 24;
      const bx = Math.max(0, Math.min(x, imgW - boxW));
      const by = Math.max(0, y - boxH - 2);

      ctx.globalAlpha = 0.85;
      ctx.fillRect(bx, by, boxW, boxH);
      ctx.globalAlpha = 1;
      ctx.fillStyle = "#0b0b0b";
      ctx.fillText(label, bx + padX, by + (boxH - padY));
    }

    const stem =
      typeof opts.startNumber === "number"
        ? exportNumberedFileStem(1, normalizeExportStartNumber(opts.startNumber, 1))
        : imCur?.displayName.replace(/\.[^.]+$/, "") || "annotated";

    const link = document.createElement("a");
    link.href = canvas.toDataURL("image/jpeg", 0.9);
    link.download = `${stem}_annotated.jpg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    showToast("Экспорт завершён: JPG с разметкой сохранён.", {
      type: "success",
    });
  }

  /** @param {{ startNumber?: number }} [opts] */
  function downloadYoloAnnotations(opts = {}) {
    const task =
      opts.task === "segment" || opts.task === "seg" ? "segment" : "detect";
    const originalSize = effectiveOriginalSize();
    const imCur = getCurrentImage();
    if (!originalSize) {
      showToast("Нет размеров изображения для экспорта.", {
        type: "warning",
        durationMs: 4200,
      });
      return;
    }
    if (imCur?.status === "skipped") {
      showToast("Пропущенные изображения не экспортируются.", {
        type: "warning",
        durationMs: 4200,
      });
      return;
    }
    if (!imCur?.detections?.length) {
      showToast(
        "На этом кадре нет разметки. Пустое изображение без разметки не экспортируется.",
        { type: "warning", durationMs: 4500 }
      );
      return;
    }
    const detections = currentDetections();

    const im = imCur;
    const categoryState = im?.panel.categoryState ?? new Map();
    const detEnabled = im?.panel.detEnabled ?? new Map();
    const exportDetections = [];

    for (const d of detections) {
      if (d.conf < confThreshold()) continue;
      if (task === "segment" ? !isSegAnnotation(d) : !isDetectAnnotation(d)) {
        continue;
      }
      if (classHidden(d.cls_name)) continue;

      const cat = d.cls_name;
      const catEnabled = categoryState.get(cat)?.enabled ?? true;
      if (!catEnabled) continue;
      if (detEnabled.get(d.id) === false) continue;

      exportDetections.push(d);
    }

    const body = buildYoloTxtFileBody(
      { detections: exportDetections },
      originalSize.width,
      originalSize.height,
      task
    );

    if (!body) {
      showToast(
        "Нет включённой разметки для экспорта. Включите нужные объекты на панели справа.",
        { type: "warning", durationMs: 5000 }
      );
      return;
    }

    const blob = new Blob([body], {
      type: "text/plain;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);

    const stem =
      typeof opts.startNumber === "number"
        ? exportNumberedFileStem(1, normalizeExportStartNumber(opts.startNumber, 1))
        : im?.displayName.replace(/\.[^.]+$/, "") || "annotations";

    const link = document.createElement("a");
    link.href = url;
    link.download = `${stem}_${task}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    showToast(`Экспорт завершён: файл YOLO ${task === "segment" ? "Seg" : "Detect"} сохранён.`, {
      type: "success",
    });
  }

  /** @param {{ startNumber?: number }} [opts] */
  function downloadAnnotationJson(opts = {}) {
    const originalSize = effectiveOriginalSize();
    const imCur = getCurrentImage();
    if (!originalSize) {
      showToast("Нет размеров изображения для экспорта.", {
        type: "warning",
        durationMs: 4200,
      });
      return;
    }
    if (imCur?.status === "skipped") {
      showToast("Пропущенные изображения не экспортируются.", {
        type: "warning",
        durationMs: 4200,
      });
      return;
    }
    if (!imCur?.detections?.length) {
      showToast(
        "На этом кадре нет разметки. Пустое изображение без разметки не экспортируется.",
        { type: "warning", durationMs: 4500 }
      );
      return;
    }
    const detections = currentDetections();

    const im = imCur;
    const categoryState = im?.panel.categoryState ?? new Map();
    const detEnabled = im?.panel.detEnabled ?? new Map();
    const exportDetections = [];

    for (const d of detections) {
      if (d.conf < confThreshold()) continue;
      if (classHidden(d.cls_name)) continue;

      const cat = d.cls_name;
      const catEnabled = categoryState.get(cat)?.enabled ?? true;
      if (!catEnabled) continue;
      if (detEnabled.get(d.id) === false) continue;

      exportDetections.push(d);
    }

    if (!exportDetections.length) {
      showToast(
        "Нет включённой разметки для экспорта. Включите нужные объекты на панели справа.",
        { type: "warning", durationMs: 5000 }
      );
      return;
    }

    const stem =
      typeof opts.startNumber === "number"
        ? exportNumberedFileStem(1, normalizeExportStartNumber(opts.startNumber, 1))
        : im?.displayName.replace(/\.[^.]+$/, "") || "annotation";

    const annotObj = buildAnnotationExportJsonObject(
      { ...im, detections: exportDetections },
      originalSize,
      stem
    );
    const blob = new Blob([`${JSON.stringify(annotObj, null, 2)}\n`], {
      type: "application/json;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");
    link.href = url;
    link.download = `${stem}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    showToast("Экспорт завершён: файл аннотации .json сохранён.", {
      type: "success",
    });
  }

  return { downloadAnnotatedImage, downloadYoloAnnotations, downloadAnnotationJson };
}
