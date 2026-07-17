import {
  MAX_DIRECT_IMAGES,
  MAX_ZIP_IMAGES,
  MAX_IMAGE_BYTES,
  MAX_VIDEO_BYTES,
  ZIP_IMAGE_EXT_RE,
  DATA_YAML_RE,
  WORKSPACE_AUTOSAVE_DELAY_MS,
  HANDLE_HIT_PX,
  MIN_BOX_SIDE,
  DETECT_ALL_CONCURRENCY,
  DETECT_API_FETCH_TIMEOUT_MS,
  DETECT_ALL_BATCH_NAV_THROTTLE_MS,
} from "./constants.js";
import { COLORS } from "./colors.js";
import {
  trainClassesForMode,
  resetTrainClassesToDefault,
  applyTrainClassOrderFromNames,
} from "./train-classes.js";
import { newImageItemId } from "./ids.js";
import {
  parseYamlClassNames,
  normalizeZipPath,
  findZipEntryCaseInsensitive,
  findZipEntryBySuffixCaseInsensitive,
  findProjectJsonEntry,
} from "./yaml-zip.js";
import {
  EXPORT_ACTION_LABELS,
  EXPORT_ACTION_META,
} from "./export-labels.js?v=20260701-export-annot";
import {
  createWorkspaceSnapshot,
  saveWorkspaceSnapshot,
  loadLatestWorkspaceSnapshot,
  createWorkspaceAutosave,
} from "./workspace.js";
import {
  computeBatchExportEligibilitySummary,
  computeBatchCropExportSummary,
  exportCategoryDirectoriesForAction,
  exportCropClassDirectoriesForAction,
  createZipExportHandlers,
  createFrameExportHandlers,
  buildExportNamingPreview,
  normalizeExportStartNumber,
} from "./export.js?v=20260701-export-annot";
import { createEmptyBatchState } from "./batch-state.js";
import { fetchDetectApi } from "./detect-api.js";
import { formatDisplayName } from "./format-display.js";
import {
  coerceImportedImageStatus,
  reconcileImageStatusWithDetections,
  reconcileReviewedAndSkippedStatus,
  inferImageStatusFromDetections,
  rawDetectionsFromAnnotationExport,
  displayNameAfterProjectImport,
} from "./import-helpers.js?v=20260701-export-annot";
import { isServiceZipPath, isZipFile } from "./zip-path-utils.js";
import {
  isYoloLabelTxtFile,
  labelStemKeyForMatch,
  parseYoloTxtDetectionsForImport,
} from "./yolo-txt-import.js?v=20260701-export-annot";
import {
  isVideoFile,
  extractFramesFromVideoElement,
  waitForVideoMetadata,
} from "./video.js";
import { createPolygonEditor } from "./segmentation-editor/polygon-editor.js";
import { appendPolygonSimplifyBlock } from "./segmentation-editor/inspector.js";
import { createMaskContextMenu } from "./segmentation-editor/context-menu.js";
import {
  annotationSourceOf,
  annotationTypeOf,
  classesForAnnotationType,
  createEmptyModelStates,
  isDetectAnnotation,
  isSegAnnotation,
  modelStateLabel,
  normalizeModelStates,
} from "./annotation-model.js";

/**
 * @param {import("./dom-refs.js").AppDomRefs} refs
 */
export function startApp(refs) {
  const {
    uploadBtn,
    uploadVideoBtn,
    clearBtn,
    runBtn,
    runMenuToggle,
    runMenu,
    fileInput,
    videoFileInput,
    previewImage,
    previewVideo,
    placeholderText,
    imageFrame,
    imageStack,
    overlay,
    viewerPrev,
    viewerNext,
    viewerCounter,
    markReviewedBtn,
    skipImageBtn,
    headerReviewedLine,
    workspaceSaveStatus,
    headerProcessedLine,
    headerProgressTrack,
    headerProgressFill,
    headerFailedRow,
    headerFailedCount,
    workModeDetectionBtn,
    workModeSegmentationBtn,
    exportDropdownWrap,
    exportMenuToggle,
    exportMenuPanel,
    exportSummaryOverlay,
    exportSummaryIcon,
    exportSummaryScope,
    exportSummaryTitle,
    exportSummarySubtitle,
    exportSummaryBody,
    exportNamingRow,
    exportStartNumber,
    exportNamingPreview,
    exportCategoryNumbering,
    exportReviewFilter,
    exportReviewModeDetected,
    exportReviewModeManual,
    exportReviewIncludeUnreviewed,
    exportReviewFilterHint,
    exportYoloOptions,
    exportYoloEmptyLabels,
    exportYoloOptionsHint,
    exportProjectOptions,
    exportProjectIncludeAnnotations,
    exportProjectOptionsHint,
    exportSummaryConfirm,
    exportSummaryCancel,
    importSummaryOverlay,
    importSummaryTitle,
    importSummaryBody,
    importSummaryClose,
    viewerFilename,
    videoToolbar,
    videoFrameInterval,
    videoExtractBtn,
    videoShowBtn,
    batchImageListRoot,
    batchStatusFilter,
    batchSortToggle,
    batchSortMenu,
    groupsRoot,
    inspectorRoot,
    confFilterRange,
    confFilterValue,
    editorModeReviewBtn,
    editorModeEditBtn,
    editorToolsBar,
    editorToolSelectBtn,
    editorToolAddBtn,
    editorToolCropBtn,
    editorToolAddPolygonBtn,
    hotkeysHelpBtn,
    hotkeysHelpOverlay,
    hotkeysHelpClose,
    categoryModalOverlay,
    categoryModalList,
    categoryModalClose,
    categoryClearBtn,
    categoryAddBtn,
    categoryNewInput,
    classChipsRoot,
    maskContextMenu,
  } = refs;

  /** @typedef {"idle"|"queued"|"processing"|"detected"|"empty"|"failed"|"skipped"} ImageStatus */
  /** @typedef {"detection"|"segmentation"} InferenceMode */
  /**
   * Состояние правой панели для одного кадра (остаётся в памяти; для IndexedDB сериализуем позже).
   * @typedef {{
   *   categoryState: Map<string, { enabled: boolean; collapsed: boolean }>,
   *   detEnabled: Map<number, boolean>,
   *   selectedDetectionId: number|null,
   * }} ImagePanelState
   */

  /**
   * Элемент батча.
   * @typedef {{
   *   id: string,
   *   displayName: string,
   *   originalName: string,
   *   fileType: string,
   *   fileSize: number,
   *   width: number,
   *   height: number,
   *   blob: Blob,
   *   objectUrl: string,
   *   status: ImageStatus,
   *   category: string|null,
   *   reviewed: boolean,
   *   edited: boolean,
   *   detections: Array<{id:number,cls_id:number,cls_name:string,conf:number,box:[number,number,number,number],annotation_type:"detect"|"seg",source:"detect"|"seg"|"human",segment?:Array<[number,number]>}>,
   *   modelStates: {detect:any,seg:any},
   *   error: string|null,
   *   panel: ImagePanelState
   * }} BatchImageItem
   */

  /**
   * @typedef {{
   *   batchId: string,
   *   images: BatchImageItem[],
   *   currentIndex: number,
   *   createdAt: string,
   *   updatedAt: string,
   *   importSummary: {
   *     imported: number,
   *     skippedNonImage: number,
   *     skippedNonImageNames: string[],
   *     skippedOverSize: number,
   *     skippedOverSizeNames: string[],
   *     skippedOverBatchLimit: number,
   *     skippedOverBatchLimitNames: string[],
   *     importLimit?: number,
   *     skippedService?: number,
   *     skippedServiceNames?: string[],
   *     skippedUnsupported?: number,
   *     skippedUnsupportedNames?: string[],
   *   },
   *   settings: {
   *     confidenceThreshold: number,
   *     statusFilter: "all"|"idle"|"needs_review"|"reviewed"|"empty"|"failed"|"skipped",
   *     classVisibility: Record<string, boolean>,
   *     trainClassOrder: string[],
   *     editorMode: "review"|"edit",
   *     editorTool: "select"|"addBox"|"crop"|"addPolygon",
 *     inferenceMode: InferenceMode,
 *     runSelection: "both"|"detect"|"seg",
   *   }
   * }} BatchState
   */


  /** @type {BatchState} */
  let batchState = createEmptyBatchState();

  /**
   * @typedef {{ objectUrl: string, fileName: string, file: File }} VideoState
   */

  /** @type {VideoState|null} */
  let videoState = null;
  /** @type {"video"|"image"} */
  let viewerMode = "image";
  let videoExtractInFlight = false;

  /** @type {number|null} */
  let detectAllBatchUiThrottleTimer = null;
  /** Масштаб просмотра изображения (влияет на preview + overlay). */
  let viewerZoomScale = 1;
  /** Смещение изображения при зуме, в px относительно центра fit-режима. */
  let viewerPanX = 0;
  let viewerPanY = 0;
  let spacePanPressed = false;
  /** @type {{ startClientX: number, startClientY: number, startPanX: number, startPanY: number }|null} */
  let panDragState = null;
  const VIEWER_ZOOM_MIN = 1;
  const VIEWER_ZOOM_MAX = 8;
  /** Пакетное распознавание с основной кнопки */
  let detectAllInFlight = false;
  let workspaceHydrating = true;

  const IMAGE_UNDO_LIMIT = 80;
  /** @type {any[]} */
  const imageUndoStack = [];
  /** @type {any[]} */
  const imageRedoStack = [];
  /** Класс для следующей фигуры (Detect: 1–2, Seg: 1–5), сбрасывается при смене кадра. */
  /** @type {string|null} */
  let hotkeyPreferredNewBboxClassName = null;

  /**
   * Временное состояние drag на overlay (addBox / crop / move / resize bbox).
   * @type {null | {
   *   kind: "addBox"|"crop"|"move"|"resize",
   *   handle?: string,
   *   detId?: number,
   *   startIx: number,
   *   startIy: number,
   *   origBox?: [number, number, number, number],
   *   ix0?: number,
   *   iy0?: number,
   *   ix1?: number,
   *   iy1?: number,
   * }}
   */
  let pointerInteraction = null;

  /**
   * Редактор полигонов сегментации (выбор/перетаскивание вершин, перемещение
   * маски, добавление новой маски, упрощение контура) — вся логика и
   * состояние живут в segmentation-editor/*, здесь только тонкая
   * диспетчеризация вызовов из существующих обработчиков событий.
   */
  const polygonEditor = createPolygonEditor({
    getOverlayGeometry: () => getOverlayGeometry(),
    getCurrentImage: () => currentImageItem(),
    getSelectedDetection: () => {
      const im = currentImageItem();
      if (!im) return null;
      const sid = im.panel.selectedDetectionId;
      if (sid == null) return null;
      const det = currentDetections().find((d) => d.id === sid);
      if (!det || !detectionPassesUiFilters(det, im)) return null;
      return det;
    },
    setSelectedDetectionId: (id) => setSelectedDetectionId(id),
    allocateNextDetId: (im) => allocateNextDetId(im),
    getNewShapeTrainClass: (im) => getTrainClassForNewBbox(im),
    pushUndoCheckpoint: () => pushUndoCheckpoint(),
    touchBatch: () => touchBatch(),
    scheduleWorkspaceAutosave: (delayMs) => scheduleWorkspaceAutosave(delayMs),
    updateBatchNavUi: () => updateBatchNavUi(),
    buildRightPanel: () => buildRightPanel(),
    requestDraw: () => draw(),
    isEditMode: () => editorModeIsEdit(),
    getEditorTool: () => batchState.settings.editorTool,
    setEditorTool: (tool) => {
      batchState.settings.editorTool = tool;
    },
    markAnnotationHuman: (im, type) => markAnnotationTypeReviewed(im, type),
  });

  const MASK_CONTEXT_COMMANDS = [
    {
      id: "simplify",
      label: "Упростить контур",
      icon: "⌁",
      isAvailable: ({ imageId, detId }) => {
        const im = currentImageItem();
        const det = im?.detections.find((d) => d.id === detId);
        return (
          editorModeIsEdit() &&
          im?.id === imageId &&
          !!det &&
          isSegAnnotation(det)
        );
      },
      handler: (context) => focusMaskSimplifyFromContext(context),
    },
  ];
  const maskMenu = createMaskContextMenu({
    root: maskContextMenu,
    commands: MASK_CONTEXT_COMMANDS,
  });

  /** Перекрестье Add Box: CSS px на overlay, линии по краям кадра */
  let addBoxCrosshairOverlayPx =
    /** @type {{ ox: number; oy: number } | null} */ (null);

  function clearAddBoxCrosshairOverlayPx() {
    addBoxCrosshairOverlayPx = null;
  }

  /** @param {number} cssOx @param {number} cssOy */
  function syncAddBoxCrosshairOverlayPx(cssOx, cssOy) {
    if (
      !editorModeIsEdit() ||
      !["addBox", "crop"].includes(batchState.settings.editorTool) ||
      previewImage.style.display === "none" ||
      !effectiveOriginalSize()
    ) {
      clearAddBoxCrosshairOverlayPx();
      return;
    }
    const geo = getOverlayGeometry();
    if (!geo) {
      clearAddBoxCrosshairOverlayPx();
      return;
    }
    const imgLeft = geo.offsetX;
    const imgTop = geo.offsetY;
    const imgRight = geo.offsetX + geo.imgW0 * geo.sx;
    const imgBottom = geo.offsetY + geo.imgH0 * geo.sy;
    addBoxCrosshairOverlayPx = {
      ox: clamp(cssOx, imgLeft, imgRight),
      oy: clamp(cssOy, imgTop, imgBottom),
    };
  }

  function touchBatch() {
    batchState.updatedAt = new Date().toISOString();
  }

  /** @param {"saving"|"saved"|"failed"|""} state */
  function setWorkspaceSaveStatus(state) {
    workspaceSaveStatus.classList.remove(
      "is-saving",
      "is-saved",
      "is-failed"
    );
    if (state === "saving") {
      workspaceSaveStatus.textContent = "Сохранение…";
      workspaceSaveStatus.classList.add("is-saving");
    } else if (state === "saved") {
      workspaceSaveStatus.textContent = "Сохранено";
      workspaceSaveStatus.classList.add("is-saved");
    } else if (state === "failed") {
      workspaceSaveStatus.textContent = "Ошибка сохранения";
      workspaceSaveStatus.classList.add("is-failed");
    } else {
      workspaceSaveStatus.textContent = "";
    }
  }

  /** @returns {Promise<IDBDatabase>} */
  async function saveWorkspaceToIndexedDB() {
    await saveWorkspaceSnapshot(createWorkspaceSnapshot(batchState));
  }

  const workspaceAutosave = createWorkspaceAutosave({
    showToast,
    setWorkspaceSaveStatus,
    getSnapshot: () => createWorkspaceSnapshot(batchState),
  });

  function scheduleWorkspaceAutosave(delayMs = WORKSPACE_AUTOSAVE_DELAY_MS) {
    workspaceAutosave.scheduleWorkspaceAutosave(delayMs);
  }

  function isTypingInteractionTarget(target) {
    if (!target || !(target instanceof Element)) return false;
    const el = target;
    const tag = el.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return true;
    if (el.isContentEditable) return true;
    return false;
  }

  /** @param {string} value */
  function normalizeImageCategory(value) {
    const s = String(value ?? "").trim();
    return s ? s.slice(0, 64) : null;
  }

  /**
   * Перекодирование произвольного image/* blob в PNG (для экспорта и т.п.).
   * Не используйте для превью/decode текущего кадра — не ломает object URL.
   * @param {Blob|File} blob
   * @returns {Promise<Blob>}
   */
  async function convertImageBlobToPng(blob) {
    const bitmap = await createImageBitmap(blob);
    try {
      const canvas = document.createElement("canvas");
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("2d context недоступен");
      ctx.drawImage(bitmap, 0, 0);
      const out = await new Promise((resolve, reject) => {
        canvas.toBlob(
          (b) => {
            if (b) resolve(b);
            else reject(new Error("toBlob не вернул PNG"));
          },
          "image/png"
        );
      });
      return /** @type {Blob} */ (out);
    } finally {
      if ("close" in bitmap && typeof bitmap.close === "function") {
        bitmap.close();
      }
    }
  }

  /**
   * @param {Blob|File} blob
   * @param {[number, number, number, number]} crop
   */
  async function cropImageBlobToPng(blob, crop) {
    const [x1, y1, x2, y2] = crop;
    const width = x2 - x1;
    const height = y2 - y1;
    const bitmap = await createImageBitmap(blob);
    try {
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("2d context недоступен");
      ctx.drawImage(bitmap, x1, y1, width, height, 0, 0, width, height);
      return await new Promise((resolve, reject) => {
        canvas.toBlob(
          (out) => {
            if (out) resolve(out);
            else reject(new Error("Не удалось создать обрезанное изображение"));
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
   * @param {Blob|File} file
   * @param {number} ordinal1Based
   * @param {string} [originalName]
   * @returns {BatchImageItem}
   */
  function imageItemFromFile(file, ordinal1Based, originalName) {
    const url = URL.createObjectURL(file);
    const fileLikeName = file instanceof File ? file.name : formatDisplayName(ordinal1Based);
    return {
      id: newImageItemId(),
      displayName: formatDisplayName(ordinal1Based),
      originalName: originalName || fileLikeName,
      fileType: file.type || "application/octet-stream",
      fileSize: file.size,
      width: 0,
      height: 0,
      blob: file,
      objectUrl: url,
      status: /** @type {ImageStatus} */ ("idle"),
      category: null,
      reviewed: false,
      edited: false,
      detections: [],
      modelStates: createEmptyModelStates(),
      error: null,
      panel: {
        categoryState: new Map(),
        detEnabled: new Map(),
        selectedDetectionId: null,
      },
    };
  }

  /** @param {any} raw */
  function deserializePanelState(raw) {
    return {
      categoryState: new Map(Array.isArray(raw?.categoryState) ? raw.categoryState : []),
      detEnabled: new Map(Array.isArray(raw?.detEnabled) ? raw.detEnabled : []),
      selectedDetectionId:
        typeof raw?.selectedDetectionId === "number"
          ? raw.selectedDetectionId
          : null,
    };
  }

  /**
   * @param {any} raw
   * @param {number} ordinal1Based
   * @returns {BatchImageItem|null}
   */
  function deserializeImageItem(raw, ordinal1Based, legacyMode = "detection") {
    const blob =
      raw?.blob instanceof Blob
        ? raw.blob
        : new Blob([], { type: raw?.fileType || "application/octet-stream" });
    const rawAnnotations = [
      ...(Array.isArray(raw?.detections) ? raw.detections : []),
      ...(Array.isArray(raw?.masks)
        ? raw.masks.map((d) => ({ ...d, annotation_type: "seg" }))
        : []),
    ];
    const detections = rawAnnotations.map((d, i) => normalizeDetection(d, i));
    const restoredModelStates = normalizeModelStates(raw?.modelStates, {
      annotations: detections,
      legacyStatus: raw?.status,
      legacyReviewed: raw?.reviewed === true,
      legacyError: raw?.error,
      legacyMode,
    });
    for (const state of Object.values(restoredModelStates)) {
      if (state.status !== "running") continue;
      state.status = "error";
      state.error = "\u0417\u0430\u043f\u0443\u0441\u043a \u043c\u043e\u0434\u0435\u043b\u0438 \u0431\u044b\u043b \u043f\u0440\u0435\u0440\u0432\u0430\u043d \u043f\u0435\u0440\u0435\u0437\u0430\u0433\u0440\u0443\u0437\u043a\u043e\u0439 \u0441\u0442\u0440\u0430\u043d\u0438\u0446\u044b.";
      state.updatedAt = new Date().toISOString();
      state.revision = (Number(state.revision) || 0) + 1;
    }
    const usedIds = new Set();
    let nextId = detections.reduce(
      (maxId, d) => Math.max(maxId, Number.isInteger(d.id) ? d.id : -1),
      -1
    ) + 1;
    for (const d of detections) {
      if (!Number.isInteger(d.id) || usedIds.has(d.id)) d.id = nextId++;
      usedIds.add(d.id);
    }
    const item = {
      id: typeof raw?.id === "string" ? raw.id : newImageItemId(),
      displayName:
        typeof raw?.displayName === "string"
          ? raw.displayName
          : formatDisplayName(ordinal1Based),
      originalName:
        typeof raw?.originalName === "string"
          ? raw.originalName
          : `restored-${ordinal1Based}`,
      fileType: typeof raw?.fileType === "string" ? raw.fileType : blob.type,
      fileSize: typeof raw?.fileSize === "number" ? raw.fileSize : blob.size,
      width: typeof raw?.width === "number" ? raw.width : 0,
      height: typeof raw?.height === "number" ? raw.height : 0,
      blob,
      objectUrl: URL.createObjectURL(blob),
      status: /** @type {ImageStatus} */ (
        typeof raw?.status === "string" ? raw.status : "idle"
      ),
      category: typeof raw?.category === "string" ? raw.category : null,
      reviewed: raw?.reviewed === true,
      edited: raw?.edited === true,
      detections,
      modelStates: restoredModelStates,
      error: typeof raw?.error === "string" ? raw.error : null,
      panel: deserializePanelState(raw?.panel),
    };
    syncLegacyImageState(item, { preserveSkipped: true });
    reconcileReviewedAndSkippedStatus(item);
    return item;
  }

  /** @returns {BatchImageItem|null} */
  function currentImageItem() {
    const { images, currentIndex } = batchState;
    if (currentIndex < 0 || currentIndex >= images.length) return null;
    return images[currentIndex];
  }

  /** Effective size для разметки: размер после детекта с сервера или decode превью. */
  /** @returns {{ width: number; height: number }|null} */
  function effectiveOriginalSize() {
    const im = currentImageItem();
    if (!im || im.width <= 0 || im.height <= 0) return null;
    return { width: im.width, height: im.height };
  }

  /** @returns {BatchImageItem["detections"]} */
  function currentDetections() {
    return currentImageItem()?.detections ?? [];
  }

  function confThreshold() {
    return batchState.settings.confidenceThreshold;
  }

  /** @param {string} name */
  function canonicalTrainClassName(name) {
    const s = String(name ?? "").toLowerCase();
    const found = allTrainClasses().find((t) => t.name === s);
    return found ? found.name : null;
  }

  function classHidden(className) {
    const cv = batchState.settings.classVisibility;
    const canon = canonicalTrainClassName(className);
    if (canon !== null && cv[canon] === false) return true;
    return cv[className] === false;
  }

  /**
   * @param {any} raw
   * @param {number} fallbackIndex
   * @returns {BatchImageItem["detections"][0]}
   */
  function normalizeDetection(raw, fallbackIndex) {
    const id = typeof raw?.id === "number" ? raw.id : fallbackIndex;
    const segment = Array.isArray(raw?.segment)
      ? raw.segment
          .map((p) =>
            Array.isArray(p) && p.length >= 2
              ? /** @type {[number, number]} */ ([Number(p[0]), Number(p[1])])
              : p && typeof p === "object"
                ? /** @type {[number, number]} */ ([Number(p.x), Number(p.y)])
              : null
          )
          .filter((p) => p && Number.isFinite(p[0]) && Number.isFinite(p[1]))
      : [];
    const annotation_type = annotationTypeOf({ ...raw, segment });
    const classes = classesForAnnotationType(annotation_type);
    const rawName = String(raw?.cls_name ?? raw?.class_name ?? "")
      .trim()
      .toLowerCase();
    const rawId = Number(raw?.cls_id ?? raw?.class_id);
    const fallbackClass = classes[0];
    const tc =
      classes.find((candidate) => candidate.name === rawName) ||
      (Number.isInteger(rawId)
        ? classes.find((candidate) => candidate.id === rawId)
        : null);
    const cls_id = tc?.id ?? (Number.isInteger(rawId) ? rawId : fallbackClass.id);
    const cls_name =
      tc?.name || rawName || fallbackClass.name;
    const conf = typeof raw?.conf === "number"
      ? raw.conf
      : typeof raw?.confidence === "number"
        ? raw.confidence
        : 0;
    let box =
      Array.isArray(raw?.box) && raw.box.length >= 4
        ? [
            Number(raw.box[0]),
            Number(raw.box[1]),
            Number(raw.box[2]),
            Number(raw.box[3]),
          ]
        : [0, 0, 0, 0];
    if (segment.length >= 3 && !(box[2] > box[0] && box[3] > box[1])) {
      const xs = segment.map((p) => p[0]);
      const ys = segment.map((p) => p[1]);
      box = [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
    }
    const source = annotationSourceOf({ ...raw, annotation_type });
    const detection =
      segment.length >= 3
        ? { id, cls_id, cls_name, conf, box, source, annotation_type, segment }
        : { id, cls_id, cls_name, conf, box, source, annotation_type };
    return detection;
  }

  /** Производные legacy-поля оставлены для существующих фильтров списка. */
  function syncLegacyImageState(im, opts = {}) {
    if (!im) return;
    im.modelStates = normalizeModelStates(im.modelStates);
    const states = [im.modelStates.detect, im.modelStates.seg];
    const previousStatus = im.status;
    if (opts.preserveSkipped && previousStatus === "skipped") return;
    const hasRunning = states.some((state) => state.status === "running");
    const hasReady = states.some((state) =>
      state.status === "ready" || state.status === "reviewed"
    );
    const hasError = states.some((state) => state.status === "error");
    const hasAnnotations = Array.isArray(im.detections) && im.detections.length > 0;
    if (hasRunning) im.status = "processing";
    else if (hasReady || hasAnnotations) im.status = hasAnnotations ? "detected" : "empty";
    else if (hasError) im.status = "failed";
    else im.status = "idle";

    const activeStates = states.filter((state) => state.status !== "not_run");
    im.reviewed =
      activeStates.length > 0 &&
      activeStates.every((state) => state.status === "reviewed");
    const errors = [
      im.modelStates.detect.error ? `Detect: ${im.modelStates.detect.error}` : "",
      im.modelStates.seg.error ? `Seg: ${im.modelStates.seg.error}` : "",
    ].filter(Boolean);
    im.error = errors.length ? errors.join("\n") : null;
  }

  function markAnnotationTypeReviewed(im, type) {
    if (!im) return;
    im.modelStates = normalizeModelStates(im.modelStates);
    const state = im.modelStates[type];
    state.status = "reviewed";
    state.error = null;
    state.updatedAt = new Date().toISOString();
    state.revision = (Number(state.revision) || 0) + 1;
    syncLegacyImageState(im, { preserveSkipped: false });
    /** Общая галочка кадра остаётся отдельным явным действием. */
    im.reviewed = false;
  }


  /** @param {BatchImageItem["detections"][0]} d */
  function detectionSourceLabel(d) {
    if (d.source === "human") return "вручную";
    return d.source === "seg" ? "Seg" : "Detect";
  }

  /** @param {ImageStatus|string} status */
  function formatImageStatusForUi(status) {
    switch (status) {
      case "idle":
        return "не обработано";
      case "queued":
        return "в очереди";
      case "processing":
        return "распознавание";
      case "detected":
        return "распознано";
      case "empty":
        return "пусто";
      case "failed":
        return "ошибка";
      case "skipped":
        return "пропущено";
      default:
        return String(status);
    }
  }

  /** @param {[number,number,number,number]} box */
  function fmtBoxCoords(box) {
    const r = (n) => (Math.round(Number(n) * 10) / 10).toFixed(1);
    const [x1, y1, x2, y2] = box;
    return `${r(x1)}, ${r(y1)} — ${r(x2)}, ${r(y2)}`;
  }

  /**
   * Смена класса в редакторе: human-источник, сброс reviewed, пометка edited.
   * @param {BatchImageItem} im
   * @param {BatchImageItem["detections"][0]} d
   * @param {{ name: string; id: number }} tc
   */
  function applyDetectionClassHuman(im, d, tc) {
    const annotationType = annotationTypeOf(d);
    const targetClass = classesForAnnotationType(annotationType).find(
      (candidate) => candidate.name === tc.name
    );
    if (!targetClass) return false;
    if (
      d.cls_id === targetClass.id &&
      d.cls_name === targetClass.name &&
      d.source === "human"
    ) {
      return false;
    }
    pushUndoCheckpoint();
    /* Если класс был скрыт чипом «скрыть», после смены на него bbox перестаёт проходить фильтры — сбрасывается выделение и рамка пропадает. */
    if (batchState.settings.classVisibility[targetClass.name] === false) {
      delete batchState.settings.classVisibility[targetClass.name];
    }
    let catSt = im.panel.categoryState.get(targetClass.name);
    if (!catSt) {
      catSt = { enabled: true, collapsed: false };
      im.panel.categoryState.set(targetClass.name, catSt);
    } else {
      catSt.enabled = true;
    }

    d.cls_id = targetClass.id;
    d.cls_name = targetClass.name;
    d.source = "human";
    d.annotation_type = annotationType;
    markAnnotationTypeReviewed(im, annotationType);
    im.edited = true;
    im.reviewed = false;
    touchBatch();
    scheduleWorkspaceAutosave(0);
    updateBatchNavUi();
    buildRightPanel();
    draw();
    return true;
  }

  /** Сумма bbox класса по всем изображениям батча (тот же порог уверенности). */
  function countTrainClassBboxesAcrossBatch(canonicalName) {
    const lc = canonicalName.toLowerCase();
    const thr = confThreshold();
    let sum = 0;
    for (const im of batchState.images) {
      const list = Array.isArray(im.detections) ? im.detections : [];
      for (const d of list) {
        if (d.conf >= thr && String(d.cls_name).toLowerCase() === lc) sum += 1;
      }
    }
    return sum;
  }

  function revokeBatchObjectUrls() {
    for (const im of batchState.images) {
      if (im.objectUrl) URL.revokeObjectURL(im.objectUrl);
    }
  }

  function clearVideoState() {
    if (videoState?.objectUrl) URL.revokeObjectURL(videoState.objectUrl);
    videoState = null;
    viewerMode = "image";
    previewVideo.pause();
    previewVideo.removeAttribute("src");
    previewVideo.load();
    previewVideo.hidden = true;
    videoToolbar.hidden = true;
    videoShowBtn.hidden = true;
    videoFileInput.value = "";
  }

  function syncVideoToolbar() {
    const hasVideo = !!videoState;
    videoToolbar.hidden = !hasVideo;
    videoShowBtn.hidden = !hasVideo || viewerMode === "video";
    videoExtractBtn.disabled = !hasVideo || videoExtractInFlight || detectAllInFlight;
  }

  function showVideoView() {
    viewerMode = "video";
    placeholderText.style.display = "none";
    previewImage.style.display = "none";
    previewVideo.hidden = false;
    clearCanvas();
    buildRightPanel();
    syncVideoToolbar();
    updateMainViewerNav();
  }

  function hideVideoView() {
    previewVideo.hidden = true;
    syncVideoToolbar();
  }

  /**
   * @param {File} file
   */
  function ingestVideoFile(file) {
    if (!isVideoFile(file)) {
      showToast("Выберите файл видео (MP4, WebM, MOV…).", {
        type: "error",
        durationMs: 3800,
      });
      videoFileInput.value = "";
      return;
    }
    if (file.size > MAX_VIDEO_BYTES) {
      showToast(
        `Видео больше ${MAX_VIDEO_BYTES / (1024 * 1024)} МБ. Выберите файл меньше.`,
        { type: "error", durationMs: 4200 }
      );
      videoFileInput.value = "";
      return;
    }

    if (batchState.images.length) {
      revokeBatchObjectUrls();
      batchState = createEmptyBatchState();
      syncInferenceModeUi();
      groupsRoot.innerHTML = "";
      showToast("Батч изображений очищен — загружено новое видео.", {
        type: "info",
        durationMs: 3200,
      });
    }

    clearVideoState();
    const objectUrl = URL.createObjectURL(file);
    videoState = { objectUrl, fileName: file.name, file };
    previewVideo.src = objectUrl;
    previewVideo.onloadedmetadata = () => {
      syncVideoToolbar();
      updateMainViewerNav();
    };
    showVideoView();
    touchBatch();
    scheduleWorkspaceAutosave(0);
    setStatus("Извлеките кадры или запустите распознавание");
    videoFileInput.value = "";
    updateBatchNavUi();
  }

  /** @returns {Promise<boolean>} */
  async function extractVideoFramesToBatch() {
    if (!videoState) return false;
    if (videoExtractInFlight) return false;

    videoExtractInFlight = true;
    syncVideoToolbar();
    const prevStatus = runBtn.textContent;
    runBtn.disabled = true;
    runMenuToggle.disabled = true;
    videoExtractBtn.textContent = "Извлекаем…";

    try {
      await waitForVideoMetadata(previewVideo);
      previewVideo.pause();
      const intervalSec = Number.parseFloat(videoFrameInterval.value) || 1;
      const frames = await extractFramesFromVideoElement(previewVideo, {
        intervalSec,
        maxFrames: MAX_DIRECT_IMAGES - batchState.images.length,
        baseName: videoState.fileName,
        onProgress: (done, total) => {
          videoExtractBtn.textContent = `Извлекаем… ${done}/${total}`;
        },
      });

      if (!frames.length) {
        showToast("Не удалось извлечь кадры из видео.", {
          type: "error",
          durationMs: 4200,
        });
        return false;
      }

      viewerMode = "image";
      hideVideoView();

      finalizeImageImport(frames, {
        imported: frames.length,
        skippedNonImage: 0,
        skippedNonImageNames: [],
        skippedOverSize: 0,
        skippedOverSizeNames: [],
        skippedOverBatchLimit: 0,
        skippedOverBatchLimitNames: [],
        importLimit: MAX_DIRECT_IMAGES,
        skippedService: 0,
        skippedServiceNames: [],
        skippedUnsupported: 0,
        skippedUnsupportedNames: [],
      });

      showToast(`Из видео извлечено кадров: ${frames.length}.`, {
        type: "success",
        durationMs: 3400,
      });
      return true;
    } catch (e) {
      showToast(String(e?.message || e), { type: "error", durationMs: 4800 });
      return false;
    } finally {
      videoExtractInFlight = false;
      videoExtractBtn.textContent = "Извлечь кадры";
      syncRecognizeBusy();
      if (!detectAllInFlight) setStatus(prevStatus || runButtonIdleText());
      syncVideoToolbar();
    }
  }

  /** @returns {Blob|null} */
  function currentBlobForApi() {
    const im = currentImageItem();
    if (!im) return null;
    return im.blob;
  }


  function fmtConf(x) {
    return (Math.round(x * 100) / 100).toFixed(2);
  }

  /** Подсветка пресетов: активен только при совпадении с шагом слайдера. */
  function syncConfidencePresetActive(v) {
    const wrap = confFilterRange.closest(".confidence-filter");
    if (!wrap) return;
    wrap.querySelectorAll(".confidence-preset").forEach((el) => {
      if (!(el instanceof HTMLButtonElement)) return;
      const raw = Number(el.dataset.confidence);
      const on =
        Number.isFinite(raw) && Math.abs(raw - v) < 0.004;
      el.classList.toggle("is-active", on);
    });
  }

  /** Слайдер и подпись из `batchState.settings.confidenceThreshold`. */
  function updateConfidenceFilterDom() {
    const v = Math.max(
      0,
      Math.min(1, Number(batchState.settings.confidenceThreshold) || 0)
    );
    batchState.settings.confidenceThreshold = v;
    confFilterRange.value = String(v);
    confFilterValue.textContent = fmtConf(v);
    confFilterRange.style.setProperty("--value-pct", `${v * 100}%`);
    syncConfidencePresetActive(v);
  }

  /** Только визуал от текущего значения ползунка (без перезаписи `value`). */
  function refreshConfidenceFilterVisual() {
    const v = Math.max(0, Math.min(1, Number(confFilterRange.value) || 0));
    confFilterValue.textContent = fmtConf(v);
    confFilterRange.style.setProperty("--value-pct", `${v * 100}%`);
    syncConfidencePresetActive(v);
  }

  function setStatus(text) {
    runBtn.textContent = text;
  }

  /** @param {unknown} mode @returns {InferenceMode} */
  function normalizeInferenceMode(mode) {
    return mode === "segmentation" ? "segmentation" : "detection";
  }

  /** @returns {InferenceMode} */
  function currentInferenceMode() {
    return normalizeInferenceMode(batchState.settings.inferenceMode);
  }

  function allTrainClasses() {
    return [
      ...classesForAnnotationType("detect"),
      ...classesForAnnotationType("seg"),
    ];
  }

  function annotationClasses(d) {
    return classesForAnnotationType(annotationTypeOf(d));
  }

  function editorAnnotationType() {
    if (batchState.settings.editorTool === "addPolygon") return "seg";
    if (batchState.settings.editorTool === "addBox") return "detect";
    const im = currentImageItem();
    const sid = im?.panel.selectedDetectionId;
    const selected = sid == null ? null : im?.detections.find((d) => d.id === sid);
    if (selected) return annotationTypeOf(selected);
    return "detect";
  }

  function currentTrainClasses() {
    return classesForAnnotationType(editorAnnotationType());
  }

  function storeCurrentClassOrders(settings) {
    const classOrders = {
      detect: trainClassesForMode("detection").map((tc) => tc.name),
      seg: trainClassesForMode("segmentation").map((tc) => tc.name),
    };
    settings.classOrders = classOrders;
    settings.inferenceMode = normalizeInferenceMode(settings.inferenceMode);
    settings.trainClassOrder = [
      ...(settings.inferenceMode === "segmentation"
        ? classOrders.seg
        : classOrders.detect),
    ];
    return classOrders;
  }

  function applyClassOrdersFromSettings(settings) {
    resetTrainClassesToDefault();
    const explicit =
      settings?.classOrders && typeof settings.classOrders === "object"
        ? settings.classOrders
        : {};
    let detectOrder = Array.isArray(explicit.detect) ? explicit.detect : null;
    let segOrder = Array.isArray(explicit.seg) ? explicit.seg : null;
    const legacyOrder = Array.isArray(settings?.trainClassOrder)
      ? settings.trainClassOrder
      : null;
    const legacyMode = normalizeInferenceMode(settings?.inferenceMode);
    if (legacyOrder && legacyMode === "segmentation" && !segOrder) {
      segOrder = legacyOrder;
    }
    if (legacyOrder && legacyMode !== "segmentation" && !detectOrder) {
      detectOrder = legacyOrder;
    }
    if (detectOrder) {
      applyTrainClassOrderFromNames(detectOrder, "detection");
    }
    if (segOrder) {
      applyTrainClassOrderFromNames(segOrder, "segmentation");
    }
    return storeCurrentClassOrders(settings);
  }

  function isSegmentationMode() {
    return editorAnnotationType() === "seg";
  }

  function runButtonIdleText() {
    return "Распознать";
  }

  function syncWorkModeUi() {
    workModeDetectionBtn.classList.add("is-active");
    workModeSegmentationBtn.classList.remove("is-active");
    workModeDetectionBtn.setAttribute("aria-checked", "true");
    workModeSegmentationBtn.setAttribute("aria-checked", "false");
    if (!detectAllInFlight) setStatus(runButtonIdleText());
  }

  function syncInferenceModeUi() {
    if (!["both", "detect", "seg"].includes(batchState.settings.runSelection)) {
      batchState.settings.runSelection = "both";
    }
    syncWorkModeUi();
    syncEditorChrome();
  }

  /** @param {InferenceMode} mode */
  function setInferenceMode(mode) {
    if (mode !== "segmentation") return;
    maskMenu?.close();
    showToast("Раздел «Анализ» пока в разработке.", {
      type: "info",
      durationMs: 2800,
    });
  }

  /**
   * @param {string} message
   * @param {number | { type?: "success"|"info"|"warning"|"error"; durationMs?: number }} [optsOrMs]
   */
  function showToast(message, optsOrMs = 3400) {
    const opts =
      typeof optsOrMs === "number"
        ? { durationMs: optsOrMs }
        : { ...(optsOrMs || {}) };
    const durationMs = Math.max(800, opts.durationMs ?? 3400);
    /** @type {"success"|"info"|"warning"|"error"} */
    const type = opts.type ?? "info";

    let root = document.getElementById("workspace-toast-root");
    if (!root) {
      root = document.createElement("div");
      root.id = "workspace-toast-root";
      root.setAttribute("aria-live", "polite");
      document.body.appendChild(root);
    }
    while (root.children.length >= 6) {
      root.firstElementChild?.remove();
    }
    const t = document.createElement("div");
    t.className = `workspace-toast workspace-toast--${type}`;
    t.setAttribute("role", type === "error" || type === "warning" ? "alert" : "status");
    t.textContent = message;
    root.appendChild(t);
    requestAnimationFrame(() => t.classList.add("is-visible"));
    window.setTimeout(() => {
      t.classList.remove("is-visible");
      window.setTimeout(() => t.remove(), 220);
    }, durationMs);
  }


  function closeExportMenu() {
    exportMenuPanel.hidden = true;
    exportMenuToggle.setAttribute("aria-expanded", "false");
  }

  /** Порядок и подписи причин пропуска кадров при экспорте — для попапа сводки. */
  const EXPORT_SKIP_REASON_LABELS = [
    { key: "pending", label: "ожидают обработки нейросетью" },
    { key: "notReviewed", label: "не проверены (отфильтрованы)" },
    { key: "noBbox", label: "нет разметки" },
    { key: "noCrops", label: "нет валидных bbox для кропа" },
    { key: "excludedClass", label: "класс не выбран для экспорта" },
    { key: "empty", label: "пустой кадр (без объектов)" },
    { key: "skipped", label: "отмечены «пропустить»" },
    { key: "failed", label: "ошибка распознавания" },
    { key: "noBlob", label: "нет файла изображения" },
  ];

  /**
   * @param {{ value: number|string, label: string, variant?: string }} cfg
   * @returns {HTMLDivElement}
   */
  function createExportStatTile(cfg) {
    const tile = document.createElement("div");
    tile.className = `export-stat-tile export-stat-tile--${cfg.variant || "default"}`;
    const value = document.createElement("div");
    value.className = "export-stat-tile-value";
    value.textContent = String(cfg.value);
    const label = document.createElement("div");
    label.className = "export-stat-tile-label";
    label.textContent = cfg.label;
    tile.appendChild(value);
    tile.appendChild(label);
    return tile;
  }

  /**
   * Сворачиваемый список причин, по которым кадры не попадут в экспорт.
   * Рисуется только если есть хотя бы одна причина — чтобы не загромождать попап
   * в обычном случае, когда экспортируется всё.
   * @param {HTMLElement} container
   * @param {{ reasons: Record<string, number> }} batchSummary
   */
  function appendExportSkipReasonsDetails(container, batchSummary) {
    /** @type {Array<{ label: string, count: number }>} */
    const present = [];
    for (const { key, label } of EXPORT_SKIP_REASON_LABELS) {
      const count = batchSummary.reasons[key] || 0;
      if (count > 0) present.push({ label, count });
    }
    if (!present.length) return;

    const details = document.createElement("details");
    details.className = "export-skip-reasons";

    const summary = document.createElement("summary");
    summary.className = "export-skip-reasons-summary";
    const totalSkipped = present.reduce((sum, r) => sum + r.count, 0);
    summary.textContent = `Почему пропущено: ${totalSkipped}`;
    details.appendChild(summary);

    const ul = document.createElement("ul");
    ul.className = "export-skip-reason-list";
    for (const { label, count } of present) {
      const li = document.createElement("li");
      li.className = "export-skip-reason-item";
      const badge = document.createElement("span");
      badge.className = "export-skip-reason-count";
      badge.textContent = String(count);
      li.appendChild(badge);
      li.appendChild(document.createTextNode(label));
      ul.appendChild(li);
    }
    details.appendChild(ul);
    container.appendChild(details);
  }

  /** @param {number} exportableCount */
  function syncExportNamingUi(exportableCount) {
    const count = Math.max(1, exportableCount || 1);
    const start = normalizeExportStartNumber(exportStartNumber.value, count);
    exportStartNumber.value = String(start);
    exportNamingPreview.textContent = `Пример: ${buildExportNamingPreview(start, count)}`;
  }

  /** @param {boolean} visible @param {number} [exportableCount] */
  function setExportNamingRowVisible(visible, exportableCount = 1) {
    exportNamingRow.hidden = !visible;
    if (visible) syncExportNamingUi(exportableCount);
  }

  /** @returns {number} */
  function readExportStartNumberFromUi(exportableCount = 1) {
    return normalizeExportStartNumber(exportStartNumber.value, exportableCount);
  }

  /** @param {string[]} categoryDirs @param {string} [title="Стартовый номер по категориям"] @param {{ cropMode?: boolean, classCounts?: Map<string, number>, preservedInclude?: Set<string>|null, preservedStartNumbers?: Record<string, number> }} [renderOpts] */
  function renderExportCategoryNumberingRows(
    categoryDirs,
    title = "Стартовый номер по категориям",
    renderOpts = {}
  ) {
    exportCategoryStartInputs.clear();
    exportCropClassIncludeInputs.clear();
    exportCategoryNumbering.innerHTML = "";
    if (!categoryDirs.length) {
      exportCategoryNumbering.hidden = true;
      return;
    }
    exportCategoryNumbering.hidden = false;

    const titleEl = document.createElement("div");
    titleEl.className = "export-category-numbering-title";
    titleEl.textContent = renderOpts.cropMode
      ? "Классы для экспорта и нумерация"
      : title;
    exportCategoryNumbering.appendChild(titleEl);

    for (const dir of categoryDirs) {
      const row = document.createElement("div");
      row.className = "export-category-numbering-row";
      if (renderOpts.cropMode) row.classList.add("export-crop-class-row");

      const label = document.createElement("label");
      label.className = "export-category-numbering-label";
      const count = renderOpts.classCounts?.get(dir);
      label.textContent =
        renderOpts.cropMode && typeof count === "number"
          ? `${dir} (${count})`
          : dir;

      const input = document.createElement("input");
      input.type = "number";
      input.className = "export-category-numbering-input";
      input.min = "1";
      input.step = "1";
      input.value = String(
        renderOpts.preservedStartNumbers?.[dir] ??
          exportCategoryStartInputs.get(dir)?.value ??
          "1"
      );
      input.inputMode = "numeric";
      input.title = renderOpts.cropMode
        ? `Начать нумерацию класса ${dir}`
        : `Начать нумерацию категории ${dir}`;
      input.addEventListener("input", () => {
        input.value = String(normalizeExportStartNumber(input.value, 1));
      });

      const inputId = `export-cat-start-${dir.replace(/[^a-z0-9_-]+/gi, "_")}`;
      input.id = inputId;

      if (renderOpts.cropMode) {
        const includeCb = document.createElement("input");
        includeCb.type = "checkbox";
        includeCb.className = "export-crop-class-include";
        includeCb.checked =
          renderOpts.preservedInclude === null ||
          renderOpts.preservedInclude === undefined ||
          renderOpts.preservedInclude.has(dir);
        includeCb.title = `Экспортировать класс ${dir}`;
        includeCb.addEventListener("change", () => {
          row.classList.toggle("is-excluded", !includeCb.checked);
          input.disabled = !includeCb.checked;
          refreshExportModalFromCropClassSelection();
        });

        input.disabled = !includeCb.checked;
        if (!includeCb.checked) row.classList.add("is-excluded");

        label.textContent =
          typeof count === "number" ? `${dir} (${count})` : dir;
        label.htmlFor = inputId;

        row.appendChild(includeCb);
        row.appendChild(label);
        row.appendChild(input);
        exportCropClassIncludeInputs.set(dir, includeCb);
      } else {
        label.htmlFor = inputId;
        row.appendChild(label);
        row.appendChild(input);
      }

      exportCategoryNumbering.appendChild(row);
      exportCategoryStartInputs.set(dir, input);
    }
  }

  /** @returns {Set<string>|null} null — все классы (ещё не отрисованы чекбоксы) */
  function readExportIncludedCropClassesFromUi() {
    if (!exportCropClassIncludeInputs.size) return null;
    /** @type {Set<string>} */
    const out = new Set();
    for (const [dir, cb] of exportCropClassIncludeInputs.entries()) {
      if (cb.checked) out.add(dir);
    }
    return out;
  }

  function buildCropExportRenderOpts(
    filter,
    preservedInclude = null,
    preservedStartNumbers
  ) {
    const allDirs = exportCropClassDirectoriesForAction(batchState.images, filter);
    const fullSummary = computeBatchCropExportSummary(batchState.images, filter, null);
    return {
      cropMode: true,
      classCounts: fullSummary.classCounts,
      preservedInclude,
      preservedStartNumbers,
      allDirs,
      fullSummary,
    };
  }

  function refreshExportModalFromCropClassSelection() {
    if (exportModalBatchActionKind !== "batch-crops-zip") return;
    const filter = readExportReviewFilterFromUi();
    const included = readExportIncludedCropClassesFromUi();
    const startNumbers = readExportCategoryStartNumbersFromUi();
    const cropOpts = buildCropExportRenderOpts(filter, included, startNumbers);
    const summary = computeBatchCropExportSummary(
      batchState.images,
      filter,
      included
    );
    const zipOk = typeof JSZip === "function";
    fillExportSummaryBody(summary, {
      mode: "batch",
      actionKind: "batch-crops-zip",
    });
    renderExportCategoryNumberingRows(
      cropOpts.allDirs,
      exportNumberingTitleForAction("batch-crops-zip"),
      cropOpts
    );
    exportSummaryConfirm.disabled = !(summary.exportable > 0 && zipOk);
  }

  /** @returns {Record<string, number>} */
  function readExportCategoryStartNumbersFromUi() {
    /** @type {Record<string, number>} */
    const out = {};
    for (const [dir, input] of exportCategoryStartInputs.entries()) {
      out[dir] = normalizeExportStartNumber(input.value, 1);
    }
    return out;
  }

  /** @returns {{ manualOnly: boolean, includeUnreviewed: boolean, includeEmptyYoloLabels: boolean, includeAnnotations: boolean }} */
  function readExportReviewFilterFromUi() {
    return {
      manualOnly: exportReviewModeManual.checked,
      includeUnreviewed: exportReviewIncludeUnreviewed.checked,
      includeEmptyYoloLabels: exportYoloEmptyLabels.checked,
      includeAnnotations: exportProjectIncludeAnnotations.checked,
    };
  }

  /** @param {string} actionKind */
  function exportActionSupportsEmptyYoloLabels(actionKind) {
    return (
      actionKind === "batch-yolo-zip" ||
      actionKind === "batch-yolo-detect-zip" ||
      actionKind === "batch-yolo-seg-zip" ||
      actionKind === "batch-project-zip" ||
      actionKind === "batch-annotations-zip"
    );
  }

  /** @param {string} actionKind */
  function syncExportYoloOptionsUi(actionKind) {
    const show = exportActionSupportsEmptyYoloLabels(actionKind);
    exportYoloOptions.hidden = !show;
    if (!show) {
      exportYoloEmptyLabels.checked = false;
      return;
    }
    if (exportYoloEmptyLabels.checked) {
      exportYoloOptionsHint.textContent =
        "Кадры без разметки попадут в экспорт с пустым labels/*.txt — как negative samples для YOLO.";
    } else {
      exportYoloOptionsHint.textContent =
        "Без галочки экспортируются только кадры с разметкой. Для обучения YOLO часто нужны и пустые .txt.";
    }
  }

  /** @param {string} actionKind */
  function exportActionSupportsAnnotationsToggle(actionKind) {
    return actionKind === "batch-project-zip";
  }

  /** @param {string} actionKind */
  function syncExportProjectOptionsUi(actionKind) {
    const show = exportActionSupportsAnnotationsToggle(actionKind);
    exportProjectOptions.hidden = !show;
    if (!show) {
      exportProjectIncludeAnnotations.checked = true;
      return;
    }
    if (exportProjectIncludeAnnotations.checked) {
      exportProjectOptionsHint.textContent =
        "В архив попадёт папка annotations/ с подробной JSON-разметкой каждого кадра.";
    } else {
      exportProjectOptionsHint.textContent =
        "Папка annotations/ не будет создана — останутся только images/ и labels/ (YOLO .txt). При повторном импорте такого проекта разметка восстановится не полностью.";
    }
  }

  function syncExportReviewFilterUi() {
    const manualOnly = exportReviewModeManual.checked;
    exportReviewIncludeUnreviewed.disabled = manualOnly;
    if (manualOnly) {
      exportReviewFilterHint.textContent =
        "Режим «только вручную»: в архив попадут только кадры с отметкой «проверено».";
    } else if (!exportReviewIncludeUnreviewed.checked) {
      exportReviewFilterHint.textContent =
        "Непроверенные кадры исключены — останутся только отмеченные «проверено».";
    } else {
      exportReviewFilterHint.textContent =
        "Включены результаты нейросети и проверенные вручную; непроверенные с разметкой тоже попадут в экспорт.";
    }
  }

  /**
   * @param {{ total:number, exportable:number, skipped:number, reasons: Record<string, number>, isCropSummary?: boolean, exportableImages?: number }} batchSummary
   * @param {{ mode: "batch"|"single", singleExports?: number, currentLabel?: string, actionKind?: string }} opts
   */
  function fillExportSummaryBody(batchSummary, opts) {
    exportSummaryBody.textContent = "";
    const isCrop =
      batchSummary.isCropSummary || opts.actionKind === "batch-crops-zip";

    if (opts.mode === "batch") {
      if (batchSummary.total === 0) {
        const note = document.createElement("p");
        note.className = "export-summary-empty-note";
        note.textContent =
          "В батче пока нет изображений — сначала загрузите фото или видео.";
        exportSummaryBody.appendChild(note);
        return;
      }

      const stats = document.createElement("div");
      stats.className = "export-stats-row";
      stats.appendChild(
        createExportStatTile({
          value: batchSummary.exportable,
          label: isCrop ? "кропов попадёт в архив" : "файлов попадёт в архив",
          variant: "accent",
        })
      );
      stats.appendChild(
        createExportStatTile({
          value: batchSummary.skipped,
          label: "кадров отфильтровано",
          variant: batchSummary.skipped > 0 ? "muted" : "muted-zero",
        })
      );
      exportSummaryBody.appendChild(stats);

      if (isCrop) {
        const note = document.createElement("div");
        note.className = "export-summary-note";
        const images =
          typeof batchSummary.exportableImages === "number"
            ? batchSummary.exportableImages
            : 0;
        note.textContent = `Кропы взяты из ${images} кадров батча.`;
        exportSummaryBody.appendChild(note);
      }

      if (
        exportActionSupportsEmptyYoloLabels(opts.actionKind || "") &&
        typeof batchSummary.exportableEmptyLabels === "number" &&
        batchSummary.exportableEmptyLabels > 0
      ) {
        const note = document.createElement("div");
        note.className = "export-summary-note export-summary-note--info";
        note.textContent = `Из них ${batchSummary.exportableEmptyLabels} без разметки — войдут как пустые negative samples.`;
        exportSummaryBody.appendChild(note);
      }

      appendExportSkipReasonsDetails(exportSummaryBody, batchSummary);
      return;
    }

    const nSingle =
      typeof opts.singleExports === "number" ? opts.singleExports : 1;

    const card = document.createElement("div");
    card.className = "export-current-frame-card";
    const cardTitle = document.createElement("div");
    cardTitle.className = "export-current-frame-card-title";
    cardTitle.textContent =
      nSingle > 1 ? `Будет сохранено файлов: ${nSingle}` : "Будет сохранён текущий кадр";
    card.appendChild(cardTitle);
    if (opts.currentLabel) {
      const cardDesc = document.createElement("div");
      cardDesc.className = "export-current-frame-card-desc";
      cardDesc.textContent = opts.currentLabel;
      card.appendChild(cardDesc);
    }
    exportSummaryBody.appendChild(card);

    const stats = document.createElement("div");
    stats.className = "export-stats-row export-stats-row--secondary";
    stats.appendChild(
      createExportStatTile({
        value: batchSummary.exportable,
        label: "попало бы из всего батча",
        variant: "accent-soft",
      })
    );
    stats.appendChild(
      createExportStatTile({
        value: batchSummary.skipped,
        label: "пропущено бы по батчу",
        variant: "muted",
      })
    );
    exportSummaryBody.appendChild(stats);

    appendExportSkipReasonsDetails(exportSummaryBody, batchSummary);
  }

  /** @type {((ev: KeyboardEvent) => void)|null} */
  let exportSummaryEscapeHandler = null;
  /** @type {Map<string, HTMLInputElement>} */
  const exportCategoryStartInputs = new Map();
  /** @type {Map<string, HTMLInputElement>} */
  const exportCropClassIncludeInputs = new Map();
  /** @type {string|null} */
  let exportModalBatchActionKind = null;
  /** @type {((startNumber: number, startNumbersByCategory?: Record<string, number>, exportFilter?: { manualOnly: boolean, includeUnreviewed: boolean }) => void)|null} */
  let exportModalBatchOnConfirm = null;
  /** @type {((ev: Event) => void)|null} */
  let exportReviewFilterChangeHandler = null;
  /** @type {((ev: Event) => void)|null} */
  let exportYoloOptionsChangeHandler = null;
  /** @type {((ev: Event) => void)|null} */
  let exportProjectOptionsChangeHandler = null;

  function closeExportSummaryModal() {
    exportSummaryOverlay.hidden = true;
    exportNamingRow.hidden = true;
    exportCategoryNumbering.hidden = true;
    exportCategoryNumbering.innerHTML = "";
    exportReviewFilter.hidden = true;
    exportYoloOptions.hidden = true;
    exportYoloEmptyLabels.checked = false;
    exportProjectOptions.hidden = true;
    exportProjectIncludeAnnotations.checked = true;
    exportCategoryStartInputs.clear();
    exportCropClassIncludeInputs.clear();
    exportModalBatchActionKind = null;
    exportModalBatchOnConfirm = null;
    if (exportReviewFilterChangeHandler) {
      exportReviewModeDetected.removeEventListener("change", exportReviewFilterChangeHandler);
      exportReviewModeManual.removeEventListener("change", exportReviewFilterChangeHandler);
      exportReviewIncludeUnreviewed.removeEventListener(
        "change",
        exportReviewFilterChangeHandler
      );
      exportReviewFilterChangeHandler = null;
    }
    if (exportYoloOptionsChangeHandler) {
      exportYoloEmptyLabels.removeEventListener("change", exportYoloOptionsChangeHandler);
      exportYoloOptionsChangeHandler = null;
    }
    if (exportProjectOptionsChangeHandler) {
      exportProjectIncludeAnnotations.removeEventListener(
        "change",
        exportProjectOptionsChangeHandler
      );
      exportProjectOptionsChangeHandler = null;
    }
    exportStartNumber.oninput = null;
    exportSummaryConfirm.onclick = null;
    exportSummaryCancel.onclick = null;
    if (exportSummaryEscapeHandler) {
      document.removeEventListener("keydown", exportSummaryEscapeHandler);
      exportSummaryEscapeHandler = null;
    }
  }

  /** @type {((ev: KeyboardEvent) => void)|null} */
  let importSummaryEscapeHandler = null;

  function closeImportSummaryModal() {
    importSummaryOverlay.hidden = true;
    importSummaryClose.onclick = null;
    if (importSummaryEscapeHandler) {
      document.removeEventListener("keydown", importSummaryEscapeHandler);
      importSummaryEscapeHandler = null;
    }
  }

  /** @type {((ev: KeyboardEvent) => void)|null} */
  let categoryModalEscapeHandler = null;

  function categoryChoicesForCurrentImage() {
    const set = new Set();
    for (const im of batchState.images) {
      const c = normalizeImageCategory(im.category ?? "");
      if (c) set.add(c);
    }
    const current = normalizeImageCategory(currentImageItem()?.category ?? "");
    if (current) set.add(current);
    return Array.from(set).sort((a, b) => a.localeCompare(b, "ru"));
  }

  function closeCategoryModal() {
    categoryModalOverlay.hidden = true;
    categoryModalClose.onclick = null;
    categoryClearBtn.onclick = null;
    categoryAddBtn.onclick = null;
    categoryNewInput.onkeydown = null;
    categoryNewInput.oninput = null;
    if (categoryModalEscapeHandler) {
      document.removeEventListener("keydown", categoryModalEscapeHandler);
      categoryModalEscapeHandler = null;
    }
  }

  function applyCategoryToCurrentImage(category) {
    const im = currentImageItem();
    if (!im) return;
    im.category = normalizeImageCategory(category ?? "");
    touchBatch();
    scheduleWorkspaceAutosave(0);
    updateBatchNavUi();
  }

  function renderCategoryModalChoices() {
    categoryModalList.innerHTML = "";
    const choices = categoryChoicesForCurrentImage();
    const active = normalizeImageCategory(currentImageItem()?.category ?? "");
    if (!choices.length) {
      const empty = document.createElement("div");
      empty.className = "category-modal-empty";
      empty.textContent = "Категории пока не добавлены.";
      categoryModalList.appendChild(empty);
      return;
    }
    for (const category of choices) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "category-pill";
      if (active === category) btn.classList.add("is-active");
      btn.textContent = category;
      btn.onclick = () => {
        applyCategoryToCurrentImage(category);
        renderCategoryModalChoices();
      };
      categoryModalList.appendChild(btn);
    }
  }

  function stableValidationHash(value) {
    let hash = 2166136261;
    const text = String(value ?? "");
    for (let i = 0; i < text.length; i++) {
      hash ^= text.charCodeAt(i);
      hash = Math.imul(hash, 16777619);
    }
    return hash >>> 0;
  }

  /** @param {BatchImageItem} im */
  function modelQualityScoreForValidation(im) {
    if (im.status === "empty") return 0.48;
    const all = Array.isArray(im.detections) ? im.detections : [];
    const model = all.filter((d) => d.source !== "human");
    const detections = model.length ? model : all;
    if (!detections.length) return 0;
    let sum = 0;
    let min = 1;
    for (const det of detections) {
      const conf = clamp(Number(det.conf) || 0, 0, 1);
      sum += conf;
      min = Math.min(min, conf);
    }
    let score = (sum / detections.length) * 0.68 + min * 0.32;
    score -= tinyBoxFraction(im) * 0.18;
    if (im.edited) score -= 0.08;
    return clamp(score, 0, 1);
  }

  /** @param {BatchImageItem} im */
  function isAutomaticValidationCandidate(im) {
    const category = normalizeImageCategory(im.category ?? "");
    const categoryAllowed =
      category == null || category === "train" || category === "val";
    const statusAllowed = im.status === "detected" || im.status === "empty";
    return categoryAllowed && statusAllowed && im.blob instanceof Blob && im.blob.size > 0;
  }

  function chooseRepresentativeValidationIds(items, targetCount) {
    const sorted = [...items].sort((a, b) => {
      const qualityDiff =
        modelQualityScoreForValidation(a) - modelQualityScoreForValidation(b);
      if (qualityDiff !== 0) return qualityDiff;
      return compareDisplayName(a, b);
    });
    const selectedIds = new Set();
    for (let i = 0; i < targetCount; i++) {
      const start = Math.floor((i * sorted.length) / targetCount);
      const end = Math.max(
        start + 1,
        Math.floor(((i + 1) * sorted.length) / targetCount)
      );
      const bucket = sorted.slice(start, end);
      let selected = bucket[0];
      let bestHash = Number.POSITIVE_INFINITY;
      for (const im of bucket) {
        const hash = stableValidationHash(`${im.id}|${im.displayName}|${i}`);
        if (hash < bestHash) {
          bestHash = hash;
          selected = im;
        }
      }
      selectedIds.add(selected.id);
    }
    return selectedIds;
  }

  function applyAutomaticValidationSplit(rawPercent) {
    const percent = clamp(Math.round(Number(rawPercent) || 20), 5, 50);
    const candidates = batchState.images.filter(isAutomaticValidationCandidate);
    if (candidates.length < 2) {
      showToast("\u0414\u043b\u044f \u0430\u0432\u0442\u043e\u0440\u0430\u0437\u0434\u0435\u043b\u0435\u043d\u0438\u044f \u043d\u0443\u0436\u043d\u043e \u043c\u0438\u043d\u0438\u043c\u0443\u043c 2 \u043e\u0431\u0440\u0430\u0431\u043e\u0442\u0430\u043d\u043d\u044b\u0445 \u043a\u0430\u0434\u0440\u0430.", {
        type: "warning",
      });
      return;
    }
    const valCount = clamp(
      Math.round((candidates.length * percent) / 100),
      1,
      candidates.length - 1
    );
    if (
      !window.confirm(
        `\u0420\u0430\u0441\u043f\u0440\u0435\u0434\u0435\u043b\u0438\u0442\u044c ${candidates.length} \u043a\u0430\u0434\u0440\u043e\u0432: train \u2014 ${
          candidates.length - valCount
        }, val \u2014 ${valCount}? \u041a\u0430\u0442\u0435\u0433\u043e\u0440\u0438\u0438 train/val \u0431\u0443\u0434\u0443\u0442 \u043e\u0431\u043d\u043e\u0432\u043b\u0435\u043d\u044b.`
      )
    ) {
      return;
    }
    const validationIds = chooseRepresentativeValidationIds(candidates, valCount);
    for (const im of candidates) {
      im.category = validationIds.has(im.id) ? "val" : "train";
    }
    touchBatch();
    scheduleWorkspaceAutosave(0);
    updateBatchNavUi();
    renderCategoryModalChoices();
    showToast(
      `\u0413\u043e\u0442\u043e\u0432\u043e: train \u2014 ${candidates.length - valCount}, val \u2014 ${valCount}. \u0412 val \u0441\u043c\u0435\u0448\u0430\u043d\u044b \u0441\u043b\u043e\u0436\u043d\u044b\u0435, \u0441\u0440\u0435\u0434\u043d\u0438\u0435 \u0438 \u0445\u043e\u0440\u043e\u0448\u0438\u0435 \u043a\u0430\u0434\u0440\u044b.`,
      { type: "success", durationMs: 5200 }
    );
  }

  function ensureCategoryAutoValControls() {
    let root = categoryModalOverlay.querySelector(".category-auto-val");
    if (!(root instanceof HTMLElement)) {
      root = document.createElement("section");
      root.className = "category-auto-val";

      const title = document.createElement("div");
      title.className = "category-auto-val-title";
      title.textContent = "\u0410\u0432\u0442\u043e\u0440\u0430\u0437\u0434\u0435\u043b\u0435\u043d\u0438\u0435 train / val";

      const row = document.createElement("div");
      row.className = "category-auto-val-row";
      const label = document.createElement("label");
      label.className = "category-auto-val-label";
      label.textContent = "\u0414\u043e\u043b\u044f val";
      const input = document.createElement("input");
      input.id = "category-val-percent";
      input.className = "category-auto-val-input";
      input.type = "number";
      input.min = "5";
      input.max = "50";
      input.step = "1";
      input.value = "20";
      input.setAttribute("aria-label", "\u0414\u043e\u043b\u044f val \u0432 \u043f\u0440\u043e\u0446\u0435\u043d\u0442\u0430\u0445");
      const suffix = document.createElement("span");
      suffix.textContent = "%";
      const button = document.createElement("button");
      button.id = "category-auto-val-btn";
      button.type = "button";
      button.className = "category-auto-val-btn";
      button.textContent = "\u0420\u0430\u0441\u043f\u0440\u0435\u0434\u0435\u043b\u0438\u0442\u044c";
      row.append(label, input, suffix, button);

      const hint = document.createElement("div");
      hint.className = "category-auto-val-hint";
      hint.textContent =
        "Val \u0432\u044b\u0431\u0438\u0440\u0430\u0435\u0442\u0441\u044f \u043f\u043e \u0432\u0441\u0435\u043c\u0443 \u0434\u0438\u0430\u043f\u0430\u0437\u043e\u043d\u0443 \u043a\u0430\u0447\u0435\u0441\u0442\u0432\u0430 \u043c\u043e\u0434\u0435\u043b\u0438: \u0441\u043b\u043e\u0436\u043d\u044b\u0435, \u0441\u0440\u0435\u0434\u043d\u0438\u0435 \u0438 \u0445\u043e\u0440\u043e\u0448\u0438\u0435 \u043a\u0430\u0434\u0440\u044b.";

      root.append(title, row, hint);
      categoryModalOverlay
        .querySelector(".category-modal-actions")
        ?.before(root);
    }
    return {
      input: root.querySelector("#category-val-percent"),
      button: root.querySelector("#category-auto-val-btn"),
    };
  }

  function openCategoryModal() {
    if (!currentImageItem()) {
      showToast("Сначала выберите кадр из списка.", { type: "warning" });
      return;
    }
    closeCategoryModal();
    closeHotkeysHelp();
    closeExportSummaryModal();
    closeImportSummaryModal();
    categoryModalOverlay.hidden = false;
    categoryNewInput.value = "";
    renderCategoryModalChoices();
    const autoValControls = ensureCategoryAutoValControls();
    if (
      autoValControls.input instanceof HTMLInputElement &&
      autoValControls.button instanceof HTMLButtonElement
    ) {
      autoValControls.button.onclick = () => {
        const percent = clamp(
          Math.round(Number(autoValControls.input.value) || 20),
          5,
          50
        );
        autoValControls.input.value = String(percent);
        applyAutomaticValidationSplit(percent);
      };
    }


    const addFromInput = () => {
      const next = normalizeImageCategory(categoryNewInput.value);
      if (!next) return;
      applyCategoryToCurrentImage(next);
      categoryNewInput.value = "";
      renderCategoryModalChoices();
    };

    categoryAddBtn.onclick = () => addFromInput();
    categoryClearBtn.onclick = () => {
      applyCategoryToCurrentImage(null);
      renderCategoryModalChoices();
    };
    categoryModalClose.onclick = () => closeCategoryModal();
    categoryNewInput.onkeydown = (ev) => {
      if (ev.key === "Enter") {
        ev.preventDefault();
        addFromInput();
      }
    };
    categoryNewInput.oninput = () => {
      categoryNewInput.value = categoryNewInput.value.slice(0, 64);
    };

    categoryModalEscapeHandler = (ev) => {
      if (ev.key === "Escape") closeCategoryModal();
    };
    document.addEventListener("keydown", categoryModalEscapeHandler);
    categoryNewInput.focus();
  }

  function openHotkeysHelp() {
    closeBatchSortMenu();
    closeExportMenu();
    hotkeysHelpOverlay.hidden = false;
    hotkeysHelpClose.focus();
  }

  function closeHotkeysHelp() {
    hotkeysHelpOverlay.hidden = true;
    hotkeysHelpBtn.focus();
  }

  function cloneImageWorkspaceForUndo() {
    const im = currentImageItem();
    if (!im) return null;
    return {
      blob: im.blob,
      fileType: im.fileType,
      fileSize: im.fileSize,
      width: im.width,
      height: im.height,
      detections: im.detections.map((d) => ({
        id: d.id,
        cls_id: d.cls_id,
        cls_name: d.cls_name,
        conf: d.conf,
        box: /** @type {[number, number, number, number]} */ ([
          d.box[0],
          d.box[1],
          d.box[2],
          d.box[3],
        ]),
        segment: Array.isArray(d.segment) ? d.segment.map((p) => [...p]) : undefined,
        source: d.source,
        annotation_type: annotationTypeOf(d),
      })),
      modelStates: {
        detect: { ...im.modelStates.detect },
        seg: { ...im.modelStates.seg },
      },
      selectedDetectionId: im.panel.selectedDetectionId,
      edited: im.edited,
      reviewed: im.reviewed,
      status: im.status,
      detEnabled: Array.from(im.panel.detEnabled.entries()),
      error: im.error,
      categoryState: Array.from(im.panel.categoryState.entries()).map(([k, v]) => [
        k,
        { enabled: v.enabled, collapsed: v.collapsed },
      ]),
    };
  }

  function applyImageWorkspaceUndoSnapshot(snap) {
    const im = currentImageItem();
    if (!im || !snap) return;
    const liveModelStates = normalizeModelStates(im.modelStates);
    const imageChanged = snap.blob instanceof Blob && snap.blob !== im.blob;
    if (imageChanged) {
      if (im.objectUrl) URL.revokeObjectURL(im.objectUrl);
      im.blob = snap.blob;
      im.objectUrl = URL.createObjectURL(snap.blob);
      im.fileType = snap.fileType || snap.blob.type || "image/png";
      im.fileSize = Number(snap.fileSize) || snap.blob.size;
      im.width = Number(snap.width) || 0;
      im.height = Number(snap.height) || 0;
    }
    im.detections = snap.detections.map((d, i) => normalizeDetection(d, i));
    im.panel.selectedDetectionId = snap.selectedDetectionId;
    const restoredModelStates = normalizeModelStates(snap.modelStates, {
      annotations: im.detections,
      legacyStatus: snap.status,
      legacyReviewed: snap.reviewed,
    });
    for (const type of ["detect", "seg"]) {
      const restoredState = restoredModelStates[type];
      const liveState = liveModelStates[type];
      restoredState.revision =
        Math.max(
          Number(restoredState.revision) || 0,
          Number(liveState.revision) || 0
        ) + 1;
      if (restoredState.status === "running") {
        const typeAnnotations = im.detections.filter(
          (d) => annotationTypeOf(d) === type
        );
        restoredState.status = typeAnnotations.some((d) => d.source === "human")
          ? "reviewed"
          : typeAnnotations.length
            ? "ready"
            : "not_run";
        restoredState.error = null;
        restoredState.updatedAt = new Date().toISOString();
      }
    }
    im.modelStates = restoredModelStates;
    im.edited = snap.edited;
    im.reviewed = snap.reviewed;
    im.status = snap.status;
    reconcileReviewedAndSkippedStatus(im);
    im.panel.detEnabled = new Map(snap.detEnabled);
    im.error = snap.error ?? null;
    syncLegacyImageState(im, { preserveSkipped: snap.status === "skipped" });
    im.panel.categoryState = new Map(
      snap.categoryState.map(([k, v]) => [k, { ...v }])
    );
    reconcileSelectedDetectionWithFilters(im);
    touchBatch();
    scheduleWorkspaceAutosave(0);
    updateBatchNavUi();
    if (imageChanged) {
      resetViewerZoom();
      showCurrentPreview();
    } else {
      buildRightPanel();
      draw();
    }
  }

  function pushUndoCheckpoint() {
    if (!currentImageItem()) return;
    const snap = cloneImageWorkspaceForUndo();
    imageUndoStack.push(snap);
    if (imageUndoStack.length > IMAGE_UNDO_LIMIT) imageUndoStack.shift();
    imageRedoStack.length = 0;
  }

  function undoImageWorkspace() {
    if (!imageUndoStack.length) return false;
    const prev = imageUndoStack.pop();
    const cur = cloneImageWorkspaceForUndo();
    if (cur) imageRedoStack.push(cur);
    applyImageWorkspaceUndoSnapshot(prev);
    return true;
  }

  function redoImageWorkspace() {
    if (!imageRedoStack.length) return false;
    const next = imageRedoStack.pop();
    const cur = cloneImageWorkspaceForUndo();
    if (cur) imageUndoStack.push(cur);
    applyImageWorkspaceUndoSnapshot(next);
    return true;
  }

  function cancelActiveEditorTool() {
    maskMenu.close();
    if (!editorModeIsEdit()) return;
    if (polygonEditor.cancel()) {
      touchBatch();
      buildInspector();
      draw();
      return;
    }
    if (pointerInteraction) {
      pointerInteraction = null;
      clearAddBoxCrosshairOverlayPx();
      touchBatch();
      buildInspector();
      draw();
      return;
    }
    if (
      batchState.settings.editorTool === "addBox" ||
      batchState.settings.editorTool === "crop" ||
      batchState.settings.editorTool === "addPolygon"
    ) {
      batchState.settings.editorTool = "select";
      clearAddBoxCrosshairOverlayPx();
      touchBatch();
      syncEditorChrome();
      buildInspector();
      buildRightPanel();
      draw();
    }
  }

  /** @param {HTMLElement} body @param {string} text */
  function appendImportSummaryTextRow(body, text) {
    const row = document.createElement("div");
    row.className = "export-summary-row";
    row.textContent = text;
    body.appendChild(row);
  }

  /** @param {HTMLElement} body @param {string} title */
  function appendImportSummarySectionTitle(body, title) {
    const h = document.createElement("div");
    h.className = "import-summary-section-title";
    h.textContent = title;
    body.appendChild(h);
  }

  /** @param {HTMLElement} body @param {string[]} names @param {number} [max] */
  function appendImportSummaryNameList(body, names, max = 14) {
    if (!names.length) return;
    const ul = document.createElement("ul");
    ul.className = "export-summary-list";
    const nShow = Math.min(names.length, max);
    for (let i = 0; i < nShow; i++) {
      const li = document.createElement("li");
      li.textContent = names[i];
      ul.appendChild(li);
    }
    if (names.length > nShow) {
      const li = document.createElement("li");
      li.className = "import-summary-muted";
      li.textContent = `… и ещё ${names.length - nShow}`;
      ul.appendChild(li);
    }
    body.appendChild(ul);
  }

  /**
   * @param {{
   *   title?: string,
   *   kind: "images-zip"|"project-zip",
   *   addedImages: number,
   *   skippedFiles: number,
   *   skipIssueLines: string[],
   *   skipNameBuckets?: Array<{ label: string, names: string[] }>,
   * }} cfg
   */
  function openImportSummaryModal(cfg) {
    closeImportSummaryModal();
    importSummaryTitle.textContent = cfg.title || "Сводка импорта ZIP";
    importSummaryBody.innerHTML = "";

    appendImportSummaryTextRow(
      importSummaryBody,
      `Добавлено изображений: ${cfg.addedImages}`
    );
    appendImportSummaryTextRow(
      importSummaryBody,
      `Пропущено файлов: ${cfg.skippedFiles}`
    );

    if (cfg.skipIssueLines.length) {
      appendImportSummarySectionTitle(importSummaryBody, "Причины пропуска");
      for (const line of cfg.skipIssueLines) {
        appendImportSummaryTextRow(importSummaryBody, line);
      }
    }

    const buckets = cfg.skipNameBuckets || [];
    const bucketsWithNames = buckets.filter((b) => b.names?.length);
    if (bucketsWithNames.length) {
      appendImportSummarySectionTitle(importSummaryBody, "Примеры имён");
      for (const b of bucketsWithNames) {
        appendImportSummaryTextRow(importSummaryBody, `${b.label}:`);
        appendImportSummaryNameList(importSummaryBody, b.names);
      }
    }

    importSummaryOverlay.hidden = false;
    importSummaryClose.onclick = () => closeImportSummaryModal();

    importSummaryEscapeHandler = (ev) => {
      if (ev.key === "Escape") closeImportSummaryModal();
    };
    document.addEventListener("keydown", importSummaryEscapeHandler);
    importSummaryClose.focus();
  }

  /** @returns {number} */
  function defaultExportStartNumberForCurrentFrame() {
    const im = currentImageItem();
    if (!im) return 1;
    const m = /^(\d+)/.exec(String(im.displayName || ""));
    if (m) return normalizeExportStartNumber(Number(m[1]), 1);
    return normalizeExportStartNumber(batchState.currentIndex + 1, 1);
  }

  /** @param {string} actionKind @param {{ manualOnly: boolean, includeUnreviewed: boolean }} exportFilter @param {Set<string>|null} [includedCropClasses] */
  function buildBatchExportSummary(actionKind, exportFilter, includedCropClasses = null) {
    if (actionKind === "batch-crops-zip") {
      return computeBatchCropExportSummary(
        batchState.images,
        exportFilter,
        includedCropClasses
      );
    }
    return computeBatchExportEligibilitySummary(
      batchState.images,
      actionKind,
      exportFilter
    );
  }

  /** @param {string} actionKind @param {{ manualOnly: boolean, includeUnreviewed: boolean }} exportFilter */
  function buildBatchExportNumberingDirs(actionKind, exportFilter) {
    if (actionKind === "batch-crops-zip") {
      return exportCropClassDirectoriesForAction(batchState.images, exportFilter);
    }
    return exportCategoryDirectoriesForAction(
      batchState.images,
      actionKind,
      exportFilter
    );
  }

  /** @param {string} actionKind */
  function exportNumberingTitleForAction(actionKind) {
    return actionKind === "batch-crops-zip"
      ? "Стартовый номер по классам"
      : "Стартовый номер по категориям";
  }

  /** @param {string} actionKind @param {ReturnType<typeof computeBatchExportEligibilitySummary>} summary @param {string[]} numberingDirs @param {boolean} zipOk @param {boolean} needsZip @param {{ cropMode?: boolean, classCounts?: Map<string, number>, preservedInclude?: Set<string>|null, preservedStartNumbers?: Record<string, number> }} [cropRenderOpts] */
  function refreshBatchExportSummaryModal(
    actionKind,
    summary,
    numberingDirs,
    zipOk,
    needsZip,
    cropRenderOpts
  ) {
    fillExportSummaryBody(summary, { mode: "batch", actionKind });
    renderExportCategoryNumberingRows(
      numberingDirs,
      exportNumberingTitleForAction(actionKind),
      cropRenderOpts
    );
    const namingCount = Math.max(0, summary.exportable);
    setExportNamingRowVisible(false, namingCount || 1);
    exportSummaryConfirm.disabled = !(summary.exportable > 0 && (!needsZip || zipOk));
  }

  function refreshExportModalFromReviewFilters() {
    if (!exportModalBatchActionKind) return;
    const filter = readExportReviewFilterFromUi();
    syncExportReviewFilterUi();
    syncExportYoloOptionsUi(exportModalBatchActionKind);
    const included =
      exportModalBatchActionKind === "batch-crops-zip"
        ? readExportIncludedCropClassesFromUi()
        : null;
    const startNumbers =
      exportModalBatchActionKind === "batch-crops-zip"
        ? readExportCategoryStartNumbersFromUi()
        : undefined;
    const summary = buildBatchExportSummary(
      exportModalBatchActionKind,
      filter,
      included
    );
    const numberingDirs = buildBatchExportNumberingDirs(
      exportModalBatchActionKind,
      filter
    );
    const needsZip = exportModalBatchActionKind.startsWith("batch-");
    const zipOk = typeof JSZip === "function";
    const cropRenderOpts =
      exportModalBatchActionKind === "batch-crops-zip"
        ? buildCropExportRenderOpts(filter, included, startNumbers)
        : undefined;
    refreshBatchExportSummaryModal(
      exportModalBatchActionKind,
      summary,
      numberingDirs,
      zipOk,
      needsZip,
      cropRenderOpts
    );
  }

  /** @param {{ title?: string, subtitle: string, summary: ReturnType<typeof computeBatchExportEligibilitySummary>, mode: "batch"|"single", actionKind?: string, singleExports?: number, currentLabel?: string, canConfirm: boolean, categoryDirs?: string[], onConfirm?: (startNumber: number, startNumbersByCategory?: Record<string, number>, exportFilter?: { manualOnly: boolean, includeUnreviewed: boolean }) => void, initialStartNumber?: number }} cfg */
  function openExportSummaryModal(cfg) {
    const kind = cfg.actionKind ? String(cfg.actionKind) : "";
    const meta = EXPORT_ACTION_META[/** @type {keyof typeof EXPORT_ACTION_META} */ (kind)];
    exportSummaryIcon.textContent = meta?.icon || "📦";
    exportSummaryTitle.textContent = meta?.short || cfg.title || "Экспорт";
    exportSummarySubtitle.textContent = meta?.desc || cfg.subtitle || "";
    const isBatchScope = cfg.mode === "batch";
    exportSummaryScope.textContent = isBatchScope ? "Весь батч" : "Текущий кадр";
    exportSummaryScope.className = `export-summary-scope export-summary-scope--${
      isBatchScope ? "batch" : "current"
    }`;
    fillExportSummaryBody(cfg.summary, { ...cfg, actionKind: kind });
    const needsZip = kind.startsWith("batch-");
    const namingCount =
      cfg.mode === "batch"
        ? Math.max(0, cfg.summary.exportable)
        : Math.max(1, cfg.singleExports ?? 1);
    const showNaming =
      cfg.mode === "single" &&
      (kind === "current-jpg" ||
        kind === "current-yolo-detect" ||
        kind === "current-yolo-seg" ||
        kind === "current-annotation-json");
    if (showNaming) {
      exportStartNumber.value = String(
        cfg.initialStartNumber ??
          (cfg.mode === "batch" ? 1 : defaultExportStartNumberForCurrentFrame())
      );
      setExportNamingRowVisible(true, namingCount || 1);
    } else {
      setExportNamingRowVisible(false, 1);
    }
    renderExportCategoryNumberingRows(
      cfg.mode === "batch" ? cfg.categoryDirs || [] : [],
      exportNumberingTitleForAction(kind),
      kind === "batch-crops-zip" && cfg.mode === "batch"
        ? buildCropExportRenderOpts(
            readExportReviewFilterFromUi(),
            null,
            undefined
          )
        : undefined
    );

    const zipOk = typeof JSZip === "function";

    if (cfg.mode === "batch" && kind.startsWith("batch-")) {
      exportReviewFilter.hidden = false;
      exportReviewModeDetected.checked = true;
      exportReviewModeManual.checked = false;
      exportReviewIncludeUnreviewed.checked = true;
      exportReviewIncludeUnreviewed.disabled = false;
      exportYoloEmptyLabels.checked = false;
      exportProjectIncludeAnnotations.checked = true;
      syncExportReviewFilterUi();
      syncExportYoloOptionsUi(kind);
      syncExportProjectOptionsUi(kind);
      exportModalBatchActionKind = kind;
      exportModalBatchOnConfirm = cfg.onConfirm ?? null;
      exportReviewFilterChangeHandler = () => refreshExportModalFromReviewFilters();
      exportYoloOptionsChangeHandler = () => refreshExportModalFromReviewFilters();
      exportProjectOptionsChangeHandler = () => syncExportProjectOptionsUi(kind);
      exportReviewModeDetected.addEventListener("change", exportReviewFilterChangeHandler);
      exportReviewModeManual.addEventListener("change", exportReviewFilterChangeHandler);
      exportReviewIncludeUnreviewed.addEventListener(
        "change",
        exportReviewFilterChangeHandler
      );
      exportYoloEmptyLabels.addEventListener("change", exportYoloOptionsChangeHandler);
      exportProjectIncludeAnnotations.addEventListener(
        "change",
        exportProjectOptionsChangeHandler
      );
    } else {
      exportReviewFilter.hidden = true;
      exportYoloOptions.hidden = true;
      exportProjectOptions.hidden = true;
      exportModalBatchActionKind = null;
      exportModalBatchOnConfirm = null;
    }

    let canConfirm = !!cfg.canConfirm;
    if (needsZip && !zipOk) {
      canConfirm = false;
      const warn = document.createElement("p");
      warn.className = "export-summary-row";
      warn.style.marginTop = "10px";
      warn.style.color = "var(--text-muted)";
      warn.textContent =
        "JSZip не загружен. Обновите страницу и попробуйте снова.";
      exportSummaryBody.appendChild(warn);
    }

    exportSummaryConfirm.disabled = !canConfirm;
    exportSummaryOverlay.hidden = false;

    exportStartNumber.oninput = () => syncExportNamingUi(namingCount || 1);

    exportSummaryConfirm.onclick = () => {
      const startNumber = readExportStartNumberFromUi(namingCount || 1);
      const startNumbersByCategory =
        cfg.mode === "batch" ? readExportCategoryStartNumbersFromUi() : undefined;
      const exportFilter =
        cfg.mode === "batch" ? readExportReviewFilterFromUi() : undefined;
      const includedCropClasses =
        cfg.mode === "batch" && kind === "batch-crops-zip"
          ? readExportIncludedCropClassesFromUi()
          : undefined;
      const onConfirm = exportModalBatchOnConfirm || cfg.onConfirm;
      const confirmEnabled = !exportSummaryConfirm.disabled;
      closeExportSummaryModal();
      if (confirmEnabled && onConfirm) {
        onConfirm(
          startNumber,
          startNumbersByCategory,
          exportFilter,
          includedCropClasses
        );
      }
    };
    exportSummaryCancel.onclick = () => closeExportSummaryModal();

    exportSummaryEscapeHandler = (ev) => {
      if (ev.key === "Escape") closeExportSummaryModal();
    };
    document.addEventListener("keydown", exportSummaryEscapeHandler);
  }

  /** @param {string} kind */
  function describeCurrentFrameForExport(kind) {
    const im = currentImageItem();
    if (!im) return { ok: false, detail: "Нет выбранного кадра." };

    if (!im.blob || im.blob.size === 0)
      return { ok: false, detail: "Нет файла изображения." };
    if (im.status === "skipped")
      return {
        ok: false,
        detail: "Пропущенный кадр не экспортируется.",
      };
    if (im.status === "failed")
      return { ok: false, detail: "Кадр со статусом «ошибка» не экспортируется." };
    if (im.status === "empty")
      return { ok: false, detail: "Кадр со статусом «пусто» не экспортируется." };
    if (
      (im.status === "idle" ||
        im.status === "queued" ||
        im.status === "processing") &&
      !im.reviewed
    ) {
      return {
        ok: false,
        detail:
          "Кадр ещё не готов (ожидание / очередь / распознавание) — дождитесь обработки или отметьте «проверено».",
      };
    }
    if (!Array.isArray(im.detections) || !im.detections.length) {
      return { ok: false, detail: "На кадре нет разметки." };
    }
    if (
      kind === "current-yolo-detect" &&
      !im.detections.some(isDetectAnnotation)
    ) {
      return { ok: false, detail: "На кадре нет bbox Detect." };
    }
    if (kind === "current-yolo-seg" && !im.detections.some(isSegAnnotation)) {
      return { ok: false, detail: "На кадре нет масок Seg." };
    }

    if (kind === "current-jpg") {
      const oz = effectiveOriginalSize();
      if (!oz || !previewImage.src) {
        return {
          ok: false,
          detail: "Нет превью или размеров изображения для сохранения.",
        };
      }
      return {
        ok: true,
        detail: `Текущий кадр «${im.displayName}» будет сохранён.`,
      };
    }

    if (!effectiveOriginalSize()) {
      return {
        ok: false,
        detail: "Нет размеров изображения для экспорта.",
      };
    }

    return {
      ok: true,
      detail: `Текущий кадр «${im.displayName}» будет сохранён.`,
    };
  }

  /** @param {string} actionKind */
  function beginExportFlow(actionKind) {
    const defaultFilter = {
      manualOnly: false,
      includeUnreviewed: true,
      includeEmptyYoloLabels: false,
      includeAnnotations: true,
    };
    const summary = buildBatchExportSummary(actionKind, defaultFilter);
    const subtitle =
      EXPORT_ACTION_LABELS[
        /** @type {keyof typeof EXPORT_ACTION_LABELS} */ (actionKind)
      ] || actionKind;

    if (actionKind.startsWith("batch-")) {
      const zipOk = typeof JSZip === "function";
      openExportSummaryModal({
        title: "Экспорт",
        subtitle,
        summary,
        mode: "batch",
        actionKind,
        canConfirm: summary.exportable > 0 && zipOk,
        categoryDirs: buildBatchExportNumberingDirs(actionKind, defaultFilter),
        onConfirm: (_startNumber, startNumbersByCategory, exportFilter, includedCropClasses) =>
          runBatchExportAction(
            actionKind,
            1,
            startNumbersByCategory,
            exportFilter ?? defaultFilter,
            includedCropClasses ?? null
          ),
      });
      return;
    }

    const cur = describeCurrentFrameForExport(actionKind);
    if (!cur.ok) {
      showToast(cur.detail, { type: "warning", durationMs: 3800 });
      return;
    }
    openExportSummaryModal({
      title: "Экспорт",
      subtitle,
      summary,
      mode: "single",
      actionKind,
      singleExports: 1,
      currentLabel: cur.detail,
      canConfirm: true,
      onConfirm: (startNumber) => runCurrentExportAction(actionKind, startNumber),
    });
  }

  /** @param {string} kind @param {number} [startNumber=1] @param {Record<string, number>} [startNumbersByCategory] @param {{ manualOnly: boolean, includeUnreviewed: boolean }} [exportFilter] @param {Set<string>|null} [includedCropClasses] */
  function runBatchExportAction(
    kind,
    startNumber = 1,
    startNumbersByCategory,
    exportFilter,
    includedCropClasses = null
  ) {
    const opts = {
      startNumber,
      startNumbersByCategory,
      exportFilter: exportFilter ?? {
        manualOnly: false,
        includeUnreviewed: true,
        includeEmptyYoloLabels: false,
        includeAnnotations: true,
      },
      includedCropClasses,
    };
    switch (kind) {
      case "batch-png-clean":
        void zipExport.exportBatchPngZip(false, opts);
        break;
      case "batch-png-marked":
        void zipExport.exportBatchPngZip(true, opts);
        break;
      case "batch-yolo-detect-zip":
        void zipExport.exportBatchYoloTxtZip({ ...opts, task: "detect" });
        break;
      case "batch-yolo-seg-zip":
        void zipExport.exportBatchYoloTxtZip({ ...opts, task: "segment" });
        break;
      case "batch-annotations-zip":
        void zipExport.exportBatchAnnotationsZip(opts);
        break;
      case "batch-project-zip":
        void zipExport.exportFullProjectZip(opts);
        break;
      case "batch-crops-zip":
        void zipExport.exportBatchCropsZip(opts);
        break;
      default:
        break;
    }
  }

  /** @param {string} kind @param {number} [startNumber=1] */
  function runCurrentExportAction(kind, startNumber = 1) {
    const opts = { startNumber };
    switch (kind) {
      case "current-jpg":
        frameExport.downloadAnnotatedImage(opts);
        break;
      case "current-yolo-detect":
        frameExport.downloadYoloAnnotations({ ...opts, task: "detect" });
        break;
      case "current-yolo-seg":
        frameExport.downloadYoloAnnotations({ ...opts, task: "segment" });
        break;
      case "current-annotation-json":
        frameExport.downloadAnnotationJson(opts);
        break;
      default:
        break;
    }
  }

  function scheduleThrottledBatchNavDuringDetectAll() {
    if (!detectAllInFlight) {
      updateBatchNavUi();
      return;
    }
    if (detectAllBatchUiThrottleTimer !== null) {
      window.clearTimeout(detectAllBatchUiThrottleTimer);
    }
    detectAllBatchUiThrottleTimer = window.setTimeout(() => {
      detectAllBatchUiThrottleTimer = null;
      updateBatchNavUi();
    }, DETECT_ALL_BATCH_NAV_THROTTLE_MS);
  }

  /** Успех одной модели заменяет только её автоматические аннотации. */
  function applyModelSuccessFromResponse(im, data, type, runRevision) {
    const state = im.modelStates?.[type];
    if (!state || state.revision !== runRevision || state.status !== "running") {
      return false;
    }
    const meta = data.image;
    if (
      meta &&
      typeof meta.width === "number" &&
      typeof meta.height === "number" &&
      meta.width > 0 &&
      meta.height > 0
    ) {
      im.width = meta.width;
      im.height = meta.height;
    }

    const rawDets = Array.isArray(data.detections) ? data.detections : [];
    const allowedClassNames = new Set(
      classesForAnnotationType(type).map((tc) => tc.name)
    );
    const preserved = im.detections.filter(
      (d) => !(annotationTypeOf(d) === type && d.source === type)
    );
    let nextId =
      preserved.reduce((maxId, d) => Math.max(maxId, Number(d.id) || 0), -1) + 1;
    const automatic = rawDets
      .filter((d) =>
        allowedClassNames.has(String(d?.cls_name ?? "").trim().toLowerCase())
      )
      .filter((d) =>
        type === "seg"
          ? Array.isArray(d?.segment) && d.segment.length >= 3
          : true
      )
      .map((d) =>
        normalizeDetection(
          {
            ...d,
            id: nextId++,
            annotation_type: type,
            source: type,
            ...(type === "detect" ? { segment: undefined } : {}),
          },
          nextId
        )
      );
    im.detections = [...preserved, ...automatic];

    const keptIds = new Set(im.detections.map((d) => d.id));
    im.panel.detEnabled = new Map(
      Array.from(im.panel.detEnabled.entries()).filter(([id]) => keptIds.has(id))
    );
    for (const d of automatic) im.panel.detEnabled.set(d.id, true);
    if (
      im.panel.selectedDetectionId != null &&
      !keptIds.has(im.panel.selectedDetectionId)
    ) {
      im.panel.selectedDetectionId = null;
    }

    state.status = "ready";
    state.error = null;
    state.updatedAt = new Date().toISOString();
    syncLegacyImageState(im, { preserveSkipped: false });
    return true;
  }

  /** Ошибка одной модели не меняет аннотации или состояние другой. */
  function applyModelFailureToItem(im, type, msg, runRevision) {
    const state = im.modelStates?.[type];
    if (!state || state.revision !== runRevision) return false;
    state.status = "error";
    state.error = msg;
    state.updatedAt = new Date().toISOString();
    syncLegacyImageState(im, { preserveSkipped: false });
    return true;
  }

  /** @param {number} idx @param {boolean} [autosave] */
  function refreshUiAfterImageProcessed(idx, autosave = true) {
    touchBatch();
    if (autosave && !detectAllInFlight) {
      scheduleWorkspaceAutosave(0);
    }
    if (batchState.currentIndex === idx) {
      buildRightPanel();
      resizeCanvasToFrame();
      draw();
    }
    if (detectAllInFlight) {
      scheduleThrottledBatchNavDuringDetectAll();
    } else {
      updateBatchNavUi();
    }
  }

  /** @param {BatchImageItem} im */
  function matchesBatchStatusFilter(im) {
    const sf = batchState.settings.statusFilter || "all";
    if (sf === "all") return true;
    if (sf === "idle") return im.status === "idle";
    if (sf === "needs_review") {
      return !im.reviewed && (im.status === "detected" || im.edited === true);
    }
    if (sf === "reviewed") return im.reviewed === true && im.status !== "skipped";
    if (sf === "empty") return im.status === "empty";
    if (sf === "failed") return im.status === "failed";
    if (sf === "skipped") return im.status === "skipped";
    return true;
  }

  /** @returns {number[]} индексы в batchState.images */
  function getFilteredBatchIndices() {
    const out = [];
    for (let i = 0; i < batchState.images.length; i++) {
      const im = batchState.images[i];
      if (matchesBatchStatusFilter(im)) out.push(i);
    }
    return out;
  }

  function syncBatchListFilterUiFromState() {
    batchStatusFilter.value = batchState.settings.statusFilter ?? "all";
  }

  function ensureRestoredBatchListVisible() {
    if (!batchState.images.length) return false;
    const visible = getFilteredBatchIndices();
    if (visible.includes(batchState.currentIndex)) return false;
    batchState.settings.statusFilter = "all";
    syncBatchListFilterUiFromState();
    return true;
  }

  function reconcileSelectionAfterListFilterChange() {
    const visible = getFilteredBatchIndices();
    if (!batchState.images.length) {
      updateBatchNavUi();
      return;
    }
    if (visible.length && !visible.includes(batchState.currentIndex)) {
      selectBatchIndex(visible[0]);
      return;
    }
    updateBatchNavUi();
  }

  function onBatchListFiltersChanged() {
    batchState.settings.statusFilter = /** @type {BatchState["settings"]["statusFilter"]} */ (
      batchStatusFilter.value || "all"
    );
    touchBatch();
    reconcileSelectionAfterListFilterChange();
  }

  function closeBatchSortMenu() {
    if (batchSortMenu) batchSortMenu.hidden = true;
    batchSortToggle?.setAttribute("aria-expanded", "false");
  }

  /** @param {BatchImageItem} im */
  function meanDetectionConfidence(im) {
    const d = im.detections;
    if (!Array.isArray(d) || !d.length) return -1;
    let s = 0;
    for (const x of d) s += Math.max(0, Math.min(1, Number(x.conf) || 0));
    return s / d.length;
  }

  /** @param {BatchImageItem} im */
  function minDetectionConfidence(im) {
    const d = im.detections;
    if (!Array.isArray(d) || !d.length) return -1;
    let m = 1;
    for (const x of d) m = Math.min(m, Math.max(0, Math.min(1, Number(x.conf) || 0)));
    return m;
  }

  /** @param {BatchImageItem} im @returns {number} 0..1 доля «слишком маленьких» bbox */
  function tinyBoxFraction(im) {
    const d = im.detections;
    if (!Array.isArray(d) || !d.length || !im.width || !im.height) return 0;
    const minShort = Math.max(
      MIN_BOX_SIDE * 3,
      Math.min(im.width, im.height) * 0.012
    );
    let bad = 0;
    for (const det of d) {
      const [x1, y1, x2, y2] = det.box;
      const w = Math.abs(x2 - x1);
      const h = Math.abs(y2 - y1);
      if (Math.min(w, h) < minShort) bad++;
    }
    return bad / d.length;
  }

  /**
   * Выше — «лучше» разметка (для сортировки по убыванию).
   * @param {BatchImageItem} im
   */
  function annotationQualityScore(im) {
    if (im.status === "failed" || im.status === "skipped") return -1e6;
    if (im.status === "idle" || im.status === "queued" || im.status === "processing")
      return -5e5;
    const d = im.detections || [];
    if (!d.length) return im.status === "empty" ? -1e3 : -200;
    const avg = meanDetectionConfidence(im);
    const minC = minDetectionConfidence(im);
    let score = avg * 200 + minC * 90;
    score -= tinyBoxFraction(im) * 160;
    if (im.reviewed) score += 130;
    return score;
  }

  /** @param {BatchImageItem} a @param {BatchImageItem} b */
  function compareDisplayName(a, b) {
    return a.displayName.localeCompare(b.displayName, "ru", { sensitivity: "base" });
  }

  /** @param {BatchImageItem} im */
  function bboxCountForSort(im) {
    return Array.isArray(im.detections) ? im.detections.length : 0;
  }

  /**
   * @param {"quality-desc"|"quality-asc"|"bbox-count-desc"|"bbox-count-asc"} mode
   */
  function applyBatchImageSort(mode) {
    const images = batchState.images;
    if (!images.length) return;
    const curId = images[batchState.currentIndex]?.id ?? null;

    /** @type {BatchImageItem[]} */
    const next = [...images];
    next.sort((a, b) => {
      let c = 0;
      if (mode === "quality-desc" || mode === "quality-asc") {
        const sa = annotationQualityScore(a);
        const sb = annotationQualityScore(b);
        c = mode === "quality-desc" ? sb - sa : sa - sb;
      } else {
        const na = bboxCountForSort(a);
        const nb = bboxCountForSort(b);
        c = mode === "bbox-count-desc" ? nb - na : na - nb;
      }
      if (c !== 0) return c;
      return compareDisplayName(a, b);
    });

    batchState.images = next;
    if (curId) {
      const ni = batchState.images.findIndex((x) => x.id === curId);
      if (ni >= 0) batchState.currentIndex = ni;
    }
    touchBatch();
    scheduleWorkspaceAutosave(0);
    updateBatchNavUi();
    closeBatchSortMenu();
  }

  /** @param {BatchImageItem} im */
  function batchItemStatusIcon(im) {
    if (im.status === "skipped") return "⏭";
    if (im.reviewed) return "✅";
    if (im.status === "processing") return "⏳";
    if (im.status === "failed") return "❌";
    if (im.edited) return "✏️";
    switch (im.status) {
      case "idle":
        return "○";
      case "queued":
        return "\u2026";
      case "processing":
        return "⏳";
      case "detected":
        return "\u26A0\uFE0F";
      case "empty":
        return "\u2205";
      case "failed":
        return "\u274C";
      case "skipped":
        return "\u23ED";
      default:
        return "○";
    }
  }

  function renderBatchImageList() {
    if (!batchImageListRoot) return;
    batchImageListRoot.innerHTML = "";

    const list = batchState.images;
    if (!list.length) {
      const empty = document.createElement("div");
      empty.className = "batch-image-list-empty";
      empty.textContent = "Загрузите одно или несколько изображений.";
      batchImageListRoot.appendChild(empty);
      return;
    }

    const filteredIdx = getFilteredBatchIndices();
    if (!filteredIdx.length) {
      const empty = document.createElement("div");
      empty.className = "batch-image-list-empty";
      empty.textContent = "Ничего не найдено по фильтру.";
      batchImageListRoot.appendChild(empty);
      return;
    }

    for (const i of filteredIdx) {
      const im = list[i];

      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "batch-image-card";
      if (i === batchState.currentIndex) btn.classList.add("is-active");
      btn.setAttribute("aria-pressed", i === batchState.currentIndex ? "true" : "false");

      const iconEl = document.createElement("span");
      iconEl.className = "bic-icon";
      iconEl.setAttribute("aria-hidden", "true");
      iconEl.textContent = batchItemStatusIcon(im);

      const body = document.createElement("span");
      body.className = "bic-body";

      const nameEl = document.createElement("span");
      nameEl.className = "bic-name";
      nameEl.textContent = im.displayName;

      const metaRow = document.createElement("span");
      metaRow.className = "bic-meta-row";

      const metaEl = document.createElement("span");
      metaEl.className = "bic-meta";
      const nBoxes = Array.isArray(im.detections) ? im.detections.length : 0;
      metaEl.textContent = nBoxes > 0 ? `${nBoxes}\u00A0аннот.` : "";

      const categoryLabel = normalizeImageCategory(im.category ?? "");

      body.appendChild(nameEl);
      metaRow.appendChild(metaEl);
      if (categoryLabel) {
        const catEl = document.createElement("span");
        catEl.className = "bic-category";
        catEl.textContent = categoryLabel;
        metaRow.appendChild(catEl);
      }
      body.appendChild(metaRow);

      btn.appendChild(iconEl);
      btn.appendChild(body);

      btn.addEventListener("click", () => selectBatchIndex(i));

      batchImageListRoot.appendChild(btn);
    }
  }

  function scrollActiveBatchImageToTop() {
    if (!batchImageListRoot) return;
    const activeCard = batchImageListRoot.querySelector(
      ".batch-image-card.is-active"
    );
    if (!(activeCard instanceof HTMLElement)) return;
    batchImageListRoot.scrollTop = activeCard.offsetTop;
  }

  /** @param {number} index */
  function selectBatchIndex(index) {
    maskMenu.close();
    if (index < 0 || index >= batchState.images.length) return;
    if (batchState.currentIndex === index) {
      updateBatchNavUi();
      return;
    }
    pointerInteraction = null;
    clearAddBoxCrosshairOverlayPx();
    imageUndoStack.length = 0;
    imageRedoStack.length = 0;
    hotkeyPreferredNewBboxClassName = null;
    resetViewerZoom();
    batchState.currentIndex = index;
    viewerMode = "image";
    touchBatch();
    scheduleWorkspaceAutosave(0);
    updateBatchNavUi();
    scrollActiveBatchImageToTop();
    showCurrentPreview();
  }

  function updateMainViewerNav() {
    const n = batchState.images.length;
    const idx = batchState.currentIndex;
    const im = currentImageItem();

    if (!n) {
      if (videoState && viewerMode === "video") {
        viewerFilename.textContent = videoState.fileName;
        viewerCounter.textContent = "Видео";
      } else {
        viewerFilename.textContent = videoState?.fileName ?? "";
        viewerCounter.textContent = videoState ? "Видео (скрыто)" : "Нет кадров";
      }
      viewerPrev.disabled = true;
      viewerNext.disabled = true;
      markReviewedBtn.disabled = true;
      skipImageBtn.disabled = true;
      syncVideoToolbar();
      return;
    }

    viewerFilename.textContent = im?.displayName ?? "";
    viewerCounter.textContent = `${idx + 1} из ${n}`;
    viewerPrev.disabled = idx <= 0;
    viewerNext.disabled = idx >= n - 1;
    markReviewedBtn.disabled = !im || im.reviewed === true;
    skipImageBtn.disabled =
      !n || !im || im.status === "skipped" || im.reviewed === true;
    syncVideoToolbar();
  }

  /**
   * Пропуск текущего кадра (очередь разметки).
   */
  function skipCurrentImage() {
    const im = currentImageItem();
    if (!im || im.status === "skipped" || im.reviewed === true) return;
    const idx = batchState.currentIndex;
    im.status = "skipped";
    im.reviewed = false;
    im.error = null;
    showToast(`Файл пропущен: «${im.displayName}».`, {
      type: "info",
      durationMs: 2800,
    });
    touchBatch();
    scheduleWorkspaceAutosave(0);
    updateBatchNavUi();
    const nextIdx = pickNextSequentialImageIndex(idx);
    if (nextIdx !== idx) selectBatchIndex(nextIdx);
  }

  /**
   * После изменения bbox на текущем кадре: `edited`, затем обновление UI (header/list/canvas при активном кадре).
   * Из консоли/редактора: `globalThis.markWorkspaceBboxEdited()`.
   */
  function markCurrentImageBboxEdited() {
    const im = currentImageItem();
    if (!im) return;
    im.edited = true;
    markAnnotationTypeReviewed(im, "detect");
    touchBatch();
    updateBatchNavUi();
  }

  globalThis.markWorkspaceBboxEdited = markCurrentImageBboxEdited;

  function updateBatchNavUi() {
    updateMainViewerNav();
    updateHeaderProgressSummary();
    syncEditorChrome();
    renderBatchImageList();
    syncTrainClassesUi();
    buildInspector();
  }

  /**
   * Показывает превью для batchState.images[batchState.currentIndex].
   */
  function showCurrentPreview() {
    if (viewerMode === "video" && videoState) {
      showVideoView();
      syncPanCursorUi();
      return;
    }

    hideVideoView();
    const im = currentImageItem();
    previewImage.onload = null;
    previewImage.onerror = null;

    if (!im?.objectUrl) {
      previewImage.src = "";
      previewImage.style.display = "none";
      previewImage.style.transform = "";
      if (videoState) {
        showVideoView();
        syncPanCursorUi();
        return;
      }
      placeholderText.style.display = "block";
      clearCanvas();
      buildRightPanel();
      updateBatchNavUi();
      return;
    }

    previewImage.onload = () => {
      if (!im) return;
      const w = previewImage.naturalWidth;
      const h = previewImage.naturalHeight;
      let sizeChanged = false;
      if (w > 0 && h > 0) {
        sizeChanged = im.width !== w || im.height !== h;
        im.width = w;
        im.height = h;
      }
      placeholderText.style.display = "none";
      previewImage.style.display = "block";
      applyPreviewImageTransform();
      syncPanCursorUi();
      clearCanvas();
      resizeCanvasToFrame();
      buildRightPanel();
      draw();
      updateBatchNavUi();
      if (sizeChanged) {
        touchBatch();
        scheduleWorkspaceAutosave(0);
      }
    };
    previewImage.onerror = () => {
      if (im) {
        im.status = "failed";
        im.error = "Не удалось отобразить изображение";
        touchBatch();
        updateBatchNavUi();
      }
      placeholderText.style.display = "block";
      previewImage.style.display = "none";
      previewImage.style.transform = "";
      syncPanCursorUi();
      clearCanvas();
    };

    previewImage.src = im.objectUrl;
  }

  /** @param {number} delta -1 или +1 */
  function navigateBatch(delta) {
    selectBatchIndex(batchState.currentIndex + delta);
  }

  /** Кадры, по которым ещё предполагается проход review (не отмечено проверенным и не пропущено). */
  function batchImageNeedsAttention(im) {
    return !im.reviewed && im.status !== "skipped";
  }

  /**
   * Следующий по порядку индекс с `batchImageNeedsAttention`, начиная после `fromIndex` (циклически).
   * @returns {number|null}
   */
  function findNextUnreviewedNonSkippedIndex(fromIndex) {
    const images = batchState.images;
    const n = images.length;
    if (!n) return null;
    for (let step = 1; step < n; step++) {
      const i = (fromIndex + step) % n;
      if (batchImageNeedsAttention(images[i])) return i;
    }
    return null;
  }

  /** Следующий кадр по порядку в списке (только вперёд). */
  function pickNextSequentialImageIndex(fromIndex) {
    const n = batchState.images.length;
    if (fromIndex + 1 < n) return fromIndex + 1;
    return fromIndex;
  }

  function clearAll(persist = false) {
    maskMenu.close();
    closeRunMenu();
    revokeBatchObjectUrls();
    pointerInteraction = null;
    clearAddBoxCrosshairOverlayPx();
    imageUndoStack.length = 0;
    imageRedoStack.length = 0;
    hotkeyPreferredNewBboxClassName = null;
    resetViewerZoom();
    batchState = createEmptyBatchState();

    fileInput.value = "";
    videoFileInput.value = "";
    closeCategoryModal();
    clearVideoState();
    previewImage.src = "";
    previewImage.style.display = "none";
    placeholderText.style.display = "block";
    groupsRoot.innerHTML = "";
    batchState.settings.confidenceThreshold = 0;
    updateConfidenceFilterDom();
    syncBatchListFilterUiFromState();
    detectAllInFlight = false;
    syncRecognizeBusy();
    updateBatchNavUi();
    setStatus(runButtonIdleText());
    clearCanvas();
    if (persist) scheduleWorkspaceAutosave(0);
    syncInferenceModeUi();
  }

  /** @param {any} snap */
  function restoreWorkspaceSnapshot(snap) {
    revokeBatchObjectUrls();
    resetViewerZoom();
    const empty = createEmptyBatchState();
    const legacyMode = normalizeInferenceMode(snap?.settings?.inferenceMode);
    const images = Array.isArray(snap?.images)
      ? snap.images
          .map((im, i) => deserializeImageItem(im, i + 1, legacyMode))
          .filter(Boolean)
      : [];

    batchState = {
      batchId: typeof snap?.batchId === "string" ? snap.batchId : empty.batchId,
      images: /** @type {BatchImageItem[]} */ (images),
      currentIndex: clamp(
        typeof snap?.currentIndex === "number" ? snap.currentIndex : 0,
        0,
        Math.max(0, images.length - 1)
      ),
      createdAt: typeof snap?.createdAt === "string" ? snap.createdAt : empty.createdAt,
      updatedAt: typeof snap?.updatedAt === "string" ? snap.updatedAt : empty.updatedAt,
      importSummary: {
        ...empty.importSummary,
        ...(snap?.importSummary && typeof snap.importSummary === "object"
          ? snap.importSummary
          : {}),
      },
      settings: {
        ...empty.settings,
        ...(snap?.settings && typeof snap.settings === "object" ? snap.settings : {}),
        classVisibility: {
          ...(snap?.settings?.classVisibility &&
          typeof snap.settings.classVisibility === "object"
            ? snap.settings.classVisibility
            : {}),
        },
      },
    };

    applyClassOrdersFromSettings(batchState.settings);

    fileInput.value = "";
    updateConfidenceFilterDom();
    syncBatchListFilterUiFromState();
    const filtersReset = ensureRestoredBatchListVisible();
    detectAllInFlight = false;
    syncRecognizeBusy();
    setStatus(runButtonIdleText());

    if (batchState.images.length) {
      if (filtersReset) {
        touchBatch();
        scheduleWorkspaceAutosave();
      }
      updateBatchNavUi();
      showCurrentPreview();
    } else {
      previewImage.src = "";
      previewImage.style.display = "none";
      placeholderText.style.display = "block";
      groupsRoot.innerHTML = "";
      clearCanvas();
      updateBatchNavUi();
    }
    syncInferenceModeUi();
  }

  /** @param {BatchState["importSummary"]} s */
  function describeImportIssues(s) {
    const lines = [];
    if (s.skippedNonImage)
      lines.push(`Не изображения (не image/*): ${s.skippedNonImage}`);
    if (s.skippedOverSize)
      lines.push(`Больше ${MAX_IMAGE_BYTES / (1024 * 1024)} МБ: ${s.skippedOverSize}`);
    if (s.skippedOverBatchLimit)
      lines.push(`Лимит ${s.importLimit ?? MAX_DIRECT_IMAGES} файлов за раз: не загружено ${s.skippedOverBatchLimit}`);
    if (s.skippedService)
      lines.push(`Служебные файлы архива: ${s.skippedService}`);
    if (s.skippedUnsupported)
      lines.push(`Неподдерживаемые файлы в ZIP: ${s.skippedUnsupported}`);
    return lines;
  }

  /** @param {BatchState["importSummary"]} s */
  function collectImportSkipBuckets(s) {
    /** @type {Array<{ label: string; names: string[] }>} */
    const out = [];
    const lim = s.importLimit ?? MAX_DIRECT_IMAGES;
    /**
     * @param {string} label
     * @param {number} count
     * @param {string[]|undefined} names
     */
    const push = (label, count, names) => {
      if (!count) return;
      const arr = Array.isArray(names) ? names.filter(Boolean) : [];
      if (arr.length) out.push({ label, names: arr });
    };
    push(
      "Не изображения (не image/*)",
      s.skippedNonImage,
      s.skippedNonImageNames
    );
    push(
      `Больше ${MAX_IMAGE_BYTES / (1024 * 1024)} МБ`,
      s.skippedOverSize,
      s.skippedOverSizeNames
    );
    push(
      `Лимит ${lim} файлов за раз`,
      s.skippedOverBatchLimit,
      s.skippedOverBatchLimitNames
    );
    push(
      "Служебные пути в архиве",
      s.skippedService,
      s.skippedServiceNames
    );
    push(
      "Неподдерживаемые файлы в ZIP",
      s.skippedUnsupported,
      s.skippedUnsupportedNames
    );
    return out;
  }

  /** @param {BatchImageItem} im */
  async function ensureImageItemDecodedDimensions(im) {
    if (im.width > 0 && im.height > 0) return;
    try {
      const bmp = await createImageBitmap(im.blob);
      try {
        im.width = bmp.width;
        im.height = bmp.height;
      } finally {
        if ("close" in bmp && typeof bmp.close === "function") bmp.close();
      }
    } catch {
      /* оставить 0 */
    }
  }

  /**
   * Импорт архива Export Full Project ZIP: восстановление batch, сохранение в IndexedDB.
   * @param {any} zip
   * @param {any} payload project.json
   */
  async function finalizeProjectZipImport(zip, payload) {
    revokeBatchObjectUrls();

    const payloadSettings =
      payload?.settings && typeof payload.settings === "object"
        ? payload.settings
        : {};
    const projectClassSettings = {
      ...payloadSettings,
      classOrders: {
        ...(payloadSettings.classOrders &&
        typeof payloadSettings.classOrders === "object"
          ? payloadSettings.classOrders
          : {}),
      },
    };
    const classNamesFromPayload = (rows) =>
      Array.isArray(rows)
        ? rows
            .map((row) =>
              String(typeof row === "string" ? row : row?.name ?? "")
                .trim()
                .toLowerCase()
            )
            .filter(Boolean)
        : [];
    const payloadDetectOrder = classNamesFromPayload(payload?.classes?.detect);
    const payloadSegOrder = classNamesFromPayload(
      payload?.classes?.segment ?? payload?.classes?.seg
    );
    if (payloadDetectOrder.length) projectClassSettings.classOrders.detect = payloadDetectOrder;
    if (payloadSegOrder.length) projectClassSettings.classOrders.seg = payloadSegOrder;
    applyClassOrdersFromSettings(projectClassSettings);

    const empty = createEmptyBatchState();
    const rows = Array.isArray(payload?.images) ? payload.images : [];

    /** @type {BatchImageItem[]} */
    const builtImages = [];
    let skippedManifestEntries = 0;

    for (let i = 0; i < rows.length; i++) {
      if (builtImages.length >= MAX_ZIP_IMAGES) break;

      const row = rows[i];
      const stem =
        typeof row?.exportStem === "string" ? row.exportStem.trim() : "";
      const categoryDir =
        typeof row?.exportCategoryDir === "string"
          ? row.exportCategoryDir.trim()
          : typeof row?.category === "string" && row.category.trim()
            ? row.category.trim().replace(/[\\/:*?"<>|]/g, "_").replace(/\s+/g, "_")
            : "";
      if (!stem) {
        skippedManifestEntries++;
        continue;
      }

      const imgRelPath = categoryDir
        ? `images/${categoryDir}/${stem}.png`
        : `images/${stem}.png`;
      let imgEntry =
        findZipEntryCaseInsensitive(zip, imgRelPath) ||
        findZipEntryCaseInsensitive(zip, `images/${stem}.png`) ||
        findZipEntryBySuffixCaseInsensitive(zip, `/${stem}.png`);
      if (!imgEntry) {
        skippedManifestEntries++;
        console.warn("[project zip] нет файла images/", stem, ".png");
        continue;
      }

      let pngBlob;
      try {
        pngBlob = await imgEntry.async("blob");
      } catch (err) {
        skippedManifestEntries++;
        console.warn("[project zip] чтение изображения", stem, err);
        continue;
      }

      const typedBlob = new Blob([pngBlob], { type: "image/png" });

      let annotRaw = null;
      const annotRelPath = categoryDir
        ? `annotations/${categoryDir}/${stem}.json`
        : `annotations/${stem}.json`;
      const annotEntry =
        findZipEntryCaseInsensitive(zip, annotRelPath) ||
        findZipEntryCaseInsensitive(zip, `annotations/${stem}.json`) ||
        findZipEntryBySuffixCaseInsensitive(zip, `/${stem}.json`);
      if (annotEntry) {
        try {
          annotRaw = JSON.parse(await annotEntry.async("string"));
        } catch (err) {
          console.warn("[project zip] annotations", stem, err);
        }
      }

      let detRaw = annotRaw
        ? rawDetectionsFromAnnotationExport(annotRaw)
        : [];
      if (
        !annotRaw &&
        typeof row?.width === "number" &&
        row.width > 0 &&
        typeof row?.height === "number" &&
        row.height > 0
      ) {
        const labelPaths = [
          `labels/detect/${categoryDir ? `${categoryDir}/` : ""}${stem}.txt`,
          `labels/segment/${categoryDir ? `${categoryDir}/` : ""}${stem}.txt`,
          `labels/${categoryDir ? `${categoryDir}/` : ""}${stem}.txt`,
        ];
        const usedEntries = new Set();
        for (const labelPath of labelPaths) {
          const entry =
            findZipEntryCaseInsensitive(zip, labelPath) ||
            findZipEntryBySuffixCaseInsensitive(zip, `/${labelPath}`);
          if (!entry || usedEntries.has(entry.name)) continue;
          usedEntries.add(entry.name);
          try {
            const text = await entry.async("string");
            detRaw.push(
              ...parseYoloTxtDetectionsForImport(
                text,
                row.width,
                row.height
              )
            );
          } catch (err) {
            console.warn("[project zip] labels", labelPath, err);
          }
        }
      }

      const rawIm = {
        id: typeof row?.id === "string" ? row.id : newImageItemId(),
        displayName: displayNameAfterProjectImport(
          row,
          stem,
          builtImages.length + 1
        ),
        originalName:
          typeof row?.originalName === "string"
            ? row.originalName
            : `${stem}.png`,
        fileType:
          typeof row?.fileType === "string" ? row.fileType : "image/png",
        fileSize:
          typeof row?.fileSize === "number" ? row.fileSize : typedBlob.size,
        width: typeof row?.width === "number" ? row.width : 0,
        height: typeof row?.height === "number" ? row.height : 0,
        blob: typedBlob,
        status: coerceImportedImageStatus(row?.status),
        category: typeof row?.category === "string" ? row.category : null,
        reviewed: row?.reviewed === true,
        edited: row?.edited === true,
        detections: detRaw,
        error: null,
        modelStates: annotRaw?.modelStates || row?.modelStates,
        panel: row?.panel,
      };

      const im = deserializeImageItem(
        rawIm,
        builtImages.length + 1,
        normalizeInferenceMode(payload?.settings?.inferenceMode)
      );
      reconcileImageStatusWithDetections(im);
      reconcileReviewedAndSkippedStatus(im);
      await ensureImageItemDecodedDimensions(im);
      builtImages.push(im);
      await yieldToMain();
    }

    if (!builtImages.length) {
      batchState = createEmptyBatchState();
      previewImage.src = "";
      previewImage.style.display = "none";
      placeholderText.style.display = "block";
      groupsRoot.innerHTML = "";
      clearCanvas();
      updateBatchNavUi();
      fileInput.value = "";
      syncInferenceModeUi();
      showToast(
        "Не удалось импортировать проект: нет изображений в images/ или повреждён архив.",
        { type: "error", durationMs: 5200 }
      );
      return;
    }

    const mergedSettings = {
      ...empty.settings,
      ...projectClassSettings,
      classVisibility: {
        ...(payload?.settings?.classVisibility &&
        typeof payload.settings.classVisibility === "object"
          ? payload.settings.classVisibility
          : {}),
      },
    };

    batchState = {
      batchId:
        typeof payload?.batchId === "string" ? payload.batchId : empty.batchId,
      images: builtImages,
      currentIndex: clamp(
        typeof payload?.currentIndex === "number" ? payload.currentIndex : 0,
        0,
        Math.max(0, builtImages.length - 1)
      ),
      createdAt:
        typeof payload?.createdAt === "string"
          ? payload.createdAt
          : empty.createdAt,
      updatedAt:
        typeof payload?.updatedAt === "string"
          ? payload.updatedAt
          : new Date().toISOString(),
      importSummary: {
        ...empty.importSummary,
        imported: builtImages.length,
        importLimit: MAX_ZIP_IMAGES,
      },
      settings: { ...mergedSettings },
    };

    storeCurrentClassOrders(batchState.settings);

    fileInput.value = "";
    updateConfidenceFilterDom();
    syncBatchListFilterUiFromState();

    ensureRestoredBatchListVisible();
    detectAllInFlight = false;
    syncRecognizeBusy();
    setStatus(runButtonIdleText());

    touchBatch();
    try {
      setWorkspaceSaveStatus("saving");
      await saveWorkspaceToIndexedDB();
      setWorkspaceSaveStatus("saved");
    } catch (err) {
      console.warn("[project zip] IndexedDB save failed:", err);
      setWorkspaceSaveStatus("failed");
      showToast(
        "Автосохранение не удалось. Проверьте место на диске и доступ к IndexedDB.",
        { type: "error", durationMs: 5200 }
      );
    }

    groupsRoot.innerHTML = "";
    pointerInteraction = null;
    updateBatchNavUi();
    showCurrentPreview();

    syncInferenceModeUi();

    openImportSummaryModal({
      title: "Сводка импорта проекта (ZIP)",
      kind: "project-zip",
      addedImages: builtImages.length,
      skippedFiles: skippedManifestEntries,
      skipIssueLines: [
        ...(skippedManifestEntries > 0
          ? [
              `Записей в манифесте без файла в images/ или с ошибкой чтения: ${skippedManifestEntries}`,
            ]
          : []),
        ...(payload?.includesAnnotationsFolder === false
          ? [
              "Папка annotations/ отсутствует — разметка восстановлена из отдельных YOLO Detect и YOLO Seg файлов.",
            ]
          : []),
      ],
      skipNameBuckets: [],
    });
  }

  /**
   * @param {Array<{ blob: Blob|File; originalName: string }>} accepted
   * @param {BatchState["importSummary"]} summary
   * @param {{ openZipSummaryModal?: boolean }} [opts]
   */
  function finalizeImageImport(accepted, summary, opts = {}) {
    const firstNewIndex = batchState.images.length;
    const issues = describeImportIssues(summary);
    const skippedTotal =
      (summary.skippedNonImage || 0) +
      (summary.skippedOverSize || 0) +
      (summary.skippedOverBatchLimit || 0) +
      (summary.skippedService || 0) +
      (summary.skippedUnsupported || 0);

    if (!accepted.length) {
      showToast("Не удалось добавить ни одного изображения.", {
        type: "error",
        durationMs: 4200,
      });
      if (opts.openZipSummaryModal) {
        openImportSummaryModal({
          title: "Сводка импорта ZIP",
          kind: "images-zip",
          addedImages: 0,
          skippedFiles: skippedTotal,
          skipIssueLines: issues,
          skipNameBuckets: collectImportSkipBuckets(summary),
        });
      } else if (issues.length) {
        showToast(issues.join("; "), { type: "info", durationMs: 4200 });
      }
      syncBatchListFilterUiFromState();
      updateBatchNavUi();
      fileInput.value = "";
      return;
    }

    batchState.importSummary = summary;
    storeCurrentClassOrders(batchState.settings);

    accepted.forEach((file, i) => {
      batchState.images.push(
        imageItemFromFile(file.blob, firstNewIndex + i + 1, file.originalName)
      );
    });

    selectBatchIndex(firstNewIndex);
    touchBatch();
    scheduleWorkspaceAutosave(0);

    if (opts.openZipSummaryModal) {
      openImportSummaryModal({
        title: "Сводка импорта ZIP",
        kind: "images-zip",
        addedImages: accepted.length,
        skippedFiles: skippedTotal,
        skipIssueLines: issues,
        skipNameBuckets: collectImportSkipBuckets(summary),
      });
    } else if (skippedTotal > 0) {
      showToast(
        `Загружено ${accepted.length}, пропущено ${skippedTotal}.`,
        { type: "info", durationMs: 2800 }
      );
    }

    syncBatchListFilterUiFromState();
    updateBatchNavUi();
    groupsRoot.innerHTML = "";
    setStatus(runButtonIdleText());
    showCurrentPreview();
    fileInput.value = "";
  }

  /**
   * @param {File[]} txtFiles
   * @returns {Promise<{ appliedImages: number, unmatched: string[], noSize: string[] }>}
   */
  async function applyYoloTxtLabelFiles(txtFiles) {
    if (!batchState.images.length) {
      showToast(
        "Сначала загрузите изображения — TXT с разметкой привязываются по имени файла.",
        { type: "warning", durationMs: 4200 }
      );
      return { appliedImages: 0, unmatched: txtFiles.map((f) => f.name), noSize: [] };
    }

    /** @type {Set<string>} */
    const affectedIds = new Set();
    /** @type {string[]} */
    const unmatched = [];
    /** @type {string[]} */
    const noSize = [];
    let appliedImages = 0;
    let protectedImportApproved = null;

    for (const file of txtFiles) {
      const stem = labelStemKeyForMatch(file.name);
      const targets = batchState.images.filter(
        (im) => labelStemKeyForMatch(im.originalName) === stem
      );
      if (!targets.length) {
        unmatched.push(file.name);
        continue;
      }

      let text = "";
      try {
        text = await file.text();
      } catch {
        unmatched.push(file.name);
        continue;
      }

      for (const im of targets) {
        await ensureImageItemDecodedDimensions(im);
        if (!(im.width > 0 && im.height > 0)) {
          noSize.push(im.originalName || im.displayName);
          continue;
        }

        const raw = parseYoloTxtDetectionsForImport(text, im.width, im.height);
        if (!raw.length) continue;
        const imported = raw.map((d, i) => normalizeDetection(d, i));
        const importedTypes = new Set(imported.map(annotationTypeOf));
        im.modelStates = normalizeModelStates(im.modelStates);
        const hasProtectedType = Array.from(importedTypes).some(
          (type) =>
            im.modelStates[type].status === "reviewed" ||
            im.detections.some(
              (d) => annotationTypeOf(d) === type && d.source === "human"
            )
        );
        if (hasProtectedType && protectedImportApproved !== true) {
          if (protectedImportApproved === false) continue;
          protectedImportApproved = window.confirm(
            "\u0415\u0441\u0442\u044c \u0440\u0443\u0447\u043d\u0430\u044f \u0438\u043b\u0438 \u043f\u0440\u043e\u0432\u0435\u0440\u0435\u043d\u043d\u0430\u044f " +
              "\u0440\u0430\u0437\u043c\u0435\u0442\u043a\u0430. \u0417\u0430\u043c\u0435\u043d\u0438\u0442\u044c \u0435\u0451 \u0438\u043c\u043f\u043e\u0440\u0442\u043e\u043c TXT?"
          );
          if (!protectedImportApproved) continue;
        }
        const preserved = im.detections.filter(
          (d) => !importedTypes.has(annotationTypeOf(d))
        );
        let nextId =
          preserved.reduce((maxId, d) => Math.max(maxId, d.id), -1) + 1;
        for (const d of imported) d.id = nextId++;
        im.detections = [...preserved, ...imported];
        const keptIds = new Set(im.detections.map((d) => d.id));
        im.panel.detEnabled = new Map(
          Array.from(im.panel.detEnabled.entries()).filter(([id]) =>
            keptIds.has(id)
          )
        );
        if (
          im.panel.selectedDetectionId != null &&
          !keptIds.has(im.panel.selectedDetectionId)
        ) {
          im.panel.selectedDetectionId = null;
        }
        for (const d of imported) {
          im.panel.detEnabled.set(d.id, true);
        }
        im.reviewed = false;
        im.edited = true;
        for (const type of importedTypes) markAnnotationTypeReviewed(im, type);
        affectedIds.add(im.id);
        appliedImages++;
      }
    }

    if (appliedImages > 0) {
      touchBatch();
      scheduleWorkspaceAutosave(0);
      updateBatchNavUi();
      const cur = currentImageItem();
      if (cur && affectedIds.has(cur.id)) {
        buildRightPanel();
        draw();
      }
    }

    if (unmatched.length) {
      showToast(
        unmatched.length === 1
          ? `Нет кадра с именем «${labelStemKeyForMatch(unmatched[0])}» для ${unmatched[0]}.`
          : `Не найдены кадры для ${unmatched.length} TXT (по имени без расширения).`,
        { type: "warning", durationMs: 4500 }
      );
    } else if (appliedImages > 0) {
      showToast(
        `Разметка из TXT применена к ${appliedImages} кадр(ам).`,
        { type: "success", durationMs: 3200 }
      );
    } else if (noSize.length) {
      showToast(
        "Не удалось прочитать размеры изображения для применения TXT.",
        { type: "warning", durationMs: 4200 }
      );
    }

    return { appliedImages, unmatched, noSize };
  }

  /**
   * @param {FileList|File[]|Iterable<File>} files
   */
  async function processIncomingFiles(files) {
    const list = Array.from(files);
    if (!list.length) return;
    const zipFiles = list.filter((file) => isZipFile(file));
    if (zipFiles.length) {
      if (zipFiles.length > 1 || list.length > 1) {
        showToast(
          "Импортируется первый ZIP; остальные файлы игнорируются.",
          { type: "info", durationMs: 3800 }
        );
      }
      void ingestZipFile(zipFiles[0]);
      return;
    }

    const txtFiles = list.filter((f) => isYoloLabelTxtFile(f));
    const imageFiles = list.filter(
      (f) => !isYoloLabelTxtFile(f) && f.type.startsWith("image/")
    );
    const rest = list.filter((f) => !txtFiles.includes(f) && !imageFiles.includes(f));

    if (imageFiles.length) {
      ingestImageFiles(imageFiles);
    }
    if (txtFiles.length) {
      await applyYoloTxtLabelFiles(txtFiles);
    }
    if (rest.length && !imageFiles.length && !txtFiles.length) {
      ingestImageFiles(list);
    } else if (rest.length) {
      showToast(
        `Пропущено файлов: ${rest.length} (нужны изображения или YOLO .txt).`,
        { type: "info", durationMs: 3600 }
      );
    }
  }

  /**
   * @param {DragEvent} e
   */
  function dragEventHasImportableFiles(e) {
    const dt = e.dataTransfer;
    if (!dt) return false;
    if (dt.types && Array.from(dt.types).includes("Files")) return true;
    return (dt.files?.length ?? 0) > 0;
  }

  /** @param {ClipboardEvent} e */
  function clipboardImageFiles(e) {
    const cd = e.clipboardData;
    if (!cd) return [];
    /** @type {File[]} */
    const out = [];
    const items = cd.items;
    if (items?.length) {
      for (let i = 0; i < items.length; i++) {
        const item = items[i];
        if (item.kind !== "file") continue;
        const type = item.type || "";
        if (!type.startsWith("image/")) continue;
        const file = item.getAsFile();
        if (!file) continue;
        out.push(normalizePastedImageFile(file, out.length));
      }
    }
    if (!out.length && cd.files?.length) {
      for (let i = 0; i < cd.files.length; i++) {
        const file = cd.files[i];
        if (!file.type.startsWith("image/")) continue;
        out.push(normalizePastedImageFile(file, out.length));
      }
    }
    return out;
  }

  /** @param {File} file @param {number} index */
  function normalizePastedImageFile(file, index) {
    if (file.name && file.name.trim()) return file;
    const mime = file.type && file.type.startsWith("image/") ? file.type : "image/png";
    const ext = mime.split("/")[1] || "png";
    return new File([file], `pasted-${Date.now()}-${index + 1}.${ext}`, { type: mime });
  }

  let imageFrameDragDepth = 0;

  function setImageFrameDragOverActive(active) {
    imageFrame.classList.toggle("image-frame-drag-over", active);
  }

  imageFrame.addEventListener("dragenter", (e) => {
    if (!dragEventHasImportableFiles(e)) return;
    e.preventDefault();
    imageFrameDragDepth++;
    setImageFrameDragOverActive(true);
  });

  imageFrame.addEventListener("dragover", (e) => {
    if (!dragEventHasImportableFiles(e)) return;
    e.preventDefault();
    if (e.dataTransfer) e.dataTransfer.dropEffect = "copy";
  });

  imageFrame.addEventListener("dragleave", (e) => {
    if (!dragEventHasImportableFiles(e)) return;
    e.preventDefault();
    imageFrameDragDepth = Math.max(0, imageFrameDragDepth - 1);
    if (imageFrameDragDepth === 0) setImageFrameDragOverActive(false);
  });

  imageFrame.addEventListener("drop", (e) => {
    if (!dragEventHasImportableFiles(e)) return;
    e.preventDefault();
    imageFrameDragDepth = 0;
    setImageFrameDragOverActive(false);
    const files = e.dataTransfer?.files;
    if (!files?.length) return;
    void processIncomingFiles(files);
  });

  document.addEventListener("paste", (e) => {
    if (isTypingInteractionTarget(e.target)) return;
    const images = clipboardImageFiles(e);
    if (!images.length) return;
    e.preventDefault();
    void processIncomingFiles(images);
    showToast(
      images.length === 1
        ? "Изображение вставлено из буфера."
        : `Вставлено изображений: ${images.length}.`,
      { type: "info", durationMs: 2400 }
    );
  });

  /**
   * @param {FileList|File[]} files
   */
  function ingestImageFiles(files) {
    if (videoState) {
      clearVideoState();
    }
    if (!batchState.images.length) resetTrainClassesToDefault();
    const skippedNonImageNames = [];
    const skippedOverSizeNames = [];
    const skippedOverBatchLimitNames = [];
    const skippedUnsupportedNames = [];
    /** @type {Array<{ blob: Blob|File; originalName: string }>} */
    const accepted = [];

    for (const f of files) {
      if (isZipFile(f)) {
        skippedUnsupportedNames.push(f.name);
        continue;
      }
      if (!f.type.startsWith("image/")) {
        skippedNonImageNames.push(f.name);
        continue;
      }
      if (f.size > MAX_IMAGE_BYTES) {
        skippedOverSizeNames.push(f.name);
        continue;
      }
      if (accepted.length >= MAX_DIRECT_IMAGES) {
        skippedOverBatchLimitNames.push(f.name);
        continue;
      }
      accepted.push({ blob: f, originalName: f.name });
    }

    finalizeImageImport(accepted, {
      imported: accepted.length,
      skippedNonImage: skippedNonImageNames.length,
      skippedNonImageNames: [...skippedNonImageNames],
      skippedOverSize: skippedOverSizeNames.length,
      skippedOverSizeNames: [...skippedOverSizeNames],
      skippedOverBatchLimit: skippedOverBatchLimitNames.length,
      skippedOverBatchLimitNames: [...skippedOverBatchLimitNames],
      importLimit: MAX_DIRECT_IMAGES,
      skippedService: 0,
      skippedServiceNames: [],
      skippedUnsupported: skippedUnsupportedNames.length,
      skippedUnsupportedNames,
    });
  }

  /**
   * Если в архиве есть служебный YAML со списком классов — тихо подстроить порядок при точном совпадении с фиксированным набором сайта.
   * @param {any} zip
   */
  async function applyTrainClassesFromZipIfPresent(zip) {
    const yamlEntry = Object.values(zip.files)
      .filter((entry) => !entry.dir && DATA_YAML_RE.test(normalizeZipPath(entry.name)))
      .sort((a, b) =>
        normalizeZipPath(a.name).localeCompare(normalizeZipPath(b.name))
      )[0];
    if (!yamlEntry) return null;
    try {
      const text = await yamlEntry.async("string");
      const parsedNames = parseYamlClassNames(text);
      if (!parsedNames?.length) return null;
      const normalized = parsedNames.map((name) =>
        String(name).trim().toLowerCase()
      );
      const sameSet = (type) => {
        const expected = classesForAnnotationType(type)
          .map((tc) => tc.name)
          .sort();
        const actual = [...normalized].sort();
        return (
          expected.length === actual.length &&
          expected.every((name, i) => name === actual[i])
        );
      };
      const type = sameSet("seg") ? "seg" : sameSet("detect") ? "detect" : null;
      if (!type) return null;
      applyTrainClassOrderFromNames(
        normalized,
        type === "seg" ? "segmentation" : "detection"
      );
      return type;
    } catch {
      /* игнор */
    }
  }

  async function restoreAnnotationsFromOrdinaryZip(
    zip,
    images,
    genericLabelType = null
  ) {
    let restoredImages = 0;
    const findFirstEntry = (candidates) => {
      for (const candidate of candidates) {
        const entry =
          findZipEntryCaseInsensitive(zip, candidate) ||
          findZipEntryBySuffixCaseInsensitive(zip, candidate);
        if (entry) return entry;
      }
      return null;
    };

    const annotationRecords = [];
    for (const entry of Object.values(zip.files)) {
      const path = normalizeZipPath(entry.name);
      if (entry.dir || !/(^|\/)annotations\/.+\.json$/i.test(path)) continue;
      try {
        const payload = JSON.parse(await entry.async("string"));
        if (!payload || typeof payload !== "object") continue;
        annotationRecords.push({
          path: path.toLowerCase(),
          payload,
        });
      } catch {
        /* Повреждённый JSON не мешает восстановить остальные кадры. */
      }
    }
    const usedAnnotationPaths = new Set();
    const normalizeMatchName = (value) =>
      normalizeZipPath(String(value ?? "")).trim().toLowerCase();
    const matchNameScore = (left, right) => {
      const a = normalizeMatchName(left);
      const b = normalizeMatchName(right);
      if (!a || !b) return 0;
      if (a === b) return 60;
      const aBase = a.split("/").pop() || a;
      const bBase = b.split("/").pop() || b;
      if (aBase === bBase) return 40;
      const aStem = aBase.replace(/\.[^./]+$/, "");
      const bStem = bBase.replace(/\.[^./]+$/, "");
      return aStem && aStem === bStem ? 20 : 0;
    };
    const annotationRecordScore = (record, im) => {
      const imageMeta = record.payload?.image || {};
      if (
        imageMeta.id != null &&
        im.id != null &&
        String(imageMeta.id) === String(im.id)
      ) {
        return 1000;
      }
      let score = 0;
      const originalNames = [im.originalName];
      const payloadOriginalNames = [
        imageMeta.originalName,
        imageMeta.imageNames?.beforeUpload,
        record.payload?.originalName,
      ];
      for (const left of originalNames) {
        for (const right of payloadOriginalNames) {
          score = Math.max(score, matchNameScore(left, right));
        }
      }
      const displayNames = [im.displayName];
      const payloadDisplayNames = [
        imageMeta.displayName,
        imageMeta.imageNames?.afterUploadOnSite,
        imageMeta.imageNames?.afterExport,
      ];
      for (const left of displayNames) {
        for (const right of payloadDisplayNames) {
          score = Math.max(score, matchNameScore(left, right));
        }
      }
      return score;
    };
    const findAnnotationRecord = (im, preferredEntry) => {
      const preferredPath = preferredEntry
        ? normalizeZipPath(preferredEntry.name).toLowerCase()
        : "";
      const preferred = annotationRecords.find(
        (record) =>
          record.path === preferredPath && !usedAnnotationPaths.has(record.path)
      );
      if (preferred) return preferred;

      let best = null;
      let bestScore = 0;
      let tied = false;
      for (const record of annotationRecords) {
        if (usedAnnotationPaths.has(record.path)) continue;
        const score = annotationRecordScore(record, im);
        if (score > bestScore) {
          best = record;
          bestScore = score;
          tied = false;
        } else if (score > 0 && score === bestScore) {
          tied = true;
        }
      }
      return bestScore > 0 && !tied ? best : null;
    };

    for (const im of images) {
      const imagePath = normalizeZipPath(im.originalName);
      const imageRelMatch = imagePath.match(/(?:^|\/)images\/(.+)$/i);
      const imageRel = imageRelMatch ? imageRelMatch[1] : imagePath;
      const stemRel = imageRel.replace(/\.[^./]+$/, "");
      const baseStem = stemRel.split("/").pop() || stemRel;
      let annotationPayload = null;
      let rawAnnotations = [];
      const completedTypes = new Set();

      const annotationEntry = findFirstEntry([
        `annotations/${stemRel}.json`,
        `annotations/${baseStem}.json`,
      ]);
      const annotationRecord = findAnnotationRecord(im, annotationEntry);
      if (annotationRecord) {
        annotationPayload = annotationRecord.payload;
        rawAnnotations = rawDetectionsFromAnnotationExport(annotationPayload);
        usedAnnotationPaths.add(annotationRecord.path);
      }

      if (!annotationPayload) {
        await ensureImageItemDecodedDimensions(im);
        if (!(im.width > 0 && im.height > 0)) continue;
        const labelSpecs = [
          {
            type: "detect",
            candidates: [
              `labels/detect/${stemRel}.txt`,
              `labels/detect/${baseStem}.txt`,
            ],
          },
          {
            type: "seg",
            candidates: [
              `labels/segment/${stemRel}.txt`,
              `labels/seg/${stemRel}.txt`,
              `labels/segment/${baseStem}.txt`,
              `labels/seg/${baseStem}.txt`,
            ],
          },
          {
            type: genericLabelType,
            candidates: [
              `labels/${stemRel}.txt`,
              `labels/${baseStem}.txt`,
            ],
          },
        ];
        const usedEntries = new Set();
        for (const spec of labelSpecs) {
          const entry = findFirstEntry(spec.candidates);
          if (!entry) continue;
          const entryPath = normalizeZipPath(entry.name).toLowerCase();
          if (usedEntries.has(entryPath)) continue;
          usedEntries.add(entryPath);
          const parsed = parseYoloTxtDetectionsForImport(
            await entry.async("string"),
            im.width,
            im.height
          );
          rawAnnotations.push(...parsed);
          if (spec.type) completedTypes.add(spec.type);
          for (const row of parsed) completedTypes.add(annotationTypeOf(row));
        }
      }

      if (!annotationPayload && !rawAnnotations.length && !completedTypes.size) {
        continue;
      }

      const normalized = rawAnnotations.map((d, i) => normalizeDetection(d, i));
      const usedIds = new Set();
      let nextId = normalized.reduce(
        (maxId, d) => Math.max(maxId, Number.isInteger(d.id) ? d.id : -1),
        -1
      ) + 1;
      for (const d of normalized) {
        if (!Number.isInteger(d.id) || usedIds.has(d.id)) d.id = nextId++;
        usedIds.add(d.id);
      }
      im.detections = normalized;
      im.modelStates = normalizeModelStates(annotationPayload?.modelStates, {
        annotations: normalized,
        legacyStatus: normalized.length ? "detected" : "empty",
        legacyReviewed: false,
        legacyMode: "detection",
      });
      for (const type of completedTypes) {
        if (im.modelStates[type].status === "not_run") {
          im.modelStates[type].status = normalized.some(
            (d) => annotationTypeOf(d) === type && d.source === "human"
          )
            ? "reviewed"
            : "ready";
        }
      }
      for (const state of Object.values(im.modelStates)) {
        if (state.status !== "running") continue;
        state.status = "error";
        state.error = "Model run was interrupted before ZIP export completed.";
        state.revision = (Number(state.revision) || 0) + 1;
      }
      im.panel.detEnabled = new Map(normalized.map((d) => [d.id, true]));
      im.panel.selectedDetectionId = null;
      im.edited = normalized.some((d) => d.source === "human");
      syncLegacyImageState(im, { preserveSkipped: false });
      restoredImages++;
    }

    if (restoredImages > 0) {
      touchBatch();
      scheduleWorkspaceAutosave(0);
      updateBatchNavUi();
      showCurrentPreview();
    }
    return restoredImages;
  }

  /** @param {File} zipFile */
  async function ingestZipFile(zipFile) {
    const JSZipCtor = globalThis.JSZip;
    if (!JSZipCtor) {
      showToast(
        "JSZip не загружен. Проверьте подключение к интернету и перезагрузите страницу.",
        { type: "error", durationMs: 6500 }
      );
      fileInput.value = "";
      return;
    }

    setStatus("Распаковываю ZIP…");

    try {
      const zip = await JSZipCtor.loadAsync(zipFile);

      const projEntry = findProjectJsonEntry(zip);
      if (projEntry) {
        let projectPayload = null;
        try {
          projectPayload = JSON.parse(await projEntry.async("string"));
        } catch {
          projectPayload = null;
        }
        if (
          projectPayload &&
          typeof projectPayload === "object" &&
          Array.isArray(projectPayload.images)
        ) {
          await finalizeProjectZipImport(zip, projectPayload);
          return;
        }
      }

      if (!batchState.images.length) resetTrainClassesToDefault();
      const genericLabelType = await applyTrainClassesFromZipIfPresent(zip);
      const skippedServiceNames = [];
      const skippedUnsupportedNames = [];
      const skippedOverSizeNames = [];
      const skippedOverBatchLimitNames = [];
      /** @type {Array<{ path: string; blob: Blob }>} */
      const imageEntries = [];

      const entries = Object.values(zip.files).sort((a, b) =>
        normalizeZipPath(a.name).localeCompare(normalizeZipPath(b.name))
      );
      const hasAnnotationJsonEntries = entries.some((entry) =>
        !entry.dir &&
        /(^|\/)annotations\/.+\.json$/i.test(normalizeZipPath(entry.name))
      );

      for (const entry of entries) {
        if (entry.dir) continue;
        const path = normalizeZipPath(entry.name);
        if (DATA_YAML_RE.test(path)) continue;
        if (/(^|\/)project\.json$/i.test(path)) continue;
        if (
          /(^|\/)labels\/.+\.txt$/i.test(path) ||
          /(^|\/)annotations\/.+\.json$/i.test(path) ||
          /(^|\/)classes(?:\.(?:detect|segment))?\.txt$/i.test(path)
        ) {
          continue;
        }
        if (isServiceZipPath(path)) {
          skippedServiceNames.push(path);
          continue;
        }
        if (!ZIP_IMAGE_EXT_RE.test(path)) {
          skippedUnsupportedNames.push(path);
          continue;
        }
        const blob = await entry.async("blob");
        if (blob.size > MAX_IMAGE_BYTES) {
          skippedOverSizeNames.push(path);
          continue;
        }
        if (imageEntries.length >= MAX_ZIP_IMAGES) {
          skippedOverBatchLimitNames.push(path);
          continue;
        }
        imageEntries.push({
          path,
          blob: new Blob([blob], {
            type: path.toLowerCase().endsWith(".webp")
              ? "image/webp"
              : path.toLowerCase().endsWith(".png")
                ? "image/png"
                : "image/jpeg",
          }),
        });
      }

      const firstNewIndex = batchState.images.length;
      const annotationsOnlyImport =
        imageEntries.length === 0 && hasAnnotationJsonEntries;
      if (annotationsOnlyImport) {
        const protectedCount = batchState.images.filter((im) => {
          const states = normalizeModelStates(im.modelStates);
          return (
            states.detect.status === "reviewed" ||
            states.seg.status === "reviewed" ||
            im.detections.some((d) => d.source === "human")
          );
        }).length;
        if (
          protectedCount > 0 &&
          !window.confirm(
            `На ${protectedCount} кадр(ах) есть проверенная или ручная разметка. ` +
              "Импорт JSON заменит разметку совпавших кадров. Продолжить?"
          )
        ) {
          fileInput.value = "";
          return;
        }
      } else {
        finalizeImageImport(
          imageEntries.map((entry) => ({
            blob: entry.blob,
            originalName: entry.path,
          })),
          {
            imported: imageEntries.length,
            skippedNonImage: 0,
            skippedNonImageNames: [],
            skippedOverSize: skippedOverSizeNames.length,
            skippedOverSizeNames,
            skippedOverBatchLimit: skippedOverBatchLimitNames.length,
            skippedOverBatchLimitNames,
            importLimit: MAX_ZIP_IMAGES,
            skippedService: skippedServiceNames.length,
            skippedServiceNames,
            skippedUnsupported: skippedUnsupportedNames.length,
            skippedUnsupportedNames,
          },
          { openZipSummaryModal: true }
        );
      }
      const annotationTargets = annotationsOnlyImport
        ? batchState.images
        : batchState.images.slice(firstNewIndex);
      const restoredAnnotations = await restoreAnnotationsFromOrdinaryZip(
        zip,
        annotationTargets,
        genericLabelType
      );
      if (annotationsOnlyImport) {
        showToast(
          restoredAnnotations > 0
            ? `JSON-аннотации восстановлены для ${restoredAnnotations} кадр(ов).`
            : "В архиве не найдено аннотаций, совпадающих с загруженными кадрами.",
          {
            type: restoredAnnotations > 0 ? "success" : "warning",
            durationMs: 4400,
          }
        );
        fileInput.value = "";
      }
    } catch (err) {
      console.error("[zip import] failed:", err);
      showToast(`Не удалось прочитать ZIP: ${String(err?.message || err)}`, {
        type: "error",
        durationMs: 5200,
      });
      fileInput.value = "";
    } finally {
      setStatus(runButtonIdleText());
    }
  }

  function clearCanvas() {
    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
  }

  function resizeCanvasToFrame() {
    const frameRect = imageFrame.getBoundingClientRect();
    if (frameRect.width <= 0 || frameRect.height <= 0) return;

    const dpr = window.devicePixelRatio || 1;
    overlay.style.width = `${frameRect.width}px`;
    overlay.style.height = `${frameRect.height}px`;
    overlay.width = Math.max(1, Math.floor(frameRect.width * dpr));
    overlay.height = Math.max(1, Math.floor(frameRect.height * dpr));

    const ctx = overlay.getContext("2d");
    if (ctx) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function clamp(v, a, b) {
    return Math.max(a, Math.min(b, v));
  }

  function resetViewerZoom() {
    viewerZoomScale = 1;
    viewerPanX = 0;
    viewerPanY = 0;
    applyPreviewImageTransform();
  }

  function clampViewerPan(frameW, frameH, drawnW, drawnH) {
    const maxPanX = Math.max(0, (drawnW - frameW) / 2);
    const maxPanY = Math.max(0, (drawnH - frameH) / 2);
    viewerPanX = clamp(viewerPanX, -maxPanX, maxPanX);
    viewerPanY = clamp(viewerPanY, -maxPanY, maxPanY);
  }

  function applyPreviewImageTransform() {
    if (previewImage.style.display === "none" || viewerMode === "video") {
      previewImage.style.transform = "";
      return;
    }
    if (
      viewerZoomScale <= VIEWER_ZOOM_MIN + 1e-6 &&
      Math.abs(viewerPanX) < 0.01 &&
      Math.abs(viewerPanY) < 0.01
    ) {
      previewImage.style.transform = "";
      return;
    }
    previewImage.style.transformOrigin = "center center";
    previewImage.style.transform = `translate(${viewerPanX}px, ${viewerPanY}px) scale(${viewerZoomScale})`;
  }

  function syncPanCursorUi() {
    const panReady =
      spacePanPressed &&
      viewerMode !== "video" &&
      !!effectiveOriginalSize() &&
      previewImage.style.display !== "none";
    imageFrame.classList.toggle("image-frame-pan-ready", panReady && !panDragState);
    imageFrame.classList.toggle("image-frame-pan-dragging", !!panDragState);
  }

  function editorModeIsReview() {
    return batchState.settings.editorMode === "review";
  }

  function editorModeIsEdit() {
    return batchState.settings.editorMode === "edit";
  }

  /**
   * @returns {null | {
   *   frameRect: DOMRect,
   *   offsetX: number,
   *   offsetY: number,
   *   sx: number,
   *   sy: number,
   *   imgW0: number,
   *   imgH0: number,
   * }}
   */
  function getOverlayGeometry() {
    const originalSize = effectiveOriginalSize();
    if (!originalSize) return null;
    const frameRect = imageFrame.getBoundingClientRect();
    if (frameRect.width <= 0 || frameRect.height <= 0) return null;

    const imgW0 = originalSize.width;
    const imgH0 = originalSize.height;
    if (!imgW0 || !imgH0) return null;

    const frameAR = frameRect.width / frameRect.height;
    const imgAR = imgW0 / imgH0;

    let drawnW;
    let drawnH;
    let offsetX;
    let offsetY;

    if (frameAR > imgAR) {
      drawnH = frameRect.height;
      drawnW = drawnH * imgAR;
      offsetX = (frameRect.width - drawnW) / 2;
      offsetY = 0;
    } else {
      drawnW = frameRect.width;
      drawnH = drawnW / imgAR;
      offsetX = 0;
      offsetY = (frameRect.height - drawnH) / 2;
    }

    drawnW *= viewerZoomScale;
    drawnH *= viewerZoomScale;

    clampViewerPan(frameRect.width, frameRect.height, drawnW, drawnH);

    offsetX = (frameRect.width - drawnW) / 2 + viewerPanX;
    offsetY = (frameRect.height - drawnH) / 2 + viewerPanY;

    const sx = drawnW / imgW0;
    const sy = drawnH / imgH0;

    return { frameRect, offsetX, offsetY, sx, sy, imgW0, imgH0 };
  }

  /** Canvas overlay (CSS px, как offsetX/offsetY у события) → координаты исходного изображения. */
  function canvasToImageCoords(geo, ox, oy) {
    const ix = (ox - geo.offsetX) / geo.sx;
    const iy = (oy - geo.offsetY) / geo.sy;
    return { ix, iy };
  }

  /** Координаты изображения → canvas overlay (CSS px). */
  function imageToCanvasCoords(geo, ix, iy) {
    return {
      ox: geo.offsetX + ix * geo.sx,
      oy: geo.offsetY + iy * geo.sy,
    };
  }

  /** Точка на canvas внутри bbox (box в координатах изображения). */
  function hitTestBox(geo, ox, oy, box) {
    const [x1, y1, x2, y2] = box;
    const p1 = imageToCanvasCoords(geo, x1, y1);
    const p2 = imageToCanvasCoords(geo, x2, y2);
    const left = Math.min(p1.ox, p2.ox);
    const right = Math.max(p1.ox, p2.ox);
    const top = Math.min(p1.oy, p2.oy);
    const bottom = Math.max(p1.oy, p2.oy);
    return ox >= left && ox <= right && oy >= top && oy <= bottom;
  }

  /**
   * Маркеры изменения размера: 4 угла и 4 середины сторон.
   * @returns {"nw"|"ne"|"se"|"sw"|"n"|"e"|"s"|"w"|null}
   */
  function hitTestHandle(geo, ox, oy, box) {
    const [x1, y1, x2, y2] = box;
    /** @type {[number, number, string][]} */
    const corners = [
      [x1, y1, "nw"],
      [x2, y1, "ne"],
      [x2, y2, "se"],
      [x1, y2, "sw"],
    ];
    for (const [ix, iy, name] of corners) {
      const p = imageToCanvasCoords(geo, ix, iy);
      if (Math.hypot(ox - p.ox, oy - p.oy) <= HANDLE_HIT_PX)
        return /** @type {"nw"|"ne"|"se"|"sw"} */ (name);
    }
    const mx = (x1 + x2) / 2;
    const my = (y1 + y2) / 2;
    /** @type {[number, number, string][]} */
    const edges = [
      [mx, y1, "n"],
      [mx, y2, "s"],
      [x2, my, "e"],
      [x1, my, "w"],
    ];
    for (const [ix, iy, name] of edges) {
      const p = imageToCanvasCoords(geo, ix, iy);
      if (Math.hypot(ox - p.ox, oy - p.oy) <= HANDLE_HIT_PX)
        return /** @type {"n"|"e"|"s"|"w"} */ (name);
    }
    return null;
  }

  /** @param {BatchImageItem} im */
  function detectionPassesUiFilters(d, im) {
    if (d.conf < confThreshold()) return false;
    if (classHidden(d.cls_name)) return false;
    const cat = d.cls_name;
    const catEnabled = im.panel.categoryState.get(cat)?.enabled ?? true;
    if (!catEnabled) return false;
    if (im.panel.detEnabled.get(d.id) === false) return false;
    return true;
  }

  /**
   * Hit-test только по bbox, проходящим порог confidence и видимость класса / группы (как на canvas).
   * При перекрытии — верхний в списке (последний в массиве).
   * @param {{ offsetX: number; offsetY: number; sx: number; sy: number }} geo
   */
  function hitTestDetectionAtOverlay(geo, ox, oy) {
    const im = currentImageItem();
    if (!im) return null;
    const list = currentDetections().filter((d) => detectionPassesUiFilters(d, im));
    for (let i = list.length - 1; i >= 0; i--) {
      const d = list[i];
      const hasSegment = Array.isArray(d.segment) && d.segment.length >= 3;
      if (hasSegment ? polygonEditor.hitTestSelection(geo, ox, oy, d) : hitTestBox(geo, ox, oy, d.box)) {
        return d;
      }
    }
    return null;
  }

  function hitTestSegMaskAtOverlay(geo, ox, oy) {
    const im = currentImageItem();
    if (!im) return null;
    const list = currentDetections().filter(
      (d) => isSegAnnotation(d) && detectionPassesUiFilters(d, im)
    );
    for (let i = list.length - 1; i >= 0; i--) {
      if (polygonEditor.hitTestSelection(geo, ox, oy, list[i])) return list[i];
    }
    return null;
  }

  function focusMaskSimplifyFromContext({ imageId, detId }) {
    const im = currentImageItem();
    const det = im?.detections.find((d) => d.id === detId);
    if (!editorModeIsEdit() || im?.id !== imageId || !isSegAnnotation(det)) return;
    pointerInteraction = null;
    polygonEditor.cancel();
    batchState.settings.editorTool = "select";
    setSelectedDetectionId(detId);
    syncEditorChrome();
    buildRightPanel();
    draw();
    window.requestAnimationFrame(() => {
      const block = inspectorRoot.querySelector(
        `.inspector-simplify[data-det-id="${detId}"]`
      );
      if (!(block instanceof HTMLElement)) return;
      block.classList.add("is-context-target");
      block.scrollIntoView({ block: "nearest", behavior: "smooth" });
      block.querySelector(".inspector-simplify-slider")?.focus();
      window.setTimeout(() => block.classList.remove("is-context-target"), 1200);
    });
  }

  function onOverlayContextMenu(event) {
    const geo = getOverlayGeometry();
    if (!geo || previewImage.style.display === "none") return;
    const rect = overlay.getBoundingClientRect();
    const hit = hitTestSegMaskAtOverlay(
      geo,
      event.clientX - rect.left,
      event.clientY - rect.top
    );
    maskMenu.close();
    if (!hit) return;
    event.preventDefault();
    event.stopPropagation();
    setSelectedDetectionId(hit.id);
    buildRightPanel();
    draw();
    const im = currentImageItem();
    if (!im) return;
    maskMenu.open({
      clientX: event.clientX,
      clientY: event.clientY,
      context: { imageId: im.id, detId: hit.id },
    });
  }

  /**
   * Сброс выделения, если bbox больше не проходит UI-фильтры.
   * @returns {boolean} true, если выделение сброшено
   */
  function reconcileSelectedDetectionWithFilters(im) {
    const sid = im.panel.selectedDetectionId;
    if (sid == null) return false;
    const d = im.detections.find((x) => x.id === sid);
    if (!d || !detectionPassesUiFilters(d, im)) {
      im.panel.selectedDetectionId = null;
      touchBatch();
      return true;
    }
    return false;
  }

  /** @param {number|null} id */
  function setSelectedDetectionId(id) {
    const im = currentImageItem();
    if (!im) return;
    im.panel.selectedDetectionId = id;
  }

  function allocateNextDetId(im) {
    return im.detections.reduce((m, d) => Math.max(m, d.id), -1) + 1;
  }

  /** Обрезка bbox по границам изображения (координаты изображения). */
  function clampBoxToImage(imgW, imgH, box) {
    let [x1, y1, x2, y2] = box;
    x1 = clamp(x1, 0, imgW);
    x2 = clamp(x2, 0, imgW);
    y1 = clamp(y1, 0, imgH);
    y2 = clamp(y2, 0, imgH);
    if (Math.abs(x2 - x1) < 1e-3) x2 = clamp(x1 + 1, 0, imgW);
    if (Math.abs(y2 - y1) < 1e-3) y2 = clamp(y1 + 1, 0, imgH);
    return [
      Math.min(x1, x2),
      Math.min(y1, y2),
      Math.max(x1, x2),
      Math.max(y1, y2),
    ];
  }

  /** Закреплённые рёбра при resize по маркеру (противоположная сторона не «ездит» при min size). */
  function getResizeFixedEdges(handle) {
    switch (handle) {
      case "nw":
        return { x: "x2", y: "y2" };
      case "ne":
        return { x: "x1", y: "y2" };
      case "se":
        return { x: "x1", y: "y1" };
      case "sw":
        return { x: "x2", y: "y1" };
      case "w":
        return { x: "x2" };
      case "e":
        return { x: "x1" };
      case "n":
        return { y: "y2" };
      case "s":
        return { y: "y1" };
      default:
        return {};
    }
  }

  /**
   * Гарантирует ширину/высоту ≥ minSide с учётом того, какой маркер тянули.
   * @param {[number,number,number,number]} box
   */
  function enforceMinBoxDimensions(box, imgW, imgH, minSide, handle) {
    let x1 = Math.min(box[0], box[2]);
    let y1 = Math.min(box[1], box[3]);
    let x2 = Math.max(box[0], box[2]);
    let y2 = Math.max(box[1], box[3]);
    const fe = getResizeFixedEdges(handle);

    if (x2 - x1 < minSide) {
      if (fe.x === "x2") {
        x1 = x2 - minSide;
        if (x1 < 0) {
          x1 = 0;
          x2 = minSide;
        }
      } else if (fe.x === "x1") {
        x2 = x1 + minSide;
        if (x2 > imgW) {
          x2 = imgW;
          x1 = x2 - minSide;
        }
      } else {
        const cx = (x1 + x2) / 2;
        x1 = clamp(cx - minSide / 2, 0, imgW - minSide);
        x2 = x1 + minSide;
      }
    }

    if (y2 - y1 < minSide) {
      if (fe.y === "y2") {
        y1 = y2 - minSide;
        if (y1 < 0) {
          y1 = 0;
          y2 = minSide;
        }
      } else if (fe.y === "y1") {
        y2 = y1 + minSide;
        if (y2 > imgH) {
          y2 = imgH;
          y1 = y2 - minSide;
        }
      } else {
        const cy = (y1 + y2) / 2;
        y1 = clamp(cy - minSide / 2, 0, imgH - minSide);
        y2 = y1 + minSide;
      }
    }

    x1 = clamp(x1, 0, imgW);
    x2 = clamp(x2, 0, imgW);
    y1 = clamp(y1, 0, imgH);
    y2 = clamp(y2, 0, imgH);
    return [
      Math.min(x1, x2),
      Math.min(y1, y2),
      Math.max(x1, x2),
      Math.max(y1, y2),
    ];
  }

  /**
   * Новый bbox после drag маркера resize (координаты изображения).
   * @param {[number,number,number,number]} ob
   * @param {"nw"|"ne"|"se"|"sw"|"n"|"e"|"s"|"w"} handle
   */
  function computeResizedBox(ob, handle, imx, imy, startIx, startIy, imgW, imgH) {
    const dx = imx - startIx;
    const dy = imy - startIy;
    let x1 = ob[0];
    let y1 = ob[1];
    let x2 = ob[2];
    let y2 = ob[3];
    switch (handle) {
      case "nw":
        x1 += dx;
        y1 += dy;
        break;
      case "ne":
        x2 += dx;
        y1 += dy;
        break;
      case "se":
        x2 += dx;
        y2 += dy;
        break;
      case "sw":
        x1 += dx;
        y2 += dy;
        break;
      case "n":
        y1 += dy;
        break;
      case "s":
        y2 += dy;
        break;
      case "e":
        x2 += dx;
        break;
      case "w":
        x1 += dx;
        break;
      default:
        break;
    }
    const clamped = clampBoxToImage(imgW, imgH, [x1, y1, x2, y2]);
    return enforceMinBoxDimensions(clamped, imgW, imgH, MIN_BOX_SIDE, handle);
  }

  /** После move/resize геометрии на canvas: human + edited + reviewed + autosave-хук. */
  function finalizeBBoxGeometryEdit(im, det) {
    det.annotation_type = "detect";
    det.source = "human";
    im.edited = true;
    im.reviewed = false;
    markAnnotationTypeReviewed(im, "detect");
    touchBatch();
    scheduleWorkspaceAutosave(0);
    updateBatchNavUi();
    buildRightPanel();
  }

  /** @param {[number,number,number,number]} ob @param {number} dx @param {number} dy */
  function translateBoxWithinImage(ob, dx, dy, imgW, imgH) {
    const w = ob[2] - ob[0];
    const h = ob[3] - ob[1];
    let x1 = ob[0] + dx;
    let y1 = ob[1] + dy;
    x1 = clamp(x1, 0, Math.max(0, imgW - w));
    y1 = clamp(y1, 0, Math.max(0, imgH - h));
    return [x1, y1, x1 + w, y1 + h];
  }

  /**
   * Класс для новой фигуры: у выделенного объекта или первый класс текущего режима.
   * @param {BatchImageItem} im
   */
  function getTrainClassForNewBbox(im) {
    const type = batchState.settings.editorTool === "addPolygon" ? "seg" : "detect";
    const classes = classesForAnnotationType(type);
    if (hotkeyPreferredNewBboxClassName) {
      const hit = classes.find((t) => t.name === hotkeyPreferredNewBboxClassName);
      if (hit) return hit;
    }
    const sid = im.panel.selectedDetectionId;
    if (sid != null) {
      const d = im.detections.find((x) => x.id === sid);
      if (d && annotationTypeOf(d) === type) {
        const byName = classes.find((t) => t.name === d.cls_name);
        if (byName) return byName;
        const byId = classes.find((t) => t.id === d.cls_id);
        if (byId) return byId;
      }
    }
    return classes[0];
  }

  function tryDeleteSelectedDetection() {
    if (!editorModeIsEdit()) return false;
    const im = currentImageItem();
    if (!im) return false;
    const sid = im.panel.selectedDetectionId;
    if (sid == null) return false;
    const det = im.detections.find((d) => d.id === sid);
    if (!det) return false;
    pushUndoCheckpoint();
    const before = im.detections.length;
    im.detections = im.detections.filter((d) => d.id !== sid);
    if (im.detections.length === before) return false;

    im.panel.selectedDetectionId = null;
    im.edited = true;
    im.reviewed = false;
    markAnnotationTypeReviewed(im, annotationTypeOf(det));
    if (im.detections.length === 0) {
      im.status = /** @type {ImageStatus} */ ("empty");
    }

    touchBatch();
    scheduleWorkspaceAutosave(0);
    updateBatchNavUi();
    buildRightPanel();
    draw();
    return true;
  }

  function finalizeAddBoxInteraction() {
    const pi = pointerInteraction;
    if (!pi || pi.kind !== "addBox") return;
    const geo = getOverlayGeometry();
    if (!geo) return;
    const ix0 = pi.ix0 ?? 0;
    const iy0 = pi.iy0 ?? 0;
    const ix1 = pi.ix1 ?? ix0;
    const iy1 = pi.iy1 ?? iy0;
    let x1 = Math.min(ix0, ix1);
    let x2 = Math.max(ix0, ix1);
    let y1 = Math.min(iy0, iy1);
    let y2 = Math.max(iy0, iy1);
    if (x2 - x1 < MIN_BOX_SIDE || y2 - y1 < MIN_BOX_SIDE) return;

    const im = currentImageItem();
    if (!im) return;

    const box = clampBoxToImage(geo.imgW0, geo.imgH0, [x1, y1, x2, y2]);
    const bw = box[2] - box[0];
    const bh = box[3] - box[1];
    if (bw < MIN_BOX_SIDE || bh < MIN_BOX_SIDE) return;

    pushUndoCheckpoint();

    const tc = getTrainClassForNewBbox(im);
    const newDet = {
      id: allocateNextDetId(im),
      cls_id: tc.id,
      cls_name: tc.name,
      conf: 1,
      box,
      source: /** @type {"human"} */ ("human"),
      annotation_type: "detect",
    };
    im.detections.push(newDet);
    im.panel.selectedDetectionId = newDet.id;
    im.status = /** @type {ImageStatus} */ ("detected");
    im.edited = true;
    im.reviewed = false;
    /* После добавления возвращаем Select: проще сразу двигать/менять класс нового bbox без второго случайного drag. */
    markAnnotationTypeReviewed(im, "detect");
    batchState.settings.editorTool = "select";

    touchBatch();
    scheduleWorkspaceAutosave(0);
    syncEditorChrome();
    updateBatchNavUi();
    buildRightPanel();
  }

  /**
   * Отсекает полигон по прямоугольнику crop и переносит его начало в (0, 0).
   * @param {Array<[number, number]>} segment
   * @param {[number, number, number, number]} crop
   */
  function clipPolygonToCrop(segment, crop) {
    const [x1, y1, x2, y2] = crop;
    let points = segment
      .map((p) => [Number(p[0]), Number(p[1])])
      .filter((p) => Number.isFinite(p[0]) && Number.isFinite(p[1]));

    const clipEdge = (input, inside, intersect) => {
      if (!input.length) return [];
      const output = [];
      let previous = input[input.length - 1];
      let previousInside = inside(previous);
      for (const current of input) {
        const currentInside = inside(current);
        if (currentInside) {
          if (!previousInside) output.push(intersect(previous, current));
          output.push(current);
        } else if (previousInside) {
          output.push(intersect(previous, current));
        }
        previous = current;
        previousInside = currentInside;
      }
      return output;
    };

    const verticalIntersection = (edgeX, a, b) => {
      const dx = b[0] - a[0];
      const t = Math.abs(dx) < 1e-9 ? 0 : (edgeX - a[0]) / dx;
      return [edgeX, a[1] + (b[1] - a[1]) * t];
    };
    const horizontalIntersection = (edgeY, a, b) => {
      const dy = b[1] - a[1];
      const t = Math.abs(dy) < 1e-9 ? 0 : (edgeY - a[1]) / dy;
      return [a[0] + (b[0] - a[0]) * t, edgeY];
    };

    points = clipEdge(points, (p) => p[0] >= x1, (a, b) =>
      verticalIntersection(x1, a, b)
    );
    points = clipEdge(points, (p) => p[0] <= x2, (a, b) =>
      verticalIntersection(x2, a, b)
    );
    points = clipEdge(points, (p) => p[1] >= y1, (a, b) =>
      horizontalIntersection(y1, a, b)
    );
    points = clipEdge(points, (p) => p[1] <= y2, (a, b) =>
      horizontalIntersection(y2, a, b)
    );

    const translated = [];
    for (const [x, y] of points) {
      const point = [
        clamp(x - x1, 0, x2 - x1),
        clamp(y - y1, 0, y2 - y1),
      ];
      const previous = translated[translated.length - 1];
      if (
        !previous ||
        Math.abs(previous[0] - point[0]) > 1e-6 ||
        Math.abs(previous[1] - point[1]) > 1e-6
      ) {
        translated.push(point);
      }
    }
    if (translated.length > 1) {
      const first = translated[0];
      const last = translated[translated.length - 1];
      if (
        Math.abs(first[0] - last[0]) <= 1e-6 &&
        Math.abs(first[1] - last[1]) <= 1e-6
      ) {
        translated.pop();
      }
    }
    return translated;
  }

  /** @param {any} detection @param {[number, number, number, number]} crop */
  function cropDetectionToRect(detection, crop) {
    const [x1, y1, x2, y2] = crop;
    if (Array.isArray(detection.segment) && detection.segment.length >= 3) {
      const segment = clipPolygonToCrop(detection.segment, crop);
      if (segment.length < 3) return null;
      const xs = segment.map((p) => p[0]);
      const ys = segment.map((p) => p[1]);
      const minX = Math.min(...xs);
      const minY = Math.min(...ys);
      const maxX = Math.max(...xs);
      const maxY = Math.max(...ys);
      if (!(maxX > minX && maxY > minY)) return null;
      return {
        ...detection,
        box: [minX, minY, maxX, maxY],
        segment,
        source: "human",
      };
    }

    const bx1 = Math.max(Number(detection.box?.[0]) || 0, x1);
    const by1 = Math.max(Number(detection.box?.[1]) || 0, y1);
    const bx2 = Math.min(Number(detection.box?.[2]) || 0, x2);
    const by2 = Math.min(Number(detection.box?.[3]) || 0, y2);
    if (!(bx2 > bx1 && by2 > by1)) return null;
    return {
      ...detection,
      box: [bx1 - x1, by1 - y1, bx2 - x1, by2 - y1],
      source: "human",
    };
  }

  /** @param {NonNullable<typeof pointerInteraction>} interaction */
  async function finalizeCropInteraction(interaction) {
    if (interaction.kind !== "crop") return;
    const geo = getOverlayGeometry();
    const im = currentImageItem();
    if (!geo || !im) return;

    const maxW = Math.max(1, Math.round(geo.imgW0));
    const maxH = Math.max(1, Math.round(geo.imgH0));
    const ix0 = interaction.ix0 ?? 0;
    const iy0 = interaction.iy0 ?? 0;
    const ix1 = interaction.ix1 ?? ix0;
    const iy1 = interaction.iy1 ?? iy0;
    const crop = /** @type {[number, number, number, number]} */ ([
      clamp(Math.floor(Math.min(ix0, ix1)), 0, maxW),
      clamp(Math.floor(Math.min(iy0, iy1)), 0, maxH),
      clamp(Math.ceil(Math.max(ix0, ix1)), 0, maxW),
      clamp(Math.ceil(Math.max(iy0, iy1)), 0, maxH),
    ]);
    const width = crop[2] - crop[0];
    const height = crop[3] - crop[1];
    if (width < MIN_BOX_SIDE || height < MIN_BOX_SIDE) {
      showToast("Область обрезки слишком маленькая.", {
        type: "warning",
        durationMs: 2800,
      });
      return;
    }

    const originalBlob = im.blob;
    const originalCount = im.detections.length;
    const originalTypes = new Set(im.detections.map(annotationTypeOf));
    try {
      const croppedBlob = await cropImageBlobToPng(originalBlob, crop);
      if (currentImageItem()?.id !== im.id || im.blob !== originalBlob) {
        showToast("Обрезка отменена: активный кадр изменился.", {
          type: "info",
          durationMs: 2800,
        });
        return;
      }

      pushUndoCheckpoint();
      const detections = im.detections
        .map((d) => cropDetectionToRect(d, crop))
        .filter(Boolean);
      const keptIds = new Set(detections.map((d) => d.id));

      if (im.objectUrl) URL.revokeObjectURL(im.objectUrl);
      im.blob = croppedBlob;
      im.objectUrl = URL.createObjectURL(croppedBlob);
      im.fileType = "image/png";
      im.fileSize = croppedBlob.size;
      im.width = width;
      im.height = height;
      im.detections = detections;
      im.panel.detEnabled = new Map(
        Array.from(im.panel.detEnabled.entries()).filter(([id]) => keptIds.has(id))
      );
      for (const d of detections) {
        if (!im.panel.detEnabled.has(d.id)) im.panel.detEnabled.set(d.id, true);
      }
      if (
        im.panel.selectedDetectionId != null &&
        !keptIds.has(im.panel.selectedDetectionId)
      ) {
        im.panel.selectedDetectionId = null;
      }
      im.status = detections.length ? "detected" : "empty";
      im.reviewed = false;
      im.edited = true;
      im.modelStates = normalizeModelStates(im.modelStates);
      const cropUpdatedAt = new Date().toISOString();
      for (const type of ["detect", "seg"]) {
        const state = im.modelStates[type];
        state.status = originalTypes.has(type) ? "reviewed" : "not_run";
        state.error = null;
        state.updatedAt = cropUpdatedAt;
        state.revision = (Number(state.revision) || 0) + 1;
      }
      syncLegacyImageState(im, { preserveSkipped: false });
      im.reviewed = false;
      im.error = null;

      batchState.settings.editorTool = "select";
      clearAddBoxCrosshairOverlayPx();
      resetViewerZoom();
      touchBatch();
      scheduleWorkspaceAutosave(0);
      updateBatchNavUi();
      showCurrentPreview();

      const removed = originalCount - detections.length;
      showToast(
        `Изображение обрезано до ${width} × ${height}${removed > 0 ? `; удалено объектов: ${removed}` : ""}.`,
        { type: "success", durationMs: 3600 }
      );
    } catch (err) {
      console.warn("[crop image]", err);
      showToast(
        `Не удалось обрезать изображение: ${err?.message || String(err)}`,
        { type: "error", durationMs: 5200 }
      );
    }
  }

  /** @param {number|null} ox @param {number|null} oy @param {boolean} [altKey] */
  function refreshOverlayCursorHint(ox, oy, altKey = false) {
    overlay.classList.remove(
      "overlay-cursor-crosshair",
      "overlay-cursor-move",
      "overlay-cursor-nwse",
      "overlay-cursor-nesw",
      "overlay-cursor-ns",
      "overlay-cursor-ew",
      "overlay-cursor-vertex",
      "overlay-cursor-remove-vertex",
      "overlay-cursor-insert-vertex"
    );
    if (!overlay.classList.contains("overlay-pointer-on")) return;

    if (editorModeIsReview()) {
      if (ox == null || oy == null) {
        overlay.style.cursor = "";
        return;
      }
      const geo = getOverlayGeometry();
      if (!geo) return;
      overlay.style.cursor = hitTestDetectionAtOverlay(geo, ox, oy) ? "pointer" : "";
      return;
    }

    overlay.style.cursor = "";

    if (
      batchState.settings.editorTool === "addBox" ||
      batchState.settings.editorTool === "crop"
    ) {
      overlay.classList.add("overlay-cursor-crosshair");
      return;
    }

    if (batchState.settings.editorTool === "addPolygon") {
      overlay.classList.add("overlay-cursor-crosshair");
      return;
    }

    if (ox == null || oy == null) return;
    const geo = getOverlayGeometry();
    if (!geo) return;
    const im = currentImageItem();
    if (!im) return;
    const sid = im.panel.selectedDetectionId;
    const det =
      sid != null ? currentDetections().find((d) => d.id === sid) : null;
    if (!det || !detectionPassesUiFilters(det, im)) return;

    if (Array.isArray(det.segment) && det.segment.length >= 3) {
      const cls = polygonEditor.cursorClassFor(geo, ox, oy, altKey);
      if (cls) overlay.classList.add(cls);
      return;
    }

    const h = hitTestHandle(geo, ox, oy, det.box);
    if (h === "nw" || h === "se") overlay.classList.add("overlay-cursor-nwse");
    else if (h === "ne" || h === "sw") overlay.classList.add("overlay-cursor-nesw");
    else if (h === "n" || h === "s") overlay.classList.add("overlay-cursor-ns");
    else if (h === "e" || h === "w") overlay.classList.add("overlay-cursor-ew");
    else if (hitTestBox(geo, ox, oy, det.box))
      overlay.classList.add("overlay-cursor-move");
  }

  function syncEditorChrome() {
    const review = editorModeIsReview();
    if (
      review ||
      !["addBox", "crop"].includes(batchState.settings.editorTool)
    ) {
      clearAddBoxCrosshairOverlayPx();
    }
    editorModeReviewBtn.classList.toggle("is-active", review);
    editorModeEditBtn.classList.toggle("is-active", !review);
    editorModeReviewBtn.setAttribute("aria-checked", review ? "true" : "false");
    editorModeEditBtn.setAttribute("aria-checked", !review ? "true" : "false");
    editorToolsBar.hidden = review;

    editorToolAddBtn.hidden = false;
    editorToolAddPolygonBtn.hidden = false;

    const selTool = batchState.settings.editorTool === "select";
    editorToolSelectBtn.classList.toggle("is-active", selTool);
    editorToolAddBtn.classList.toggle("is-active", batchState.settings.editorTool === "addBox");
    editorToolCropBtn.classList.toggle(
      "is-active",
      batchState.settings.editorTool === "crop"
    );
    editorToolAddPolygonBtn.classList.toggle(
      "is-active",
      batchState.settings.editorTool === "addPolygon"
    );

    const showOverlay =
      !!previewImage.src &&
      previewImage.style.display !== "none" &&
      !!effectiveOriginalSize();
    overlay.classList.toggle("overlay-pointer-on", showOverlay);

    refreshOverlayCursorHint(null, null);
  }

  function getColorByClass(clsId) {
    return COLORS[Math.abs(clsId) % COLORS.length];
  }

  function draw() {
    resizeCanvasToFrame();
    const geo = getOverlayGeometry();
    if (!geo) return;
    applyPreviewImageTransform();
    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    const {
      frameRect,
      offsetX,
      offsetY,
      sx,
      sy,
      imgW0,
      imgH0,
    } = geo;

    ctx.lineWidth = 2;
    ctx.font = "12px system-ui, sans-serif";

    const im = currentImageItem();
    if (!im) return;

    if (reconcileSelectedDetectionWithFilters(im)) {
      buildRightPanel();
    }

    const categoryState = im.panel.categoryState;
    const detEnabled = im.panel.detEnabled;
    const detections = currentDetections();
    const selId = im.panel.selectedDetectionId;

    const drawOneMask = (
      /** @type {BatchImageItem["detections"][0]} */ d,
      /** @type {boolean} */ isSel
    ) => {
      if (!Array.isArray(d.segment) || d.segment.length < 3) return;
      const color = getColorByClass(d.cls_id);
      ctx.save();
      ctx.beginPath();
      d.segment.forEach(([px, py], i) => {
        const x = offsetX + px * sx;
        const y = offsetY + py * sy;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.closePath();
      const selectedForEdit =
        isSel &&
        editorModeIsEdit() &&
        batchState.settings.editorTool === "select";
      ctx.globalAlpha = 1;
      ctx.fillStyle = selectedForEdit
        ? "rgba(255,255,255,0.30)"
        : color;
      if (!selectedForEdit) ctx.globalAlpha = isSel ? 0.38 : 0.24;
      ctx.fill();
      ctx.globalAlpha = isSel ? 0.95 : 0.65;
      ctx.lineWidth = isSel ? 3 : 1.5;
      ctx.strokeStyle = selectedForEdit ? "rgba(15,15,15,0.95)" : color;
      ctx.stroke();
      ctx.restore();
    };

    const drawOneBox = (
      /** @type {BatchImageItem["detections"][0]} */ d,
      /** @type {boolean} */ isSel
    ) => {
      const [x1, y1, x2, y2] = d.box;
      const x = offsetX + x1 * sx;
      const y = offsetY + y1 * sy;
      const w = (x2 - x1) * sx;
      const h = (y2 - y1) * sy;

      const color = getColorByClass(d.cls_id);
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      if (
        isSel &&
        editorModeIsEdit() &&
        batchState.settings.editorTool === "select" &&
        (!Array.isArray(d.segment) || d.segment.length < 3)
      ) {
        ctx.fillStyle = "rgba(255,255,255,0.30)";
        ctx.fillRect(x, y, w, h);
        ctx.fillStyle = color;
      }
      if (isSel) {
        ctx.strokeStyle = "rgba(255,255,255,0.92)";
        ctx.lineWidth = 6;
        ctx.strokeRect(x, y, w, h);
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);
      } else {
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);
      }
      ctx.lineWidth = 2;

      const label = `${d.cls_name} #${d.id + 1} (${fmtConf(d.conf)})`;
      const padX = 6;
      const padY = 4;
      const textW = ctx.measureText(label).width;
      const boxW = textW + padX * 2;
      const boxH = 18;
      const bx = Math.max(0, Math.min(x, frameRect.width - boxW));
      const by = Math.max(0, y - boxH - 2);
      ctx.globalAlpha = 0.85;
      ctx.fillRect(bx, by, boxW, boxH);
      ctx.globalAlpha = 1;
      ctx.fillStyle = "#0b0b0b";
      ctx.fillText(label, bx + padX, by + 13);

      if (
        editorModeIsEdit() &&
        batchState.settings.editorTool === "select" &&
        isSel
      ) {
        const hs = 8;
        const xm = x + w / 2;
        const ym = y + h / 2;
        ctx.strokeStyle = "#ffffff";
        ctx.fillStyle = "rgba(255,255,255,0.95)";
        /** Углы и середины сторон (canvas px). */
        const pts = [
          [x, y],
          [xm, y],
          [x + w, y],
          [x + w, ym],
          [x + w, y + h],
          [xm, y + h],
          [x, y + h],
          [x, ym],
        ];
        for (const [cx, cy] of pts) {
          ctx.fillRect(cx - hs / 2, cy - hs / 2, hs, hs);
          ctx.strokeRect(cx - hs / 2, cy - hs / 2, hs, hs);
        }
      }
    };

    for (const d of detections) {
      const isSel = selId === d.id;
      if (!isSel && !detectionPassesUiFilters(d, im)) continue;
      if (!isSegAnnotation(d)) continue;
      drawOneMask(d, isSel);
    }

    for (const d of detections) {
      const isSel = selId === d.id;
      if (!isSel && !detectionPassesUiFilters(d, im)) continue;
      if (!isDetectAnnotation(d)) continue;
      drawOneBox(d, isSel);
    }

    if (
      editorModeIsEdit() &&
      ["addBox", "crop"].includes(batchState.settings.editorTool) &&
      addBoxCrosshairOverlayPx
    ) {
      const { ox, oy } = addBoxCrosshairOverlayPx;
      const imgLeft = offsetX;
      const imgTop = offsetY;
      const imgRight = offsetX + imgW0 * sx;
      const imgBottom = offsetY + imgH0 * sy;

      ctx.save();
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.5)";
      ctx.beginPath();
      ctx.moveTo(imgLeft, oy);
      ctx.lineTo(imgRight, oy);
      ctx.moveTo(ox, imgTop);
      ctx.lineTo(ox, imgBottom);
      ctx.stroke();
      ctx.strokeStyle = "rgba(63, 130, 247, 0.75)";
      ctx.lineDashOffset = 5;
      ctx.beginPath();
      ctx.moveTo(imgLeft, oy);
      ctx.lineTo(imgRight, oy);
      ctx.moveTo(ox, imgTop);
      ctx.lineTo(ox, imgBottom);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.lineDashOffset = 0;
      ctx.restore();
    }

    if (
      pointerInteraction &&
      (pointerInteraction.kind === "addBox" ||
        pointerInteraction.kind === "crop") &&
      pointerInteraction.ix0 != null
    ) {
      const ix0 = pointerInteraction.ix0;
      const iy0 = pointerInteraction.iy0 ?? 0;
      const ix1 = pointerInteraction.ix1 ?? ix0;
      const iy1 = pointerInteraction.iy1 ?? iy0;
      const rx1 = Math.min(ix0, ix1);
      const ry1 = Math.min(iy0, iy1);
      const rx2 = Math.max(ix0, ix1);
      const ry2 = Math.max(iy0, iy1);
      const x = offsetX + rx1 * sx;
      const y = offsetY + ry1 * sy;
      const w = (rx2 - rx1) * sx;
      const h = (ry2 - ry1) * sy;
      const isCropSelection = pointerInteraction.kind === "crop";

      ctx.save();
      if (isCropSelection) {
        ctx.fillStyle = "rgba(0, 0, 0, 0.48)";
        ctx.beginPath();
        ctx.rect(offsetX, offsetY, imgW0 * sx, imgH0 * sy);
        ctx.rect(x, y, Math.max(w, 1), Math.max(h, 1));
        ctx.fill("evenodd");
      } else {
        ctx.fillStyle = "rgba(255,255,255,0.30)";
        ctx.fillRect(x, y, Math.max(w, 1), Math.max(h, 1));
      }
      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = isCropSelection
        ? "rgba(245, 158, 11, 0.98)"
        : "rgba(63, 130, 247, 0.95)";
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, Math.max(w, 1), Math.max(h, 1));
      ctx.restore();
    }

    polygonEditor.renderOverlay(ctx, geo);

    updateAddBoxInspectorZoom();
  }

  function onOverlayPointerDown(e) {
    if (e.button !== 0) return;
    if (spacePanPressed && e.button === 0) {
      panDragState = {
        startClientX: e.clientX,
        startClientY: e.clientY,
        startPanX: viewerPanX,
        startPanY: viewerPanY,
      };
      syncPanCursorUi();
      e.preventDefault();
      return;
    }
    if (!effectiveOriginalSize() || previewImage.style.display === "none")
      return;

    const geo = getOverlayGeometry();
    if (!geo) return;

    const ox = e.offsetX;
    const oy = e.offsetY;

    if (editorModeIsReview()) {
      const hit = hitTestDetectionAtOverlay(geo, ox, oy);
      setSelectedDetectionId(hit ? hit.id : null);
      touchBatch();
      buildRightPanel();
      draw();
      return;
    }

    if (
      batchState.settings.editorTool === "addBox" ||
      batchState.settings.editorTool === "crop"
    ) {
      const { ix, iy } = canvasToImageCoords(geo, ox, oy);
      pointerInteraction = {
        kind: batchState.settings.editorTool,
        ix0: clamp(ix, 0, geo.imgW0),
        iy0: clamp(iy, 0, geo.imgH0),
        ix1: clamp(ix, 0, geo.imgW0),
        iy1: clamp(iy, 0, geo.imgH0),
        startIx: ix,
        startIy: iy,
      };
      syncAddBoxCrosshairOverlayPx(ox, oy);
      e.preventDefault();
      draw();
      return;
    }

    if (batchState.settings.editorTool === "addPolygon") {
      polygonEditor.handlePointerDown(e, ox, oy);
      e.preventDefault();
      draw();
      return;
    }

    const imCur = currentImageItem();
    if (!imCur) return;

    const sid = imCur.panel.selectedDetectionId;
    const selDet =
      sid != null ? currentDetections().find((d) => d.id === sid) : null;

    if (selDet && detectionPassesUiFilters(selDet, imCur)) {
      const selDetHasSegment = Array.isArray(selDet.segment) && selDet.segment.length >= 3;

      if (selDetHasSegment) {
        if (polygonEditor.handlePointerDown(e, ox, oy)) {
          e.preventDefault();
          draw();
          return;
        }
      } else {
        const handle = hitTestHandle(geo, ox, oy, selDet.box);
        if (handle) {
          const { ix, iy } = canvasToImageCoords(geo, ox, oy);
          pointerInteraction = {
            kind: "resize",
            handle,
            detId: selDet.id,
            startIx: ix,
            startIy: iy,
            origBox: /** @type {[number,number,number,number]} */ ([
              selDet.box[0],
              selDet.box[1],
              selDet.box[2],
              selDet.box[3],
            ]),
          };
          pushUndoCheckpoint();
          selDet.annotation_type = "detect";
          selDet.source = "human";
          markAnnotationTypeReviewed(imCur, "detect");
          e.preventDefault();
          draw();
          return;
        }
        if (hitTestBox(geo, ox, oy, selDet.box)) {
          const { ix, iy } = canvasToImageCoords(geo, ox, oy);
          pointerInteraction = {
            kind: "move",
            detId: selDet.id,
            startIx: ix,
            startIy: iy,
            origBox: /** @type {[number,number,number,number]} */ ([
              selDet.box[0],
              selDet.box[1],
              selDet.box[2],
              selDet.box[3],
            ]),
          };
          pushUndoCheckpoint();
          selDet.annotation_type = "detect";
          selDet.source = "human";
          markAnnotationTypeReviewed(imCur, "detect");
          e.preventDefault();
          draw();
          return;
        }
      }
    }

    const hit = hitTestDetectionAtOverlay(geo, ox, oy);
    setSelectedDetectionId(hit ? hit.id : null);
    touchBatch();
    buildRightPanel();
    draw();
  }

  /** @param {WheelEvent} e */
  function onImageFrameWheel(e) {
    if (!effectiveOriginalSize() || previewImage.style.display === "none") return;
    if (viewerMode === "video") return;

    const geoBefore = getOverlayGeometry();
    if (!geoBefore) return;

    const frameRect = imageFrame.getBoundingClientRect();
    const ox = e.clientX - frameRect.left;
    const oy = e.clientY - frameRect.top;
    const { ix, iy } = canvasToImageCoords(geoBefore, ox, oy);

    const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    const nextZoom = clamp(
      viewerZoomScale * factor,
      VIEWER_ZOOM_MIN,
      VIEWER_ZOOM_MAX
    );
    if (Math.abs(nextZoom - viewerZoomScale) < 1e-6) return;

    viewerZoomScale = nextZoom;
    if (viewerZoomScale <= VIEWER_ZOOM_MIN + 1e-6) {
      resetViewerZoom();
      draw();
      e.preventDefault();
      return;
    }

    const frameAR = frameRect.width / frameRect.height;
    const imgAR = geoBefore.imgW0 / geoBefore.imgH0;
    let baseW;
    let baseH;
    if (frameAR > imgAR) {
      baseH = frameRect.height;
      baseW = baseH * imgAR;
    } else {
      baseW = frameRect.width;
      baseH = baseW / imgAR;
    }
    const drawnW = baseW * viewerZoomScale;
    const drawnH = baseH * viewerZoomScale;
    const sx = drawnW / geoBefore.imgW0;
    const sy = drawnH / geoBefore.imgH0;

    viewerPanX = ox - ((frameRect.width - drawnW) / 2 + ix * sx);
    viewerPanY = oy - ((frameRect.height - drawnH) / 2 + iy * sy);
    clampViewerPan(frameRect.width, frameRect.height, drawnW, drawnH);

    draw();
    e.preventDefault();
  }

  function onDocumentPointerMove(e) {
    if (panDragState) {
      viewerPanX = panDragState.startPanX + (e.clientX - panDragState.startClientX);
      viewerPanY = panDragState.startPanY + (e.clientY - panDragState.startClientY);
      draw();
      return;
    }
    if (polygonEditor.isInteracting()) {
      const r = overlay.getBoundingClientRect();
      polygonEditor.handlePointerMove(e, e.clientX - r.left, e.clientY - r.top);
      return;
    }
    if (!pointerInteraction) return;
    const geo = getOverlayGeometry();
    if (!geo) return;

    const r = overlay.getBoundingClientRect();
    const ox = e.clientX - r.left;
    const oy = e.clientY - r.top;
    const { ix, iy } = canvasToImageCoords(geo, ox, oy);
    const imx = clamp(ix, 0, geo.imgW0);
    const imy = clamp(iy, 0, geo.imgH0);

    const im = currentImageItem();
    if (!im) return;

    if (
      pointerInteraction.kind === "addBox" ||
      pointerInteraction.kind === "crop"
    ) {
      syncAddBoxCrosshairOverlayPx(ox, oy);
      pointerInteraction.ix1 = imx;
      pointerInteraction.iy1 = imy;
      draw();
      return;
    }

    const det = im.detections.find((d) => d.id === pointerInteraction.detId);
    if (!det || !pointerInteraction.origBox) return;

    const ob = pointerInteraction.origBox;

    if (pointerInteraction.kind === "move") {
      const dx = imx - pointerInteraction.startIx;
      const dy = imy - pointerInteraction.startIy;
      det.box = clampBoxToImage(
        geo.imgW0,
        geo.imgH0,
        translateBoxWithinImage(ob, dx, dy, geo.imgW0, geo.imgH0)
      );
      draw();
      return;
    }

    if (pointerInteraction.kind === "resize" && pointerInteraction.handle) {
      det.box = computeResizedBox(
        ob,
        /** @type {"nw"|"ne"|"se"|"sw"|"n"|"e"|"s"|"w"} */ (
          pointerInteraction.handle
        ),
        imx,
        imy,
        pointerInteraction.startIx,
        pointerInteraction.startIy,
        geo.imgW0,
        geo.imgH0
      );
      draw();
    }
  }

  function onDocumentPointerUp() {
    if (panDragState) {
      panDragState = null;
      syncPanCursorUi();
      return;
    }
    if (polygonEditor.isInteracting()) {
      polygonEditor.handlePointerUp();
      return;
    }
    if (!pointerInteraction) return;

    if (pointerInteraction.kind === "addBox") {
      finalizeAddBoxInteraction();
    } else if (pointerInteraction.kind === "crop") {
      const cropInteraction = pointerInteraction;
      pointerInteraction = null;
      clearAddBoxCrosshairOverlayPx();
      draw();
      void finalizeCropInteraction(cropInteraction);
      return;
    } else if (
      pointerInteraction.kind === "move" ||
      pointerInteraction.kind === "resize"
    ) {
      const imUp = currentImageItem();
      const detUp =
        imUp?.detections.find((d) => d.id === pointerInteraction.detId) ?? null;
      if (imUp && detUp) finalizeBBoxGeometryEdit(imUp, detUp);
    }

    pointerInteraction = null;
    draw();
  }

  /**
   * Обработано: каждый кадр считается один раз, если выполняется хотя бы одно из условий:
   * detected, empty, failed, skipped или проверено (`reviewed === true`).
   */
  /** @param {BatchImageItem} im */
  function imageCountsAsProcessedForHeader(im) {
    return (
      im.status === "detected" ||
      im.status === "empty" ||
      im.status === "failed" ||
      im.status === "skipped" ||
      im.reviewed === true
    );
  }

  /** @returns {{ total: number, reviewed: number, processed: number, failed: number }} */
  function computeHeaderProgressMetrics() {
    const images = batchState.images;
    const total = images.length;
    let reviewed = 0;
    let processed = 0;
    let failed = 0;
    for (const im of images) {
      if (im.reviewed && im.status !== "skipped") reviewed++;
      if (im.status === "failed") failed++;
      if (imageCountsAsProcessedForHeader(im)) processed++;
    }
    return { total, reviewed, processed, failed };
  }

  function updateHeaderProgressSummary() {
    const { total, reviewed, processed, failed } = computeHeaderProgressMetrics();

    headerReviewedLine.textContent = total ? `${reviewed} / ${total}` : "0 / 0";
    headerProcessedLine.textContent = total ? `${processed} / ${total}` : "0 / 0";

    const pct = total ? Math.min(100, Math.round((processed / total) * 100)) : 0;
    headerProgressFill.style.width = `${pct}%`;
    headerProgressTrack.setAttribute("aria-valuenow", String(pct));

    if (failed > 0) {
      headerFailedRow.hidden = false;
      headerFailedCount.textContent = `${failed}`;
    } else {
      headerFailedRow.hidden = true;
    }
  }

  function yieldToMain() {
    return new Promise((resolve) => {
      window.requestAnimationFrame(() => resolve(undefined));
    });
  }
  let zipExport;
  function syncExportMenuBusyWithFlags() {
    zipExport.syncLocalExportMenu();
  }
  zipExport = createZipExportHandlers({
    getBatchState: () => batchState,
    showToast,
    convertImageBlobToPng,
    yieldToMain,
    syncExportMenuBusyWithFlags,
    exportMenuToggle,
    exportMenuPanel,
    fmtConf,
  });

  const frameExport = createFrameExportHandlers({
    showToast,
    getCurrentImage: currentImageItem,
    previewImage,
    effectiveOriginalSize,
    currentDetections,
    confThreshold,
    classHidden,
    detectionSourceLabel,
    fmtConf,
  });

  /** @param {BatchImageItem} im */
  function isEligibleForDetectAll(im) {
    if (!im.blob || im.blob.size === 0) return false;
    if (im.status === "skipped") return false;
    if (im.status === "queued" || im.status === "processing") return false;
    return (
      im.status === "idle" ||
      im.status === "failed" ||
      im.status === "empty" ||
      im.status === "detected"
    );
  }

  function groupByCategory(list) {
    /** @type {Map<string, typeof list>} */
    const m = new Map();
    for (const d of list) {
      const k = d.cls_name;
      if (!m.has(k)) m.set(k, []);
      m.get(k).push(d);
    }
    return m;
  }

  function inspectorAppendRow(panel, label, valueText) {
    const row = document.createElement("div");
    row.className = "inspector-row";
    const lab = document.createElement("span");
    lab.className = "inspector-label";
    lab.textContent = label;
    const val = document.createElement("span");
    val.className = "inspector-value";
    val.textContent = valueText;
    row.appendChild(lab);
    row.appendChild(val);
    panel.appendChild(row);
  }

  function addBoxZoomImagePoint() {
    if (!editorModeIsEdit() || batchState.settings.editorTool !== "addBox") {
      return null;
    }
    const geo = getOverlayGeometry();
    if (!geo) return null;

    if (pointerInteraction?.kind === "addBox") {
      const ix = pointerInteraction.ix1 ?? pointerInteraction.ix0;
      const iy = pointerInteraction.iy1 ?? pointerInteraction.iy0;
      if (typeof ix === "number" && typeof iy === "number") {
        return {
          ix: clamp(ix, 0, geo.imgW0),
          iy: clamp(iy, 0, geo.imgH0),
          imgW: geo.imgW0,
          imgH: geo.imgH0,
        };
      }
    }

    if (addBoxCrosshairOverlayPx) {
      const { ix, iy } = canvasToImageCoords(
        geo,
        addBoxCrosshairOverlayPx.ox,
        addBoxCrosshairOverlayPx.oy
      );
      return {
        ix: clamp(ix, 0, geo.imgW0),
        iy: clamp(iy, 0, geo.imgH0),
        imgW: geo.imgW0,
        imgH: geo.imgH0,
      };
    }

    return null;
  }

  function updateAddBoxInspectorZoom() {
    const canvas = inspectorRoot.querySelector(".inspector-addbox-zoom-canvas");
    const hint = inspectorRoot.querySelector(".inspector-addbox-zoom-hint");
    if (!(canvas instanceof HTMLCanvasElement)) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const point = addBoxZoomImagePoint();
    const ready =
      point &&
      previewImage.complete &&
      previewImage.naturalWidth > 0 &&
      previewImage.naturalHeight > 0;

    canvas.width = 260;
    canvas.height = 170;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!ready) {
      ctx.fillStyle = "#0f1116";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      if (hint) hint.textContent = "Наведите курсор на изображение.";
      return;
    }

    const INSPECTOR_ZOOM_FACTOR = 0.12;
    const cropH = Math.max(24, Math.min(point.imgH, point.imgH * INSPECTOR_ZOOM_FACTOR));
    const cropW = Math.max(32, Math.min(point.imgW, cropH * (canvas.width / canvas.height)));
    const sx0 = clamp(point.ix - cropW / 2, 0, Math.max(0, point.imgW - cropW));
    const sy0 = clamp(point.iy - cropH / 2, 0, Math.max(0, point.imgH - cropH));

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.fillStyle = "#0f1116";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    // Рисуем только исходное фото: overlay с bbox/крестом не участвует.
    ctx.drawImage(
      previewImage,
      sx0,
      sy0,
      cropW,
      cropH,
      0,
      0,
      canvas.width,
      canvas.height
    );

    const cursorX = ((point.ix - sx0) / cropW) * canvas.width;
    const cursorY = ((point.iy - sy0) / cropH) * canvas.height;

    ctx.save();
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.5)";
    ctx.beginPath();
    ctx.moveTo(0, cursorY);
    ctx.lineTo(canvas.width, cursorY);
    ctx.moveTo(cursorX, 0);
    ctx.lineTo(cursorX, canvas.height);
    ctx.stroke();
    ctx.strokeStyle = "rgba(63, 130, 247, 0.85)";
    ctx.lineDashOffset = 5;
    ctx.beginPath();
    ctx.moveTo(0, cursorY);
    ctx.lineTo(canvas.width, cursorY);
    ctx.moveTo(cursorX, 0);
    ctx.lineTo(cursorX, canvas.height);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.lineDashOffset = 0;
    ctx.restore();

    if (hint) {
      hint.textContent = `${Math.round(point.ix)}, ${Math.round(point.iy)} px`;
    }
  }

  function appendAddBoxZoomInspector(panel) {
    if (
      !editorModeIsEdit() ||
      batchState.settings.editorTool !== "addBox" ||
      previewImage.style.display === "none" ||
      !effectiveOriginalSize()
    ) {
      return;
    }

    const wrap = document.createElement("div");
    wrap.className = "inspector-addbox-zoom";

    const title = document.createElement("div");
    title.className = "inspector-addbox-zoom-title";
    title.textContent = "Приближение";

    const canvas = document.createElement("canvas");
    canvas.className = "inspector-addbox-zoom-canvas";
    canvas.setAttribute("aria-label", "Приближенный фрагмент изображения без разметки");

    const hint = document.createElement("div");
    hint.className = "inspector-addbox-zoom-hint";
    hint.textContent = "Наведите курсор на изображение.";

    wrap.appendChild(title);
    wrap.appendChild(canvas);
    wrap.appendChild(hint);
    panel.appendChild(wrap);
    updateAddBoxInspectorZoom();
  }

  function buildInspector() {
    inspectorRoot.innerHTML = "";
    const panel = document.createElement("div");
    panel.className = "inspector-panel";

    const h = document.createElement("h4");
    h.className = "inspector-heading";
    h.textContent = "Инспектор";
    panel.appendChild(h);

    const im = currentImageItem();
    if (!im) {
      const p = document.createElement("p");
      p.className = "inspector-empty";
      p.textContent = "Нет загруженного изображения.";
      panel.appendChild(p);
      inspectorRoot.appendChild(panel);
      return;
    }

    appendAddBoxZoomInspector(panel);

    const sid = im.panel.selectedDetectionId;
    const det =
      sid != null ? (im.detections.find((d) => d.id === sid) ?? null) : null;

    const originalFileLabel =
      (im.originalName && String(im.originalName).trim()) ||
      im.displayName ||
      "—";

    if (!det) {
      inspectorAppendRow(panel, "Оригинал", originalFileLabel);
      inspectorAppendRow(panel, "Статус", formatImageStatusForUi(im.status));
      inspectorAppendRow(panel, "Проверено", im.reviewed ? "да" : "нет");
      inspectorAppendRow(panel, "Изменено", im.edited ? "да" : "нет");
      const wh =
        im.width > 0 && im.height > 0 ? `${im.width} × ${im.height}` : "—";
      inspectorAppendRow(panel, "Размер", wh);
      inspectorAppendRow(panel, "Detect", modelStateLabel(im.modelStates.detect.status));
      inspectorAppendRow(panel, "Seg", modelStateLabel(im.modelStates.seg.status));
      inspectorAppendRow(
        panel,
        "Кол-во bbox",
        String(im.detections.filter(isDetectAnnotation).length)
      );
      inspectorAppendRow(
        panel,
        "Кол-во масок",
        String(im.detections.filter(isSegAnnotation).length)
      );
      if (im.error) {
        const err = document.createElement("div");
        err.className = "inspector-error";
        err.textContent = im.error;
        panel.appendChild(err);
      }
    } else {
      inspectorAppendRow(panel, "Оригинальное имя", originalFileLabel);
      inspectorAppendRow(panel, "Класс", det.cls_name);
      inspectorAppendRow(panel, "Уверенность", fmtConf(det.conf));
      inspectorAppendRow(panel, "Источник", detectionSourceLabel(det));
      if (isSegAnnotation(det)) {
        inspectorAppendRow(panel, "Точек", String(det.segment?.length ?? 0));
      } else {
        inspectorAppendRow(panel, "bbox", fmtBoxCoords(det.box));
      }

      inspectorAppendRow(
        panel,
        "Тип",
        isSegAnnotation(det) ? "маска Seg" : "bbox Detect"
      );
      const actions = document.createElement("div");
      actions.className = "inspector-actions";

      const sel = document.createElement("select");
      sel.className = "inspector-class-select";
      sel.title = "Класс объекта";
      sel.setAttribute(
        "aria-label",
        isSegAnnotation(det) ? "Класс маски" : "Класс bbox"
      );
      const detClasses = annotationClasses(det);
      for (const tc of detClasses) {
        const opt = document.createElement("option");
        opt.value = tc.name;
        opt.textContent = tc.name;
        sel.appendChild(opt);
      }
      sel.value = detClasses.some((t) => t.name === det.cls_name)
        ? det.cls_name
        : detClasses[0].name;
      sel.disabled = !editorModeIsEdit();
      sel.addEventListener("change", () => {
        const tc = detClasses.find((t) => t.name === sel.value);
        if (tc) applyDetectionClassHuman(im, det, tc);
      });
      actions.appendChild(sel);

      if (editorModeIsEdit()) {
        const delBtn = document.createElement("button");
        delBtn.type = "button";
        delBtn.className = "inspector-delete-btn";
        delBtn.textContent = "Удалить";
        delBtn.addEventListener("click", () => {
          tryDeleteSelectedDetection();
        });
        actions.appendChild(delBtn);
      }

      panel.appendChild(actions);

      if (editorModeIsEdit() && isSegAnnotation(det)) {
        appendPolygonSimplifyBlock(panel, det, {
          pushUndoCheckpoint: () => pushUndoCheckpoint(),
          requestDraw: () => draw(),
          setSimplifyPreview: (detId, points) => polygonEditor.setSimplifyPreview(detId, points),
          clearSimplifyPreview: (detId) => polygonEditor.clearSimplifyPreview(detId),
          applySimplify: (points) => polygonEditor.applySimplify(det, points),
        });
      }
    }

    inspectorRoot.appendChild(panel);
  }

  function buildRightPanel() {
    const imPre = currentImageItem();
    if (imPre) reconcileSelectedDetectionWithFilters(imPre);

    groupsRoot.innerHTML = "";

    const im = currentImageItem();
    const detections = currentDetections();
    const categoryState = im?.panel.categoryState ?? new Map();
    const detEnabled = im?.panel.detEnabled ?? new Map();

    const filtered = detections.filter((d) => d.conf >= confThreshold());

    const grouped = groupByCategory(filtered);
    const cats = Array.from(grouped.keys()).sort((a, b) => a.localeCompare(b));

    const renderGroupsEmpty = (message) => {
      const empty = document.createElement("div");
      empty.className = "groups-empty";
      empty.textContent = message;
      groupsRoot.appendChild(empty);
    };

    const visibleCats = cats.filter((c) => !classHidden(c));
    if (!visibleCats.length) {
      if (!im) {
        renderGroupsEmpty("Выберите кадр в списке слева.");
      } else if (filtered.length === 0 && detections.length > 0) {
        renderGroupsEmpty(
          "Нет объектов выше текущего порога уверенности — ослабьте фильтр выше."
        );
      } else if (filtered.length === 0) {
        renderGroupsEmpty(
          "На этом кадре пока нет объектов. Запустите распознавание или добавьте bbox либо полигон."
        );
      } else {
        renderGroupsEmpty(
          "Все классы скрыты — включите их на полоске классов над изображением."
        );
      }
    }

    for (const cat of cats) {
      if (classHidden(cat)) continue;

      if (!categoryState.has(cat)) {
        categoryState.set(cat, { enabled: true, collapsed: false });
      }
      const st = categoryState.get(cat);

      const details = document.createElement("details");
      details.className = "group";
      details.open = !st.collapsed;
      details.addEventListener("toggle", () => {
        const s = categoryState.get(cat);
        if (s) s.collapsed = !details.open;
      });

      const summary = document.createElement("summary");
      summary.className = "group-summary";

      const left = document.createElement("div");
      left.className = "group-left";

      const catCheckbox = document.createElement("input");
      catCheckbox.type = "checkbox";
      catCheckbox.className = "group-cat-toggle";
      catCheckbox.checked = st.enabled;
      catCheckbox.title = "Показывать класс на кадре и в экспорте";
      catCheckbox.addEventListener("click", (ev) => ev.stopPropagation());
      catCheckbox.addEventListener("change", () => {
        const s = categoryState.get(cat);
        if (s) s.enabled = catCheckbox.checked;
        touchBatch();
        scheduleWorkspaceAutosave();
        draw();
      });

      const title = document.createElement("span");
      title.className = "group-title";
      title.textContent = cat;

      const count = document.createElement("span");
      count.className = "group-count";
      count.textContent = String(grouped.get(cat).length);

      const chevron = document.createElement("span");
      chevron.className = "group-chevron";
      chevron.setAttribute("aria-hidden", "true");

      const meta = document.createElement("div");
      meta.className = "group-summary-meta";
      meta.appendChild(count);
      meta.appendChild(chevron);

      left.appendChild(catCheckbox);
      left.appendChild(title);
      summary.appendChild(left);
      summary.appendChild(meta);

      const items = document.createElement("div");
      items.className = "group-items";

      for (const d of grouped.get(cat)) {
        if (!detEnabled.has(d.id)) detEnabled.set(d.id, true);

        const row = document.createElement("div");
        row.className = "det-row";
        if (im.panel.selectedDetectionId === d.id) row.classList.add("is-selected");

        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.className = "det-row-toggle";
        cb.title = "Учитывать объект при отрисовке и экспорте";
        cb.checked = detEnabled.get(d.id) !== false;
        cb.addEventListener("click", (ev) => ev.stopPropagation());
        cb.addEventListener("change", () => {
          detEnabled.set(d.id, cb.checked);
          touchBatch();
          scheduleWorkspaceAutosave();
          draw();
        });

        row.addEventListener("click", (ev) => {
          const t = ev.target;
          if (t instanceof HTMLInputElement && t.type === "checkbox") return;
          if (t instanceof HTMLSelectElement) return;
          setSelectedDetectionId(d.id);
          touchBatch();
          buildRightPanel();
          draw();
        });

        const chip = document.createElement("span");
        chip.className = "det-chip";
        chip.style.background = getColorByClass(d.cls_id);

        const text = document.createElement("span");
        text.className = "det-text";
        text.textContent = `#${d.id + 1} (${fmtConf(d.conf)})`;

        row.appendChild(cb);
        row.appendChild(chip);
        row.appendChild(text);

        if (editorModeIsEdit() && im.panel.selectedDetectionId === d.id) {
          const sel = document.createElement("select");
          sel.className = "det-class-select";
          sel.title = "Класс объекта";
          const rowClasses = annotationClasses(d);
          for (const tc of rowClasses) {
            const opt = document.createElement("option");
            opt.value = tc.name;
            opt.textContent = tc.name;
            sel.appendChild(opt);
          }
          sel.value = rowClasses.some((t) => t.name === d.cls_name)
            ? d.cls_name
            : rowClasses[0].name;
          sel.addEventListener("click", (ev) => ev.stopPropagation());
          sel.addEventListener("change", () => {
            const tc = rowClasses.find((t) => t.name === sel.value);
            if (tc) applyDetectionClassHuman(im, d, tc);
          });
          row.appendChild(sel);
        }

        items.appendChild(row);
      }

      details.appendChild(summary);
      details.appendChild(items);
      groupsRoot.appendChild(details);
    }
    syncTrainClassesUi();
    buildInspector();
  }

  function syncTrainClassesUi() {
    renderClassChips();
  }

  function renderClassChips() {
    classChipsRoot.innerHTML = "";
    for (const tc of allTrainClasses()) {
      const n = countTrainClassBboxesAcrossBatch(tc.name);
      const hidden = classHidden(tc.name);
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "class-chip";
      if (hidden) chip.classList.add("is-hidden");
      chip.title = `${tc.name}: ${n} по всему батчу (порог ${fmtConf(confThreshold())}+). Нажмите, чтобы ${hidden ? "показать" : "скрыть"} класс.`;
      chip.setAttribute("aria-pressed", hidden ? "false" : "true");
      chip.addEventListener("click", () => {
        batchState.settings.classVisibility[tc.name] = hidden;
        touchBatch();
        scheduleWorkspaceAutosave();
        buildRightPanel();
        draw();
      });

      const dot = document.createElement("span");
      dot.className = "class-chip-dot";
      dot.style.background = getColorByClass(tc.id);

      const text = document.createElement("span");
      text.textContent = `${tc.name} ${n}`;

      chip.appendChild(dot);
      chip.appendChild(text);
      classChipsRoot.appendChild(chip);
    }
  }

  uploadBtn.addEventListener("click", () => fileInput.click());
  uploadVideoBtn.addEventListener("click", () => videoFileInput.click());

  videoFileInput.addEventListener("change", (event) => {
    const list = event.target.files;
    if (!list?.length) return;
    ingestVideoFile(list[0]);
  });

  videoExtractBtn.addEventListener("click", () => {
    void extractVideoFramesToBatch();
  });

  videoShowBtn.addEventListener("click", () => {
    if (!videoState) return;
    showVideoView();
    updateBatchNavUi();
  });

  viewerPrev.addEventListener("click", () => navigateBatch(-1));
  viewerNext.addEventListener("click", () => navigateBatch(1));

  function applyMarkReviewedFromToolbar() {
    const im = currentImageItem();
    if (!im || im.reviewed === true) return;
    const idx = batchState.currentIndex;
    if (im.status === "skipped") {
      im.status = inferImageStatusFromDetections(im);
      reconcileImageStatusWithDetections(im);
    }
    im.modelStates = normalizeModelStates(im.modelStates);
    let marked = false;
    for (const type of ["detect", "seg"]) {
      const state = im.modelStates[type];
      const hasType = im.detections.some((d) => annotationTypeOf(d) === type);
      if (state.status === "ready" || (state.status === "not_run" && hasType)) {
        state.status = "reviewed";
        state.error = null;
        state.updatedAt = new Date().toISOString();
        state.revision = (Number(state.revision) || 0) + 1;
        marked = true;
      }
    }
    if (!marked && Object.values(im.modelStates).every((s) => s.status === "not_run")) {
      for (const state of Object.values(im.modelStates)) {
        state.status = "reviewed";
        state.updatedAt = new Date().toISOString();
        state.revision = (Number(state.revision) || 0) + 1;
      }
    }
    syncLegacyImageState(im, { preserveSkipped: false });
    touchBatch();
    scheduleWorkspaceAutosave(0);
    updateBatchNavUi();
    const next = findNextUnreviewedNonSkippedIndex(idx);
    if (next !== null) selectBatchIndex(next);
  }

  markReviewedBtn.addEventListener("click", () => {
    applyMarkReviewedFromToolbar();
  });

  skipImageBtn.addEventListener("click", () => {
    skipCurrentImage();
  });

  document.addEventListener("keydown", (e) => {
    if (e.defaultPrevented) return;

    if (e.code === "Escape" && !categoryModalOverlay.hidden) {
      closeCategoryModal();
      e.preventDefault();
      return;
    }

    if (e.code === "Escape" && !hotkeysHelpOverlay.hidden) {
      closeHotkeysHelp();
      e.preventDefault();
      return;
    }

    const typing = isTypingInteractionTarget(e.target);
    const mod = e.ctrlKey || e.metaKey;
    if (mod && e.code === "KeyZ" && !e.shiftKey) {
      if (!typing && undoImageWorkspace()) e.preventDefault();
      return;
    }
    if (mod && (e.code === "KeyY" || (e.code === "KeyZ" && e.shiftKey))) {
      if (!typing && redoImageWorkspace()) e.preventDefault();
      return;
    }

    if (mod || e.altKey) return;
    if (e.code === "Space") {
      if (!typing) {
        spacePanPressed = true;
        syncPanCursorUi();
        e.preventDefault();
      }
      return;
    }
    if (!importSummaryOverlay.hidden || !exportSummaryOverlay.hidden) return;
    if (!categoryModalOverlay.hidden) return;
    if (!hotkeysHelpOverlay.hidden) return;
    if (typing) return;

    const code = e.code;

    if (code === "Escape") {
      cancelActiveEditorTool();
      e.preventDefault();
      return;
    }

    if (code === "Enter") {
      if (editorModeIsEdit() && polygonEditor.handleEnterKey()) {
        touchBatch();
        syncEditorChrome();
        buildInspector();
        buildRightPanel();
        draw();
        e.preventDefault();
      }
      return;
    }

    if (code === "ArrowLeft") {
      if (!batchState.images.length) return;
      e.preventDefault();
      navigateBatch(-1);
      return;
    }
    if (code === "ArrowRight") {
      if (!batchState.images.length) return;
      e.preventDefault();
      navigateBatch(1);
      return;
    }

    if (code === "KeyR") {
      applyMarkReviewedFromToolbar();
      e.preventDefault();
      return;
    }
    if (code === "KeyS") {
      skipCurrentImage();
      e.preventDefault();
      return;
    }
    if (code === "KeyE") {
      maskMenu.close();
      batchState.settings.editorMode = editorModeIsEdit() ? "review" : "edit";
      pointerInteraction = null;
      touchBatch();
      syncEditorChrome();
      buildInspector();
      buildRightPanel();
      draw();
      e.preventDefault();
      return;
    }
    if (code === "KeyA") {
      maskMenu.close();
      batchState.settings.editorMode = "edit";
      batchState.settings.editorTool = isSegmentationMode() ? "addPolygon" : "addBox";
      pointerInteraction = null;
      polygonEditor.cancel();
      touchBatch();
      syncEditorChrome();
      buildInspector();
      buildRightPanel();
      draw();
      e.preventDefault();
      return;
    }
    if (code === "KeyM") {
      openCategoryModal();
      e.preventDefault();
      return;
    }

    /** Индекс класса по цифре (1–5 и NumPad); совпадает с подсказкой в UI. */
    const digitClassIndexByCode = {
      Digit1: 0,
      Numpad1: 0,
      Digit2: 1,
      Numpad2: 1,
      Digit3: 2,
      Numpad3: 2,
      Digit4: 3,
      Numpad4: 3,
      Digit5: 4,
      Numpad5: 4,
    };
    const classIdx = digitClassIndexByCode[code];
    if (
      classIdx !== undefined &&
      classIdx < currentTrainClasses().length &&
      editorModeIsEdit()
    ) {
      const tc = currentTrainClasses()[classIdx];
      if (tc) {
        const im = currentImageItem();
        const sid = im?.panel.selectedDetectionId ?? null;
        const activeType = editorAnnotationType();
        if (im && sid != null) {
          const det = im.detections.find((d) => d.id === sid);
          if (det && annotationTypeOf(det) === activeType) {
            applyDetectionClassHuman(im, det, tc);
            buildInspector();
            buildRightPanel();
            draw();
            e.preventDefault();
            return;
          }
        }
        if (
          batchState.settings.editorTool === "addBox" ||
          batchState.settings.editorTool === "addPolygon"
        ) {
          hotkeyPreferredNewBboxClassName = tc.name;
          e.preventDefault();
          return;
        }
      }
      return;
    }

    if (code === "Backspace" || code === "Delete") {
      if (editorModeIsEdit() && tryDeleteSelectedDetection()) {
        e.preventDefault();
      }
    }
  });

  document.addEventListener("keyup", (e) => {
    if (e.code !== "Space") return;
    spacePanPressed = false;
    panDragState = null;
    syncPanCursorUi();
  });

  overlay.addEventListener("mousedown", onOverlayPointerDown);
  overlay.addEventListener("contextmenu", onOverlayContextMenu);
  overlay.addEventListener("mousemove", (e) => {
    refreshOverlayCursorHint(e.offsetX, e.offsetY, e.altKey || e.ctrlKey || e.metaKey);
    if (
      editorModeIsEdit() &&
      ["addBox", "crop"].includes(batchState.settings.editorTool)
    ) {
      syncAddBoxCrosshairOverlayPx(e.offsetX, e.offsetY);
      draw();
    }
    if (editorModeIsEdit() && batchState.settings.editorTool === "addPolygon") {
      polygonEditor.handleOverlayHover(e.offsetX, e.offsetY);
    }
  });
  overlay.addEventListener("mouseleave", () => {
    refreshOverlayCursorHint(null, null);
    if (
      !(
        pointerInteraction &&
        (pointerInteraction.kind === "addBox" ||
          pointerInteraction.kind === "crop")
      )
    ) {
      clearAddBoxCrosshairOverlayPx();
      draw();
    }
  });
  imageFrame.addEventListener("wheel", onImageFrameWheel, { passive: false });
  document.addEventListener("mousemove", onDocumentPointerMove);
  document.addEventListener("mouseup", onDocumentPointerUp);

  editorModeReviewBtn.addEventListener("click", () => {
    maskMenu.close();
    batchState.settings.editorMode = "review";
    pointerInteraction = null;
    touchBatch();
    syncEditorChrome();
    buildInspector();
    buildRightPanel();
    draw();
  });

  editorModeEditBtn.addEventListener("click", () => {
    maskMenu.close();
    batchState.settings.editorMode = "edit";
    pointerInteraction = null;
    touchBatch();
    syncEditorChrome();
    buildInspector();
    buildRightPanel();
    draw();
  });

  editorToolSelectBtn.addEventListener("click", () => {
    maskMenu.close();
    batchState.settings.editorTool = "select";
    pointerInteraction = null;
    polygonEditor.cancel();
    touchBatch();
    syncEditorChrome();
    buildInspector();
    draw();
  });

  editorToolAddBtn.addEventListener("click", () => {
    maskMenu.close();
    batchState.settings.editorTool = "addBox";
    pointerInteraction = null;
    polygonEditor.cancel();
    touchBatch();
    syncEditorChrome();
    buildInspector();
    draw();
  });

  editorToolCropBtn.addEventListener("click", () => {
    maskMenu.close();
    batchState.settings.editorTool = "crop";
    pointerInteraction = null;
    polygonEditor.cancel();
    touchBatch();
    syncEditorChrome();
    buildInspector();
    draw();
  });

  editorToolAddPolygonBtn.addEventListener("click", () => {
    maskMenu.close();
    batchState.settings.editorTool = "addPolygon";
    pointerInteraction = null;
    touchBatch();
    syncEditorChrome();
    buildInspector();
    draw();
  });

  fileInput.addEventListener("change", (event) => {
    const list = event.target.files;
    if (!list || !list.length) return;
    void processIncomingFiles(list);
  });

  batchStatusFilter.addEventListener("change", () => {
    onBatchListFiltersChanged();
  });

  batchSortToggle.addEventListener("click", (e) => {
    e.stopPropagation();
    if (!batchState.images.length) return;
    closeExportMenu();
    closeHotkeysHelp();
    const willOpen = batchSortMenu.hidden;
    batchSortMenu.hidden = !willOpen;
    batchSortToggle.setAttribute("aria-expanded", willOpen ? "true" : "false");
  });

  batchSortMenu.addEventListener("click", (e) => {
    e.stopPropagation();
    const btn = e.target.closest("[data-batch-sort]");
    if (!(btn instanceof HTMLButtonElement)) return;
    const mode = btn.getAttribute("data-batch-sort");
    if (
      mode === "quality-desc" ||
      mode === "quality-asc" ||
      mode === "bbox-count-desc" ||
      mode === "bbox-count-asc"
    ) {
      applyBatchImageSort(
        /** @type {"quality-desc"|"quality-asc"|"bbox-count-desc"|"bbox-count-asc"} */ (mode)
      );
    }
  });

  hotkeysHelpBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    openHotkeysHelp();
  });
  hotkeysHelpClose.addEventListener("click", (e) => {
    e.stopPropagation();
    closeHotkeysHelp();
  });
  hotkeysHelpOverlay.addEventListener("click", (e) => {
    e.stopPropagation();
    if (e.target === hotkeysHelpOverlay) closeHotkeysHelp();
  });

  clearBtn.addEventListener("click", () => clearAll(true));

  exportMenuToggle.addEventListener("click", (e) => {
    e.stopPropagation();
    closeBatchSortMenu();
    closeHotkeysHelp();
    const willOpen = exportMenuPanel.hidden;
    exportMenuPanel.hidden = !willOpen;
    exportMenuToggle.setAttribute(
      "aria-expanded",
      willOpen ? "true" : "false"
    );
  });

  document.addEventListener("click", () => {
    closeExportMenu();
    closeBatchSortMenu();
  });

  document.addEventListener(
    "click",
    (e) => {
      if (hotkeysHelpOverlay.hidden) return;
      const t = /** @type {Node} */ (e.target);
      if (hotkeysHelpOverlay.contains(t)) return;
      closeHotkeysHelp();
    },
    true
  );

  exportMenuPanel.addEventListener("click", (e) => {
    e.stopPropagation();
    const btn = e.target.closest("[data-export]");
    if (!(btn instanceof HTMLButtonElement)) return;
    const actionKind = btn.getAttribute("data-export");
    if (!actionKind) return;
    closeExportMenu();
    beginExportFlow(actionKind);
  });

  exportSummaryOverlay.addEventListener("click", (e) => {
    if (e.target === exportSummaryOverlay) closeExportSummaryModal();
  });

  importSummaryOverlay.addEventListener("click", (e) => {
    if (e.target === importSummaryOverlay) closeImportSummaryModal();
  });

  categoryModalOverlay.addEventListener("click", (e) => {
    if (e.target === categoryModalOverlay) closeCategoryModal();
  });

  confFilterRange.addEventListener("input", () => {
    batchState.settings.confidenceThreshold = Number(confFilterRange.value) || 0;
    refreshConfidenceFilterVisual();
    touchBatch();
    scheduleWorkspaceAutosave();
    buildRightPanel();
    draw();
  });

  confFilterRange.closest(".confidence-filter")?.addEventListener("click", (e) => {
    const btn = e.target.closest(".confidence-preset");
    if (!(btn instanceof HTMLButtonElement)) return;
    const v = Math.max(0, Math.min(1, Number(btn.dataset.confidence) || 0));
    batchState.settings.confidenceThreshold = v;
    confFilterRange.value = String(v);
    refreshConfidenceFilterVisual();
    touchBatch();
    scheduleWorkspaceAutosave();
    buildRightPanel();
    draw();
  });

  function closeRunMenu() {
    runMenu.hidden = true;
    runMenuToggle.setAttribute("aria-expanded", "false");
  }

  function syncRecognizeBusy() {
    runBtn.disabled = detectAllInFlight;
    runMenuToggle.disabled = detectAllInFlight;
    runMenu.querySelectorAll("button").forEach((button) => {
      button.disabled = detectAllInFlight;
    });
  }

  function runTypes(runMode) {
    if (runMode === "detect") return ["detect"];
    if (runMode === "seg") return ["seg"];
    return ["detect", "seg"];
  }

  function isEligibleForModelRun(im, type) {
    if (!im?.blob || im.blob.size === 0 || im.status === "skipped") return false;
    im.modelStates = normalizeModelStates(im.modelStates);
    return im.modelStates[type].status !== "running";
  }

  function modelRunIsProtected(im, type) {
    const state = im.modelStates[type];
    return (
      state.status === "reviewed" ||
      im.detections.some(
        (d) => annotationTypeOf(d) === type && d.source === "human"
      )
    );
  }

  function confirmProtectedModelRuns(tasks) {
    const protectedTasks = tasks.filter(({ im, type }) =>
      modelRunIsProtected(im, type)
    );
    if (!protectedTasks.length) return true;
    const images = new Set(protectedTasks.map(({ im }) => im.id)).size;
    return window.confirm(
      `На ${images} кадр(ах) есть проверенные или изменённые вручную результаты ` +
        `(${protectedTasks.length} запусков моделей). Автоматические результаты выбранных ` +
        `моделей будут заменены, ручные аннотации сохранятся. Продолжить?`
    );
  }

  /** Очередь независимых заданий «кадр × модель». */
  async function detectAllImages(runMode = "both") {
    if (!batchState.images.length) {
      if (videoState) {
        const extracted = await extractVideoFramesToBatch();
        if (!extracted || !batchState.images.length) return;
      } else {
        showToast("Нет изображений. Сначала загрузите файлы или видео.", {
          type: "warning",
        });
        return;
      }
    }

    if (detectAllInFlight) return;
    const normalizedRunMode = ["both", "detect", "seg"].includes(runMode)
      ? runMode
      : "both";
    const types = runTypes(normalizedRunMode);
    const tasks = [];
    for (let i = 0; i < batchState.images.length; i++) {
      const im = batchState.images[i];
      for (const type of types) {
        if (isEligibleForModelRun(im, type)) tasks.push({ idx: i, im, type });
      }
    }

    if (!tasks.length) {
      showToast(
        "Нет доступных кадров для выбранных моделей: пропущенные и уже выполняющиеся задания не запускаются.",
        { type: "info", durationMs: 4200 }
      );
      return;
    }
    if (!confirmProtectedModelRuns(tasks)) return;

    closeRunMenu();
    detectAllInFlight = true;
    batchState.settings.runSelection = normalizedRunMode;
    syncRecognizeBusy();
    setStatus(
      normalizedRunMode === "both"
        ? "Detect + Seg…"
        : normalizedRunMode === "seg"
          ? "Seg…"
          : "Detect…"
    );

    const total = tasks.length;
    let completed = 0;
    let succeeded = 0;
    let failed = 0;
    let ignored = 0;

    for (const task of tasks) {
      const state = task.im.modelStates[task.type];
      state.revision = (Number(state.revision) || 0) + 1;
      task.runRevision = state.revision;
      state.status = "running";
      state.error = null;
      state.updatedAt = new Date().toISOString();
      syncLegacyImageState(task.im, { preserveSkipped: false });
    }
    touchBatch();
    scheduleWorkspaceAutosave(0);
    updateBatchNavUi();

    let nextSlot = 0;

    async function worker() {
      while (true) {
        const slot = nextSlot++;
        if (slot >= tasks.length) break;

        const task = tasks[slot];
        const { idx, im, type, runRevision } = task;
        refreshUiAfterImageProcessed(idx, false);
        await yieldToMain();

        try {
          const data = await fetchDetectApi(
            im,
            type === "seg" ? "segmentation" : "detection"
          );
          if (applyModelSuccessFromResponse(im, data, type, runRevision)) {
            succeeded++;
          } else {
            ignored++;
          }
        } catch (e) {
          if (
            applyModelFailureToItem(
              im,
              type,
              String(e?.message || e),
              runRevision
            )
          ) {
            failed++;
          } else {
            ignored++;
          }
        }

        completed++;
        touchBatch();
        refreshUiAfterImageProcessed(idx);
        await yieldToMain();
      }
    }

    try {
      await Promise.all(
        Array.from({ length: DETECT_ALL_CONCURRENCY }, () => worker())
      );
    } finally {
      if (detectAllBatchUiThrottleTimer !== null) {
        window.clearTimeout(detectAllBatchUiThrottleTimer);
        detectAllBatchUiThrottleTimer = null;
      }
      detectAllInFlight = false;
      syncRecognizeBusy();
      setStatus(runButtonIdleText());
      scheduleWorkspaceAutosave(0);
      updateBatchNavUi();
      showToast(
        `Распознавание завершено: ${succeeded} готово, ${failed} ошибок` +
          `${ignored ? `, ${ignored} ответов не применено из-за ручных изменений` : ""}. ` +
          `Обработано ${completed} из ${total} заданий.`,
        {
          type: failed ? "warning" : "success",
          durationMs: 4300,
        }
      );
    }
  }

  runBtn.addEventListener("click", () => {
    closeRunMenu();
    void detectAllImages("both");
  });

  runMenuToggle.addEventListener("click", (event) => {
    event.stopPropagation();
    const willOpen = runMenu.hidden;
    runMenu.hidden = !willOpen;
    runMenuToggle.setAttribute("aria-expanded", String(willOpen));
  });
  runMenu.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    const button = target.closest("[data-run-mode]");
    if (!(button instanceof HTMLButtonElement)) return;
    const mode = button.dataset.runMode || "both";
    closeRunMenu();
    void detectAllImages(mode);
  });
  document.addEventListener("click", (event) => {
    const target = event.target;
    if (
      !runMenu.hidden &&
      (!(target instanceof Element) || !target.closest("#recognize-split"))
    ) {
      closeRunMenu();
    }
  });

  workModeDetectionBtn.addEventListener("click", () => setInferenceMode("detection"));
  workModeSegmentationBtn.addEventListener("click", () => setInferenceMode("segmentation"));
  workModeDetectionBtn.disabled = true;
  workModeSegmentationBtn.disabled = true;

  window.addEventListener("resize", () => {
    if (!previewImage.src) return;
    draw();
  });

  async function initializeWorkspace() {
    try {
      const snap = await loadLatestWorkspaceSnapshot();
      if (snap) {
        restoreWorkspaceSnapshot(snap);
        setWorkspaceSaveStatus("saved");
      } else {
        clearAll(false);
      }
    } catch (err) {
      console.warn("[workspace autosave] restore failed:", err);
      setWorkspaceSaveStatus("failed");
      clearAll(false);
    } finally {
      workspaceHydrating = false;
      workModeDetectionBtn.disabled = false;
      workModeSegmentationBtn.disabled = false;
      syncInferenceModeUi();
    }
  }

  void initializeWorkspace();
}
