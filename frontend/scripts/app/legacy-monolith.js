(() => {
  /** @typedef {"idle"|"queued"|"processing"|"detected"|"empty"|"failed"|"skipped"} ImageStatus */

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
   *   reviewed: boolean,
   *   edited: boolean,
   *   detections: Array<{id:number,cls_id:number,cls_name:string,conf:number,box:[number,number,number,number],source?:"model"|"human"}>,
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
   *     editorTool: "select"|"addBox",
   *   }
   * }} BatchState
   */

  const MAX_DIRECT_IMAGES = 100;
  const MAX_ZIP_IMAGES = 1000;
  const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
  const ZIP_IMAGE_EXT_RE = /\.(jpe?g|png|webp)$/i;
  const DATA_YAML_RE = /(^|\/)data\.ya?ml$/i;
  const WORKSPACE_DB_NAME = "ai-train-workspace";
  const WORKSPACE_DB_VERSION = 1;
  const WORKSPACE_STORE_NAME = "workspaces";
  const WORKSPACE_AUTOSAVE_DELAY_MS = 300;

  /** Фиксированные классы сайта: редактор не становится универсальным редактором любых классов. */
  const TRAIN_PART_CLASS_NAMES = ["body", "autocoupler", "axlebox", "bogie"];
  /** Индексы классов в batch-экспорте YOLO TXT (0–3), фиксированный порядок классов сайта. */
  const YOLO_TXT_EXPORT_CLASS_ORDER = Object.freeze([...TRAIN_PART_CLASS_NAMES]);
  /** Версия манифеста Full Project ZIP (project.json). */
  const PROJECT_EXPORT_JSON_VERSION = "1.0";
  const DEFAULT_TRAIN_CLASSES = TRAIN_PART_CLASS_NAMES.map((name, id) => ({
    name,
    id,
  }));

  /** @type {Array<{ name: string; id: number }>} */
  let TRAIN_CLASSES = DEFAULT_TRAIN_CLASSES.map((tc) => ({ ...tc }));
  const HANDLE_HIT_PX = 12;
  /** Минимальная сторона bbox в координатах изображения (resize / новый box). */
  const MIN_BOX_SIDE = 5;

  /** @type {HTMLButtonElement|null} */
  const uploadBtn = document.getElementById("upload-btn");
  /** @type {HTMLButtonElement|null} */
  const clearBtn = document.getElementById("clear-btn");
  /** @type {HTMLButtonElement|null} */
  const runBtn = document.getElementById("run-btn");
  /** @type {HTMLInputElement|null} */
  const fileInput = document.getElementById("file-input");
  /** @type {HTMLImageElement|null} */
  const previewImage = document.getElementById("preview-image");
  /** @type {HTMLElement|null} */
  const placeholderText = document.getElementById("placeholder-text");
  /** @type {HTMLElement|null} */
  const imageFrame = document.getElementById("image-frame");
  /** @type {HTMLElement|null} */
  const imageStack = document.getElementById("image-stack");
  /** @type {HTMLCanvasElement|null} */
  const overlay = document.getElementById("overlay-canvas");

  /** @type {HTMLButtonElement|null} */
  const viewerPrev = document.getElementById("viewer-prev");
  /** @type {HTMLButtonElement|null} */
  const viewerNext = document.getElementById("viewer-next");
  /** @type {HTMLElement|null} */
  const viewerCounter = document.getElementById("viewer-counter");
  /** @type {HTMLButtonElement|null} */
  const markReviewedBtn = document.getElementById("mark-reviewed-btn");
  /** @type {HTMLButtonElement|null} */
  const skipImageBtn = document.getElementById("skip-image-btn");

  /** @type {HTMLElement|null} */
  const headerReviewedLine = document.getElementById("header-reviewed-line");
  /** @type {HTMLElement|null} */
  const workspaceSaveStatus = document.getElementById("workspace-save-status");
  /** @type {HTMLElement|null} */
  const headerProcessedLine = document.getElementById("header-processed-line");
  /** @type {HTMLElement|null} */
  const headerProgressTrack = document.getElementById("header-progress-track");
  /** @type {HTMLElement|null} */
  const headerProgressFill = document.getElementById("header-progress-fill");
  /** @type {HTMLElement|null} */
  const headerFailedRow = document.getElementById("header-failed-row");
  /** @type {HTMLElement|null} */
  const headerFailedCount = document.getElementById("header-failed-count");

  /** @type {HTMLElement|null} */
  const exportDropdownWrap = document.getElementById("export-dropdown-wrap");
  /** @type {HTMLButtonElement|null} */
  const exportMenuToggle = document.getElementById("export-menu-toggle");
  /** @type {HTMLElement|null} */
  const exportMenuPanel = document.getElementById("export-menu-panel");
  /** @type {HTMLElement|null} */
  const exportSummaryOverlay = document.getElementById("export-summary-overlay");
  /** @type {HTMLElement|null} */
  const exportSummaryTitle = document.getElementById("export-summary-title");
  /** @type {HTMLElement|null} */
  const exportSummarySubtitle = document.getElementById("export-summary-subtitle");
  /** @type {HTMLElement|null} */
  const exportSummaryBody = document.getElementById("export-summary-body");
  /** @type {HTMLButtonElement|null} */
  const exportSummaryConfirm = document.getElementById("export-summary-confirm");
  /** @type {HTMLButtonElement|null} */
  const exportSummaryCancel = document.getElementById("export-summary-cancel");

  /** @type {HTMLElement|null} */
  const importSummaryOverlay = document.getElementById("import-summary-overlay");
  /** @type {HTMLElement|null} */
  const importSummaryTitle = document.getElementById("import-summary-title");
  /** @type {HTMLElement|null} */
  const importSummaryBody = document.getElementById("import-summary-body");
  /** @type {HTMLButtonElement|null} */
  const importSummaryClose = document.getElementById("import-summary-close");

  /** @type {HTMLElement|null} */
  const viewerFilename = document.getElementById("viewer-filename");
  /** @type {HTMLElement|null} */
  const batchImageListRoot = document.getElementById("batch-image-list-root");
  /** @type {HTMLSelectElement|null} */
  const batchStatusFilter = document.getElementById("batch-status-filter");

  /** @type {HTMLElement|null} */
  const totalObjectsEl = document.getElementById("total-objects");
  /** @type {HTMLElement|null} */
  const groupsRoot = document.getElementById("groups-root");
  /** @type {HTMLElement|null} */
  const inspectorRoot = document.getElementById("inspector-root");
  /** @type {HTMLInputElement|null} */
  const confFilterRange = document.getElementById("confidence-filter-range");
  /** @type {HTMLElement|null} */
  const confFilterValue = document.getElementById("confidence-filter-value");

  /** @type {HTMLButtonElement|null} */
  const editorModeReviewBtn = document.getElementById("editor-mode-review");
  /** @type {HTMLButtonElement|null} */
  const editorModeEditBtn = document.getElementById("editor-mode-edit");
  /** @type {HTMLElement|null} */
  const editorToolsBar = document.getElementById("editor-tools-bar");
  /** @type {HTMLButtonElement|null} */
  const editorToolSelectBtn = document.getElementById("editor-tool-select");
  /** @type {HTMLButtonElement|null} */
  const editorToolAddBtn = document.getElementById("editor-tool-add");
  /** @type {HTMLElement|null} */
  const classChipsRoot = document.getElementById("class-chips-root");

  if (
    !uploadBtn ||
    !clearBtn ||
    !runBtn ||
    !fileInput ||
    !previewImage ||
    !placeholderText ||
    !imageFrame ||
    !imageStack ||
    !overlay ||
    !viewerPrev ||
    !viewerNext ||
    !viewerCounter ||
    !markReviewedBtn ||
    !skipImageBtn ||
    !headerReviewedLine ||
    !workspaceSaveStatus ||
    !headerProcessedLine ||
    !headerProgressTrack ||
    !headerProgressFill ||
    !headerFailedRow ||
    !headerFailedCount ||
    !totalObjectsEl ||
    !groupsRoot ||
    !inspectorRoot ||
    !confFilterRange ||
    !confFilterValue ||
    !exportDropdownWrap ||
    !exportMenuToggle ||
    !exportMenuPanel ||
    !exportSummaryOverlay ||
    !exportSummaryTitle ||
    !exportSummarySubtitle ||
    !exportSummaryBody ||
    !exportSummaryConfirm ||
    !exportSummaryCancel ||
    !importSummaryOverlay ||
    !importSummaryTitle ||
    !importSummaryBody ||
    !importSummaryClose ||
    !viewerFilename ||
    !batchImageListRoot ||
    !batchStatusFilter ||
    !editorModeReviewBtn ||
    !editorModeEditBtn ||
    !editorToolsBar ||
    !editorToolSelectBtn ||
    !editorToolAddBtn ||
    !classChipsRoot
  ) {
    return;
  }

  /** @returns {string} */
  function newBatchId() {
    if (crypto.randomUUID) return crypto.randomUUID();
    return `batch_${Date.now()}_${Math.random().toString(16).slice(2)}`;
  }

  /** @returns {string} */
  function newImageItemId() {
    if (crypto.randomUUID) return crypto.randomUUID();
    return `img_${Date.now()}_${Math.random().toString(16).slice(2)}`;
  }

  /** @returns {BatchState} */
  function createEmptyBatchState() {
    const now = new Date().toISOString();
    return {
      batchId: newBatchId(),
      images: [],
      currentIndex: 0,
      createdAt: now,
      updatedAt: now,
      importSummary: {
        imported: 0,
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
      },
      settings: {
        confidenceThreshold: 0,
        statusFilter: "all",
        classVisibility: {},
        trainClassOrder: TRAIN_CLASSES.map((tc) => tc.name),
        editorMode: "review",
        editorTool: "select",
      },
    };
  }

  function resetTrainClassesToDefault() {
    TRAIN_CLASSES = DEFAULT_TRAIN_CLASSES.map((tc) => ({ ...tc }));
  }

  /** @param {string[]} names */
  function trainClassNamesMatchFixedSet(names) {
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
  function applyTrainClassOrderFromNames(names) {
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

  /** @type {BatchState} */
  let batchState = createEmptyBatchState();

  const DETECT_ALL_CONCURRENCY = 2;
  /** Ожидание ответа /api/detect; без этого зависший сервер держит кадр в «processing» бесконечно */
  const DETECT_API_FETCH_TIMEOUT_MS = 180000;
  /** Мин. интервал между полными перерисовками списка батча во время пакета */
  const DETECT_ALL_BATCH_NAV_THROTTLE_MS = 150;
  /** @type {number|null} */
  let detectAllBatchUiThrottleTimer = null;
  /** Пакетное распознавание с основной кнопки */
  let detectAllInFlight = false;
  /** ZIP-экспорт PNG по батчу */
  let pngZipExportInFlight = false;
  /** ZIP-экспорт YOLO labels по батчу */
  let yoloTxtZipExportInFlight = false;
  /** Полный экспорт проекта (images + labels + annotations + manifest) */
  let fullProjectZipExportInFlight = false;
  /** @type {Promise<IDBDatabase>|null} */
  let workspaceDbPromise = null;
  /** @type {number|null} */
  let workspaceAutosaveTimer = null;
  let workspaceAutosaveGeneration = 0;

  /**
   * Временное состояние drag на overlay (addBox / move / resize bbox).
   * @type {null | {
   *   kind: "addBox"|"move"|"resize",
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
      batchState.settings.editorTool !== "addBox" ||
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
      workspaceSaveStatus.textContent = "Saving...";
      workspaceSaveStatus.classList.add("is-saving");
    } else if (state === "saved") {
      workspaceSaveStatus.textContent = "Saved";
      workspaceSaveStatus.classList.add("is-saved");
    } else if (state === "failed") {
      workspaceSaveStatus.textContent = "Save failed";
      workspaceSaveStatus.classList.add("is-failed");
    } else {
      workspaceSaveStatus.textContent = "";
    }
  }

  /** @returns {Promise<IDBDatabase>} */
  function openWorkspaceDb() {
    if (!("indexedDB" in window)) {
      return Promise.reject(new Error("IndexedDB is not available"));
    }
    if (workspaceDbPromise) return workspaceDbPromise;
    workspaceDbPromise = new Promise((resolve, reject) => {
      const req = indexedDB.open(WORKSPACE_DB_NAME, WORKSPACE_DB_VERSION);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(WORKSPACE_STORE_NAME)) {
          db.createObjectStore(WORKSPACE_STORE_NAME, { keyPath: "batchId" });
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () =>
        reject(req.error || new Error("Failed to open IndexedDB"));
    });
    return workspaceDbPromise;
  }

  /** @param {IDBTransaction} tx */
  function waitForTransaction(tx) {
    return new Promise((resolve, reject) => {
      tx.oncomplete = () => resolve(undefined);
      tx.onabort = () => reject(tx.error || new Error("IndexedDB tx aborted"));
      tx.onerror = () => reject(tx.error || new Error("IndexedDB tx failed"));
    });
  }

  /** @param {IDBRequest} req */
  function waitForRequest(req) {
    return new Promise((resolve, reject) => {
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error || new Error("IndexedDB request failed"));
    });
  }

  /** @param {ImagePanelState} panel */
  function serializePanelState(panel) {
    return {
      categoryState: Array.from(panel.categoryState.entries()),
      detEnabled: Array.from(panel.detEnabled.entries()),
      selectedDetectionId: panel.selectedDetectionId,
    };
  }

  /** @param {BatchImageItem} im */
  function serializeImageItem(im) {
    return {
      id: im.id,
      displayName: im.displayName,
      originalName: im.originalName,
      fileType: im.fileType,
      fileSize: im.fileSize,
      width: im.width,
      height: im.height,
      blob: im.blob,
      status: im.status,
      reviewed: im.reviewed,
      edited: im.edited,
      detections: im.detections.map((d) => ({
        id: d.id,
        cls_id: d.cls_id,
        cls_name: d.cls_name,
        conf: d.conf,
        box: [...d.box],
        source: d.source ?? "model",
      })),
      error: im.error,
      panel: serializePanelState(im.panel),
    };
  }

  function createWorkspaceSnapshot() {
    return {
      batchId: batchState.batchId,
      currentIndex: batchState.currentIndex,
      createdAt: batchState.createdAt,
      updatedAt: batchState.updatedAt,
      savedAt: new Date().toISOString(),
      importSummary: { ...batchState.importSummary },
      settings: {
        ...batchState.settings,
        classVisibility: { ...batchState.settings.classVisibility },
      },
      images: batchState.images.map((im) => serializeImageItem(im)),
    };
  }

  async function loadLatestWorkspaceSnapshot() {
    const db = await openWorkspaceDb();
    const tx = db.transaction(WORKSPACE_STORE_NAME, "readonly");
    const records = /** @type {any[]} */ (
      await waitForRequest(tx.objectStore(WORKSPACE_STORE_NAME).getAll())
    );
    if (!records.length) return null;
    records.sort((a, b) => {
      const at = Date.parse(a?.savedAt || a?.updatedAt || a?.createdAt || "");
      const bt = Date.parse(b?.savedAt || b?.updatedAt || b?.createdAt || "");
      return bt - at;
    });
    return records[0] ?? null;
  }

  async function saveWorkspaceToIndexedDB() {
    const db = await openWorkspaceDb();
    const tx = db.transaction(WORKSPACE_STORE_NAME, "readwrite");
    tx.objectStore(WORKSPACE_STORE_NAME).put(createWorkspaceSnapshot());
    await waitForTransaction(tx);
  }

  function runWorkspaceAutosave() {
    const generation = ++workspaceAutosaveGeneration;
    saveWorkspaceToIndexedDB()
      .then(() => {
        if (generation === workspaceAutosaveGeneration) {
          setWorkspaceSaveStatus("saved");
        }
      })
      .catch((err) => {
        console.warn("[workspace autosave] failed:", err);
        if (generation === workspaceAutosaveGeneration) {
          setWorkspaceSaveStatus("failed");
          showToast(
            "Автосохранение не удалось. Проверьте место на диске и доступ к IndexedDB.",
            { type: "error", durationMs: 5200 }
          );
        }
      });
  }

  /** Debounced autosave. Use 0 for important discrete changes, 300ms for UI settings. */
  function scheduleWorkspaceAutosave(delayMs = WORKSPACE_AUTOSAVE_DELAY_MS) {
    setWorkspaceSaveStatus("saving");
    if (workspaceAutosaveTimer !== null) {
      window.clearTimeout(workspaceAutosaveTimer);
      workspaceAutosaveTimer = null;
    }
    if (delayMs <= 0) {
      runWorkspaceAutosave();
      return;
    }
    workspaceAutosaveTimer = window.setTimeout(() => {
      workspaceAutosaveTimer = null;
      runWorkspaceAutosave();
    }, delayMs);
  }

  /** @param {EventTarget|null} target */
  function isTypingInteractionTarget(target) {
    if (!target || !(target instanceof Element)) return false;
    const el = target;
    const tag = el.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return true;
    if (el.isContentEditable) return true;
    return false;
  }

  /**
   * Логическое имя слота в UI/экспорте всегда .png без перекодирования исходного файла.
   * @param {number} ordinal1Based 1 .. n (в пределах batch)
   */
  function formatDisplayName(ordinal1Based) {
    const n = String(Math.max(1, ordinal1Based)).padStart(3, "0");
    return `${n}.png`;
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
      reviewed: false,
      edited: false,
      detections: [],
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
  function deserializeImageItem(raw, ordinal1Based) {
    const blob =
      raw?.blob instanceof Blob
        ? raw.blob
        : new Blob([], { type: raw?.fileType || "application/octet-stream" });
    return {
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
      reviewed: raw?.reviewed === true,
      edited: raw?.edited === true,
      detections: Array.isArray(raw?.detections)
        ? raw.detections.map((d, i) => normalizeDetection(d, i))
        : [],
      error: typeof raw?.error === "string" ? raw.error : null,
      panel: deserializePanelState(raw?.panel),
    };
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
    const found = TRAIN_CLASSES.find((t) => t.name === s);
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
    const cls_id = typeof raw?.cls_id === "number" ? raw.cls_id : 0;
    const cls_name =
      typeof raw?.cls_name === "string"
        ? raw.cls_name
        : TRAIN_CLASSES[0]?.name ?? "body";
    const conf = typeof raw?.conf === "number" ? raw.conf : 0;
    const box =
      Array.isArray(raw?.box) && raw.box.length >= 4
        ? [
            Number(raw.box[0]),
            Number(raw.box[1]),
            Number(raw.box[2]),
            Number(raw.box[3]),
          ]
        : [0, 0, 0, 0];
    const source = raw?.source === "human" ? "human" : "model";
    return { id, cls_id, cls_name, conf, box, source };
  }

  /** @param {BatchImageItem["detections"][0]} d */
  function detectionSourceLabel(d) {
    return d.source === "human" ? "human" : "model";
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
    /* Если класс был скрыт чипом «скрыть», после смены на него bbox перестаёт проходить фильтры — сбрасывается выделение и рамка пропадает. */
    if (batchState.settings.classVisibility[tc.name] === false) {
      delete batchState.settings.classVisibility[tc.name];
    }
    let catSt = im.panel.categoryState.get(tc.name);
    if (!catSt) {
      catSt = { enabled: true, collapsed: false };
      im.panel.categoryState.set(tc.name, catSt);
    } else {
      catSt.enabled = true;
    }

    d.cls_id = tc.id;
    d.cls_name = tc.name;
    d.source = "human";
    im.edited = true;
    im.reviewed = false;
    touchBatch();
    scheduleWorkspaceAutosave(0);
    updateBatchNavUi();
    buildRightPanel();
    draw();
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

  /** @returns {Blob|null} */
  function currentBlobForApi() {
    const im = currentImageItem();
    if (!im) return null;
    return im.blob;
  }

  const COLORS = [
    "#4ade80",
    "#60a5fa",
    "#f97316",
    "#f472b6",
    "#facc15",
    "#a78bfa",
    "#22d3ee",
    "#fb7185",
  ];

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
      document.body.appendChild(root);
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

  const EXPORT_ACTION_LABELS = {
    "current-jpg": "Текущий кадр: фото с разметкой (.jpg)",
    "current-yolo": "Текущий кадр: разметка YOLO (.txt)",
    "current-json": "Текущий кадр: экспорт JSON",
    "batch-png-clean": "Весь батч: PNG ZIP без разметки",
    "batch-png-marked": "Весь батч: PNG ZIP с разметкой",
    "batch-yolo-zip": "Весь батч: YOLO TXT (ZIP)",
    "batch-project-zip": "Весь батч: Full Project ZIP",
  };

  /**
   * Сводка для диалога экспорта (соответствует правилам batch PNG/YOLO/Project).
   */
  function computeBatchExportEligibilitySummary() {
    const images = batchState.images;
    const reasons = {
      empty: 0,
      skipped: 0,
      failed: 0,
      noBbox: 0,
      pending: 0,
      noBlob: 0,
    };
    let exportable = 0;

    for (const im of images) {
      if (isImageValidForPngZipExport(im)) {
        exportable++;
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
      if (
        im.status === "idle" ||
        im.status === "queued" ||
        im.status === "processing"
      ) {
        reasons.pending++;
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
      skipped: total - exportable,
      reasons,
    };
  }

  function syncExportMenuBusyWithFlags() {
    const busy =
      pngZipExportInFlight ||
      yoloTxtZipExportInFlight ||
      fullProjectZipExportInFlight;
    exportMenuToggle.disabled = !!busy;
    exportMenuPanel.querySelectorAll(".export-menu-item").forEach((btn) => {
      btn.disabled = !!busy;
    });
  }

  function closeExportMenu() {
    exportMenuPanel.hidden = true;
    exportMenuToggle.setAttribute("aria-expanded", "false");
  }

  /** @param {HTMLUListElement} ul */
  function appendExportReasonLine(ul, label, count) {
    if (count <= 0) return;
    const li = document.createElement("li");
    li.textContent = `${label}: ${count}`;
    ul.appendChild(li);
  }

  /** @param {{ total:number, exportable:number, skipped:number, reasons: Record<string, number> }} batchSummary */
  function appendStandardSkipReasons(ul, batchSummary) {
    appendExportReasonLine(ul, "empty", batchSummary.reasons.empty);
    appendExportReasonLine(ul, "skipped", batchSummary.reasons.skipped);
    appendExportReasonLine(ul, "failed", batchSummary.reasons.failed);
    appendExportReasonLine(ul, "no bbox", batchSummary.reasons.noBbox);
    appendExportReasonLine(
      ul,
      "не готовы (idle / очередь / распознавание)",
      batchSummary.reasons.pending
    );
    appendExportReasonLine(
      ul,
      "нет файла изображения",
      batchSummary.reasons.noBlob
    );
  }

  /**
   * @param {{ total:number, exportable:number, skipped:number, reasons: Record<string, number> }} batchSummary
   * @param {{ mode: "batch"|"single", singleExports?: number, currentLabel?: string }} opts
   */
  function fillExportSummaryBody(batchSummary, opts) {
    exportSummaryBody.textContent = "";

    if (opts.mode === "batch") {
      const row1 = document.createElement("div");
      row1.className = "export-summary-row";
      row1.textContent = `Будет экспортировано изображений: ${batchSummary.exportable}`;
      exportSummaryBody.appendChild(row1);

      const row2 = document.createElement("div");
      row2.className = "export-summary-row";
      row2.textContent = `Пропущено: ${batchSummary.skipped}`;
      exportSummaryBody.appendChild(row2);

      const head = document.createElement("div");
      head.className = "export-summary-row";
      head.textContent = "Причины:";
      exportSummaryBody.appendChild(head);

      const ul = document.createElement("ul");
      ul.className = "export-summary-list";
      appendStandardSkipReasons(ul, batchSummary);
      exportSummaryBody.appendChild(ul);

      if (batchSummary.total === 0) {
        const note = document.createElement("p");
        note.className = "export-summary-row";
        note.textContent = "В батче нет изображений.";
        exportSummaryBody.appendChild(note);
      }
      return;
    }

    const nSingle =
      typeof opts.singleExports === "number" ? opts.singleExports : 1;
    const rowCur = document.createElement("div");
    rowCur.className = "export-summary-row";
    rowCur.textContent = `Будет экспортировано файлов: ${nSingle} (текущий кадр).`;
    exportSummaryBody.appendChild(rowCur);

    if (opts.currentLabel) {
      const rl = document.createElement("div");
      rl.className = "export-summary-row";
      rl.textContent = opts.currentLabel;
      exportSummaryBody.appendChild(rl);
    }

    const sep = document.createElement("div");
    sep.className = "export-summary-row";
    sep.style.marginTop = "12px";
    sep.textContent = `По всему батчу при таком экспорте попало бы: ${batchSummary.exportable} из ${batchSummary.total}`;
    exportSummaryBody.appendChild(sep);

    const rowSkip = document.createElement("div");
    rowSkip.className = "export-summary-row";
    rowSkip.textContent = `Пропущено было бы: ${batchSummary.skipped}`;
    exportSummaryBody.appendChild(rowSkip);

    const head = document.createElement("div");
    head.className = "export-summary-row";
    head.textContent = "Причины:";
    exportSummaryBody.appendChild(head);

    const ul = document.createElement("ul");
    ul.className = "export-summary-list";
    appendStandardSkipReasons(ul, batchSummary);
    exportSummaryBody.appendChild(ul);
  }

  /** @type {((ev: KeyboardEvent) => void)|null} */
  let exportSummaryEscapeHandler = null;

  function closeExportSummaryModal() {
    exportSummaryOverlay.hidden = true;
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

  const IMPORT_SUMMARY_NAME_PREVIEW = 14;

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
  function appendImportSummaryNameList(body, names, max = IMPORT_SUMMARY_NAME_PREVIEW) {
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

  /** @param {{ title?: string, subtitle: string, summary: ReturnType<typeof computeBatchExportEligibilitySummary>, mode: "batch"|"single", actionKind?: string, singleExports?: number, currentLabel?: string, canConfirm: boolean, onConfirm?: () => void }} cfg */
  function openExportSummaryModal(cfg) {
    exportSummaryTitle.textContent = cfg.title || "Экспорт";
    exportSummarySubtitle.textContent = cfg.subtitle || "";
    fillExportSummaryBody(cfg.summary, cfg);

    const kind = cfg.actionKind ? String(cfg.actionKind) : "";
    const needsZip =
      kind.startsWith("batch-png") ||
      kind === "batch-yolo-zip" ||
      kind === "batch-project-zip";
    const zipOk = typeof JSZip === "function";
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

    exportSummaryConfirm.onclick = () => {
      closeExportSummaryModal();
      if (canConfirm && cfg.onConfirm) cfg.onConfirm();
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
      return { ok: false, detail: "Кадр со статусом skipped не экспортируется." };
    if (im.status === "failed")
      return { ok: false, detail: "Кадр со статусом failed не экспортируется." };
    if (im.status === "empty")
      return { ok: false, detail: "Кадр со статусом empty не экспортируется." };
    if (
      im.status === "idle" ||
      im.status === "queued" ||
      im.status === "processing"
    ) {
      return {
        ok: false,
        detail:
          "Кадр ещё не готов (idle / очередь / распознавание) — дождитесь обработки.",
      };
    }
    if (!Array.isArray(im.detections) || !im.detections.length) {
      return { ok: false, detail: "На кадре нет bbox." };
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
    const summary = computeBatchExportEligibilitySummary();
    const subtitle =
      EXPORT_ACTION_LABELS[
        /** @type {keyof typeof EXPORT_ACTION_LABELS} */ (actionKind)
      ] || actionKind;

    if (actionKind.startsWith("batch-")) {
      const needsZip =
        actionKind.startsWith("batch-png") ||
        actionKind === "batch-yolo-zip" ||
        actionKind === "batch-project-zip";
      const zipOk = typeof JSZip === "function";
      openExportSummaryModal({
        title: "Экспорт",
        subtitle,
        summary,
        mode: "batch",
        actionKind,
        canConfirm: summary.exportable > 0 && (!needsZip || zipOk),
        onConfirm: () => runBatchExportAction(actionKind),
      });
      return;
    }

    const cur = describeCurrentFrameForExport(actionKind);
    if (!cur.ok) {
      showToast(cur.detail, { type: "warning", durationMs: 3800 });
      return;
    }
    runCurrentExportAction(actionKind);
  }

  /** @param {string} kind */
  function runBatchExportAction(kind) {
    switch (kind) {
      case "batch-png-clean":
        void exportBatchPngZip(false);
        break;
      case "batch-png-marked":
        void exportBatchPngZip(true);
        break;
      case "batch-yolo-zip":
        void exportBatchYoloTxtZip();
        break;
      case "batch-project-zip":
        void exportFullProjectZip();
        break;
      default:
        break;
    }
  }

  /** @param {string} kind */
  function runCurrentExportAction(kind) {
    switch (kind) {
      case "current-jpg":
        downloadAnnotatedImage();
        break;
      case "current-yolo":
        downloadYoloAnnotations();
        break;
      case "current-json":
        downloadJson();
        break;
      default:
        break;
    }
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
      if (im.reviewed) reviewed++;
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

  /**
   * Кадры для PNG ZIP: есть хотя бы один bbox, финальный статус (не очередь/ошибка/пропуск).
   * @param {BatchImageItem} im
   */
  function isImageValidForPngZipExport(im) {
    if (!im.blob || im.blob.size === 0) return false;
    if (!Array.isArray(im.detections) || im.detections.length === 0) return false;
    if (im.status === "skipped" || im.status === "failed") return false;
    if (
      im.status === "idle" ||
      im.status === "queued" ||
      im.status === "processing"
    ) {
      return false;
    }
    return true;
  }

  /** @param {number} ordinal1Based */
  function pngZipEntryBaseName(ordinal1Based) {
    const n = String(Math.max(1, ordinal1Based)).padStart(3, "0");
    return `${n}.png`;
  }

  /** @param {Blob} blob @param {string} filename */
  function triggerBlobDownload(blob, filename) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.setTimeout(() => URL.revokeObjectURL(url), 400);
  }

  /**
   * Полноразмерное PNG с bbox/подписями (все сохранённые detections).
   * @param {BatchImageItem} im
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
        ctx.strokeRect(x, y, w, h);

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
   */
  async function exportBatchPngZip(withMarkup) {
    if (pngZipExportInFlight) return;
    if (typeof JSZip !== "function") {
      alert("JSZip не загружен. Обновите страницу и попробуйте снова.");
      return;
    }

    const items = batchState.images.filter(isImageValidForPngZipExport);
    if (!items.length) {
      showToast(
        "Нет кадров для экспорта: нужны bbox и статус после распознавания (не idle / очередь / ошибка / пропуск).",
        { type: "warning", durationMs: 4200 }
      );
      return;
    }

    pngZipExportInFlight = true;
    syncExportMenuBusyWithFlags();

    try {
      const zip = new JSZip();
      const folder = zip.folder("images");
      if (!folder) throw new Error("Не удалось создать папку images в архиве.");

      let ordinal = 0;
      for (const im of items) {
        ordinal++;
        const name = pngZipEntryBaseName(ordinal);
        const pngBlob = withMarkup
          ? await renderAnnotatedPngBlobFromItem(im)
          : await convertImageBlobToPng(im.blob);
        folder.file(name, pngBlob);
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
      alert(
        `Ошибка экспорта ZIP: ${err && err.message ? err.message : String(err)}`
      );
    } finally {
      pngZipExportInFlight = false;
      syncExportMenuBusyWithFlags();
    }
  }

  /** @param {string} clsName */
  function yoloTxtExportClassId(clsName) {
    const n = String(clsName ?? "").trim().toLowerCase();
    const idx = YOLO_TXT_EXPORT_CLASS_ORDER.indexOf(n);
    return idx >= 0 ? idx : 0;
  }

  /**
   * Размер изображения для нормализации bbox (метаданные кадра или decode blob).
   * @param {BatchImageItem} im
   * @returns {Promise<{width:number,height:number}|null>}
   */
  async function labelNormalizationSize(im) {
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

  /**
   * Положительные размеры для экспорта (bbox/YOLO), чтобы каждый валидный кадр получил свой ordinal в архиве.
   * @param {BatchImageItem} im
   */
  async function ensurePositiveExportDimensions(im) {
    const s = await labelNormalizationSize(im);
    if (s && s.width > 0 && s.height > 0) return s;
    const w = typeof im.width === "number" && im.width > 0 ? im.width : 1;
    const h = typeof im.height === "number" && im.height > 0 ? im.height : 1;
    return { width: w, height: h };
  }

  /**
   * YOLO: class_id x_center y_center width height (все 0..1 относительно размера кадра).
   * Все сохранённые bbox; неизвестное имя класса мапится в 0 (body).
   * @param {BatchImageItem} im
   * @param {number} iw
   * @param {number} ih
   * @returns {string} тело файла или пустая строка, если строк нет
   */
  function buildYoloTxtFileBody(im, iw, ih) {
    if (!(iw > 0 && ih > 0)) return "";
    const lines = [];
    for (const d of im.detections) {
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
      const clsId = yoloTxtExportClassId(d.cls_name);
      lines.push(
        [clsId, nx.toFixed(6), ny.toFixed(6), nw.toFixed(6), nh.toFixed(6)].join(
          " "
        )
      );
    }
    if (!lines.length) return "";
    return `${lines.join("\n")}\n`;
  }

  async function exportBatchYoloTxtZip() {
    if (yoloTxtZipExportInFlight) return;
    if (typeof JSZip !== "function") {
      alert("JSZip не загружен. Обновите страницу и попробуйте снова.");
      return;
    }

    const items = batchState.images.filter(isImageValidForPngZipExport);
    if (!items.length) {
      showToast(
        "Нет кадров для экспорта YOLO: нужны bbox и статус после распознавания (не idle / очередь / ошибка / пропуск).",
        { type: "warning", durationMs: 4200 }
      );
      return;
    }

    yoloTxtZipExportInFlight = true;
    syncExportMenuBusyWithFlags();

    try {
      const zip = new JSZip();
      const folder = zip.folder("labels");
      if (!folder) throw new Error("Не удалось создать папку labels в архиве.");

      let ordinal = 0;
      for (const im of items) {
        ordinal++;
        const stem = pngZipEntryBaseName(ordinal).replace(/\.png$/i, "");
        const size = await labelNormalizationSize(im);
        const body =
          size && size.width > 0 && size.height > 0
            ? buildYoloTxtFileBody(im, size.width, size.height)
            : "";
        folder.file(`${stem}.txt`, body || "");
        await yieldToMain();
      }

      const zipBlob = await zip.generateAsync({ type: "blob" });
      const stamp = new Date().toISOString().replace(/[:.]/g, "-");
      triggerBlobDownload(zipBlob, `labels_yolo_${stamp}.zip`);
      showToast(
        `Экспорт завершён: архив YOLO, ${items.length} файлов в labels/.`,
        { type: "success" }
      );
    } catch (err) {
      console.warn("[export yolo txt zip]", err);
      alert(
        `Ошибка экспорта ZIP: ${err && err.message ? err.message : String(err)}`
      );
    } finally {
      yoloTxtZipExportInFlight = false;
      syncExportMenuBusyWithFlags();
    }
  }

  function buildProjectExportDataYaml() {
    const lines = ["names:"];
    for (let i = 0; i < YOLO_TXT_EXPORT_CLASS_ORDER.length; i++) {
      lines.push(`  ${i}: ${YOLO_TXT_EXPORT_CLASS_ORDER[i]}`);
    }
    return `${lines.join("\n")}\n`;
  }

  /**
   * Один кадр для annotations/*.json (полные bbox, class_id в порядке классов сайта).
   * @param {BatchImageItem} im
   * @param {{width:number,height:number}} size
   * @param {string} exportStem без расширения: "001"
   */
  function buildAnnotationExportJsonObject(im, size, exportStem) {
    return {
      schema: "ai-train.annotation.v1",
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
      detections: im.detections.map((d) => {
        const [bx1, by1, bx2, by2] = d.box;
        return {
          id: d.id,
          class_id: yoloTxtExportClassId(d.cls_name),
          class_name: d.cls_name,
          confidence: d.conf,
          source: d.source ?? "model",
          bbox: {
            x1: bx1,
            y1: by1,
            x2: bx2,
            y2: by2,
          },
        };
      }),
    };
  }

  async function exportFullProjectZip() {
    if (fullProjectZipExportInFlight) return;
    if (typeof JSZip !== "function") {
      alert("JSZip не загружен. Обновите страницу и попробуйте снова.");
      return;
    }

    const items = batchState.images.filter(isImageValidForPngZipExport);
    if (!items.length) {
      showToast(
        "Нет кадров для экспорта: нужны bbox и статус после распознавания (не idle / очередь / ошибка / пропуск).",
        { type: "warning", durationMs: 4200 }
      );
      return;
    }

    fullProjectZipExportInFlight = true;
    syncExportMenuBusyWithFlags();

    try {
      const zip = new JSZip();
      const folderImages = zip.folder("images");
      const folderLabels = zip.folder("labels");
      const folderAnnot = zip.folder("annotations");
      if (!folderImages || !folderLabels || !folderAnnot) {
        throw new Error("Не удалось создать папки images/labels/annotations в архиве.");
      }

      /** @type {Array<Record<string, unknown>>} */
      const imagesMeta = [];

      let exportOrdinal = 0;
      for (const im of items) {
        exportOrdinal++;
        const stem = pngZipEntryBaseName(exportOrdinal).replace(/\.png$/i, "");

        const size = await ensurePositiveExportDimensions(im);

        const pngBlob = await convertImageBlobToPng(im.blob);
        folderImages.file(`${stem}.png`, pngBlob);

        const labelBody = buildYoloTxtFileBody(im, size.width, size.height);
        folderLabels.file(`${stem}.txt`, labelBody || "");

        const annotObj = buildAnnotationExportJsonObject(im, size, stem);
        folderAnnot.file(`${stem}.json`, `${JSON.stringify(annotObj, null, 2)}\n`);

        imagesMeta.push({
          exportStem: stem,
          exportedImageFileName: `${stem}.png`,
          id: im.id,
          displayName: im.displayName,
          originalName: im.originalName,
          imageNames: {
            beforeUpload: im.originalName,
            afterUploadOnSite: im.displayName,
            afterExport: `${stem}.png`,
          },
          status: im.status,
          reviewed: im.reviewed,
          edited: im.edited,
          width: size.width,
          height: size.height,
          detectionsCount: im.detections.length,
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
        schema: "ai-train.project-export.v1",
        version: PROJECT_EXPORT_JSON_VERSION,
        batchId: batchState.batchId,
        createdAt: batchState.createdAt,
        updatedAt: batchState.updatedAt,
        exportedAt,
        currentIndex: exportedCurrentIndex,
        classes: YOLO_TXT_EXPORT_CLASS_ORDER.map((name, id) => ({ id, name })),
        settings: {
          ...batchState.settings,
          classVisibility: { ...batchState.settings.classVisibility },
        },
        images: imagesMeta.map((row) => ({
          exportStem: row.exportStem,
          exportedImageFileName: row.exportedImageFileName,
          id: row.id,
          displayName: row.displayName,
          originalName: row.originalName,
          imageNames: row.imageNames,
          status: row.status,
          reviewed: row.reviewed,
          edited: row.edited,
          width: row.width,
          height: row.height,
          detectionsCount: row.detectionsCount,
          fileType: row.fileType,
          fileSize: row.fileSize,
          panel: row.panel,
        })),
      };

      zip.file("project.json", `${JSON.stringify(projectPayload, null, 2)}\n`);
      zip.file("data.yaml", buildProjectExportDataYaml());

      const zipBlob = await zip.generateAsync({ type: "blob" });
      const stamp = new Date().toISOString().replace(/[:.]/g, "-");
      triggerBlobDownload(zipBlob, `project_${stamp}.zip`);
      showToast(
        `Экспорт завершён: полный проект, ${imagesMeta.length} кадров.`,
        { type: "success", durationMs: 3800 }
      );
    } catch (err) {
      console.warn("[export full project zip]", err);
      alert(
        `Ошибка экспорта ZIP: ${err && err.message ? err.message : String(err)}`
      );
    } finally {
      fullProjectZipExportInFlight = false;
      syncExportMenuBusyWithFlags();
    }
  }

  /** @param {BatchImageItem} im */
  async function fetchDetectApi(im) {
    const fd = new FormData();
    fd.append("file", im.blob, im.displayName);
    const ac = new AbortController();
    const abortTimer = window.setTimeout(() => {
      ac.abort();
    }, DETECT_API_FETCH_TIMEOUT_MS);
    let resp;
    try {
      resp = await fetch("/api/detect", {
        method: "POST",
        body: fd,
        signal: ac.signal,
      });
    } catch (e) {
      const name = e && typeof e === "object" && "name" in e ? String(e.name) : "";
      if (name === "AbortError") {
        throw new Error(
          `Таймаут распознавания (${Math.round(DETECT_API_FETCH_TIMEOUT_MS / 1000)} с). Проверь сервер или это изображение.`
        );
      }
      throw e;
    } finally {
      window.clearTimeout(abortTimer);
    }
    const raw = await resp.text();
    let data = null;
    try {
      data = raw ? JSON.parse(raw) : null;
    } catch {
      throw new Error(
        "Сервер вернул не‑JSON ответ. Проверь, запущен ли бэкенд /api/detect."
      );
    }
    if (!resp.ok || !data) {
      throw new Error(data?.error || "Detect failed");
    }
    return data;
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

  /** @param {BatchImageItem} im @param {any} data */
  function applyDetectSuccessFromResponse(im, data) {
    im.panel.categoryState.clear();
    im.panel.detEnabled.clear();
    im.panel.selectedDetectionId = null;

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
    im.detections = rawDets.map((d, i) => normalizeDetection(d, i));
    im.status = im.detections.length > 0 ? "detected" : "empty";
    im.reviewed = false;
    im.edited = false;
    im.error = null;
  }

  /** @param {BatchImageItem} im @param {string} msg */
  function applyDetectFailureToItem(im, msg) {
    im.status = "failed";
    im.error = msg;
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
    if (sf === "reviewed") return im.reviewed === true;
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
      batchState.currentIndex = visible[0];
      touchBatch();
      scheduleWorkspaceAutosave(0);
      showCurrentPreview();
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

  /** @param {BatchImageItem} im */
  function batchItemStatusIcon(im) {
    if (im.reviewed) return "✅";
    if (im.status === "processing") return "⏳";
    if (im.status === "failed") return "❌";
    if (im.status === "skipped") return "⏭";
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

      const metaEl = document.createElement("span");
      metaEl.className = "bic-meta";
      const nBoxes = Array.isArray(im.detections) ? im.detections.length : 0;
      metaEl.textContent = nBoxes > 0 ? `${nBoxes}\u00A0bbox` : "";

      body.appendChild(nameEl);
      body.appendChild(metaEl);

      btn.appendChild(iconEl);
      btn.appendChild(body);

      btn.addEventListener("click", () => selectBatchIndex(i));

      batchImageListRoot.appendChild(btn);
    }
  }

  /** @param {number} index */
  function selectBatchIndex(index) {
    if (index < 0 || index >= batchState.images.length) return;
    if (batchState.currentIndex === index) {
      updateBatchNavUi();
      return;
    }
    pointerInteraction = null;
    clearAddBoxCrosshairOverlayPx();
    batchState.currentIndex = index;
    touchBatch();
    scheduleWorkspaceAutosave(0);
    updateBatchNavUi();
    showCurrentPreview();
  }

  function updateMainViewerNav() {
    const n = batchState.images.length;
    const idx = batchState.currentIndex;
    const im = currentImageItem();

    if (!n) {
      viewerFilename.textContent = "";
      viewerCounter.textContent = "Нет кадров";
      viewerPrev.disabled = true;
      viewerNext.disabled = true;
      markReviewedBtn.disabled = true;
      skipImageBtn.disabled = true;
      return;
    }

    viewerFilename.textContent = im?.displayName ?? "";
    viewerCounter.textContent = `${idx + 1} из ${n}`;
    viewerPrev.disabled = idx <= 0;
    viewerNext.disabled = idx >= n - 1;
    markReviewedBtn.disabled =
      !im || im.reviewed === true || im.status === "skipped";
    skipImageBtn.disabled = !n || !im || im.status === "skipped";
  }

  /**
   * Пропуск текущего кадра (очередь разметки).
   */
  function skipCurrentImage() {
    const im = currentImageItem();
    if (!im || im.status === "skipped") return;
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
    const im = currentImageItem();
    previewImage.onload = null;
    previewImage.onerror = null;

    if (!im?.objectUrl) {
      previewImage.src = "";
      previewImage.style.display = "none";
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
    revokeBatchObjectUrls();
    pointerInteraction = null;
    clearAddBoxCrosshairOverlayPx();
    batchState = createEmptyBatchState();

    fileInput.value = "";
    previewImage.src = "";
    previewImage.style.display = "none";
    placeholderText.style.display = "block";
    groupsRoot.innerHTML = "";
    totalObjectsEl.textContent = "0";
    batchState.settings.confidenceThreshold = 0;
    updateConfidenceFilterDom();
    syncBatchListFilterUiFromState();
    detectAllInFlight = false;
    runBtn.disabled = false;
    updateBatchNavUi();
    setStatus("Запустить распознавание");
    clearCanvas();
    if (persist) scheduleWorkspaceAutosave(0);
  }

  /** @param {any} snap */
  function restoreWorkspaceSnapshot(snap) {
    revokeBatchObjectUrls();
    const empty = createEmptyBatchState();
    const images = Array.isArray(snap?.images)
      ? snap.images
          .map((im, i) => deserializeImageItem(im, i + 1))
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

    if (
      Array.isArray(batchState.settings.trainClassOrder) &&
      applyTrainClassOrderFromNames(batchState.settings.trainClassOrder)
    ) {
      batchState.settings.trainClassOrder = TRAIN_CLASSES.map((tc) => tc.name);
    } else {
      resetTrainClassesToDefault();
      batchState.settings.trainClassOrder = TRAIN_CLASSES.map((tc) => tc.name);
    }

    fileInput.value = "";
    updateConfidenceFilterDom();
    syncBatchListFilterUiFromState();
    const filtersReset = ensureRestoredBatchListVisible();
    detectAllInFlight = false;
    runBtn.disabled = false;
    setStatus("Запустить распознавание");

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
      totalObjectsEl.textContent = "0";
      clearCanvas();
      updateBatchNavUi();
    }
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

  /** @param {string} value */
  function stripYamlQuotes(value) {
    return value.trim().replace(/^['"]|['"]$/g, "");
  }

  /** @param {string} text */
  function parseYamlClassNames(text) {
    const lines = text.replace(/\r/g, "").split("\n");
    const inlineList = text.match(/^(names|classes)\s*:\s*\[([^\]]*)\]/m);
    if (inlineList) {
      return inlineList[2]
        .split(",")
        .map((x) => stripYamlQuotes(x.replace(/#.*/, "")))
        .filter(Boolean);
    }

    const inlineObject = text.match(/^(names|classes)\s*:\s*\{([^}]*)\}/m);
    if (inlineObject) {
      return inlineObject[2]
        .split(",")
        .map((part) => part.split(":").slice(1).join(":"))
        .map((x) => stripYamlQuotes(x.replace(/#.*/, "")))
        .filter(Boolean);
    }

    for (let i = 0; i < lines.length; i++) {
      if (!/^\s*(names|classes)\s*:\s*(#.*)?$/.test(lines[i])) continue;
      const out = [];
      for (let j = i + 1; j < lines.length; j++) {
        const line = lines[j];
        if (!/^\s+/.test(line)) break;
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith("#")) continue;
        const listItem = trimmed.match(/^-\s*(.+)$/);
        if (listItem) {
          out.push(stripYamlQuotes(listItem[1].replace(/#.*/, "")));
          continue;
        }
        const keyed = trimmed.match(/^\d+\s*:\s*(.+)$/);
        if (keyed) out.push(stripYamlQuotes(keyed[1].replace(/#.*/, "")));
      }
      if (out.length) return out.filter(Boolean);
    }

    return null;
  }

  /** @param {string} path */
  function normalizeZipPath(path) {
    return String(path || "").replace(/\\/g, "/").replace(/^\.\/+/, "");
  }

  /** @param {any} zip */
  function findZipEntryCaseInsensitive(zip, relativePath) {
    const target = normalizeZipPath(relativePath).toLowerCase();
    for (const e of Object.values(zip.files)) {
      if (e.dir) continue;
      if (normalizeZipPath(e.name).toLowerCase() === target) return e;
    }
    return null;
  }

  /** @param {any} zip */
  function findProjectJsonEntry(zip) {
    for (const e of Object.values(zip.files)) {
      if (e.dir) continue;
      const p = normalizeZipPath(e.name);
      if (/(^|\/)project\.json$/i.test(p)) return e;
    }
    return null;
  }

  /** @param {string} s */
  function coerceImportedImageStatus(s) {
    const allowed = new Set([
      "idle",
      "queued",
      "processing",
      "detected",
      "empty",
      "failed",
      "skipped",
    ]);
    return allowed.has(s) ? /** @type {ImageStatus} */ (s) : "detected";
  }

  /** @param {BatchImageItem} im */
  function reconcileImageStatusWithDetections(im) {
    const n = Array.isArray(im.detections) ? im.detections.length : 0;
    if (n > 0 && im.status === "empty") im.status = "detected";
    if (n === 0 && im.status === "detected") im.status = "empty";
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
   * Восстановление детекций из annotations/*.json (формат экспорта).
   * @param {any} annot
   */
  function rawDetectionsFromAnnotationExport(annot) {
    const list = Array.isArray(annot?.detections) ? annot.detections : [];
    return list.map((d, i) => {
      const bbox = d?.bbox && typeof d.bbox === "object" ? d.bbox : {};
      const x1 = Number(bbox.x1);
      const y1 = Number(bbox.y1);
      const x2 = Number(bbox.x2);
      const y2 = Number(bbox.y2);
      const cn = String(d?.class_name ?? "")
        .trim()
        .toLowerCase();
      let tc = TRAIN_CLASSES.find((t) => t.name === cn);
      if (!tc && typeof d?.class_id === "number") {
        const byYolo = YOLO_TXT_EXPORT_CLASS_ORDER[d.class_id];
        if (byYolo)
          tc = TRAIN_CLASSES.find((t) => t.name === byYolo);
      }
      if (!tc) tc = TRAIN_CLASSES[0];
      const detId = typeof d?.id === "number" ? d.id : i;
      const conf =
        typeof d?.confidence === "number" ? d.confidence : 0;
      const src = d?.source === "human" ? "human" : "model";
      return {
        id: detId,
        cls_id: tc.id,
        cls_name: tc.name,
        conf,
        box: [x1, y1, x2, y2],
        source: src,
      };
    });
  }

  /**
   * Слот в UI после импорта — имя файла после экспорта (001.png), как в архиве.
   * @param {any} row
   * @param {string} stem exportStem
   * @param {number} ordinal1Based
   */
  function displayNameAfterProjectImport(row, stem, ordinal1Based) {
    const nested =
      row?.imageNames &&
      typeof row.imageNames.afterExport === "string"
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

  /**
   * Импорт архива Export Full Project ZIP: восстановление batch, сохранение в IndexedDB.
   * @param {any} zip
   * @param {any} payload project.json
   */
  async function finalizeProjectZipImport(zip, payload) {
    revokeBatchObjectUrls();

    resetTrainClassesToDefault();

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
      if (!stem) {
        skippedManifestEntries++;
        continue;
      }

      const imgEntry = findZipEntryCaseInsensitive(zip, `images/${stem}.png`);
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
      const annotEntry = findZipEntryCaseInsensitive(
        zip,
        `annotations/${stem}.json`
      );
      if (annotEntry) {
        try {
          annotRaw = JSON.parse(await annotEntry.async("string"));
        } catch (err) {
          console.warn("[project zip] annotations", stem, err);
        }
      }

      const detRaw = annotRaw
        ? rawDetectionsFromAnnotationExport(annotRaw)
        : [];

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
        reviewed: row?.reviewed === true,
        edited: row?.edited === true,
        detections: detRaw,
        error: null,
        panel: row?.panel,
      };

      const im = deserializeImageItem(rawIm, builtImages.length + 1);
      reconcileImageStatusWithDetections(im);
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
      totalObjectsEl.textContent = "0";
      clearCanvas();
      updateBatchNavUi();
      fileInput.value = "";
      showToast(
        "Не удалось импортировать проект: нет изображений в images/ или повреждён архив.",
        { type: "error", durationMs: 5200 }
      );
      return;
    }

    const mergedSettings = {
      ...empty.settings,
      ...(payload?.settings && typeof payload.settings === "object"
        ? payload.settings
        : {}),
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
      settings: {
        ...mergedSettings,
        trainClassOrder: TRAIN_CLASSES.map((tc) => tc.name),
      },
    };

    resetTrainClassesToDefault();
    batchState.settings.trainClassOrder = TRAIN_CLASSES.map((tc) => tc.name);

    fileInput.value = "";
    updateConfidenceFilterDom();
    syncBatchListFilterUiFromState();

    ensureRestoredBatchListVisible();
    detectAllInFlight = false;
    runBtn.disabled = false;
    setStatus("Запустить распознавание");

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
    totalObjectsEl.textContent = "0";
    pointerInteraction = null;
    updateBatchNavUi();
    showCurrentPreview();

    openImportSummaryModal({
      title: "Сводка импорта проекта (ZIP)",
      kind: "project-zip",
      addedImages: builtImages.length,
      skippedFiles: skippedManifestEntries,
      skipIssueLines:
        skippedManifestEntries > 0
          ? [
              `Записей в манифесте без файла в images/ или с ошибкой чтения: ${skippedManifestEntries}`,
            ]
          : [],
      skipNameBuckets: [],
    });
  }

  /** @param {string} path */
  function isServiceZipPath(path) {
    const normalized = normalizeZipPath(path);
    const parts = normalized.split("/");
    const base = parts[parts.length - 1] || "";
    return (
      !base ||
      parts.includes("__MACOSX") ||
      base === ".DS_Store" ||
      base.startsWith("._") ||
      (base.startsWith(".") && !DATA_YAML_RE.test(normalized))
    );
  }

  /** @param {File} file */
  function isZipFile(file) {
    return /\.zip$/i.test(file.name) || file.type === "application/zip";
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
    batchState.settings.trainClassOrder = TRAIN_CLASSES.map((tc) => tc.name);

    accepted.forEach((file, i) => {
      batchState.images.push(
        imageItemFromFile(file.blob, firstNewIndex + i + 1, file.originalName)
      );
    });

    batchState.currentIndex = firstNewIndex;
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
    totalObjectsEl.textContent = "0";
    setStatus("Запустить распознавание");
    showCurrentPreview();
    fileInput.value = "";
  }

  /**
   * @param {FileList|File[]} files
   */
  function ingestImageFiles(files) {
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
    if (!yamlEntry) return;
    try {
      const text = await yamlEntry.async("string");
      const parsedNames = parseYamlClassNames(text);
      if (parsedNames?.length) applyTrainClassOrderFromNames(parsedNames);
    } catch {
      /* игнор */
    }
  }

  /** @param {File} zipFile */
  async function ingestZipFile(zipFile) {
    const JSZipCtor = globalThis.JSZip;
    if (!JSZipCtor) {
      alert("JSZip не загружен. Проверьте подключение к интернету и перезагрузите страницу.");
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
      await applyTrainClassesFromZipIfPresent(zip);
      const skippedServiceNames = [];
      const skippedUnsupportedNames = [];
      const skippedOverSizeNames = [];
      const skippedOverBatchLimitNames = [];
      /** @type {Array<{ path: string; blob: Blob }>} */
      const imageEntries = [];

      const entries = Object.values(zip.files).sort((a, b) =>
        normalizeZipPath(a.name).localeCompare(normalizeZipPath(b.name))
      );

      for (const entry of entries) {
        if (entry.dir) continue;
        const path = normalizeZipPath(entry.name);
        if (DATA_YAML_RE.test(path)) continue;
        if (/(^|\/)project\.json$/i.test(path)) continue;
        if (
          /(^|\/)labels\/[^/]+\.txt$/i.test(path) ||
          /(^|\/)annotations\/[^/]+\.json$/i.test(path)
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
    } catch (err) {
      console.error("[zip import] failed:", err);
      showToast(`Не удалось прочитать ZIP: ${String(err?.message || err)}`, {
        type: "error",
        durationMs: 5200,
      });
      fileInput.value = "";
    } finally {
      setStatus("Запустить распознавание");
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
      if (hitTestBox(geo, ox, oy, d.box)) return d;
    }
    return null;
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
    det.source = "human";
    im.edited = true;
    im.reviewed = false;
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
   * Класс для нового bbox: у выделенного объекта (если есть), иначе body.
   * @param {BatchImageItem} im
   */
  function getTrainClassForNewBbox(im) {
    const sid = im.panel.selectedDetectionId;
    if (sid != null) {
      const d = im.detections.find((x) => x.id === sid);
      if (d) {
        const byName = TRAIN_CLASSES.find((t) => t.name === d.cls_name);
        if (byName) return byName;
        const byId = TRAIN_CLASSES.find((t) => t.id === d.cls_id);
        if (byId) return byId;
      }
    }
    return TRAIN_CLASSES[0];
  }

  function tryDeleteSelectedDetection() {
    if (!editorModeIsEdit()) return false;
    const im = currentImageItem();
    if (!im) return false;
    const sid = im.panel.selectedDetectionId;
    if (sid == null) return false;
    const before = im.detections.length;
    im.detections = im.detections.filter((d) => d.id !== sid);
    if (im.detections.length === before) return false;

    im.panel.selectedDetectionId = null;
    im.edited = true;
    im.reviewed = false;
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

    const tc = getTrainClassForNewBbox(im);
    const newDet = {
      id: allocateNextDetId(im),
      cls_id: tc.id,
      cls_name: tc.name,
      conf: 1,
      box,
      source: /** @type {"human"} */ ("human"),
    };
    im.detections.push(newDet);
    im.panel.selectedDetectionId = newDet.id;
    im.status = /** @type {ImageStatus} */ ("detected");
    im.edited = true;
    im.reviewed = false;
    /* После добавления возвращаем Select: проще сразу двигать/менять класс нового bbox без второго случайного drag. */
    batchState.settings.editorTool = "select";

    touchBatch();
    scheduleWorkspaceAutosave(0);
    syncEditorChrome();
    updateBatchNavUi();
    buildRightPanel();
  }

  function refreshOverlayCursorHint(ox, oy) {
    overlay.classList.remove(
      "overlay-cursor-crosshair",
      "overlay-cursor-move",
      "overlay-cursor-nwse",
      "overlay-cursor-nesw",
      "overlay-cursor-ns",
      "overlay-cursor-ew"
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

    if (batchState.settings.editorTool === "addBox") {
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
    if (review || batchState.settings.editorTool !== "addBox") {
      clearAddBoxCrosshairOverlayPx();
    }
    editorModeReviewBtn.classList.toggle("is-active", review);
    editorModeEditBtn.classList.toggle("is-active", !review);
    editorModeReviewBtn.setAttribute("aria-checked", review ? "true" : "false");
    editorModeEditBtn.setAttribute("aria-checked", !review ? "true" : "false");
    editorToolsBar.hidden = review;

    const selTool = batchState.settings.editorTool === "select";
    editorToolSelectBtn.classList.toggle("is-active", selTool);
    editorToolAddBtn.classList.toggle("is-active", !selTool);

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
      drawOneBox(d, isSel);
    }

    if (
      editorModeIsEdit() &&
      batchState.settings.editorTool === "addBox" &&
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
      pointerInteraction.kind === "addBox" &&
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
      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = "rgba(63, 130, 247, 0.95)";
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, Math.max(w, 1), Math.max(h, 1));
      ctx.setLineDash([]);
    }
  }

  function onOverlayPointerDown(e) {
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

    if (batchState.settings.editorTool === "addBox") {
      const { ix, iy } = canvasToImageCoords(geo, ox, oy);
      pointerInteraction = {
        kind: "addBox",
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

    const imCur = currentImageItem();
    if (!imCur) return;

    const sid = imCur.panel.selectedDetectionId;
    const selDet =
      sid != null ? currentDetections().find((d) => d.id === sid) : null;

    if (selDet && detectionPassesUiFilters(selDet, imCur)) {
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
        e.preventDefault();
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
        e.preventDefault();
        return;
      }
    }

    const hit = hitTestDetectionAtOverlay(geo, ox, oy);
    setSelectedDetectionId(hit ? hit.id : null);
    touchBatch();
    buildRightPanel();
    draw();
  }

  function onDocumentPointerMove(e) {
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

    if (pointerInteraction.kind === "addBox") {
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
    if (!pointerInteraction) return;

    if (pointerInteraction.kind === "addBox") {
      finalizeAddBoxInteraction();
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

  function downloadAnnotatedImage() {
    const originalSize = effectiveOriginalSize();
    const imCur = currentImageItem();
    if (!originalSize || !previewImage.src) {
      alert("Нет изображения или размеров для сохранения.");
      return;
    }
    if (imCur?.status === "skipped") {
      alert("Пропущенные изображения не экспортируются.");
      return;
    }
    if (!imCur?.detections?.length) {
      alert(
        "На этом кадре нет bbox. Пустое изображение без разметки не экспортируется."
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
      ctx.strokeRect(x, y, w, h);

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

    const stem = imCur?.displayName.replace(/\.[^.]+$/, "") || "annotated";

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

  function downloadYoloAnnotations() {
    const originalSize = effectiveOriginalSize();
    const imCur = currentImageItem();
    if (!originalSize) {
      alert("Нет размеров изображения для экспорта.");
      return;
    }
    if (imCur?.status === "skipped") {
      alert("Пропущенные изображения не экспортируются.");
      return;
    }
    if (!imCur?.detections?.length) {
      alert(
        "На этом кадре нет bbox. Пустое изображение без разметки не экспортируется."
      );
      return;
    }
    const detections = currentDetections();

    const lines = [];
    const im = imCur;
    const categoryState = im?.panel.categoryState ?? new Map();
    const detEnabled = im?.panel.detEnabled ?? new Map();

    for (const d of detections) {
      if (d.conf < confThreshold()) continue;
      if (classHidden(d.cls_name)) continue;

      const cat = d.cls_name;
      const catEnabled = categoryState.get(cat)?.enabled ?? true;
      if (!catEnabled) continue;
      if (detEnabled.get(d.id) === false) continue;

      const [x1, y1, x2, y2] = d.box;
      const cx = (x1 + x2) / 2;
      const cy = (y1 + y2) / 2;
      const w = x2 - x1;
      const h = y2 - y1;

      const nx = cx / originalSize.width;
      const ny = cy / originalSize.height;
      const nw = w / originalSize.width;
      const nh = h / originalSize.height;

      const clsId = d.cls_id ?? 0;
      lines.push(
        [clsId, nx.toFixed(6), ny.toFixed(6), nw.toFixed(6), nh.toFixed(6)].join(
          " "
        )
      );
    }

    if (!lines.length) {
      alert("Все боксы выключены. Включите нужные объекты перед экспортом.");
      return;
    }

    const blob = new Blob([lines.join("\n") + "\n"], {
      type: "text/plain;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);

    const stem = im?.displayName.replace(/\.[^.]+$/, "") || "annotations";

    const link = document.createElement("a");
    link.href = url;
    link.download = `${stem}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    showToast("Экспорт завершён: файл YOLO .txt сохранён.", {
      type: "success",
    });
  }

  function downloadJson() {
    const originalSize = effectiveOriginalSize();
    const imCur = currentImageItem();
    if (!originalSize) {
      alert("Нет размеров изображения для экспорта.");
      return;
    }
    if (imCur?.status === "skipped") {
      alert("Пропущенные изображения не экспортируются.");
      return;
    }
    if (!imCur?.detections?.length) {
      alert(
        "На этом кадре нет bbox. Пустое изображение без разметки не экспортируется."
      );
      return;
    }
    const detections = currentDetections();

    const width = originalSize.width;
    const height = originalSize.height;

    /** @type {Array<any>} */
    const finalDetections = [];
    const im = imCur;
    const categoryState = im?.panel.categoryState ?? new Map();
    const detEnabled = im?.panel.detEnabled ?? new Map();

    for (const d of detections) {
      if (d.conf < confThreshold()) continue;
      if (classHidden(d.cls_name)) continue;

      const cat = d.cls_name;
      const catState = categoryState.get(cat) ?? { enabled: true, collapsed: false };
      const enabled = catState.enabled && detEnabled.get(d.id) !== false;

      const [x1, y1, x2, y2] = d.box;
      const cx = (x1 + x2) / 2;
      const cy = (y1 + y2) / 2;
      const w = x2 - x1;
      const h = y2 - y1;

      finalDetections.push({
        id: d.id,
        class_id: d.cls_id,
        class_name: d.cls_name,
        confidence: d.conf,
        source: detectionSourceLabel(d),
        enabled,
        bbox: {
          x1,
          y1,
          x2,
          y2,
          cx,
          cy,
          w,
          h,
        },
        bbox_normalized: {
          x1: x1 / width,
          y1: y1 / height,
          x2: x2 / width,
          y2: y2 / height,
          cx: cx / width,
          cy: cy / height,
          w: w / width,
          h: h / height,
        },
      });
    }

    if (!finalDetections.length) {
      alert("Все боксы выключены или ниже порога. Включите нужные объекты перед экспортом.");
      return;
    }

    const categories = {};
    for (const [name, state] of categoryState.entries()) {
      categories[name] = {
        enabled: state.enabled,
        collapsed: state.collapsed,
      };
    }

    const stem = im?.displayName.replace(/\.[^.]+$/, "") || "annotations";

    const payload = {
      batch: {
        batchId: batchState.batchId,
        currentIndex: batchState.currentIndex,
        workspaceSettings: { ...batchState.settings },
      },
      image: {
        id: im?.id ?? null,
        display_name: im?.displayName ?? null,
        original_name: im?.originalName ?? null,
        width,
        height,
      },
      settings: {
        confidence_threshold: confThreshold(),
      },
      summary: {
        total_detections: detections.length,
        total_after_filter: finalDetections.length,
      },
      categories,
      detections: finalDetections,
    };

    const blob = new Blob([JSON.stringify(payload, null, 2)], {
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
    showToast("Экспорт завершён: JSON сохранён.", { type: "success" });
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

  function buildInspector() {
    inspectorRoot.innerHTML = "";
    const panel = document.createElement("div");
    panel.className = "inspector-panel";

    const h = document.createElement("h4");
    h.className = "inspector-heading";
    h.textContent = "Inspector";
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

    const sid = im.panel.selectedDetectionId;
    const det =
      sid != null ? (im.detections.find((d) => d.id === sid) ?? null) : null;

    if (!det) {
      inspectorAppendRow(panel, "Изображение", im.displayName || "—");
      inspectorAppendRow(panel, "Статус", im.status);
      inspectorAppendRow(panel, "Reviewed", im.reviewed ? "да" : "нет");
      inspectorAppendRow(panel, "Edited", im.edited ? "да" : "нет");
      const wh =
        im.width > 0 && im.height > 0 ? `${im.width} × ${im.height}` : "—";
      inspectorAppendRow(panel, "Размер", wh);
      inspectorAppendRow(panel, "Bbox", String(im.detections?.length ?? 0));
      if (im.status === "failed" && im.error) {
        const err = document.createElement("div");
        err.className = "inspector-error";
        err.textContent = im.error;
        panel.appendChild(err);
      }
    } else {
      inspectorAppendRow(panel, "Класс", det.cls_name);
      inspectorAppendRow(panel, "Confidence", fmtConf(det.conf));
      inspectorAppendRow(panel, "Source", detectionSourceLabel(det));
      inspectorAppendRow(panel, "Box", fmtBoxCoords(det.box));

      const actions = document.createElement("div");
      actions.className = "inspector-actions";

      const sel = document.createElement("select");
      sel.className = "inspector-class-select";
      sel.title = "Класс объекта";
      sel.setAttribute("aria-label", "Класс bbox");
      for (const tc of TRAIN_CLASSES) {
        const opt = document.createElement("option");
        opt.value = tc.name;
        opt.textContent = tc.name;
        sel.appendChild(opt);
      }
      sel.value = TRAIN_CLASSES.some((t) => t.name === det.cls_name)
        ? det.cls_name
        : TRAIN_CLASSES[0].name;
      sel.disabled = !editorModeIsEdit();
      sel.addEventListener("change", () => {
        const tc = TRAIN_CLASSES.find((t) => t.name === sel.value);
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
    totalObjectsEl.textContent = String(filtered.length);

    const grouped = groupByCategory(filtered);
    const cats = Array.from(grouped.keys()).sort((a, b) => a.localeCompare(b));

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
      catCheckbox.checked = st.enabled;
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

      left.appendChild(catCheckbox);
      left.appendChild(title);
      summary.appendChild(left);
      summary.appendChild(count);

      const items = document.createElement("div");
      items.className = "group-items";

      for (const d of grouped.get(cat)) {
        if (!detEnabled.has(d.id)) detEnabled.set(d.id, true);

        const row = document.createElement("div");
        row.className = "det-row";
        if (im.panel.selectedDetectionId === d.id) row.classList.add("is-selected");

        const cb = document.createElement("input");
        cb.type = "checkbox";
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
          for (const tc of TRAIN_CLASSES) {
            const opt = document.createElement("option");
            opt.value = tc.name;
            opt.textContent = tc.name;
            sel.appendChild(opt);
          }
          sel.value = TRAIN_CLASSES.some((t) => t.name === d.cls_name)
            ? d.cls_name
            : TRAIN_CLASSES[0].name;
          sel.addEventListener("click", (ev) => ev.stopPropagation());
          sel.addEventListener("change", () => {
            const tc = TRAIN_CLASSES.find((t) => t.name === sel.value);
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
    for (const tc of TRAIN_CLASSES) {
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

  viewerPrev.addEventListener("click", () => navigateBatch(-1));
  viewerNext.addEventListener("click", () => navigateBatch(1));

  markReviewedBtn.addEventListener("click", () => {
    const im = currentImageItem();
    if (!im || im.reviewed) return;
    const idx = batchState.currentIndex;
    im.reviewed = true;
    touchBatch();
    scheduleWorkspaceAutosave(0);
    updateBatchNavUi();
    const next = findNextUnreviewedNonSkippedIndex(idx);
    if (next !== null) selectBatchIndex(next);
  });

  skipImageBtn.addEventListener("click", () => {
    skipCurrentImage();
  });

  document.addEventListener("keydown", (e) => {
    if (e.defaultPrevented) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (isTypingInteractionTarget(e.target)) return;

    if (e.key === "Delete") {
      if (editorModeIsEdit() && tryDeleteSelectedDetection()) {
        e.preventDefault();
      }
      return;
    }

    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    if (!batchState.images.length) return;
    e.preventDefault();
    if (e.key === "ArrowLeft") navigateBatch(-1);
    else navigateBatch(1);
  });

  overlay.addEventListener("mousedown", onOverlayPointerDown);
  overlay.addEventListener("mousemove", (e) => {
    refreshOverlayCursorHint(e.offsetX, e.offsetY);
    if (editorModeIsEdit() && batchState.settings.editorTool === "addBox") {
      syncAddBoxCrosshairOverlayPx(e.offsetX, e.offsetY);
      draw();
    }
  });
  overlay.addEventListener("mouseleave", () => {
    refreshOverlayCursorHint(null, null);
    if (!(pointerInteraction && pointerInteraction.kind === "addBox")) {
      clearAddBoxCrosshairOverlayPx();
      draw();
    }
  });
  document.addEventListener("mousemove", onDocumentPointerMove);
  document.addEventListener("mouseup", onDocumentPointerUp);

  editorModeReviewBtn.addEventListener("click", () => {
    batchState.settings.editorMode = "review";
    pointerInteraction = null;
    touchBatch();
    syncEditorChrome();
    buildRightPanel();
    draw();
  });

  editorModeEditBtn.addEventListener("click", () => {
    batchState.settings.editorMode = "edit";
    pointerInteraction = null;
    touchBatch();
    syncEditorChrome();
    buildRightPanel();
    draw();
  });

  editorToolSelectBtn.addEventListener("click", () => {
    batchState.settings.editorTool = "select";
    pointerInteraction = null;
    touchBatch();
    syncEditorChrome();
    draw();
  });

  editorToolAddBtn.addEventListener("click", () => {
    batchState.settings.editorTool = "addBox";
    pointerInteraction = null;
    touchBatch();
    syncEditorChrome();
    draw();
  });

  fileInput.addEventListener("change", (event) => {
    const list = event.target.files;
    if (!list || !list.length) return;
    const files = Array.from(list);
    const zipFiles = files.filter((file) => isZipFile(file));
    if (zipFiles.length) {
      if (zipFiles.length > 1 || files.length > 1) {
        showToast(
          "Импортируется первый ZIP; остальные выбранные файлы игнорируются.",
          { type: "info", durationMs: 3800 }
        );
      }
      void ingestZipFile(zipFiles[0]);
      return;
    }
    ingestImageFiles(list);
  });

  batchStatusFilter.addEventListener("change", () => {
    onBatchListFiltersChanged();
  });

  clearBtn.addEventListener("click", () => clearAll(true));

  exportMenuToggle.addEventListener("click", (e) => {
    e.stopPropagation();
    const willOpen = exportMenuPanel.hidden;
    exportMenuPanel.hidden = !willOpen;
    exportMenuToggle.setAttribute(
      "aria-expanded",
      willOpen ? "true" : "false"
    );
  });

  document.addEventListener("click", () => {
    closeExportMenu();
  });

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

  /**
   * Очередь детекта по всему batch (до двух параллельных POST на `/api/detect`).
   */
  async function detectAllImages() {
    if (!batchState.images.length) {
      showToast("Нет изображений. Сначала загрузите файлы.", {
        type: "warning",
      });
      return;
    }

    if (detectAllInFlight) return;

    /** @type {number[]} */
    const indices = [];
    for (let i = 0; i < batchState.images.length; i++) {
      if (isEligibleForDetectAll(batchState.images[i])) indices.push(i);
    }

    if (!indices.length) {
      showToast(
        "Нет изображений для очереди (idle, ошибка, пусто или повтор распознанных; пропускаются skipped и уже queued/processing).",
        { type: "info", durationMs: 4500 }
      );
      return;
    }

    detectAllInFlight = true;
    runBtn.disabled = true;
    setStatus("Распознаём…");

    const total = indices.length;
    let completed = 0;

    for (const idx of indices) {
      const item = batchState.images[idx];
      item.status = "queued";
      item.error = null;
    }
    touchBatch();
    updateBatchNavUi();

    let nextSlot = 0;

    async function worker() {
      while (true) {
        const slot = nextSlot++;
        if (slot >= indices.length) break;

        const idx = indices[slot];
        const im = batchState.images[idx];

        if (!im?.blob || im.blob.size === 0) {
          if (im) applyDetectFailureToItem(im, "Нет данных изображения для распознавания.");
          completed++;
          touchBatch();
          refreshUiAfterImageProcessed(idx);
          await yieldToMain();
          continue;
        }

        im.status = "processing";
        touchBatch();
        refreshUiAfterImageProcessed(idx, false);
        await yieldToMain();

        try {
          const data = await fetchDetectApi(im);
          applyDetectSuccessFromResponse(im, data);
        } catch (e) {
          applyDetectFailureToItem(im, String(e?.message || e));
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
      runBtn.disabled = false;
      setStatus("Запустить распознавание");
      scheduleWorkspaceAutosave(0);
      updateBatchNavUi();
      showToast(`Распознавание завершено: обработано ${completed} из ${total}.`, {
        type: "success",
        durationMs: 3400,
      });
    }
  }

  runBtn.addEventListener("click", () => {
    void detectAllImages();
  });

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
        return;
      }
    } catch (err) {
      console.warn("[workspace autosave] restore failed:", err);
      setWorkspaceSaveStatus("failed");
    }
    clearAll(false);
  }

  void initializeWorkspace();
})();
