import {
  WORKSPACE_DB_NAME,
  WORKSPACE_DB_VERSION,
  WORKSPACE_STORE_NAME,
  WORKSPACE_AUTOSAVE_DELAY_MS,
} from "./constants.js";

import {
  annotationSourceOf,
  annotationTypeOf,
  isDetectAnnotation,
  isSegAnnotation,
} from "./annotation-model.js";
/** @type {Promise<IDBDatabase>|null} */
let workspaceDbPromise = null;

/** @returns {Promise<IDBDatabase>} */
export function openWorkspaceDb() {
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
export function waitForTransaction(tx) {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve(undefined);
    tx.onabort = () => reject(tx.error || new Error("IndexedDB tx aborted"));
    tx.onerror = () => reject(tx.error || new Error("IndexedDB tx failed"));
  });
}

/** @param {IDBRequest} req */
export function waitForRequest(req) {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error || new Error("IndexedDB request failed"));
  });
}

/** @param {{ categoryState: Map<string, { enabled: boolean; collapsed: boolean }>, detEnabled: Map<number, boolean>, selectedDetectionId: number|null }} panel */
export function serializePanelState(panel) {
  return {
    categoryState: Array.from(panel.categoryState.entries()),
    detEnabled: Array.from(panel.detEnabled.entries()),
    selectedDetectionId: panel.selectedDetectionId,
  };
}

/** @param {any} im */
export function serializeImageItem(im) {
  const serializeAnnotation = (d) => ({
    id: d.id,
    cls_id: d.cls_id,
    cls_name: d.cls_name,
    conf: d.conf,
    box: [...d.box],
    segment: Array.isArray(d.segment) ? d.segment.map((p) => [...p]) : undefined,
    annotation_type: annotationTypeOf(d),
    source: annotationSourceOf(d),
  });
  const annotations = Array.isArray(im.detections) ? im.detections : [];
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
    category: typeof im.category === "string" ? im.category : null,
    reviewed: im.reviewed,
    edited: im.edited,
    /** Bbox и маски намеренно сохраняются раздельно. */
    detections: annotations.filter(isDetectAnnotation).map(serializeAnnotation),
    masks: annotations.filter(isSegAnnotation).map(serializeAnnotation),
    modelStates: {
      detect: { ...(im.modelStates?.detect || {}) },
      seg: { ...(im.modelStates?.seg || {}) },
    },
    error: im.error,
    panel: serializePanelState(im.panel),
  };
}

/** @param {any} batchState */
export function createWorkspaceSnapshot(batchState) {
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

export async function loadLatestWorkspaceSnapshot() {
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

/** @param {ReturnType<typeof createWorkspaceSnapshot>} snapshot */
export async function saveWorkspaceSnapshot(snapshot) {
  const db = await openWorkspaceDb();
  const tx = db.transaction(WORKSPACE_STORE_NAME, "readwrite");
  tx.objectStore(WORKSPACE_STORE_NAME).put(snapshot);
  await waitForTransaction(tx);
}

/**
 * @param {{
 *   showToast: (msg: string, opts?: object) => void,
 *   setWorkspaceSaveStatus: (state: "saving"|"saved"|"failed"|"") => void,
 *   getSnapshot: () => ReturnType<typeof createWorkspaceSnapshot>,
 * }} cfg
 */
export function createWorkspaceAutosave(cfg) {
  /** @type {number|null} */
  let workspaceAutosaveTimer = null;
  /** Не даём нескольким put() в IndexedDB идти параллельно — каждый клонирует весь снимок с blob’ами. */
  let saveInFlight = false;
  /** Пока идёт сохранение, пришёл ещё один запрос — пересохраним сразу после текущего одним проходом. */
  let saveAgainAfterCurrent = false;

  function runWorkspaceAutosave() {
    if (saveInFlight) {
      saveAgainAfterCurrent = true;
      return;
    }
    saveInFlight = true;
    cfg.setWorkspaceSaveStatus("saving");

    const pump = () => {
      saveAgainAfterCurrent = false;
      saveWorkspaceSnapshot(cfg.getSnapshot())
        .then(() => {
          if (saveAgainAfterCurrent) {
            pump();
            return;
          }
          saveInFlight = false;
          cfg.setWorkspaceSaveStatus("saved");
        })
        .catch((err) => {
          console.warn("[workspace autosave] failed:", err);
          saveInFlight = false;
          saveAgainAfterCurrent = false;
          cfg.setWorkspaceSaveStatus("failed");
          cfg.showToast(
            "Автосохранение не удалось. Проверьте место на диске и доступ к IndexedDB.",
            { type: "error", durationMs: 5200 }
          );
        });
    };

    pump();
  }

  /** @param {number} [delayMs] */
  function scheduleWorkspaceAutosave(delayMs = WORKSPACE_AUTOSAVE_DELAY_MS) {
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

  return { scheduleWorkspaceAutosave };
}
