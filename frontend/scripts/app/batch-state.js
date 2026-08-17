import { MAX_DIRECT_IMAGES } from "./constants.js";
import { newBatchId } from "./ids.js";
import { trainClassesForMode } from "./train-classes.js";

/** @returns {any} */
export function createEmptyBatchState() {
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
      trainClassOrder: trainClassesForMode("detection").map((tc) => tc.name),
      classOrders: {
        detect: trainClassesForMode("detection").map((tc) => tc.name),
        seg: trainClassesForMode("segmentation").map((tc) => tc.name),
      },
      editorMode: "review",
      /** "select" | "addBox" | "crop" | "addPolygon" */
      editorTool: "select",
      /** Класс следующего полигона сегментации. */
      lastPolygonClassName: trainClassesForMode("segmentation")[0]?.name ?? "body",
      /** Последний пункт меню «Распознать»; основная кнопка запускает обе модели. */
      runSelection: "both",
      /** Legacy: нужен только для миграции старых проектов. */
      inferenceMode: "detection",
    },
  };
}
