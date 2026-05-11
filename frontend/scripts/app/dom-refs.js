/**
 * Ссылки на DOM приложения (после проверки — все non-null).
 * @typedef {{
 *   uploadBtn: HTMLButtonElement,
 *   clearBtn: HTMLButtonElement,
 *   runBtn: HTMLButtonElement,
 *   fileInput: HTMLInputElement,
 *   previewImage: HTMLImageElement,
 *   placeholderText: HTMLElement,
 *   imageFrame: HTMLElement,
 *   imageStack: HTMLElement,
 *   overlay: HTMLCanvasElement,
 *   viewerPrev: HTMLButtonElement,
 *   viewerNext: HTMLButtonElement,
 *   viewerCounter: HTMLElement,
 *   markReviewedBtn: HTMLButtonElement,
 *   skipImageBtn: HTMLButtonElement,
 *   headerReviewedLine: HTMLElement,
 *   workspaceSaveStatus: HTMLElement,
 *   headerProcessedLine: HTMLElement,
 *   headerProgressTrack: HTMLElement,
 *   headerProgressFill: HTMLElement,
 *   headerFailedRow: HTMLElement,
 *   headerFailedCount: HTMLElement,
 *   exportDropdownWrap: HTMLElement,
 *   exportMenuToggle: HTMLButtonElement,
 *   exportMenuPanel: HTMLElement,
 *   exportSummaryOverlay: HTMLElement,
 *   exportSummaryTitle: HTMLElement,
 *   exportSummarySubtitle: HTMLElement,
 *   exportSummaryBody: HTMLElement,
 *   exportSummaryConfirm: HTMLButtonElement,
 *   exportSummaryCancel: HTMLButtonElement,
 *   importSummaryOverlay: HTMLElement,
 *   importSummaryTitle: HTMLElement,
 *   importSummaryBody: HTMLElement,
 *   importSummaryClose: HTMLButtonElement,
 *   viewerFilename: HTMLElement,
 *   batchImageListRoot: HTMLElement,
 *   batchStatusFilter: HTMLSelectElement,
 *   totalObjectsEl: HTMLElement,
 *   groupsRoot: HTMLElement,
 *   inspectorRoot: HTMLElement,
 *   confFilterRange: HTMLInputElement,
 *   confFilterValue: HTMLElement,
 *   editorModeReviewBtn: HTMLButtonElement,
 *   editorModeEditBtn: HTMLButtonElement,
 *   editorToolsBar: HTMLElement,
 *   editorToolSelectBtn: HTMLButtonElement,
 *   editorToolAddBtn: HTMLButtonElement,
 *   classChipsRoot: HTMLElement,
 * }} AppDomRefs
 */

/** @returns {AppDomRefs|null} */
export function collectDomRefs() {
  const uploadBtn = document.getElementById("upload-btn");
  const clearBtn = document.getElementById("clear-btn");
  const runBtn = document.getElementById("run-btn");
  const fileInput = document.getElementById("file-input");
  const previewImage = document.getElementById("preview-image");
  const placeholderText = document.getElementById("placeholder-text");
  const imageFrame = document.getElementById("image-frame");
  const imageStack = document.getElementById("image-stack");
  const overlay = document.getElementById("overlay-canvas");
  const viewerPrev = document.getElementById("viewer-prev");
  const viewerNext = document.getElementById("viewer-next");
  const viewerCounter = document.getElementById("viewer-counter");
  const markReviewedBtn = document.getElementById("mark-reviewed-btn");
  const skipImageBtn = document.getElementById("skip-image-btn");
  const headerReviewedLine = document.getElementById("header-reviewed-line");
  const workspaceSaveStatus = document.getElementById("workspace-save-status");
  const headerProcessedLine = document.getElementById("header-processed-line");
  const headerProgressTrack = document.getElementById("header-progress-track");
  const headerProgressFill = document.getElementById("header-progress-fill");
  const headerFailedRow = document.getElementById("header-failed-row");
  const headerFailedCount = document.getElementById("header-failed-count");
  const exportDropdownWrap = document.getElementById("export-dropdown-wrap");
  const exportMenuToggle = document.getElementById("export-menu-toggle");
  const exportMenuPanel = document.getElementById("export-menu-panel");
  const exportSummaryOverlay = document.getElementById("export-summary-overlay");
  const exportSummaryTitle = document.getElementById("export-summary-title");
  const exportSummarySubtitle = document.getElementById("export-summary-subtitle");
  const exportSummaryBody = document.getElementById("export-summary-body");
  const exportSummaryConfirm = document.getElementById("export-summary-confirm");
  const exportSummaryCancel = document.getElementById("export-summary-cancel");
  const importSummaryOverlay = document.getElementById("import-summary-overlay");
  const importSummaryTitle = document.getElementById("import-summary-title");
  const importSummaryBody = document.getElementById("import-summary-body");
  const importSummaryClose = document.getElementById("import-summary-close");
  const viewerFilename = document.getElementById("viewer-filename");
  const batchImageListRoot = document.getElementById("batch-image-list-root");
  const batchStatusFilter = document.getElementById("batch-status-filter");
  const totalObjectsEl = document.getElementById("total-objects");
  const groupsRoot = document.getElementById("groups-root");
  const inspectorRoot = document.getElementById("inspector-root");
  const confFilterRange = document.getElementById("confidence-filter-range");
  const confFilterValue = document.getElementById("confidence-filter-value");
  const editorModeReviewBtn = document.getElementById("editor-mode-review");
  const editorModeEditBtn = document.getElementById("editor-mode-edit");
  const editorToolsBar = document.getElementById("editor-tools-bar");
  const editorToolSelectBtn = document.getElementById("editor-tool-select");
  const editorToolAddBtn = document.getElementById("editor-tool-add");
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
    return null;
  }

  return {
    uploadBtn,
    clearBtn,
    runBtn,
    fileInput,
    previewImage,
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
    exportDropdownWrap,
    exportMenuToggle,
    exportMenuPanel,
    exportSummaryOverlay,
    exportSummaryTitle,
    exportSummarySubtitle,
    exportSummaryBody,
    exportSummaryConfirm,
    exportSummaryCancel,
    importSummaryOverlay,
    importSummaryTitle,
    importSummaryBody,
    importSummaryClose,
    viewerFilename,
    batchImageListRoot,
    batchStatusFilter,
    totalObjectsEl,
    groupsRoot,
    inspectorRoot,
    confFilterRange,
    confFilterValue,
    editorModeReviewBtn,
    editorModeEditBtn,
    editorToolsBar,
    editorToolSelectBtn,
    editorToolAddBtn,
    classChipsRoot,
  };
}
