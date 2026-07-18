/**
 * Ссылки на DOM приложения (после проверки — все non-null).
 * @typedef {{
 *   uploadBtn: HTMLButtonElement,
 *   uploadVideoBtn: HTMLButtonElement,
 *   clearBtn: HTMLButtonElement,
 *   runBtn: HTMLButtonElement,
 *   runMenuToggle: HTMLButtonElement,
 *   runMenu: HTMLElement,
 *   fileInput: HTMLInputElement,
 *   videoFileInput: HTMLInputElement,
 *   previewImage: HTMLImageElement,
 *   previewVideo: HTMLVideoElement,
 *   placeholderText: HTMLElement,
 *   imageFrame: HTMLElement,
 *   imageStack: HTMLElement,
 *   overlay: HTMLCanvasElement,
 *   viewerPrev: HTMLButtonElement,
 *   viewerNext: HTMLButtonElement,
 *   viewerCounter: HTMLElement,
 *   markReviewedBtn: HTMLButtonElement,
 *   unmarkReviewedBtn: HTMLButtonElement,
 *   skipImageBtn: HTMLButtonElement,
 *   headerReviewedLine: HTMLElement,
 *   workspaceSaveStatus: HTMLElement,
 *   headerProcessedLine: HTMLElement,
 *   headerProgressTrack: HTMLElement,
 *   headerProgressFill: HTMLElement,
 *   headerFailedRow: HTMLElement,
 *   headerFailedCount: HTMLElement,
 *   workModeDetectionBtn: HTMLButtonElement,
 *   workModeSegmentationBtn: HTMLButtonElement,
 *   exportDropdownWrap: HTMLElement,
 *   exportMenuToggle: HTMLButtonElement,
 *   exportMenuPanel: HTMLElement,
 *   exportSummaryOverlay: HTMLElement,
 *   exportSummaryIcon: HTMLElement,
 *   exportSummaryScope: HTMLElement,
 *   exportSummaryTitle: HTMLElement,
 *   exportSummarySubtitle: HTMLElement,
 *   exportSummaryBody: HTMLElement,
 *   exportNamingRow: HTMLElement,
 *   exportStartNumber: HTMLInputElement,
 *   exportNamingPreview: HTMLElement,
 *   exportCategoryNumbering: HTMLElement,
 *   exportReviewFilter: HTMLElement,
 *   exportReviewModeDetected: HTMLInputElement,
 *   exportReviewModeManual: HTMLInputElement,
 *   exportReviewIncludeUnreviewed: HTMLInputElement,
 *   exportReviewFilterHint: HTMLElement,
 *   exportYoloOptions: HTMLElement,
 *   exportYoloEmptyLabels: HTMLInputElement,
 *   exportYoloOptionsHint: HTMLElement,
 *   exportProjectOptions: HTMLElement,
 *   exportProjectIncludeAnnotations: HTMLInputElement,
 *   exportProjectOptionsHint: HTMLElement,
 *   exportSummaryConfirm: HTMLButtonElement,
 *   exportSummaryCancel: HTMLButtonElement,
 *   importSummaryOverlay: HTMLElement,
 *   importSummaryTitle: HTMLElement,
 *   importSummaryBody: HTMLElement,
 *   importSummaryClose: HTMLButtonElement,
 *   viewerFilename: HTMLElement,
 *   videoToolbar: HTMLElement,
 *   videoFrameInterval: HTMLSelectElement,
 *   videoBlurEvery: HTMLInputElement,
 *   videoExtractBtn: HTMLButtonElement,
 *   videoShowBtn: HTMLButtonElement,
 *   batchImageListRoot: HTMLElement,
 *   batchStatusFilter: HTMLSelectElement,
 *   batchSortToggle: HTMLButtonElement,
 *   batchSortMenu: HTMLElement,
 *   groupsRoot: HTMLElement,
 *   inspectorRoot: HTMLElement,
 *   confFilterRange: HTMLInputElement,
 *   confFilterValue: HTMLElement,
 *   editorModeReviewBtn: HTMLButtonElement,
 *   editorModeEditBtn: HTMLButtonElement,
 *   editorToolsBar: HTMLElement,
 *   editorToolSelectBtn: HTMLButtonElement,
 *   editorToolAddBtn: HTMLButtonElement,
 *   editorToolCropBtn: HTMLButtonElement,
 *   editorToolAddPolygonBtn: HTMLButtonElement,
 *   hotkeysHelpBtn: HTMLButtonElement,
 *   hotkeysHelpOverlay: HTMLElement,
 *   hotkeysHelpClose: HTMLButtonElement,
 *   categoryModalOverlay: HTMLElement,
 *   categoryModalList: HTMLElement,
 *   categoryModalClose: HTMLButtonElement,
 *   categoryClearBtn: HTMLButtonElement,
 *   categoryAddBtn: HTMLButtonElement,
 *   categoryNewInput: HTMLInputElement,
 *   classChipsRoot: HTMLElement,
 *   maskContextMenu: HTMLElement,
 * }} AppDomRefs
 */

/** @returns {AppDomRefs|null} */
export function collectDomRefs() {
  const uploadBtn = document.getElementById("upload-btn");
  const uploadVideoBtn = document.getElementById("upload-video-btn");
  const clearBtn = document.getElementById("clear-btn");
  const runBtn = document.getElementById("run-btn");
  const runMenuToggle = document.getElementById("run-menu-toggle");
  const runMenu = document.getElementById("run-menu");
  const fileInput = document.getElementById("file-input");
  const videoFileInput = document.getElementById("video-file-input");
  const previewImage = document.getElementById("preview-image");
  const previewVideo = document.getElementById("preview-video");
  const placeholderText = document.getElementById("placeholder-text");
  const imageFrame = document.getElementById("image-frame");
  const imageStack = document.getElementById("image-stack");
  const overlay = document.getElementById("overlay-canvas");
  const viewerPrev = document.getElementById("viewer-prev");
  const viewerNext = document.getElementById("viewer-next");
  const viewerCounter = document.getElementById("viewer-counter");
  const markReviewedBtn = document.getElementById("mark-reviewed-btn");
  const unmarkReviewedBtn = document.getElementById("unmark-reviewed-btn");
  const skipImageBtn = document.getElementById("skip-image-btn");
  const headerReviewedLine = document.getElementById("header-reviewed-line");
  const workspaceSaveStatus = document.getElementById("workspace-save-status");
  const headerProcessedLine = document.getElementById("header-processed-line");
  const headerProgressTrack = document.getElementById("header-progress-track");
  const headerProgressFill = document.getElementById("header-progress-fill");
  const headerFailedRow = document.getElementById("header-failed-row");
  const headerFailedCount = document.getElementById("header-failed-count");
  const workModeDetectionBtn = document.getElementById("work-mode-detection");
  const workModeSegmentationBtn = document.getElementById("work-mode-segmentation");
  const exportDropdownWrap = document.getElementById("export-dropdown-wrap");
  const exportMenuToggle = document.getElementById("export-menu-toggle");
  const exportMenuPanel = document.getElementById("export-menu-panel");
  const exportSummaryOverlay = document.getElementById("export-summary-overlay");
  const exportSummaryIcon = document.getElementById("export-summary-icon");
  const exportSummaryScope = document.getElementById("export-summary-scope");
  const exportSummaryTitle = document.getElementById("export-summary-title");
  const exportSummarySubtitle = document.getElementById("export-summary-subtitle");
  const exportSummaryBody = document.getElementById("export-summary-body");
  const exportNamingRow = document.getElementById("export-naming-row");
  const exportStartNumber = document.getElementById("export-start-number");
  const exportNamingPreview = document.getElementById("export-naming-preview");
  const exportCategoryNumbering = document.getElementById("export-category-numbering");
  const exportReviewFilter = document.getElementById("export-review-filter");
  const exportReviewModeDetected = document.getElementById("export-review-mode-detected");
  const exportReviewModeManual = document.getElementById("export-review-mode-manual");
  const exportReviewIncludeUnreviewed = document.getElementById(
    "export-review-include-unreviewed"
  );
  const exportReviewFilterHint = document.getElementById("export-review-filter-hint");
  const exportYoloOptions = document.getElementById("export-yolo-options");
  const exportYoloEmptyLabels = document.getElementById("export-yolo-empty-labels");
  const exportYoloOptionsHint = document.getElementById("export-yolo-options-hint");
  const exportProjectOptions = document.getElementById("export-project-options");
  const exportProjectIncludeAnnotations = document.getElementById(
    "export-project-include-annotations"
  );
  const exportProjectOptionsHint = document.getElementById(
    "export-project-options-hint"
  );
  const exportSummaryConfirm = document.getElementById("export-summary-confirm");
  const exportSummaryCancel = document.getElementById("export-summary-cancel");
  const importSummaryOverlay = document.getElementById("import-summary-overlay");
  const importSummaryTitle = document.getElementById("import-summary-title");
  const importSummaryBody = document.getElementById("import-summary-body");
  const importSummaryClose = document.getElementById("import-summary-close");
  const viewerFilename = document.getElementById("viewer-filename");
  const videoToolbar = document.getElementById("video-toolbar");
  const videoFrameInterval = document.getElementById("video-frame-interval");
  const videoBlurEvery = document.getElementById("video-blur-every");
  const videoExtractBtn = document.getElementById("video-extract-btn");
  const videoShowBtn = document.getElementById("video-show-btn");
  const batchImageListRoot = document.getElementById("batch-image-list-root");
  const batchStatusFilter = document.getElementById("batch-status-filter");
  const batchSortToggle = document.getElementById("batch-sort-toggle");
  const batchSortMenu = document.getElementById("batch-sort-menu");
  const groupsRoot = document.getElementById("groups-root");
  const inspectorRoot = document.getElementById("inspector-root");
  const confFilterRange = document.getElementById("confidence-filter-range");
  const confFilterValue = document.getElementById("confidence-filter-value");
  const editorModeReviewBtn = document.getElementById("editor-mode-review");
  const editorModeEditBtn = document.getElementById("editor-mode-edit");
  const editorToolsBar = document.getElementById("editor-tools-bar");
  const editorToolSelectBtn = document.getElementById("editor-tool-select");
  const editorToolAddBtn = document.getElementById("editor-tool-add");
  const editorToolCropBtn = document.getElementById("editor-tool-crop");
  const editorToolAddPolygonBtn = document.getElementById("editor-tool-add-polygon");
  const hotkeysHelpBtn = document.getElementById("hotkeys-help-btn");
  const hotkeysHelpOverlay = document.getElementById("hotkeys-help-overlay");
  const hotkeysHelpClose = document.getElementById("hotkeys-help-close");
  const categoryModalOverlay = document.getElementById("category-modal-overlay");
  const categoryModalList = document.getElementById("category-modal-list");
  const categoryModalClose = document.getElementById("category-modal-close");
  const categoryClearBtn = document.getElementById("category-clear-btn");
  const categoryAddBtn = document.getElementById("category-add-btn");
  const categoryNewInput = document.getElementById("category-new-input");
  const classChipsRoot = document.getElementById("class-chips-root");

  const maskContextMenu = document.getElementById("mask-context-menu");
  if (
    !uploadBtn ||
    !uploadVideoBtn ||
    !clearBtn ||
    !runBtn ||
    !fileInput ||
    !runMenuToggle ||
    !runMenu ||
    !videoFileInput ||
    !previewImage ||
    !previewVideo ||
    !placeholderText ||
    !imageFrame ||
    !imageStack ||
    !overlay ||
    !viewerPrev ||
    !viewerNext ||
    !viewerCounter ||
    !markReviewedBtn ||
    !unmarkReviewedBtn ||
    !skipImageBtn ||
    !headerReviewedLine ||
    !workspaceSaveStatus ||
    !headerProcessedLine ||
    !headerProgressTrack ||
    !headerProgressFill ||
    !headerFailedRow ||
    !headerFailedCount ||
    !workModeDetectionBtn ||
    !workModeSegmentationBtn ||
    !groupsRoot ||
    !inspectorRoot ||
    !confFilterRange ||
    !confFilterValue ||
    !exportDropdownWrap ||
    !exportMenuToggle ||
    !exportMenuPanel ||
    !exportSummaryOverlay ||
    !exportSummaryIcon ||
    !exportSummaryScope ||
    !exportSummaryTitle ||
    !exportSummarySubtitle ||
    !exportSummaryBody ||
    !exportNamingRow ||
    !exportStartNumber ||
    !exportNamingPreview ||
    !exportCategoryNumbering ||
    !exportReviewFilter ||
    !exportReviewModeDetected ||
    !exportReviewModeManual ||
    !exportReviewIncludeUnreviewed ||
    !exportReviewFilterHint ||
    !exportYoloOptions ||
    !exportYoloEmptyLabels ||
    !exportYoloOptionsHint ||
    !exportProjectOptions ||
    !exportProjectIncludeAnnotations ||
    !exportProjectOptionsHint ||
    !exportSummaryConfirm ||
    !exportSummaryCancel ||
    !importSummaryOverlay ||
    !importSummaryTitle ||
    !importSummaryBody ||
    !importSummaryClose ||
    !viewerFilename ||
    !videoToolbar ||
    !videoFrameInterval ||
    !videoBlurEvery ||
    !videoExtractBtn ||
    !videoShowBtn ||
    !batchImageListRoot ||
    !batchStatusFilter ||
    !batchSortToggle ||
    !batchSortMenu ||
    !editorModeReviewBtn ||
    !editorModeEditBtn ||
    !editorToolsBar ||
    !editorToolSelectBtn ||
    !editorToolAddBtn ||
    !editorToolCropBtn ||
    !editorToolAddPolygonBtn ||
    !hotkeysHelpBtn ||
    !hotkeysHelpOverlay ||
    !hotkeysHelpClose ||
    !categoryModalOverlay ||
    !categoryModalList ||
    !categoryModalClose ||
    !categoryClearBtn ||
    !categoryAddBtn ||
    !categoryNewInput ||
    !classChipsRoot ||
    !maskContextMenu
  ) {
    return null;
  }

  return {
    uploadBtn,
    uploadVideoBtn,
    clearBtn,
    runBtn,
    fileInput,
    videoFileInput,
    runMenuToggle,
    runMenu,
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
    unmarkReviewedBtn,
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
    videoBlurEvery,
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
  };
}
