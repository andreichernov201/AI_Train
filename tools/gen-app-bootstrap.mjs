/**
 * Собирает frontend/scripts/app/bootstrap.js из frontend/scripts/app/legacy-monolith.js:
 * убирает IIFE, выносит DOM в параметр refs, вырезает блоки, перенесённые в модули.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
/** Исходный IIFE-монолит; в браузере используется `scripts/app/main.js` → `bootstrap.js`. */
const srcPath = path.join(root, "frontend", "scripts", "app", "legacy-monolith.js");
const outPath = path.join(root, "frontend", "scripts", "app", "bootstrap.js");

const raw = fs.readFileSync(srcPath, "utf8");
const lines = raw.split(/\r?\n/);

/** 1-based inclusive ranges to remove (original file line numbers) */
const ranges = [
  [2522, 2591],
  [1137, 1137],
  [927, 935],
  [843, 852],
  [329, 333],
  [296, 324],
  [251, 260],
  [93, 248],
  [66, 91],
];
ranges.sort((a, b) => b[0] - a[0]);

const L = [...lines];
for (const [a, b] of ranges) {
  const start = a - 1;
  const count = b - a + 1;
  L.splice(start, count);
}

if (!L[0]?.trim().startsWith("(() => {")) {
  throw new Error("Expected legacy monolith to start with (() => {");
}
L.shift();
while (L.length && L[L.length - 1].trim() === "") {
  L.pop();
}
const last = L[L.length - 1]?.trim();
if (last !== "})();") {
  throw new Error(`Expected legacy monolith to end with })(); got: ${JSON.stringify(L[L.length - 1])}`);
}
L.pop();

const header = `import {
  MAX_DIRECT_IMAGES,
  MAX_ZIP_IMAGES,
  MAX_IMAGE_BYTES,
  ZIP_IMAGE_EXT_RE,
  DATA_YAML_RE,
  WORKSPACE_DB_NAME,
  WORKSPACE_DB_VERSION,
  WORKSPACE_STORE_NAME,
  WORKSPACE_AUTOSAVE_DELAY_MS,
  PROJECT_EXPORT_JSON_VERSION,
  HANDLE_HIT_PX,
  MIN_BOX_SIDE,
  DETECT_ALL_CONCURRENCY,
  DETECT_API_FETCH_TIMEOUT_MS,
  DETECT_ALL_BATCH_NAV_THROTTLE_MS,
  IMPORT_SUMMARY_NAME_PREVIEW,
} from "./constants.js";
import { COLORS } from "./colors.js";
import {
  TRAIN_CLASSES,
  YOLO_TXT_EXPORT_CLASS_ORDER,
  resetTrainClassesToDefault,
  trainClassNamesMatchFixedSet,
  applyTrainClassOrderFromNames,
} from "./train-classes.js";
import { newBatchId, newImageItemId } from "./ids.js";
import {
  parseYamlClassNames,
  normalizeZipPath,
  findZipEntryCaseInsensitive,
  findProjectJsonEntry,
} from "./yaml-zip.js";
import { EXPORT_ACTION_LABELS } from "./export-labels.js";

/**
 * @param {import("./dom-refs.js").AppDomRefs} refs
 */
export function bootstrap(refs) {
  const {
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
  } = refs;

`;

const footer = "\n}\n";
const body = L.join("\n");
fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, header + body + footer, "utf8");
console.log("Wrote", path.relative(root, outPath), `(${header.length + body.length} bytes)`);
