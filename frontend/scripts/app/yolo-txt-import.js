import { yoloClassOrderForTask } from "./train-classes.js";
import { normalizeZipPath } from "./yaml-zip.js";

/**
 * Возможные пути к YOLO-лейблу для пути картинки внутри ZIP (соседний .txt и images/→labels/).
 * @param {string} imagePath
 * @returns {string[]}
 */
export function yoloLabelTxtCandidatesForImageZipPath(imagePath) {
  const norm = normalizeZipPath(imagePath);
  const out = [];
  out.push(norm.replace(/\.(jpe?g|png|webp)$/i, ".txt"));
  const m = norm.match(/^(.*\/)images(\/.+)$/i);
  if (m) {
    const rest = m[2].replace(/\.(jpe?g|png|webp)$/i, ".txt");
    out.push(`${m[1]}labels${rest}`);
  }
  const seen = new Set();
  /** @type {string[]} */
  const uniq = [];
  for (const p of out) {
    const k = normalizeZipPath(p).toLowerCase();
    if (seen.has(k)) continue;
    seen.add(k);
    uniq.push(normalizeZipPath(p));
  }
  return uniq;
}

/** Имя файла без расширения для сопоставления image ↔ label .txt */
export function labelStemKeyForMatch(name) {
  const norm = normalizeZipPath(String(name || ""));
  const base = norm.split("/").pop() || norm;
  return base.replace(/\.[^.]+$/i, "").toLowerCase();
}

/** @param {File|Blob} file */
export function isYoloLabelTxtFile(file) {
  const n = String(file?.name || "").toLowerCase();
  return n.endsWith(".txt");
}

/**
 * Пути .txt, которые относятся к импортируемым картинкам (не считаем «мусором» в ZIP).
 * @param {string[]} imagePaths
 * @returns {Set<string>}
 */
export function protectedYoloTxtPathSetForImageZipPaths(imagePaths) {
  const s = new Set();
  for (const ip of imagePaths) {
    for (const c of yoloLabelTxtCandidatesForImageZipPath(ip)) {
      s.add(c.toLowerCase());
    }
  }
  return s;
}

/**
 * YOLO TXT: строка bbox — class xc yc w h (нормализовано); сегментация — class x1 y1 x2 y2 … (чётное число координат).
 * Возвращает сырые детекции для normalizeDetection().
 * @param {string} text
 * @param {number} iw
 * @param {number} ih
 * @returns {Array<{ cls_id: number, cls_name: string, conf: number, box: [number, number, number, number], segment?: Array<[number, number]>, source: "human" }>}
 */
export function parseYoloTxtDetectionsForImport(text, iw, ih) {
  if (!(iw > 0 && ih > 0)) return [];
  const lines = String(text || "")
    .replace(/\r/g, "\n")
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith("#"));
  /** @type {Array<{ cls_id: number, cls_name: string, conf: number, box: [number, number, number, number], segment?: Array<[number, number]>, source: "human" }>} */
  const out = [];
  for (const line of lines) {
    const parts = line.split(/\s+/).filter(Boolean);
    if (parts.length < 2) continue;
    const clsRaw = Number(parts[0]);
    if (!Number.isFinite(clsRaw)) continue;
    const nums = parts.slice(1).map((x) => Number(x));
    if (nums.some((x) => !Number.isFinite(x))) continue;
    /** @type {[number, number, number, number] | null} */
    let box = null;
    /** @type {Array<[number, number]>} */
    let segment = [];
    if (nums.length === 4) {
      const [xc, yc, nw, nh] = nums;
      if (!(nw > 0 && nh > 0)) continue;
      const x1 = (xc - nw / 2) * iw;
      const y1 = (yc - nh / 2) * ih;
      const x2 = (xc + nw / 2) * iw;
      const y2 = (yc + nh / 2) * ih;
      box = [x1, y1, x2, y2];
    } else if (nums.length >= 6 && nums.length % 2 === 0) {
      let minX = Infinity;
      let minY = Infinity;
      let maxX = -Infinity;
      let maxY = -Infinity;
      for (let k = 0; k < nums.length; k += 2) {
        const px = nums[k] * iw;
        const py = nums[k + 1] * ih;
        segment.push([px, py]);
        minX = Math.min(minX, px);
        minY = Math.min(minY, py);
        maxX = Math.max(maxX, px);
        maxY = Math.max(maxY, py);
      }
      if (!(maxX > minX && maxY > minY)) continue;
      box = [minX, minY, maxX, maxY];
    } else {
      continue;
    }
    const task = segment.length >= 3 ? "segment" : "detect";
    const order = yoloClassOrderForTask(task);
    if (!Number.isInteger(clsRaw) || clsRaw < 0 || clsRaw >= order.length) continue;
    const cls_id = clsRaw;
    const cls_name = order[cls_id];
    const row = {
      cls_id,
      annotation_type: task === "segment" ? "seg" : "detect",
      cls_name,
      conf: 1,
      box,
      source: "human",
    };
    if (segment.length >= 3) row.segment = segment;
    out.push(row);
  }
  return out;
}
