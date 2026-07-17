import { DATA_YAML_RE } from "./constants.js";
import { normalizeZipPath } from "./yaml-zip.js";

/** @param {string} path */
export function isServiceZipPath(path) {
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
export function isZipFile(file) {
  return /\.zip$/i.test(file.name) || file.type === "application/zip";
}
