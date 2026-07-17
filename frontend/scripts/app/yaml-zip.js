/** @param {string} value */
export function stripYamlQuotes(value) {
  return value.trim().replace(/^['"]|['"]$/g, "");
}

/** @param {string} text */
export function parseYamlClassNames(text) {
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
export function normalizeZipPath(path) {
  return String(path || "").replace(/\\/g, "/").replace(/^\.\/+/, "");
}

/** @param {any} zip */
export function findZipEntryCaseInsensitive(zip, relativePath) {
  const target = normalizeZipPath(relativePath).toLowerCase();
  for (const e of Object.values(zip.files)) {
    if (e.dir) continue;
    if (normalizeZipPath(e.name).toLowerCase() === target) return e;
  }
  return null;
}

/** @param {any} zip */
export function findProjectJsonEntry(zip) {
  for (const e of Object.values(zip.files)) {
    if (e.dir) continue;
    const p = normalizeZipPath(e.name);
    if (/(^|\/)project\.json$/i.test(p)) return e;
  }
  return null;
}

/**
 * Ищет файл по окончанию пути (case-insensitive), например `001.png` или `cat/001.png`.
 * Возвращает первый лексикографически отсортированный матч для стабильности.
 * @param {any} zip
 * @param {string} pathSuffix
 */
export function findZipEntryBySuffixCaseInsensitive(zip, pathSuffix) {
  const suffix = normalizeZipPath(pathSuffix).toLowerCase();
  if (!suffix) return null;
  const hits = [];
  for (const e of Object.values(zip.files)) {
    if (e.dir) continue;
    const p = normalizeZipPath(e.name).toLowerCase();
    if (p.endsWith(suffix)) hits.push(e);
  }
  if (!hits.length) return null;
  hits.sort((a, b) =>
    normalizeZipPath(a.name).localeCompare(normalizeZipPath(b.name))
  );
  return hits[0];
}
