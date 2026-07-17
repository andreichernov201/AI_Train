/**
 * Логическое имя слота в UI/экспорте всегда .png без перекодирования исходного файла.
 * @param {number} ordinal1Based 1 .. n (в пределах batch)
 */
export function formatDisplayName(ordinal1Based) {
  const n = String(Math.max(1, ordinal1Based)).padStart(3, "0");
  return `${n}.png`;
}

/**
 * Ширина нумерации: минимум 3 цифры, больше если нужно для последнего номера.
 * @param {number} startNumber
 * @param {number} count
 */
export function exportNumberPadWidth(startNumber, count) {
  const start = Math.max(1, Math.floor(startNumber) || 1);
  const n = Math.max(1, Math.floor(count) || 1);
  const last = start + n - 1;
  return Math.max(3, String(last).length);
}

/**
 * @param {number} index1Based — порядковый номер в экспорте (1..count)
 * @param {number} [startNumber=1] — с какого числа начинать
 * @param {number} [padWidth]
 */
export function exportNumberedFileStem(index1Based, startNumber = 1, padWidth) {
  const num = Math.max(1, Math.floor(startNumber) || 1) + Math.max(1, index1Based) - 1;
  const width =
    typeof padWidth === "number" && padWidth > 0
      ? padWidth
      : Math.max(3, String(num).length);
  return String(num).padStart(width, "0");
}

/** @param {number} index1Based @param {number} [startNumber=1] @param {number} [padWidth] */
export function exportNumberedPngName(index1Based, startNumber = 1, padWidth) {
  return `${exportNumberedFileStem(index1Based, startNumber, padWidth)}.png`;
}
