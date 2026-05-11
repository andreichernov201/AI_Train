/** @returns {string} */
export function newBatchId() {
  if (crypto.randomUUID) return crypto.randomUUID();
  return `batch_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

/** @returns {string} */
export function newImageItemId() {
  if (crypto.randomUUID) return crypto.randomUUID();
  return `img_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}
