import { DETECT_API_FETCH_TIMEOUT_MS } from "./constants.js";

/** @param {any} im @param {"detection"|"segmentation"} [mode] */
export async function fetchDetectApi(im, mode = "detection") {
  const fd = new FormData();
  fd.append("file", im.blob, im.displayName);
  const ac = new AbortController();
  const abortTimer = window.setTimeout(() => {
    ac.abort();
  }, DETECT_API_FETCH_TIMEOUT_MS);
  let resp;
  const endpoint = mode === "segmentation" ? "/api/segment" : "/api/detect";
  try {
    resp = await fetch(endpoint, {
      method: "POST",
      body: fd,
      signal: ac.signal,
    });
  } catch (e) {
    const name = e && typeof e === "object" && "name" in e ? String(e.name) : "";
    if (name === "AbortError") {
      throw new Error(
        `Таймаут распознавания (${Math.round(DETECT_API_FETCH_TIMEOUT_MS / 1000)} с). Проверь сервер или это изображение.`
      );
    }
    throw e;
  } finally {
    window.clearTimeout(abortTimer);
  }
  const raw = await resp.text();
  let data = null;
  try {
    data = raw ? JSON.parse(raw) : null;
  } catch {
    throw new Error(
      `Сервер вернул не‑JSON ответ. Проверь, запущен ли бэкенд ${endpoint}.`
    );
  }
  if (!resp.ok || !data) {
    throw new Error(data?.error || "Распознавание не удалось");
  }
  return data;
}
