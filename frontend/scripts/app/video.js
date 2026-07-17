import {
  DEFAULT_VIDEO_FRAME_INTERVAL_SEC,
  MAX_DIRECT_IMAGES,
} from "./constants.js";

const VIDEO_MIME_RE = /^video\//i;
const VIDEO_EXT_RE = /\.(mp4|webm|mov|m4v|mkv|avi)$/i;

/** @param {HTMLVideoElement} video */
export function waitForVideoMetadata(video) {
  return new Promise((resolve, reject) => {
    if (video.readyState >= 1 && video.videoWidth > 0) {
      resolve();
      return;
    }
    const onMeta = () => {
      cleanup();
      resolve();
    };
    const onErr = () => {
      cleanup();
      reject(new Error("Не удалось загрузить метаданные видео"));
    };
    const cleanup = () => {
      video.removeEventListener("loadedmetadata", onMeta);
      video.removeEventListener("error", onErr);
    };
    video.addEventListener("loadedmetadata", onMeta);
    video.addEventListener("error", onErr);
  });
}

/** @param {File|Blob} file */
export function isVideoFile(file) {
  if (file instanceof File && file.type && VIDEO_MIME_RE.test(file.type)) {
    return true;
  }
  if (file instanceof File && VIDEO_EXT_RE.test(file.name)) return true;
  return false;
}

/**
 * Снимок кадра с видео в момент `timeSec` (сек).
 * @param {HTMLVideoElement} video
 * @param {number} timeSec
 * @returns {Promise<Blob>}
 */
export function captureVideoFrameAt(video, timeSec) {
  const duration = video.duration;
  const t =
    Number.isFinite(duration) && duration > 0
      ? Math.min(Math.max(0, timeSec), Math.max(0, duration - 0.001))
      : Math.max(0, timeSec);

  return new Promise((resolve, reject) => {
    if (!video.videoWidth || !video.videoHeight) {
      reject(new Error("Видео ещё не готово. Дождитесь загрузки метаданных."));
      return;
    }

    let settled = false;
    const finish = (fn) => {
      if (settled) return;
      settled = true;
      video.removeEventListener("seeked", onSeeked);
      video.removeEventListener("error", onError);
      fn();
    };

    const onError = () => {
      finish(() => reject(new Error("Не удалось перемотать видео")));
    };

    const onSeeked = () => {
      finish(() => {
        try {
          const canvas = document.createElement("canvas");
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          const ctx = canvas.getContext("2d");
          if (!ctx) {
            reject(new Error("2d context недоступен"));
            return;
          }
          ctx.drawImage(video, 0, 0);
          canvas.toBlob(
            (blob) => {
              if (blob) resolve(blob);
              else reject(new Error("Не удалось сохранить кадр"));
            },
            "image/png"
          );
        } catch (e) {
          reject(e);
        }
      });
    };

    video.addEventListener("seeked", onSeeked);
    video.addEventListener("error", onError);
    video.pause();
    video.currentTime = t;
  });
}

/**
 * @param {HTMLVideoElement} video
 * @param {{
 *   intervalSec?: number,
 *   maxFrames?: number,
 *   baseName: string,
 *   onProgress?: (done: number, total: number) => void,
 * }} opts
 * @returns {Promise<Array<{ blob: Blob, originalName: string }>>}
 */
export async function extractFramesFromVideoElement(video, opts) {
  await waitForVideoMetadata(video);

  const intervalSec =
    typeof opts.intervalSec === "number" && opts.intervalSec > 0
      ? opts.intervalSec
      : DEFAULT_VIDEO_FRAME_INTERVAL_SEC;
  const maxFrames = Math.min(
    typeof opts.maxFrames === "number" && opts.maxFrames > 0
      ? opts.maxFrames
      : MAX_DIRECT_IMAGES,
    MAX_DIRECT_IMAGES
  );
  const baseName = opts.baseName || "video";

  const duration = video.duration;
  if (!Number.isFinite(duration) || duration <= 0) {
    throw new Error("Не удалось определить длительность видео");
  }

  /** @type {number[]} */
  const times = [];
  for (let t = 0; t < duration && times.length < maxFrames; t += intervalSec) {
    times.push(t);
  }
  if (!times.length) {
    times.push(0);
  }

  const stem = baseName.replace(/\.[^./\\]+$/, "") || "video";
  /** @type {Array<{ blob: Blob, originalName: string }>} */
  const out = [];

  for (let i = 0; i < times.length; i++) {
    opts.onProgress?.(i, times.length);
    const blob = await captureVideoFrameAt(video, times[i]);
    const n = String(i + 1).padStart(4, "0");
    out.push({
      blob,
      originalName: `${stem}_frame_${n}.png`,
    });
  }
  opts.onProgress?.(times.length, times.length);
  return out;
}
