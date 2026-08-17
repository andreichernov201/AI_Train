import {
  DEFAULT_VIDEO_FRAME_INTERVAL_SEC,
  MAX_VIDEO_FRAMES,
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
 * Не даём выйти за границы видео.
 * @param {HTMLVideoElement} video
 * @param {number} timeSec
 */
function normalizedVideoTime(video, timeSec) {
  const duration = video.duration;
  if (Number.isFinite(duration) && duration > 0) {
    return Math.min(Math.max(0, timeSec), Math.max(0, duration - 0.001));
  }
  return Math.max(0, timeSec);
}

/**
 * Перематываем видео и ждём, пока кадр будет готов.
 * @param {HTMLVideoElement} video
 * @param {number} timeSec
 */
export function seekVideoTo(video, timeSec) {
  const t = normalizedVideoTime(video, timeSec);
  return new Promise((resolve, reject) => {
    if (!video.videoWidth || !video.videoHeight) {
      reject(new Error("Видео ещё не готово. Дождитесь загрузки метаданных."));
      return;
    }

    let settled = false;
    const timeoutId = window.setTimeout(() => {
      finish(() => reject(new Error("Видео слишком долго перематывается")));
    }, 15000);
    const cleanup = () => {
      window.clearTimeout(timeoutId);
      video.removeEventListener("seeked", onSeeked);
      video.removeEventListener("error", onError);
    };
    const finish = (fn) => {
      if (settled) return;
      settled = true;
      cleanup();
      fn();
    };
    const onError = () => {
      finish(() => reject(new Error("Не удалось перемотать видео")));
    };
    const onSeeked = () => {
      window.requestAnimationFrame(() => finish(resolve));
    };

    video.pause();
    if (Math.abs(video.currentTime - t) < 0.0005 && video.readyState >= 2) {
      window.requestAnimationFrame(() => finish(resolve));
      return;
    }
    video.addEventListener("seeked", onSeeked);
    video.addEventListener("error", onError);
    video.currentTime = t;
  });
}

/** @param {HTMLCanvasElement} canvas */
function canvasToPngBlob(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (blob) resolve(blob);
        else reject(new Error("Не удалось сохранить кадр"));
      },
      "image/png"
    );
  });
}

/**
 * Снимаем кадр в момент `timeSec`.
 * @param {HTMLVideoElement} video
 * @param {number} timeSec
 * @returns {Promise<Blob>}
 */
export async function captureVideoFrameAt(video, timeSec) {
  await waitForVideoMetadata(video);
  await seekVideoTo(video, timeSec);
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("2d context недоступен");
  ctx.drawImage(video, 0, 0);
  return canvasToPngBlob(canvas);
}

/**
 * Резкость оцениваем по дисперсии лапласиана.
 * Чем выше число, тем больше в кадре чётких мелких границ.
 * @param {Uint8ClampedArray|number[]} rgba
 * @param {number} width
 * @param {number} height
 */
export function calculateFrameSharpness(rgba, width, height) {
  if (!rgba || width < 3 || height < 3) return 0;
  const gray = new Float32Array(width * height);
  for (let i = 0, p = 0; p < gray.length; p++, i += 4) {
    gray[p] = rgba[i] * 0.299 + rgba[i + 1] * 0.587 + rgba[i + 2] * 0.114;
  }

  let sum = 0;
  let sumSq = 0;
  let count = 0;
  for (let y = 1; y < height - 1; y++) {
    const row = y * width;
    for (let x = 1; x < width - 1; x++) {
      const i = row + x;
      const laplacian =
        gray[i] * 4 -
        gray[i - 1] -
        gray[i + 1] -
        gray[i - width] -
        gray[i + width];
      sum += laplacian;
      sumSq += laplacian * laplacian;
      count++;
    }
  }
  if (!count) return 0;
  const mean = sum / count;
  return Math.max(0, sumSq / count - mean * mean);
}

/**
 * Ищем лучший кадр рядом с нужным временем, но не выходим далеко за интервал.
 * @param {number} anchorSec
 * @param {number} intervalSec
 * @param {number} durationSec
 * @param {number} [candidateCount]
 */
export function buildFrameCandidateTimes(
  anchorSec,
  intervalSec,
  durationSec,
  candidateCount = 5
) {
  const maxTime = Math.max(0, Number(durationSec) - 0.001);
  const anchor = Math.min(Math.max(0, Number(anchorSec) || 0), maxTime);
  let count = Math.max(3, Math.min(9, Math.round(candidateCount) || 5));
  if (count % 2 === 0) count = Math.min(9, count + 1);
  const radius = Math.min(Math.max(0.04, intervalSec * 0.4), 0.75);
  const result = [];
  for (let i = 0; i < count; i++) {
    const offset = -radius + (2 * radius * i) / (count - 1);
    const t = Math.min(Math.max(0, anchor + offset), maxTime);
    if (!result.some((existing) => Math.abs(existing - t) < 0.001)) {
      result.push(t);
    }
  }
  return result.length ? result : [anchor];
}

/**
 * Каждый N-й кадр берём смазанным. Ноль и один отключают этот режим.
 * @param {number} frameIndex нулевой индекс
 * @param {number} blurEvery
 */
export function shouldUseBlurredCandidate(frameIndex, blurEvery) {
  const n = Math.floor(Number(blurEvery) || 0);
  return n >= 2 && (frameIndex + 1) % n === 0;
}

/** @param {HTMLVideoElement} video */
function createFrameSelectionSurfaces(video) {
  const analysisScale = Math.min(
    1,
    320 / video.videoWidth,
    180 / video.videoHeight
  );
  const analysisCanvas = document.createElement("canvas");
  analysisCanvas.width = Math.max(3, Math.round(video.videoWidth * analysisScale));
  analysisCanvas.height = Math.max(3, Math.round(video.videoHeight * analysisScale));
  const analysisCtx = analysisCanvas.getContext("2d", {
    willReadFrequently: true,
  });
  if (!analysisCtx) throw new Error("2d context анализа недоступен");

  const selectedCanvas = document.createElement("canvas");
  selectedCanvas.width = video.videoWidth;
  selectedCanvas.height = video.videoHeight;
  const selectedCtx = selectedCanvas.getContext("2d");
  if (!selectedCtx) throw new Error("2d context кадра недоступен");

  return { analysisCanvas, analysisCtx, selectedCanvas, selectedCtx };
}

/**
 * @param {HTMLVideoElement} video
 * @param {number[]} candidateTimes
 * @param {boolean} pickBlurred
 * @param {ReturnType<typeof createFrameSelectionSurfaces>} surfaces
 */
async function selectCandidateFrame(
  video,
  candidateTimes,
  pickBlurred,
  surfaces
) {
  let selectedScore = pickBlurred ? Infinity : -Infinity;
  let selectedTime = candidateTimes[0] ?? 0;
  let hasSelected = false;

  for (const timeSec of candidateTimes) {
    await seekVideoTo(video, timeSec);
    surfaces.analysisCtx.drawImage(
      video,
      0,
      0,
      surfaces.analysisCanvas.width,
      surfaces.analysisCanvas.height
    );
    const imageData = surfaces.analysisCtx.getImageData(
      0,
      0,
      surfaces.analysisCanvas.width,
      surfaces.analysisCanvas.height
    );
    const score = calculateFrameSharpness(
      imageData.data,
      imageData.width,
      imageData.height
    );
    const better =
      !hasSelected ||
      (pickBlurred ? score < selectedScore : score > selectedScore);
    if (!better) continue;
    hasSelected = true;
    selectedScore = score;
    selectedTime = timeSec;
    surfaces.selectedCtx.drawImage(
      video,
      0,
      0,
      surfaces.selectedCanvas.width,
      surfaces.selectedCanvas.height
    );
  }

  return {
    blob: await canvasToPngBlob(surfaces.selectedCanvas),
    timeSec: selectedTime,
    sharpness: selectedScore,
  };
}

/**
 * @param {HTMLVideoElement} video
 * @param {{
 *   intervalSec?: number,
 *   maxFrames?: number,
 *   baseName: string,
 *   blurEvery?: number,
 *   candidateCount?: number,
 *   onProgress?: (done: number, total: number, detail?: { selection: "sharp"|"blur", candidates: number }) => void,
 * }} opts
 * @returns {Promise<Array<{ blob: Blob, originalName: string, selection: "sharp"|"blur", timeSec: number, sharpness: number }>>}
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
      : MAX_VIDEO_FRAMES,
    MAX_VIDEO_FRAMES
  );
  const blurEvery = Math.max(0, Math.floor(Number(opts.blurEvery) || 0));
  const candidateCount = Math.max(
    3,
    Math.min(9, Math.round(Number(opts.candidateCount) || 5))
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
  if (!times.length) times.push(0);

  const stem = baseName.replace(/\.[^./\\]+$/, "") || "video";
  const surfaces = createFrameSelectionSurfaces(video);
  /** @type {Array<{ blob: Blob, originalName: string, selection: "sharp"|"blur", timeSec: number, sharpness: number }>} */
  const out = [];

  for (let i = 0; i < times.length; i++) {
    const pickBlurred = shouldUseBlurredCandidate(i, blurEvery);
    const candidateTimes = buildFrameCandidateTimes(
      times[i],
      intervalSec,
      duration,
      candidateCount
    );
    opts.onProgress?.(i, times.length, {
      selection: pickBlurred ? "blur" : "sharp",
      candidates: candidateTimes.length,
    });
    const selected = await selectCandidateFrame(
      video,
      candidateTimes,
      pickBlurred,
      surfaces
    );
    const n = String(i + 1).padStart(4, "0");
    out.push({
      ...selected,
      selection: pickBlurred ? "blur" : "sharp",
      originalName:
        stem + "_frame_" + n + (pickBlurred ? "_blur" : "") + ".png",
    });
  }
  opts.onProgress?.(times.length, times.length, {
    selection: "sharp",
    candidates: 0,
  });
  return out;
}
