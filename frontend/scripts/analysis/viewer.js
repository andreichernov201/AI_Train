import {
  ANALYSIS_MASK_STYLE,
  analysisDetectionColor,
  analysisSegmentationColor,
} from "./colors.js";
import {
  ANALYSIS_ZOOM_MAX,
  clampAnalysisPan,
  containGeometry,
  zoomAnalysisAtPoint,
} from "./viewer-zoom.js";

function boxesForEvent(event) {
  const bestMeta = event?.metadata || {};
  const bestFrameIndex = Number(bestMeta.best_frame_reason?.frame_index ?? NaN);
  const trains = Array.isArray(event?.train_observations) ? event.train_observations : [];
  const numbers = Array.isArray(event?.number_observations) ? event.number_observations : [];
  const segments = Array.isArray(event?.segmentation_observations) ? event.segmentation_observations : [];
  if (!Number.isFinite(bestFrameIndex)) return { trains, numbers, segments };
  return {
    trains: trains.filter((item) => Number(item.frame_index) === bestFrameIndex),
    numbers: numbers.filter((item) => Number(item.frame_index) === bestFrameIndex),
    segments: segments.filter((item) => Number(item.frame_index) === bestFrameIndex),
  };
}

export function createAnalysisViewer({ image, video, canvas, empty }) {
  const context = canvas.getContext("2d");
  let currentEvent = null;
  let currentFile = null;
  let settings = { showTrain: true, showNumber: true, showSegmentation: true, showOcr: true, highlightedTrackIds: [] };
  const surface = image.parentElement;
  let zoom = 1;
  let panX = 0;
  let panY = 0;
  let dragState = null;
  let currentSource = "";

  function sourceSize() {
    return {
      width: Math.max(1, Number(currentEvent?.metadata?.frame_width || image.naturalWidth || 1)),
      height: Math.max(1, Number(currentEvent?.metadata?.frame_height || image.naturalHeight || 1)),
    };
  }

  function viewportSize() {
    const rect = surface.getBoundingClientRect();
    return { rect, width: Math.max(1, rect.width), height: Math.max(1, rect.height) };
  }

  function applyImageTransform() {
    const transformed = zoom > 1.0001 || Math.abs(panX) > 0.01 || Math.abs(panY) > 0.01;
    image.style.transformOrigin = "center center";
    image.style.transform = transformed ? `translate(${panX}px, ${panY}px) scale(${zoom})` : "";
    surface.classList.toggle("is-zoomed", zoom > 1.0001);
    surface.classList.toggle("is-panning", Boolean(dragState));
    surface.dataset.zoom = `${Math.round(zoom * 100)}%`;
  }

  function resetZoom(redraw = false) {
    zoom = 1;
    panX = 0;
    panY = 0;
    dragState = null;
    applyImageTransform();
    if (redraw) resizeAndDraw();
  }

  function clampCurrentPan() {
    const viewport = viewportSize();
    const source = sourceSize();
    const next = clampAnalysisPan({
      viewportWidth: viewport.width,
      viewportHeight: viewport.height,
      sourceWidth: source.width,
      sourceHeight: source.height,
      zoom,
      panX,
      panY,
    });
    panX = next.panX;
    panY = next.panY;
  }


  function resizeAndDraw() {
    if (image.hidden || !image.naturalWidth) {
      context.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    const viewport = viewportSize();
    const rect = viewport.rect;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.height * dpr));
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;
    canvas.style.left = "0px";
    canvas.style.top = "0px";
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, rect.width, rect.height);
    const source = sourceSize();
    clampCurrentPan();
    applyImageTransform();
    const geometry = containGeometry({
      viewportWidth: viewport.width,
      viewportHeight: viewport.height,
      sourceWidth: source.width,
      sourceHeight: source.height,
      zoom,
      panX,
      panY,
    });
    const { offsetX, offsetY, scaleX, scaleY } = geometry;
    const { trains, numbers, segments } = boxesForEvent(currentEvent);

    if (settings.showSegmentation) {
      for (const item of segments) {
        drawSegment(item);
      }
      for (const item of segments) {
        const color = analysisSegmentationColor(item.cls_name);
        const label = `${item.cls_name} ${Math.round(Number(item.conf || 0) * 100)}%`;
        drawBox(item.box, color, label, false);
      }
    }

    if (settings.showTrain) {
      for (const item of trains) {
        drawBox(item.box, analysisDetectionColor("train"), `train ${Math.round(Number(item.conf || 0) * 100)}%`, false);
      }
    }
    if (settings.showNumber) {
      for (const item of numbers) {
        const highlighted = settings.highlightedTrackIds.includes(String(item.fragment_track_id));
        const label = settings.showOcr
          ? `${item.normalized_text || "—"} ${Math.round(Number(item.ocr_confidence || 0) * 100)}%`
          : "number";
        drawBox(item.bbox, analysisDetectionColor("number"), label, highlighted);
      }
    }

    function drawSegment(item) {
      const polygon = Array.isArray(item?.segment) ? item.segment : [];
      if (polygon.length < 3) return;
      const color = analysisSegmentationColor(item.cls_name);
      context.save();
      context.beginPath();
      polygon.forEach((point, index) => {
        const x = offsetX + Number(point?.[0]) * scaleX;
        const y = offsetY + Number(point?.[1]) * scaleY;
        if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
      });
      context.closePath();
      context.fillStyle = color;
      context.globalAlpha = ANALYSIS_MASK_STYLE.fillAlpha;
      context.fill();
      context.globalAlpha = ANALYSIS_MASK_STYLE.strokeAlpha;
      context.strokeStyle = color;
      context.lineWidth = ANALYSIS_MASK_STYLE.lineWidth;
      context.stroke();
      context.restore();
    }

    function drawBox(box, color, label, selected = false) {
      if (!Array.isArray(box) || box.length !== 4) return;
      const x = offsetX + Number(box[0]) * scaleX;
      const y = offsetY + Number(box[1]) * scaleY;
      const w = Math.max(1, (Number(box[2]) - Number(box[0])) * scaleX);
      const h = Math.max(1, (Number(box[3]) - Number(box[1])) * scaleY);
      context.save();
      context.strokeStyle = color;
      if (selected) {
        context.strokeStyle = "rgba(255,255,255,0.92)";
        context.lineWidth = 6;
        context.strokeRect(x, y, w, h);
        context.strokeStyle = color;
        context.lineWidth = 3;
        context.strokeRect(x, y, w, h);
      } else {
        context.lineWidth = 2;
        context.strokeRect(x, y, w, h);
      }
      context.font = "12px system-ui, sans-serif";
      const labelWidth = context.measureText(label).width + 12;
      const labelY = Math.max(0, y - 22);
      context.globalAlpha = 0.85;
      context.fillStyle = color;
      context.fillRect(x, labelY, labelWidth, 20);
      context.globalAlpha = 1;
      context.fillStyle = "#0b0b0b";
      context.fillText(label, x + 6, labelY + 14);
      context.restore();
    }
  }

  function render(event, file, viewerSettings) {
    currentEvent = event;
    currentFile = file;
    settings = { ...settings, ...(viewerSettings || {}) };
    const assetId = event?.best_frame_asset_id;
    if (assetId) {
      video.pause();
      video.hidden = true;
      image.hidden = false;
      empty.hidden = true;
      const next = `/api/analysis/assets/${assetId}`;
      if (currentSource !== next) {
        currentSource = next;
        resetZoom();
        image.src = next;
      } else {
        resizeAndDraw();
      }
      return;
    }
    if (file?.kind === "video") {
      image.hidden = true;
      canvas.width = 1;
      video.hidden = false;
      empty.hidden = true;
      const next = file.content_url;
      if (currentSource !== next) {
        currentSource = next;
        resetZoom();
      }
      context.clearRect(0, 0, canvas.width, canvas.height);
      if (!video.src.endsWith(next)) video.src = next;
      return;
    }
    if (file?.content_url) {
      video.pause();
      video.hidden = true;
      image.hidden = false;
      empty.hidden = true;
      const next = file.content_url;
      if (currentSource !== next) {
        currentSource = next;
        resetZoom();
        image.src = next;
      } else {
        resizeAndDraw();
      }
      return;
    }
    video.hidden = true;
    image.hidden = true;
    empty.hidden = false;
    currentSource = "";
    resetZoom();
    context.clearRect(0, 0, canvas.width, canvas.height);
  }

  function onWheel(event) {
    if (image.hidden || !image.naturalWidth) return;
    event.preventDefault();
    const viewport = viewportSize();
    const source = sourceSize();
    const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
    const next = zoomAnalysisAtPoint({
      viewportWidth: viewport.width,
      viewportHeight: viewport.height,
      sourceWidth: source.width,
      sourceHeight: source.height,
      zoom,
      panX,
      panY,
      pointX: event.clientX - viewport.rect.left,
      pointY: event.clientY - viewport.rect.top,
      nextZoom: Math.max(1, Math.min(ANALYSIS_ZOOM_MAX, zoom * factor)),
    });
    zoom = next.zoom;
    panX = next.panX;
    panY = next.panY;
    applyImageTransform();
    resizeAndDraw();
  }

  function onPointerDown(event) {
    if (event.button !== 0 || zoom <= 1.0001 || image.hidden) return;
    dragState = {
      pointerId: event.pointerId,
      startClientX: event.clientX,
      startClientY: event.clientY,
      startPanX: panX,
      startPanY: panY,
    };
    surface.setPointerCapture?.(event.pointerId);
    applyImageTransform();
    event.preventDefault();
  }

  function onPointerMove(event) {
    if (!dragState || dragState.pointerId !== event.pointerId) return;
    panX = dragState.startPanX + event.clientX - dragState.startClientX;
    panY = dragState.startPanY + event.clientY - dragState.startClientY;
    clampCurrentPan();
    applyImageTransform();
    resizeAndDraw();
  }

  function finishPointerDrag(event) {
    if (!dragState || dragState.pointerId !== event.pointerId) return;
    if (surface.hasPointerCapture?.(event.pointerId)) surface.releasePointerCapture(event.pointerId);
    dragState = null;
    applyImageTransform();
  }

  surface.addEventListener("wheel", onWheel, { passive: false });
  surface.addEventListener("pointerdown", onPointerDown);
  surface.addEventListener("pointermove", onPointerMove);
  surface.addEventListener("pointerup", finishPointerDrag);
  surface.addEventListener("pointercancel", finishPointerDrag);
  surface.addEventListener("dblclick", (event) => {
    if (image.hidden || !image.naturalWidth) return;
    resetZoom(true);
    event.preventDefault();
  });
  image.addEventListener("load", resizeAndDraw);
  window.addEventListener("resize", resizeAndDraw);
  const resizeObserver = typeof ResizeObserver === "function" ? new ResizeObserver(resizeAndDraw) : null;
  resizeObserver?.observe(surface);
  return { render, redraw: resizeAndDraw, resetZoom: () => resetZoom(true) };
}
