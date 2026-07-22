export const ANALYSIS_ZOOM_MIN = 1;
export const ANALYSIS_ZOOM_MAX = 8;

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

export function containGeometry({
  viewportWidth,
  viewportHeight,
  sourceWidth,
  sourceHeight,
  zoom = 1,
  panX = 0,
  panY = 0,
}) {
  const safeViewportWidth = Math.max(1, Number(viewportWidth) || 1);
  const safeViewportHeight = Math.max(1, Number(viewportHeight) || 1);
  const safeSourceWidth = Math.max(1, Number(sourceWidth) || 1);
  const safeSourceHeight = Math.max(1, Number(sourceHeight) || 1);
  const safeZoom = clamp(Number(zoom) || 1, ANALYSIS_ZOOM_MIN, ANALYSIS_ZOOM_MAX);
  const baseScale = Math.min(
    safeViewportWidth / safeSourceWidth,
    safeViewportHeight / safeSourceHeight
  );
  const scale = baseScale * safeZoom;
  const drawWidth = safeSourceWidth * scale;
  const drawHeight = safeSourceHeight * scale;
  return {
    drawWidth,
    drawHeight,
    offsetX: (safeViewportWidth - drawWidth) / 2 + Number(panX || 0),
    offsetY: (safeViewportHeight - drawHeight) / 2 + Number(panY || 0),
    scaleX: scale,
    scaleY: scale,
  };
}

export function clampAnalysisPan({
  viewportWidth,
  viewportHeight,
  sourceWidth,
  sourceHeight,
  zoom,
  panX,
  panY,
}) {
  const geometry = containGeometry({
    viewportWidth,
    viewportHeight,
    sourceWidth,
    sourceHeight,
    zoom,
  });
  const maxPanX = Math.max(0, (geometry.drawWidth - viewportWidth) / 2);
  const maxPanY = Math.max(0, (geometry.drawHeight - viewportHeight) / 2);
  return {
    panX: clamp(Number(panX) || 0, -maxPanX, maxPanX),
    panY: clamp(Number(panY) || 0, -maxPanY, maxPanY),
  };
}

export function zoomAnalysisAtPoint({
  viewportWidth,
  viewportHeight,
  sourceWidth,
  sourceHeight,
  zoom,
  panX,
  panY,
  pointX,
  pointY,
  nextZoom,
}) {
  const before = containGeometry({
    viewportWidth,
    viewportHeight,
    sourceWidth,
    sourceHeight,
    zoom,
    panX,
    panY,
  });
  const sourceX = clamp((pointX - before.offsetX) / before.scaleX, 0, sourceWidth);
  const sourceY = clamp((pointY - before.offsetY) / before.scaleY, 0, sourceHeight);
  const safeNextZoom = clamp(nextZoom, ANALYSIS_ZOOM_MIN, ANALYSIS_ZOOM_MAX);
  const after = containGeometry({
    viewportWidth,
    viewportHeight,
    sourceWidth,
    sourceHeight,
    zoom: safeNextZoom,
  });
  const desiredPanX = pointX - ((viewportWidth - after.drawWidth) / 2 + sourceX * after.scaleX);
  const desiredPanY = pointY - ((viewportHeight - after.drawHeight) / 2 + sourceY * after.scaleY);
  return {
    zoom: safeNextZoom,
    ...clampAnalysisPan({
      viewportWidth,
      viewportHeight,
      sourceWidth,
      sourceHeight,
      zoom: safeNextZoom,
      panX: desiredPanX,
      panY: desiredPanY,
    }),
  };
}
