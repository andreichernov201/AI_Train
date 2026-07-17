/**
 * Чистая геометрия для редактора полигонов сегментации.
 * Никакого состояния, никаких зависимостей от app.js — только координаты
 * изображения/overlay и массивы точек.
 *
 * @typedef {[number, number]} Point
 * @typedef {{ offsetX: number, offsetY: number, sx: number, sy: number }} OverlayGeo
 */

/** Маркер вершины полигона: радиус хит-теста в CSS px overlay-канваса. */
export const VERTEX_HIT_PX = 10;
/** Хит-тест клика по ребру (для вставки новой точки), CSS px. */
export const EDGE_HIT_PX = 8;
/** Минимальное число точек, ниже которого полигон не имеет смысла. */
export const MIN_POLYGON_POINTS = 3;
/** Верхняя граница допуска (epsilon) слайдера упрощения, px исходного изображения. */
export const POLYGON_SIMPLIFY_MAX_EPSILON_PX = 25;

/** Координаты изображения → overlay (CSS px). */
export function imageToOverlayPoint(geo, ix, iy) {
  return {
    ox: geo.offsetX + ix * geo.sx,
    oy: geo.offsetY + iy * geo.sy,
  };
}

/** Overlay (CSS px) → координаты изображения. */
export function overlayToImagePoint(geo, ox, oy) {
  return {
    ix: (ox - geo.offsetX) / geo.sx,
    iy: (oy - geo.offsetY) / geo.sy,
  };
}

/**
 * Точка внутри полигона (ray casting), координаты point и points — в одной
 * системе координат (обычно — координаты исходного изображения).
 * @param {Point} pt
 * @param {Point[]} points
 */
export function pointInPolygon(pt, points) {
  if (!Array.isArray(points) || points.length < 3) return false;
  const [px, py] = pt;
  let inside = false;
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const [xi, yi] = points[i];
    const [xj, yj] = points[j];
    const intersects =
      yi > py !== yj > py &&
      px < ((xj - xi) * (py - yi)) / (yj - yi + Number.EPSILON) + xi;
    if (intersects) inside = !inside;
  }
  return inside;
}

/**
 * Ближайшая точка на отрезке [a,b] к точке pt + расстояние до неё.
 * @param {Point} pt @param {Point} a @param {Point} b
 * @returns {{ point: Point, distance: number, t: number }}
 */
export function closestPointOnSegment(pt, a, b) {
  const abx = b[0] - a[0];
  const aby = b[1] - a[1];
  const lenSq = abx * abx + aby * aby;
  let t = lenSq > 0 ? ((pt[0] - a[0]) * abx + (pt[1] - a[1]) * aby) / lenSq : 0;
  t = Math.max(0, Math.min(1, t));
  const point = /** @type {Point} */ ([a[0] + abx * t, a[1] + aby * t]);
  const distance = Math.hypot(pt[0] - point[0], pt[1] - point[1]);
  return { point, distance, t };
}

/** Перпендикулярное расстояние от точки до отрезка [a,b]. */
export function distanceToSegmentPx(pt, a, b) {
  return closestPointOnSegment(pt, a, b).distance;
}

/**
 * Bounding box замкнутого полигона в тех же координатах, что и points.
 * @param {Point[]} points
 * @returns {[number, number, number, number]}
 */
export function polygonBBox(points) {
  if (!Array.isArray(points) || points.length === 0) return [0, 0, 0, 0];
  let x1 = Infinity;
  let y1 = Infinity;
  let x2 = -Infinity;
  let y2 = -Infinity;
  for (const [x, y] of points) {
    if (x < x1) x1 = x;
    if (y < y1) y1 = y;
    if (x > x2) x2 = x;
    if (y > y2) y2 = y;
  }
  return [x1, y1, x2, y2];
}

/**
 * Сдвигает все точки полигона на (dx,dy), но ограничивает сдвиг так, чтобы
 * bounding box формы не выходил за границы изображения [0,imgW]x[0,imgH]
 * (аналог translateBoxWithinImage для bbox).
 * @param {Point[]} points @param {number} dx @param {number} dy
 * @param {number} imgW @param {number} imgH
 * @returns {Point[]}
 */
export function translatePolygonClamped(points, dx, dy, imgW, imgH) {
  const [x1, y1, x2, y2] = polygonBBox(points);
  let cdx = dx;
  let cdy = dy;
  if (x2 - x1 <= imgW) {
    cdx = Math.max(-x1, Math.min(cdx, imgW - x2));
  } else {
    cdx = 0;
  }
  if (y2 - y1 <= imgH) {
    cdy = Math.max(-y1, Math.min(cdy, imgH - y2));
  } else {
    cdy = 0;
  }
  return points.map(([x, y]) => [x + cdx, y + cdy]);
}

/**
 * Поиск вершины полигона в пределах hitPx от overlay-точки (ox,oy).
 * @param {OverlayGeo} geo @param {number} ox @param {number} oy
 * @param {Point[]} points @param {number} [hitPx]
 * @returns {number|null}
 */
export function hitTestPolygonVertex(geo, ox, oy, points, hitPx = VERTEX_HIT_PX) {
  if (!Array.isArray(points)) return null;
  let bestIdx = null;
  let bestDist = Infinity;
  for (let i = 0; i < points.length; i++) {
    const p = imageToOverlayPoint(geo, points[i][0], points[i][1]);
    const d = Math.hypot(ox - p.ox, oy - p.oy);
    if (d <= hitPx && d < bestDist) {
      bestDist = d;
      bestIdx = i;
    }
  }
  return bestIdx;
}

/**
 * Поиск ближайшего ребра полигона в пределах hitPx от overlay-точки.
 * Ребро i соединяет points[i] и points[(i+1) % length]; вставка новой точки
 * происходит после индекса i.
 * @param {OverlayGeo} geo @param {number} ox @param {number} oy
 * @param {Point[]} points @param {number} [hitPx]
 * @returns {{ index: number, insertAt: Point } | null}
 */
export function hitTestPolygonEdge(geo, ox, oy, points, hitPx = EDGE_HIT_PX) {
  if (!Array.isArray(points) || points.length < 2) return null;
  const { ix, iy } = overlayToImagePoint(geo, ox, oy);
  let bestIdx = null;
  let bestDist = Infinity;
  let bestPoint = null;
  for (let i = 0; i < points.length; i++) {
    const a = points[i];
    const b = points[(i + 1) % points.length];
    const { point, distance } = closestPointOnSegment([ix, iy], a, b);
    const overlayDist = distance * ((geo.sx + geo.sy) / 2);
    if (overlayDist <= hitPx && overlayDist < bestDist) {
      bestDist = overlayDist;
      bestIdx = i;
      bestPoint = point;
    }
  }
  if (bestIdx == null) return null;
  return { index: bestIdx, insertAt: /** @type {Point} */ (bestPoint) };
}

/**
 * Рамер-Дуглас-Пекер для разомкнутой цепочки точек [start..end] включительно.
 * @param {Point[]} points @param {number} epsilon
 * @returns {Point[]}
 */
function rdpOpen(points, epsilon) {
  if (points.length < 3) return points.slice();
  const first = points[0];
  const last = points[points.length - 1];
  let maxDist = -1;
  let maxIdx = -1;
  for (let i = 1; i < points.length - 1; i++) {
    const d = distanceToSegmentPx(points[i], first, last);
    if (d > maxDist) {
      maxDist = d;
      maxIdx = i;
    }
  }
  if (maxDist <= epsilon) return [first, last];
  const left = rdpOpen(points.slice(0, maxIdx + 1), epsilon);
  const right = rdpOpen(points.slice(maxIdx), epsilon);
  return left.slice(0, -1).concat(right);
}

function clonePoints(points) {
  return Array.isArray(points) ? points.map((p) => [p[0], p[1]]) : [];
}

/**
 * Упрощение ЗАМКНУТОГО контура алгоритмом Рамера-Дугласа-Пекера.
 * Контур замыкается виртуальным ребром (последняя точка → первая), поэтому
 * первая точка контура всегда остаётся на месте (как «якорь» разреза) —
 * стандартный приём для применения RDP к closed polyline без O(n^2) поиска
 * наиболее удалённой пары точек (важно: contour из Ultralytics masks.xy
 * может содержать тысячи точек).
 * @param {Point[]} points @param {number} epsilon допуск в той же системе координат, что points
 * @returns {Point[]}
 */
export function simplifyPolygonRDP(points, epsilon) {
  if (!Array.isArray(points) || points.length <= MIN_POLYGON_POINTS) {
    return clonePoints(points);
  }
  if (!(epsilon > 0)) return clonePoints(points);

  const closedChain = points.concat([points[0]]);
  const simplified = rdpOpen(closedChain, epsilon);
  const result = simplified.slice(0, -1);

  if (result.length < MIN_POLYGON_POINTS) {
    return clonePoints(points);
  }
  return result;
}
