/**
 * DOM-блок «Упрощение маски» для инспектора (правая панель).
 * Не хранит долгоживущего состояния — на каждую пересборку buildInspector()
 * стартует со слайдера в положении 0 (см. примечание про drag thumb ниже).
 */
import { simplifyPolygonRDP, POLYGON_SIMPLIFY_MAX_EPSILON_PX, MIN_POLYGON_POINTS } from "./geometry.js";

/**
 * @typedef {{
 *   pushUndoCheckpoint: () => void,
 *   requestDraw: () => void,
 *   setSimplifyPreview: (detId: number, points: Array<[number,number]>) => void,
 *   clearSimplifyPreview: (detId: number) => void,
 *   applySimplify: (points: Array<[number,number]>) => void,
 * }} SimplifyBlockCtx
 */

/**
 * @param {HTMLElement} panel
 * @param {{ id: number, segment: Array<[number,number]> }} det
 * @param {SimplifyBlockCtx} ctx
 */
export function appendPolygonSimplifyBlock(panel, det, ctx) {
  if (!Array.isArray(det.segment) || det.segment.length < MIN_POLYGON_POINTS) return;

  /*
   * buildInspector() в app.js полностью пересобирает DOM инспектора при любом
   * изменении состояния. Если бы мы вызывали такую пересборку на каждый
   * input-событие слайдера, нативный drag thumb сломался бы (узел удаляется
   * из DOM прямо во время перетаскивания). Поэтому здесь слайдер живёт внутри
   * одного вызова appendPolygonSimplifyBlock: на input мы обновляем только
   * текстовые узлы и просим перерисовать canvas (requestDraw), не трогая DOM.
   */

  ctx.clearSimplifyPreview(det.id);

  const wrap = document.createElement("div");
  wrap.className = "inspector-simplify";
  wrap.dataset.detId = String(det.id);

  const title = document.createElement("div");
  title.className = "inspector-simplify-title";
  title.textContent = "Упрощение маски";
  wrap.appendChild(title);

  const countRow = document.createElement("div");
  countRow.className = "inspector-simplify-count";
  countRow.textContent = `Точек в маске: ${det.segment.length}`;
  wrap.appendChild(countRow);

  const sliderRow = document.createElement("div");
  sliderRow.className = "inspector-simplify-slider-row";

  const slider = document.createElement("input");
  slider.type = "range";
  slider.className = "inspector-simplify-slider";
  slider.min = "0";
  slider.max = String(POLYGON_SIMPLIFY_MAX_EPSILON_PX);
  slider.step = "0.1";
  slider.value = "0";
  slider.setAttribute("aria-label", "Допуск упрощения маски");
  sliderRow.appendChild(slider);
  wrap.appendChild(sliderRow);

  const previewRow = document.createElement("div");
  previewRow.className = "inspector-simplify-preview";
  previewRow.textContent = `Допуск: 0.0 px — точек без изменений (${det.segment.length})`;
  wrap.appendChild(previewRow);

  const applyBtn = document.createElement("button");
  applyBtn.type = "button";
  applyBtn.className = "inspector-simplify-apply-btn";
  applyBtn.textContent = "Применить";
  applyBtn.disabled = true;
  wrap.appendChild(applyBtn);

  /** @type {Array<[number,number]>|null} */
  let pendingPreview = null;

  slider.addEventListener("input", () => {
    const epsilon = Number(slider.value) || 0;
    if (epsilon <= 0) {
      pendingPreview = null;
      applyBtn.disabled = true;
      previewRow.textContent = `Допуск: 0.0 px — точек без изменений (${det.segment.length})`;
      ctx.clearSimplifyPreview(det.id);
      return;
    }
    const simplified = simplifyPolygonRDP(det.segment, epsilon);
    pendingPreview = simplified;
    const changed = simplified.length < det.segment.length;
    applyBtn.disabled = !changed;
    previewRow.textContent = changed
      ? `Допуск: ${epsilon.toFixed(1)} px — точек: ${det.segment.length} → ${simplified.length}`
      : `Допуск: ${epsilon.toFixed(1)} px — точек без изменений (${det.segment.length})`;
    ctx.setSimplifyPreview(det.id, changed ? simplified : null);
  });

  applyBtn.addEventListener("click", () => {
    if (!pendingPreview || pendingPreview.length < MIN_POLYGON_POINTS) return;
    ctx.pushUndoCheckpoint();
    ctx.applySimplify(pendingPreview);
  });

  panel.appendChild(wrap);
  return wrap;
}
