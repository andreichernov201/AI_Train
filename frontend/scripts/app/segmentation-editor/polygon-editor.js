/**
 * Состояние и обработчики взаимодействий для редактирования полигонов
 * сегментации (выбор/перетаскивание вершин, перемещение всей маски,
 * добавление новой маски точка-за-точкой, упрощение контура).
 *
 * Модуль не знает о DOM/app.js напрямую — всё, что ему нужно, приходит через
 * `deps`, переданные в createPolygonEditor(). app.js делает только тонкую
 * диспетчеризацию вызовов из существующих обработчиков событий.
 */
import {
  VERTEX_HIT_PX,
  MIN_POLYGON_POINTS,
  pointInPolygon,
  polygonBBox,
  translatePolygonClamped,
  hitTestPolygonVertex,
  hitTestPolygonEdge,
  imageToOverlayPoint,
  overlayToImagePoint,
  simplifyPolygonRDP,
} from "./geometry.js";

/**
 * @typedef {{
 *   getOverlayGeometry: () => null | { offsetX: number, offsetY: number, sx: number, sy: number, imgW0: number, imgH0: number },
 *   getCurrentImage: () => any | null,
 *   getSelectedDetection: () => any | null,
 *   setSelectedDetectionId: (id: number|null) => void,
 *   allocateNextDetId: (im: any) => number,
 *   getNewShapeTrainClass: (im: any) => { id: number, name: string },
 *   pushUndoCheckpoint: () => void,
 *   touchBatch: () => void,
 *   scheduleWorkspaceAutosave: (delayMs: number) => void,
 *   updateBatchNavUi: () => void,
 *   buildRightPanel: () => void,
 *   requestDraw: () => void,
 *   isEditMode: () => boolean,
 *   getEditorTool: () => string,
 *   setEditorTool: (tool: string) => void,
 *   markAnnotationHuman: (im: any, type: "seg") => void,
 * }} PolygonEditorDeps
 */

/** @param {PolygonEditorDeps} deps */
export function createPolygonEditor(deps) {
  /**
   * @type {null | {
   *   kind: "moveVertex",
   *   detId: number,
   *   vertexIndex: number,
   *   startIx: number,
   *   startIy: number,
   * } | {
   *   kind: "movePolygon",
   *   detId: number,
   *   startIx: number,
   *   startIy: number,
   *   origSegment: Array<[number, number]>,
   * }}
   */
  let interaction = null;

  /** Черновик инструмента «Добавить полигон». */
  let draft = /** @type {null | { points: Array<[number, number]>, cursor: [number, number] | null }} */ (
    null
  );

  /** Превью упрощения маски (для слайдера в инспекторе), по detId. */
  const simplifyPreviewByDetId = new Map();

  function clampPoint(geo, ix, iy) {
    return [
      Math.max(0, Math.min(ix, geo.imgW0)),
      Math.max(0, Math.min(iy, geo.imgH0)),
    ];
  }

  function findDet(im, detId) {
    return im?.detections?.find((d) => d.id === detId) ?? null;
  }

  function hasUsableSegment(det) {
    return !!det && Array.isArray(det.segment) && det.segment.length >= MIN_POLYGON_POINTS;
  }

  /** Финализация после drag (move/resize вершины или всей маски). */
  function finalizeGeometryEdit(im, det) {
    det.box = polygonBBox(det.segment);
    det.annotation_type = "seg";
    det.source = "human";
    im.edited = true;
    im.reviewed = false;
    deps.markAnnotationHuman(im, "seg");
    deps.touchBatch();
    deps.scheduleWorkspaceAutosave(0);
    deps.updateBatchNavUi();
    deps.buildRightPanel();
  }

  /** point-in-polygon обёртка для выбора объекта по форме маски, а не по боксу. */
  function hitTestSelection(geo, ox, oy, det) {
    if (!hasUsableSegment(det)) return false;
    const { ix, iy } = overlayToImagePoint(geo, ox, oy);
    return pointInPolygon([ix, iy], det.segment);
  }

  function startAddPolygonDraft(geo, ox, oy) {
    const { ix, iy } = overlayToImagePoint(geo, ox, oy);
    const pt = clampPoint(geo, ix, iy);
    draft = { points: [pt], cursor: pt };
    deps.requestDraw();
  }

  function finishAddPolygonDraft() {
    const im = deps.getCurrentImage();
    const pts = draft?.points ?? [];
    draft = null;
    if (!im || pts.length < MIN_POLYGON_POINTS) {
      deps.requestDraw();
      return;
    }
    deps.pushUndoCheckpoint();
    const tc = deps.getNewShapeTrainClass(im);
    const segment = pts.map((p) => [p[0], p[1]]);
    const newDet = {
      id: deps.allocateNextDetId(im),
      cls_id: tc.id,
      cls_name: tc.name,
      conf: 1,
      box: polygonBBox(segment),
      annotation_type: "seg",
      source: "human",
      segment,
    };
    im.detections.push(newDet);
    deps.markAnnotationHuman(im, "seg");
    deps.setSelectedDetectionId(newDet.id);
    im.status = "detected";
    im.edited = true;
    im.reviewed = false;
    deps.setEditorTool("select");
    deps.touchBatch();
    deps.scheduleWorkspaceAutosave(0);
    deps.updateBatchNavUi();
    deps.buildRightPanel();
    deps.requestDraw();
  }

  function handleAddPolygonPointerDown(geo, ox, oy) {
    if (!draft) {
      startAddPolygonDraft(geo, ox, oy);
      return true;
    }
    const first = draft.points[0];
    const firstOverlay = imageToOverlayPoint(geo, first[0], first[1]);
    const distToFirst = Math.hypot(ox - firstOverlay.ox, oy - firstOverlay.oy);
    if (draft.points.length >= MIN_POLYGON_POINTS && distToFirst <= VERTEX_HIT_PX) {
      finishAddPolygonDraft();
      return true;
    }
    const { ix, iy } = overlayToImagePoint(geo, ox, oy);
    draft.points.push(clampPoint(geo, ix, iy));
    draft.cursor = draft.points[draft.points.length - 1];
    deps.requestDraw();
    return true;
  }

  function handleSelectPointerDown(geo, ox, oy, e) {
    const det = deps.getSelectedDetection();
    if (!hasUsableSegment(det)) return false;
    const im = deps.getCurrentImage();
    if (!im) return false;

    const modifier = !!(e?.altKey || e?.ctrlKey || e?.metaKey);
    const segment = det.segment;

    const vIdx = hitTestPolygonVertex(geo, ox, oy, segment);
    if (vIdx != null) {
      if (modifier) {
        if (segment.length <= MIN_POLYGON_POINTS) return true;
        deps.pushUndoCheckpoint();
        segment.splice(vIdx, 1);
        finalizeGeometryEdit(im, det);
        deps.requestDraw();
        return true;
      }
      deps.pushUndoCheckpoint();
      det.annotation_type = "seg";
      det.source = "human";
      deps.markAnnotationHuman(im, "seg");
      const { ix, iy } = overlayToImagePoint(geo, ox, oy);
      interaction = {
        kind: "moveVertex",
        detId: det.id,
        vertexIndex: vIdx,
        startIx: ix,
        startIy: iy,
      };
      return true;
    }

    const edge = hitTestPolygonEdge(geo, ox, oy, segment);
    if (edge && modifier) {
      deps.pushUndoCheckpoint();
      det.annotation_type = "seg";
      det.source = "human";
      deps.markAnnotationHuman(im, "seg");
      segment.splice(edge.index + 1, 0, edge.insertAt);
      const { ix, iy } = overlayToImagePoint(geo, ox, oy);
      interaction = {
        kind: "moveVertex",
        detId: det.id,
        vertexIndex: edge.index + 1,
        startIx: ix,
        startIy: iy,
      };
      deps.requestDraw();
      return true;
    }

    const { ix, iy } = overlayToImagePoint(geo, ox, oy);
    if (pointInPolygon([ix, iy], segment)) {
      deps.pushUndoCheckpoint();
      det.annotation_type = "seg";
      det.source = "human";
      deps.markAnnotationHuman(im, "seg");
      interaction = {
        kind: "movePolygon",
        detId: det.id,
        startIx: ix,
        startIy: iy,
        origSegment: segment.map((p) => [p[0], p[1]]),
      };
      return true;
    }

    return false;
  }

  /**
   * @param {MouseEvent} e
   * @param {number} ox CSS px на overlay
   * @param {number} oy CSS px на overlay
   * @returns {boolean} true, если событие обработано (app.js должен сделать return)
   */
  function handlePointerDown(e, ox, oy) {
    if (!deps.isEditMode()) return false;
    const geo = deps.getOverlayGeometry();
    if (!geo) return false;

    const tool = deps.getEditorTool();
    if (tool === "addPolygon") {
      return handleAddPolygonPointerDown(geo, ox, oy);
    }
    if (tool === "select") {
      return handleSelectPointerDown(geo, ox, oy, e);
    }
    return false;
  }

  /**
   * @param {MouseEvent} e
   * @param {number} ox @param {number} oy
   * @returns {boolean}
   */
  function handlePointerMove(e, ox, oy) {
    if (!interaction) return false;
    const geo = deps.getOverlayGeometry();
    if (!geo) return true;
    const im = deps.getCurrentImage();
    const det = findDet(im, interaction.detId);
    if (!im || !det) {
      interaction = null;
      return true;
    }

    const { ix, iy } = overlayToImagePoint(geo, ox, oy);

    if (interaction.kind === "moveVertex") {
      det.segment[interaction.vertexIndex] = clampPoint(geo, ix, iy);
      deps.requestDraw();
      return true;
    }

    if (interaction.kind === "movePolygon") {
      const dx = ix - interaction.startIx;
      const dy = iy - interaction.startIy;
      det.segment = translatePolygonClamped(
        interaction.origSegment,
        dx,
        dy,
        geo.imgW0,
        geo.imgH0
      );
      deps.requestDraw();
      return true;
    }

    return false;
  }

  /** @returns {boolean} */
  function handlePointerUp() {
    if (!interaction) return false;
    const im = deps.getCurrentImage();
    const det = findDet(im, interaction.detId);
    interaction = null;
    if (im && det) finalizeGeometryEdit(im, det);
    deps.requestDraw();
    return true;
  }

  function isInteracting() {
    return interaction != null;
  }

  /** Вызывается из overlay mousemove (не document) — обновляет превью рисуемого полигона. */
  function handleOverlayHover(ox, oy) {
    if (!draft) return;
    const geo = deps.getOverlayGeometry();
    if (!geo) return;
    const { ix, iy } = overlayToImagePoint(geo, ox, oy);
    draft.cursor = clampPoint(geo, ix, iy);
    deps.requestDraw();
  }

  /**
   * @param {{offsetX:number,offsetY:number,sx:number,sy:number,imgW0:number,imgH0:number}} geo
   * @param {number|null} ox @param {number|null} oy @param {boolean} altKey
   * @returns {string|null} CSS-класс курсора или null
   */
  function cursorClassFor(geo, ox, oy, altKey) {
    if (!deps.isEditMode()) return null;
    const tool = deps.getEditorTool();
    if (tool === "addPolygon") return "overlay-cursor-crosshair";
    if (tool !== "select") return null;
    if (ox == null || oy == null) return null;
    const det = deps.getSelectedDetection();
    if (!hasUsableSegment(det)) return null;

    const vIdx = hitTestPolygonVertex(geo, ox, oy, det.segment);
    if (vIdx != null) return altKey ? "overlay-cursor-remove-vertex" : "overlay-cursor-vertex";

    const edge = hitTestPolygonEdge(geo, ox, oy, det.segment);
    if (edge) return altKey ? "overlay-cursor-insert-vertex" : "overlay-cursor-move";

    const { ix, iy } = overlayToImagePoint(geo, ox, oy);
    if (pointInPolygon([ix, iy], det.segment)) return "overlay-cursor-move";
    return null;
  }

  /** Сброс черновика/drag без коммита (Escape, смена инструмента/кадра). */
  function cancel() {
    let handled = false;
    if (draft) {
      draft = null;
      handled = true;
    }
    if (interaction) {
      interaction = null;
      handled = true;
    }
    if (handled) deps.requestDraw();
    return handled;
  }

  /** Завершить рисуемый полигон по Enter. */
  function handleEnterKey() {
    if (!draft) return false;
    finishAddPolygonDraft();
    return true;
  }

  function setSimplifyPreview(detId, previewPoints) {
    simplifyPreviewByDetId.clear();
    if (previewPoints && previewPoints.length >= MIN_POLYGON_POINTS) {
      simplifyPreviewByDetId.set(detId, previewPoints);
    }
    deps.requestDraw();
  }

  function clearSimplifyPreview(detId) {
    simplifyPreviewByDetId.delete(detId);
    deps.requestDraw();
  }

  /** Применить упрощённый контур к объекту (коммит из инспектора). */
  function applySimplify(det, simplifiedPoints) {
    const im = deps.getCurrentImage();
    if (!im || !det || simplifiedPoints.length < MIN_POLYGON_POINTS) return;
    det.segment = simplifiedPoints.map((p) => [p[0], p[1]]);
    simplifyPreviewByDetId.delete(det.id);
    finalizeGeometryEdit(im, det);
    deps.requestDraw();
  }

  /**
   * Рисует вершины выделенного полигона, незакончённый рисуемый полигон и
   * превью упрощения. Вызывается из draw() после drawOneMask/drawOneBox.
   * @param {CanvasRenderingContext2D} ctx
   * @param {{offsetX:number,offsetY:number,sx:number,sy:number}} geo
   */
  function renderOverlay(ctx, geo) {
    const toOverlay = (ix, iy) => imageToOverlayPoint(geo, ix, iy);

    if (deps.isEditMode() && deps.getEditorTool() === "select") {
      const det = deps.getSelectedDetection();
      if (hasUsableSegment(det)) {
        const preview = simplifyPreviewByDetId.get(det.id);
        if (preview) {
          ctx.save();
          ctx.setLineDash([6, 4]);
          ctx.strokeStyle = "rgba(250, 204, 21, 0.9)";
          ctx.lineWidth = 2;
          ctx.beginPath();
          preview.forEach(([px, py], i) => {
            const p = toOverlay(px, py);
            if (i === 0) ctx.moveTo(p.ox, p.oy);
            else ctx.lineTo(p.ox, p.oy);
          });
          ctx.closePath();
          ctx.stroke();
          ctx.setLineDash([]);
          for (const [px, py] of preview) {
            const p = toOverlay(px, py);
            ctx.fillStyle = "rgba(250, 204, 21, 0.95)";
            ctx.beginPath();
            ctx.arc(p.ox, p.oy, 3, 0, Math.PI * 2);
            ctx.fill();
          }
          ctx.restore();
        }

        ctx.save();
        ctx.fillStyle = "rgba(255,255,255,0.95)";
        ctx.strokeStyle = "#0b0b0b";
        ctx.lineWidth = 1;
        const hs = 7;
        for (const [px, py] of det.segment) {
          const p = toOverlay(px, py);
          ctx.fillRect(p.ox - hs / 2, p.oy - hs / 2, hs, hs);
          ctx.strokeRect(p.ox - hs / 2, p.oy - hs / 2, hs, hs);
        }
        ctx.restore();
      }
    }

    if (draft && draft.points.length) {
      ctx.save();
      ctx.lineWidth = 2;
      ctx.strokeStyle = "rgba(15,15,15,0.95)";
      ctx.fillStyle = "rgba(255,255,255,0.30)";
      ctx.beginPath();
      draft.points.forEach(([px, py], i) => {
        const p = toOverlay(px, py);
        if (i === 0) ctx.moveTo(p.ox, p.oy);
        else ctx.lineTo(p.ox, p.oy);
      });
      if (draft.cursor) {
        const p = toOverlay(draft.cursor[0], draft.cursor[1]);
        ctx.lineTo(p.ox, p.oy);
      }
      if (draft.points.length >= 2 && draft.cursor) {
        ctx.closePath();
        ctx.fill();
      }
      ctx.stroke();

      for (let i = 0; i < draft.points.length; i++) {
        const p = toOverlay(draft.points[i][0], draft.points[i][1]);
        const isFirst = i === 0;
        ctx.beginPath();
        ctx.arc(p.ox, p.oy, isFirst ? 5 : 4, 0, Math.PI * 2);
        ctx.fillStyle = isFirst ? "rgba(255,255,255,0.95)" : "rgba(63, 130, 247, 0.95)";
        ctx.fill();
        ctx.strokeStyle = "rgba(11,11,11,0.8)";
        ctx.lineWidth = 1;
        ctx.stroke();
      }
      ctx.restore();
    }
  }

  return {
    handlePointerDown,
    handlePointerMove,
    handlePointerUp,
    isInteracting,
    handleOverlayHover,
    cursorClassFor,
    renderOverlay,
    cancel,
    handleEnterKey,
    hitTestSelection,
    setSimplifyPreview,
    clearSimplifyPreview,
    applySimplify,
    simplifyPolygonRDP,
    MIN_POLYGON_POINTS,
  };
}
