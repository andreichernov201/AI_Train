(() => {
  /** @type {HTMLButtonElement|null} */
  const uploadBtn = document.getElementById("upload-btn");
  /** @type {HTMLButtonElement|null} */
  const clearBtn = document.getElementById("clear-btn");
  /** @type {HTMLButtonElement|null} */
  const runBtn = document.getElementById("run-btn");
  /** @type {HTMLInputElement|null} */
  const fileInput = document.getElementById("file-input");
  /** @type {HTMLImageElement|null} */
  const previewImage = document.getElementById("preview-image");
  /** @type {HTMLElement|null} */
  const placeholderText = document.getElementById("placeholder-text");
  /** @type {HTMLElement|null} */
  const imageFrame = document.getElementById("image-frame");
  /** @type {HTMLElement|null} */
  const imageStack = document.getElementById("image-stack");
  /** @type {HTMLCanvasElement|null} */
  const overlay = document.getElementById("overlay-canvas");

  /** @type {HTMLButtonElement|null} */
  const downloadImageBtn = document.getElementById("download-image-btn");
  /** @type {HTMLButtonElement|null} */
  const downloadYoloBtn = document.getElementById("download-yolo-btn");
  /** @type {HTMLButtonElement|null} */
  const downloadJsonBtn = document.getElementById("download-json-btn");

  /** @type {HTMLElement|null} */
  const totalObjectsEl = document.getElementById("total-objects");
  /** @type {HTMLElement|null} */
  const groupsRoot = document.getElementById("groups-root");
  /** @type {HTMLInputElement|null} */
  const confFilterRange = document.getElementById("confidence-filter-range");
  /** @type {HTMLElement|null} */
  const confFilterValue = document.getElementById("confidence-filter-value");

  if (
    !uploadBtn ||
    !clearBtn ||
    !runBtn ||
    !fileInput ||
    !previewImage ||
    !placeholderText ||
    !imageFrame ||
    !imageStack ||
    !overlay ||
    !totalObjectsEl ||
    !groupsRoot ||
    !confFilterRange ||
    !confFilterValue ||
    !downloadImageBtn ||
    !downloadYoloBtn ||
    !downloadJsonBtn
  ) {
    return;
  }

  /** @type {File|null} */
  let currentFile = null;
  /** @type {{width:number,height:number}|null} */
  let originalSize = null;
  /** @type {Array<{id:number,cls_id:number,cls_name:string,conf:number,box:[number,number,number,number]}>} */
  let detections = [];
  /** @type {Map<number, boolean>} */
  const detEnabled = new Map();
  /** @type {Map<string, {enabled:boolean, collapsed:boolean}>} */
  const categoryState = new Map();
  /** @type {number} */
  let confThreshold = 0;

  const COLORS = [
    "#4ade80",
    "#60a5fa",
    "#f97316",
    "#f472b6",
    "#facc15",
    "#a78bfa",
    "#22d3ee",
    "#fb7185",
  ];

  function fmtConf(x) {
    return (Math.round(x * 100) / 100).toFixed(2);
  }

  function setStatus(text) {
    runBtn.textContent = text;
  }

  function clearAll() {
    currentFile = null;
    originalSize = null;
    detections = [];
    detEnabled.clear();
    categoryState.clear();

    fileInput.value = "";
    previewImage.src = "";
    previewImage.style.display = "none";
    placeholderText.style.display = "block";
    groupsRoot.innerHTML = "";
    totalObjectsEl.textContent = "0";
    confThreshold = 0;
    confFilterRange.value = "0";
    confFilterValue.textContent = fmtConf(confThreshold);
    setStatus("Запустить распознавание");
    clearCanvas();
  }

  function clearCanvas() {
    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
  }

  function resizeCanvasToFrame() {
    const frameRect = imageFrame.getBoundingClientRect();
    if (frameRect.width <= 0 || frameRect.height <= 0) return;

    const dpr = window.devicePixelRatio || 1;
    overlay.style.width = `${frameRect.width}px`;
    overlay.style.height = `${frameRect.height}px`;
    overlay.width = Math.max(1, Math.floor(frameRect.width * dpr));
    overlay.height = Math.max(1, Math.floor(frameRect.height * dpr));

    const ctx = overlay.getContext("2d");
    if (ctx) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function getColorByClass(clsId) {
    return COLORS[Math.abs(clsId) % COLORS.length];
  }

  function draw() {
    if (!originalSize) return;
    resizeCanvasToFrame();
    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    const frameRect = imageFrame.getBoundingClientRect();
    if (frameRect.width <= 0 || frameRect.height <= 0) return;

    // Рассчитываем, как именно картинка вписана в frame при object-fit: contain
    const imgW0 = originalSize.width;
    const imgH0 = originalSize.height;
    if (!imgW0 || !imgH0) return;

    const frameAR = frameRect.width / frameRect.height;
    const imgAR = imgW0 / imgH0;

    let drawnW;
    let drawnH;
    let offsetX;
    let offsetY;

    if (frameAR > imgAR) {
      // Рамка шире, чем картинка: высота совпадает, слева/справа отступы
      drawnH = frameRect.height;
      drawnW = drawnH * imgAR;
      offsetX = (frameRect.width - drawnW) / 2;
      offsetY = 0;
    } else {
      // Рамка уже: ширина совпадает, сверху/снизу отступы
      drawnW = frameRect.width;
      drawnH = drawnW / imgAR;
      offsetX = 0;
      offsetY = (frameRect.height - drawnH) / 2;
    }

    const sx = drawnW / imgW0;
    const sy = drawnH / imgH0;

    ctx.lineWidth = 2;
    ctx.font = "12px system-ui, sans-serif";

    for (const d of detections) {
      if (d.conf < confThreshold) continue;
      const cat = d.cls_name;
      const catEnabled = categoryState.get(cat)?.enabled ?? true;
      if (!catEnabled) continue;
      if (detEnabled.get(d.id) === false) continue;

      const [x1, y1, x2, y2] = d.box;
      const x = offsetX + x1 * sx;
      const y = offsetY + y1 * sy;
      const w = (x2 - x1) * sx;
      const h = (y2 - y1) * sy;

      const color = getColorByClass(d.cls_id);
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.strokeRect(x, y, w, h);

      const label = `${d.cls_name} #${d.id + 1} (${fmtConf(d.conf)})`;
      const padX = 6;
      const padY = 4;
      const textW = ctx.measureText(label).width;
      const boxW = textW + padX * 2;
      const boxH = 18;
      const bx = Math.max(0, Math.min(x, frameRect.width - boxW));
      const by = Math.max(0, y - boxH - 2);
      ctx.globalAlpha = 0.85;
      ctx.fillRect(bx, by, boxW, boxH);
      ctx.globalAlpha = 1;
      ctx.fillStyle = "#0b0b0b";
      ctx.fillText(label, bx + padX, by + 13);
    }
  }

  /**
   * Сохранение размеченной картинки как JPEG.
   * Использует натуральный размер изображения (naturalWidth/naturalHeight),
   * а боксы масштабируются из оригинальных координат.
   */
  function downloadAnnotatedImage() {
    if (!originalSize || !previewImage.src || !detections.length) {
      alert("Нет распознанных объектов для сохранения.");
      return;
    }

    const imgW = previewImage.naturalWidth || originalSize.width;
    const imgH = previewImage.naturalHeight || originalSize.height;

    const canvas = document.createElement("canvas");
    canvas.width = imgW;
    canvas.height = imgH;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Рисуем исходное изображение
    ctx.drawImage(previewImage, 0, 0, imgW, imgH);

    const sx = imgW / originalSize.width;
    const sy = imgH / originalSize.height;

    ctx.lineWidth = 2;
    ctx.font = "16px system-ui, sans-serif";

    for (const d of detections) {
      if (d.conf < confThreshold) continue;
      const cat = d.cls_name;
      const catEnabled = categoryState.get(cat)?.enabled ?? true;
      if (!catEnabled) continue;
      if (detEnabled.get(d.id) === false) continue;

      const [x1, y1, x2, y2] = d.box;
      const x = x1 * sx;
      const y = y1 * sy;
      const w = (x2 - x1) * sx;
      const h = (y2 - y1) * sy;

      const color = getColorByClass(d.cls_id);
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.strokeRect(x, y, w, h);

      const label = `${d.cls_name} #${d.id + 1} (${fmtConf(d.conf)})`;
      const padX = 8;
      const padY = 6;
      const textW = ctx.measureText(label).width;
      const boxW = textW + padX * 2;
      const boxH = 24;
      const bx = Math.max(0, Math.min(x, imgW - boxW));
      const by = Math.max(0, y - boxH - 2);

      ctx.globalAlpha = 0.85;
      ctx.fillRect(bx, by, boxW, boxH);
      ctx.globalAlpha = 1;
      ctx.fillStyle = "#0b0b0b";
      ctx.fillText(label, bx + padX, by + (boxH - padY));
    }

    const link = document.createElement("a");
    link.href = canvas.toDataURL("image/jpeg", 0.9);
    link.download = "annotated.jpg";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  /**
   * Экспорт разметки в формате YOLO:
   * <class_id> <x_center> <y_center> <width> <height>
   * Все координаты нормализованы [0,1] относительно оригинального размера.
   */
  function downloadYoloAnnotations() {
    if (!originalSize || !detections.length) {
      alert("Нет распознанных объектов для экспорта.");
      return;
    }

    const lines = [];
    for (const d of detections) {
      if (d.conf < confThreshold) continue;
      // Экспортируем только включённые в интерфейсе боксы
      const cat = d.cls_name;
      const catEnabled = categoryState.get(cat)?.enabled ?? true;
      if (!catEnabled) continue;
      if (detEnabled.get(d.id) === false) continue;

      const [x1, y1, x2, y2] = d.box;
      const cx = (x1 + x2) / 2;
      const cy = (y1 + y2) / 2;
      const w = x2 - x1;
      const h = y2 - y1;

      const nx = cx / originalSize.width;
      const ny = cy / originalSize.height;
      const nw = w / originalSize.width;
      const nh = h / originalSize.height;

      const clsId = d.cls_id ?? 0;
      lines.push(
        [
          clsId,
          nx.toFixed(6),
          ny.toFixed(6),
          nw.toFixed(6),
          nh.toFixed(6),
        ].join(" ")
      );
    }

    if (!lines.length) {
      alert("Все боксы выключены. Включите нужные объекты перед экспортом.");
      return;
    }

    const blob = new Blob([lines.join("\n") + "\n"], {
      type: "text/plain;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");
    link.href = url;
    link.download = "annotations_yolo.txt";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  /**
   * Экспорт текущего состояния в JSON.
   * Включает информацию об изображении, пороге уверенности,
   * состояниях категорий/объектов и финальный список детекций.
   */
  function downloadJson() {
    if (!originalSize || !detections.length) {
      alert("Нет распознанных объектов для экспорта.");
      return;
    }

    const width = originalSize.width;
    const height = originalSize.height;

    /** @type {Array<any>} */
    const finalDetections = [];
    for (const d of detections) {
      if (d.conf < confThreshold) continue;

      const cat = d.cls_name;
      const catState = categoryState.get(cat) ?? { enabled: true, collapsed: false };
      const enabled = catState.enabled && detEnabled.get(d.id) !== false;

      const [x1, y1, x2, y2] = d.box;
      const cx = (x1 + x2) / 2;
      const cy = (y1 + y2) / 2;
      const w = x2 - x1;
      const h = y2 - y1;

      finalDetections.push({
        id: d.id,
        class_id: d.cls_id,
        class_name: d.cls_name,
        confidence: d.conf,
        enabled,
        bbox: {
          x1,
          y1,
          x2,
          y2,
          cx,
          cy,
          w,
          h,
        },
        bbox_normalized: {
          x1: x1 / width,
          y1: y1 / height,
          x2: x2 / width,
          y2: y2 / height,
          cx: cx / width,
          cy: cy / height,
          w: w / width,
          h: h / height,
        },
      });
    }

    if (!finalDetections.length) {
      alert("Все боксы выключены или ниже порога. Включите нужные объекты перед экспортом.");
      return;
    }

    const categories = {};
    for (const [name, state] of categoryState.entries()) {
      categories[name] = {
        enabled: state.enabled,
        collapsed: state.collapsed,
      };
    }

    const payload = {
      image: {
        file_name: currentFile ? currentFile.name : null,
        width,
        height,
      },
      settings: {
        confidence_threshold: confThreshold,
      },
      summary: {
        total_detections: detections.length,
        total_after_filter: finalDetections.length,
      },
      categories,
      detections: finalDetections,
    };

    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");
    link.href = url;
    link.download = "annotations.json";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  function groupByCategory(list) {
    /** @type {Map<string, typeof detections>} */
    const m = new Map();
    for (const d of list) {
      const k = d.cls_name;
      if (!m.has(k)) m.set(k, []);
      m.get(k).push(d);
    }
    return m;
  }

  function buildRightPanel() {
    groupsRoot.innerHTML = "";

    const filtered = detections.filter((d) => d.conf >= confThreshold);
    totalObjectsEl.textContent = String(filtered.length);

    const grouped = groupByCategory(filtered);
    const cats = Array.from(grouped.keys()).sort((a, b) => a.localeCompare(b));

    for (const cat of cats) {
      if (!categoryState.has(cat)) {
        categoryState.set(cat, { enabled: true, collapsed: false });
      }
      const st = categoryState.get(cat);

      const details = document.createElement("details");
      details.className = "group";
      details.open = !st.collapsed;
      details.addEventListener("toggle", () => {
        const s = categoryState.get(cat);
        if (s) s.collapsed = !details.open;
      });

      const summary = document.createElement("summary");
      summary.className = "group-summary";

      const left = document.createElement("div");
      left.className = "group-left";

      const catCheckbox = document.createElement("input");
      catCheckbox.type = "checkbox";
      catCheckbox.checked = st.enabled;
      catCheckbox.addEventListener("change", () => {
        const s = categoryState.get(cat);
        if (s) s.enabled = catCheckbox.checked;
        draw();
      });

      const title = document.createElement("span");
      title.className = "group-title";
      title.textContent = cat;

      const count = document.createElement("span");
      count.className = "group-count";
      count.textContent = String(grouped.get(cat).length);

      left.appendChild(catCheckbox);
      left.appendChild(title);
      summary.appendChild(left);
      summary.appendChild(count);

      const items = document.createElement("div");
      items.className = "group-items";

      for (const d of grouped.get(cat)) {
        if (!detEnabled.has(d.id)) detEnabled.set(d.id, true);

        const row = document.createElement("label");
        row.className = "det-row";

        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.checked = detEnabled.get(d.id) !== false;
        cb.addEventListener("change", () => {
          detEnabled.set(d.id, cb.checked);
          draw();
        });

        const chip = document.createElement("span");
        chip.className = "det-chip";
        chip.style.background = getColorByClass(d.cls_id);

        const text = document.createElement("span");
        text.className = "det-text";
        text.textContent = `#${d.id + 1} (${fmtConf(d.conf)})`;

        row.appendChild(cb);
        row.appendChild(chip);
        row.appendChild(text);
        items.appendChild(row);
      }

      details.appendChild(summary);
      details.appendChild(items);
      groupsRoot.appendChild(details);
    }
  }

  uploadBtn.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", (event) => {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      alert("Пожалуйста, выберите файл изображения.");
      fileInput.value = "";
      return;
    }

    currentFile = file;
    detections = [];
    detEnabled.clear();
    categoryState.clear();
    groupsRoot.innerHTML = "";
    totalObjectsEl.textContent = "0";
    setStatus("Запустить распознавание");

    const url = URL.createObjectURL(file);
    previewImage.onload = () => {
      placeholderText.style.display = "none";
      previewImage.style.display = "block";
      clearCanvas();
      resizeCanvasToFrame();
      URL.revokeObjectURL(url);
    };
    previewImage.src = url;
  });

  clearBtn.addEventListener("click", clearAll);

  downloadImageBtn.addEventListener("click", downloadAnnotatedImage);
  downloadYoloBtn.addEventListener("click", downloadYoloAnnotations);
  downloadJsonBtn.addEventListener("click", downloadJson);

  confFilterRange.addEventListener("input", () => {
    confThreshold = Number(confFilterRange.value) || 0;
    confFilterValue.textContent = fmtConf(confThreshold);
    buildRightPanel();
    draw();
  });

  runBtn.addEventListener("click", async () => {
    if (!currentFile) {
      alert("Сначала загрузите фото.");
      return;
    }

    try {
      setStatus("Распознаю...");
      runBtn.disabled = true;

      const fd = new FormData();
      fd.append("file", currentFile, currentFile.name);
      const resp = await fetch("/api/detect", { method: "POST", body: fd });

      const raw = await resp.text();
      let data = null;
      try {
        data = raw ? JSON.parse(raw) : null;
      } catch {
        throw new Error("Сервер вернул не‑JSON ответ. Проверь, запущен ли бэкенд /api/detect.");
      }
      if (!resp.ok || !data) {
        throw new Error(data?.error || "Detect failed");
      }

      originalSize = data.image;
      detections = data.detections || [];
      detEnabled.clear();
      categoryState.clear();

      buildRightPanel();
      draw();
      setStatus("Готово");
      setTimeout(() => setStatus("Запустить распознавание"), 800);
    } catch (e) {
      alert(String(e?.message || e));
      setStatus("Запустить распознавание");
    } finally {
      runBtn.disabled = false;
    }
  });

  window.addEventListener("resize", () => {
    if (!previewImage.src) return;
    draw();
  });

  clearAll();
})();

