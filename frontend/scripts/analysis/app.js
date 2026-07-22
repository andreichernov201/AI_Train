import { analysisApi } from "./api.js";
import {
  analysisHotkeyAction,
  isAnalysisTypingTarget,
} from "./hotkeys.js";
import { renderLocomotiveDiagram } from "./locomotive-diagram.js";
import {
  createAnalysisState,
  eventPrimaryIdentity,
  filteredEvents,
  persistAnalysisState,
} from "./state.js";
import { createAnalysisViewer } from "./viewer.js";

const TERMINAL_JOB_STATUSES = new Set(["completed", "failed", "stopped"]);

function fmtPercent(value) {
  return `${Math.round(Number(value || 0) * 100)}%`;
}

function shortId(value) {
  const text = String(value || "");
  return text.includes("_") ? text.split("_").pop().slice(0, 7).toUpperCase() : text.slice(0, 7);
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = String(text);
  return node;
}

function statusLabel(status) {
  return {
    confirmed: "Подтверждено",
    low_confidence: "Распознано",
    number_not_recognized: "Номер не распознан",
    partial_result: "Частичный результат",
    analysis_error: "Ошибка анализа",
    stopped: "Остановлено",
    processing: "Обработка",
    queued: "В очереди",
    completed: "Готово",
    no_train: "Состав не обнаружен",
    ready: "Готов к анализу",
  }[status] || status || "Неизвестно";
}

export function startAnalysisApp() {
  const annotationApp = document.getElementById("annotation-app") || document.querySelector(".app-columns");
  const analysisApp = document.getElementById("analysis-app");
  const annotationModeButton = document.getElementById("work-mode-detection");
  const analysisModeButton = document.getElementById("work-mode-segmentation");
  if (!annotationApp || !analysisApp || !annotationModeButton || !analysisModeButton) return;

  const refs = {
    annotationHeader: document.getElementById("annotation-header-progress") || document.querySelector(".app-header-progress"),
    analysisHeader: document.getElementById("analysis-header-progress"),
    analysisHeaderFiles: document.getElementById("analysis-header-files"),
    analysisHeaderEvents: document.getElementById("analysis-header-events"),
    analysisHeaderFill: document.getElementById("analysis-header-progress-fill"),
    photoInput: document.getElementById("analysis-photo-input"),
    videoInput: document.getElementById("analysis-video-input"),
    uploadPhoto: document.getElementById("analysis-upload-photo"),
    uploadVideo: document.getElementById("analysis-upload-video"),
    clearSession: document.getElementById("analysis-clear-session"),
    start: document.getElementById("analysis-start"),
    stop: document.getElementById("analysis-stop"),
    hotkeys: document.getElementById("analysis-hotkeys"),
    fileList: document.getElementById("analysis-file-list"),
    eventList: document.getElementById("analysis-event-list"),
    eventCount: document.getElementById("analysis-event-count"),
    search: document.getElementById("analysis-search"),
    filter: document.getElementById("analysis-status-filter"),
    sort: document.getElementById("analysis-sort"),
    message: document.getElementById("analysis-message"),
    progress: document.getElementById("analysis-progress"),
    progressFill: document.getElementById("analysis-progress-fill"),
    progressPercent: document.getElementById("analysis-progress-percent"),
    progressStage: document.getElementById("analysis-progress-stage"),
    progressMeta: document.getElementById("analysis-progress-meta"),
    pipeline: document.getElementById("analysis-pipeline"),
    viewerImage: document.getElementById("analysis-viewer-image"),
    viewerVideo: document.getElementById("analysis-viewer-video"),
    viewerCanvas: document.getElementById("analysis-viewer-canvas"),
    viewerEmpty: document.getElementById("analysis-viewer-empty"),
    viewerTitle: document.getElementById("analysis-viewer-title"),
    diagram: document.getElementById("analysis-diagram"),
    detail: document.getElementById("analysis-event-detail"),
    stats: document.getElementById("analysis-stats"),
    exportJson: document.getElementById("analysis-export-json"),
    exportCsv: document.getElementById("analysis-export-csv"),
    exportZip: document.getElementById("analysis-export-zip"),
    overlayTrain: document.getElementById("analysis-overlay-train"),
    overlayNumber: document.getElementById("analysis-overlay-number"),
    overlaySegmentation: document.getElementById("analysis-overlay-segmentation"),
    overlayOcr: document.getElementById("analysis-overlay-ocr"),
  };
  const state = createAnalysisState();
  const viewer = createAnalysisViewer({
    image: refs.viewerImage,
    video: refs.viewerVideo,
    canvas: refs.viewerCanvas,
    empty: refs.viewerEmpty,
  });
  let pollTimer = null;
  let pendingReanalysis = null;

  function showMessage(text, type = "info") {
    refs.message.textContent = text || "";
    refs.message.dataset.type = type;
    refs.message.hidden = !text;
  }

  function setMode(mode) {
    state.mode = mode === "analysis" ? "analysis" : "annotation";
    const analysis = state.mode === "analysis";
    annotationApp.hidden = analysis;
    analysisApp.hidden = !analysis;
    refs.annotationHeader.hidden = analysis;
    refs.analysisHeader.hidden = !analysis;
    annotationModeButton.classList.toggle("is-active", !analysis);
    analysisModeButton.classList.toggle("is-active", analysis);
    annotationModeButton.setAttribute("aria-checked", String(!analysis));
    analysisModeButton.setAttribute("aria-checked", String(analysis));
    document.body.classList.toggle("is-analysis-mode", analysis);
    persistAnalysisState(state);
    if (analysis) {
      void refreshAll();
      window.setTimeout(() => viewer.redraw(), 0);
    }
  }

  annotationModeButton.addEventListener(
    "click",
    (event) => {
      event.preventDefault();
      event.stopImmediatePropagation();
      setMode("annotation");
    },
    true
  );
  analysisModeButton.addEventListener(
    "click",
    (event) => {
      event.preventDefault();
      event.stopImmediatePropagation();
      setMode("analysis");
    },
    true
  );

  function analysisRunning() {
    return Boolean(state.job && !TERMINAL_JOB_STATUSES.has(state.job.status));
  }

  function toggleViewerLayer(checkbox, key) {
    state.viewer[key] = !state.viewer[key];
    checkbox.checked = state.viewer[key];
    renderViewer();
  }

  function selectRelativeEvent(step) {
    const events = filteredEvents(state);
    if (!events.length) return;
    const currentIndex = events.findIndex((event) => event.id === state.selectedEventId);
    const nextIndex = currentIndex < 0
      ? (step > 0 ? 0 : events.length - 1)
      : (currentIndex + step + events.length) % events.length;
    void selectEvent(events[nextIndex].id);
  }

  function performAnalysisHotkey(action) {
    const actions = {
      start: () => { if (!analysisRunning()) void start(); },
      stop: () => {
        if (refs.hotkeys.open) refs.hotkeys.open = false;
        else if (analysisRunning()) void stop();
      },
      "next-event": () => selectRelativeEvent(1),
      "previous-event": () => selectRelativeEvent(-1),
      search: () => {
        refs.search.focus();
        refs.search.select();
      },
      reanalyze: () => {
        if (state.selectedEventId && !analysisRunning()) void reanalyzeEvent();
      },
      confirm: () => {
        if (state.selectedEventId && !analysisRunning()) void updateEvent({ action: "confirm_event" });
      },
      "toggle-train": () => toggleViewerLayer(refs.overlayTrain, "showTrain"),
      "toggle-number": () => toggleViewerLayer(refs.overlayNumber, "showNumber"),
      "toggle-segmentation": () => toggleViewerLayer(refs.overlaySegmentation, "showSegmentation"),
      "toggle-ocr": () => toggleViewerLayer(refs.overlayOcr, "showOcr"),
      help: () => {
        refs.hotkeys.open = !refs.hotkeys.open;
        if (refs.hotkeys.open) refs.hotkeys.scrollIntoView({ block: "nearest", behavior: "smooth" });
      },
    };
    actions[action]?.();
  }

  document.addEventListener(
    "keydown",
    (event) => {
      if (state.mode !== "analysis") return;
      if (isAnalysisTypingTarget(event.target)) return;
      const action = analysisHotkeyAction(event);
      if (action) {
        event.preventDefault();
        performAnalysisHotkey(action);
      }
      event.stopImmediatePropagation();
    },
    true
  );

  for (const eventName of ["dragover", "drop", "paste"]) {
    document.addEventListener(
      eventName,
      (event) => {
        if (state.mode !== "analysis") return;
        if (eventName === "dragover") event.preventDefault();
        if (eventName === "drop") {
          event.preventDefault();
          const files = event.dataTransfer?.files;
          if (files?.length) void upload(Array.from(files));
        }
        event.stopImmediatePropagation();
      },
      true
    );
  }

  function syncSettingsFromUi() {
    analysisApp.querySelectorAll("[data-analysis-setting]").forEach((input) => {
      const key = input.dataset.analysisSetting;
      if (!key) return;
      state.settings[key] = input.type === "number" || input.type === "range"
        ? Number(input.value)
        : input.value;
    });
    persistAnalysisState(state);
  }

  function syncSettingsToUi() {
    analysisApp.querySelectorAll("[data-analysis-setting]").forEach((input) => {
      const key = input.dataset.analysisSetting;
      if (key && state.settings[key] != null) input.value = String(state.settings[key]);
    });
  }

  async function ensureSession(forceNew = false) {
    if (forceNew) state.sessionId = null;
    if (state.sessionId) {
      try {
        state.session = await analysisApi.getSession(state.sessionId);
      } catch {
        state.sessionId = null;
      }
    }
    if (!state.sessionId) {
      const listed = await analysisApi.listSessions();
      const recent = forceNew ? null : listed.sessions?.[0];
      state.session = recent
        ? await analysisApi.getSession(recent.id)
        : await analysisApi.createSession(state.settings);
      state.sessionId = state.session.id;
    }
    state.files = state.session.files || [];
    if (!state.selectedFileId || !state.files.some((file) => file.id === state.selectedFileId)) {
      state.selectedFileId = state.files[0]?.id || null;
    }
    persistAnalysisState(state);
  }

  async function refreshAll() {
    if (state.busy || !state.sessionId) return;
    try {
      state.session = await analysisApi.getSession(state.sessionId);
      state.files = state.session.files || [];
      const payload = await analysisApi.listEvents(state.sessionId);
      state.events = payload.events || [];
      if (state.selectedEventId && !state.events.some((event) => event.id === state.selectedEventId)) {
        state.selectedEventId = null;
        state.selectedEvent = null;
      }
      render();
    } catch (error) {
      showMessage(`Не удалось обновить анализ: ${error.message}`, "error");
    }
  }

  async function upload(files) {
    if (!files?.length) return;
    state.busy = true;
    renderActions();
    showMessage(`Загрузка файлов: ${files.length}…`, "info");
    try {
      await ensureSession();
      const result = await analysisApi.uploadFiles(state.sessionId, files);
      state.session = await analysisApi.getSession(state.sessionId);
      state.files = state.session.files || [];
      state.selectedFileId = result.files?.[0]?.id || state.selectedFileId;
      showMessage(
        result.errors?.length
          ? `Загружено ${result.files.length}, пропущено ${result.errors.length}.`
          : `Загружено файлов: ${result.files.length}.`,
        result.errors?.length ? "warning" : "success"
      );
      render();
    } catch (error) {
      showMessage(`Ошибка загрузки: ${error.message}`, "error");
    } finally {
      state.busy = false;
      renderActions();
      refs.photoInput.value = "";
      refs.videoInput.value = "";
    }
  }

  async function start() {
    if (!state.files.length || state.job && !TERMINAL_JOB_STATUSES.has(state.job.status)) return;
    syncSettingsFromUi();
    showMessage("Анализ запущен. Результаты будут появляться по мере обработки.", "info");
    try {
      state.job = await analysisApi.start(state.sessionId, state.settings);
      pendingReanalysis = null;
      state.events = [];
      state.selectedEventId = null;
      state.selectedEvent = null;
      render();
      renderProgress();
      schedulePoll(100);
    } catch (error) {
      showMessage(`Не удалось запустить анализ: ${error.message}`, "error");
    }
  }

  async function stop() {
    if (!state.job || TERMINAL_JOB_STATUSES.has(state.job.status)) return;
    try {
      state.job = await analysisApi.stopJob(state.job.id);
      showMessage("Остановка запрошена. Уже полученные события сохраняются.", "warning");
      renderProgress();
      renderActions();
    } catch (error) {
      showMessage(`Не удалось остановить анализ: ${error.message}`, "error");
    }
  }

  function schedulePoll(delay = 700) {
    if (pollTimer) window.clearTimeout(pollTimer);
    pollTimer = window.setTimeout(() => void pollJob(), delay);
  }

  async function pollJob() {
    if (!state.job?.id) return;
    try {
      state.job = await analysisApi.getJob(state.job.id);
      const eventsPayload = await analysisApi.listEvents(state.sessionId);
      state.events = eventsPayload.events || [];
      if (pendingReanalysis) {
        const replacement = state.events.find(
          (event) => event.metadata?.reanalyze_of === pendingReanalysis.eventId
        );
        if (replacement) {
          state.selectedEventId = replacement.id;
          state.selectedEvent = replacement;
          state.selectedFileId = pendingReanalysis.fileId;
          pendingReanalysis = null;
        }
      }
      state.session = await analysisApi.getSession(state.sessionId);
      state.files = state.session.files || [];
      render();
      if (!TERMINAL_JOB_STATUSES.has(state.job.status)) {
        schedulePoll();
      } else if (state.job.status === "failed") {
        showMessage(`Анализ завершился ошибкой: ${state.job.error || "неизвестная ошибка"}`, "error");
      } else if (state.job.status === "stopped") {
        showMessage("Анализ остановлен, промежуточные результаты сохранены.", "warning");
      } else {
        showMessage("Анализ завершён.", "success");
      }
    } catch (error) {
      showMessage(`Ошибка получения прогресса: ${error.message}`, "error");
      schedulePoll(1800);
    }
  }

  async function selectEvent(eventId) {
    state.selectedEventId = eventId;
    state.selectedEvent = state.events.find((event) => event.id === eventId) || null;
    render();
    try {
      state.selectedEvent = await analysisApi.getEvent(eventId);
      const index = state.events.findIndex((event) => event.id === eventId);
      if (index >= 0) state.events[index] = state.selectedEvent;
      render();
    } catch (error) {
      showMessage(`Не удалось открыть событие: ${error.message}`, "error");
    }
  }

  function selectedFile() {
    const fileId = state.selectedEvent?.source_file_id || state.selectedFileId;
    return state.files.find((file) => file.id === fileId) || null;
  }

  function render() {
    renderActions();
    renderHeader();
    renderFiles();
    renderEvents();
    renderStats();
    renderProgress();
    renderViewer();
    renderDetail();
  }

  function renderActions() {
    const running = Boolean(state.job && !TERMINAL_JOB_STATUSES.has(state.job.status));
    refs.start.disabled = state.busy || running || !state.files.length;
    refs.stop.disabled = !running;
    refs.uploadPhoto.disabled = state.busy || running;
    refs.uploadVideo.disabled = state.busy || running;
    refs.clearSession.disabled = state.busy || running || !state.sessionId;
    for (const button of [refs.exportJson, refs.exportCsv, refs.exportZip]) {
      button.disabled = !state.sessionId || !state.events.length;
    }
  }

  function renderHeader() {
    refs.analysisHeaderFiles.textContent = String(state.files.length);
    refs.analysisHeaderEvents.textContent = String(state.events.length);
    refs.analysisHeaderFill.style.width = `${Math.max(0, Math.min(100, Number(state.job?.progress?.percent || 0)))}%`;
  }

  function renderFiles() {
    refs.fileList.innerHTML = "";
    if (!state.files.length) {
      refs.fileList.appendChild(element("div", "analysis-list-empty", "Загрузите фотографии или видео."));
      return;
    }
    for (const file of state.files) {
      const button = element("button", "analysis-file-item");
      button.type = "button";
      button.classList.toggle("is-active", file.id === state.selectedFileId);
      const icon = element("span", "analysis-file-icon", file.kind === "video" ? "▶" : "▧");
      const text = element("span", "analysis-file-text");
      text.append(
        element("strong", "", file.original_name),
        element("small", "", statusLabel(file.status))
      );
      button.append(icon, text);
      button.addEventListener("click", () => {
        state.selectedFileId = file.id;
        state.selectedEventId = null;
        state.selectedEvent = null;
        render();
      });
      refs.fileList.appendChild(button);
    }
  }

  function renderEvents() {
    refs.eventList.innerHTML = "";
    const events = filteredEvents(state);
    refs.eventCount.textContent = String(events.length);
    if (!events.length) {
      refs.eventList.appendChild(
        element("div", "analysis-list-empty", state.events.length ? "События не подходят под фильтр." : "События появятся во время анализа.")
      );
      return;
    }
    const files = new Map(state.files.map((file) => [file.id, file]));
    for (const event of events) {
      const primary = eventPrimaryIdentity(event);
      const recognizedNumber = String(primary?.recognized_number || "").trim();
      const numberMissing = event.status === "number_not_recognized"
        || !recognizedNumber
        || recognizedNumber === "—";
      const button = element("button", "analysis-event-item");
      button.type = "button";
      button.classList.toggle("is-active", event.id === state.selectedEventId);
      button.classList.toggle("is-number-missing", numberMissing);
      const thumb = element("span", "analysis-event-thumb");
      if (event.best_frame_asset_id) {
        const image = document.createElement("img");
        image.src = `/api/analysis/assets/${event.best_frame_asset_id}`;
        image.alt = "";
        thumb.appendChild(image);
      } else {
        thumb.textContent = "AI";
      }
      const body = element("span", "analysis-event-body");
      const top = element("span", "analysis-event-top");
      top.append(
        element("strong", "", primary?.recognized_number || "—"),
        element("span", `analysis-status analysis-status--${event.status}`, statusLabel(event.status))
      );
      const file = files.get(event.source_file_id);
      body.append(
        top,
        element("small", "", `${file?.original_name || "Файл"} · #${shortId(event.id)}`),
        element(
          "small",
          "analysis-event-summary",
          `${event.locomotive_identities?.length || 0} лок. · ${fmtPercent(primary?.assembly_confidence)} · ${event.metadata?.frame_count || 1} кадр.`
        )
      );
      button.append(thumb, body);
      button.addEventListener("click", () => void selectEvent(event.id));
      refs.eventList.appendChild(button);
    }
  }

  function renderStats() {
    const fragments = state.events.reduce((sum, event) => sum + (event.fragment_tracks?.length || 0), 0);
    const identities = state.events.reduce((sum, event) => sum + (event.locomotive_identities?.length || 0), 0);
    const confirmed = state.events.reduce(
      (sum, event) => sum + (event.locomotive_identities || []).filter((item) => item.status === "confirmed" || item.manually_confirmed).length,
      0
    );
    const attention = state.events.filter((event) => event.status !== "confirmed").length;
    const values = {
      files: state.files.filter((file) => ["completed", "no_train", "stopped"].includes(file.status)).length,
      events: state.events.length,
      fragments,
      identities,
      confirmed,
      attention,
    };
    refs.stats.querySelectorAll("[data-analysis-stat]").forEach((node) => {
      node.textContent = String(values[node.dataset.analysisStat] || 0);
    });
  }

  function renderProgress() {
    const job = state.job;
    const running = Boolean(job && !TERMINAL_JOB_STATUSES.has(job.status));
    refs.progress.hidden = !job;
    if (!job) return;
    const progress = job.progress || {};
    const percent = Math.max(0, Math.min(100, Number(progress.percent || 0)));
    refs.progressFill.style.width = `${percent}%`;
    refs.progressPercent.textContent = `${Math.round(percent)}%`;
    refs.progressStage.textContent = statusLabel(job.status === "processing" ? progress.stage : job.status);
    const parts = [];
    if (progress.current_file) parts.push(progress.current_file);
    if (Number.isFinite(Number(progress.current_frame))) parts.push(`кадр ${progress.current_frame}`);
    if (Number.isFinite(Number(progress.timestamp_ms))) parts.push(`${(progress.timestamp_ms / 1000).toFixed(1)} с`);
    if (progress.processed_frames) parts.push(`${progress.processed_frames} обработано`);
    if (progress.frames_per_second) parts.push(`${progress.frames_per_second} кадр/с`);
    refs.progressMeta.textContent = parts.join(" · ") || (running ? "Подготовка моделей…" : statusLabel(job.status));
    const stages = ["detect", "tracking_fragments", "file_completed", "completed"];
    const currentIndex = stages.indexOf(progress.stage);
    refs.pipeline.querySelectorAll("[data-stage]").forEach((node) => {
      const index = Number(node.dataset.stage);
      node.classList.toggle("is-active", running && index === Math.max(0, currentIndex));
      node.classList.toggle("is-done", job.status === "completed" || index < currentIndex);
    });
  }

  function renderViewer() {
    const file = selectedFile();
    refs.viewerTitle.textContent = state.selectedEvent
      ? `${file?.original_name || "Событие"} · #${shortId(state.selectedEvent.id)}`
      : file?.original_name || "Просмотр анализа";
    viewer.render(state.selectedEvent, file, state.viewer);
    renderLocomotiveDiagram(refs.diagram, state.selectedEvent, (trackIds) => {
      state.viewer.highlightedTrackIds = trackIds.map(String);
      viewer.render(state.selectedEvent, file, state.viewer);
    });
  }

  function renderDetail() {
    refs.detail.innerHTML = "";
    const event = state.selectedEvent;
    if (!event) {
      refs.detail.appendChild(
        element("div", "analysis-detail-empty", "Выберите событие в ленте, чтобы увидеть доказательства и подтвердить результат.")
      );
      return;
    }
    const heading = element("div", "analysis-detail-heading");
    const headingText = element("div", "");
    headingText.append(
      element("span", "analysis-eyebrow", `Событие #${shortId(event.id)}`),
      element("h3", "", eventPrimaryIdentity(event)?.recognized_number || "Номер не распознан")
    );
    const status = element("span", `analysis-status analysis-status--${event.status}`, statusLabel(event.status));
    heading.append(headingText, status);
    refs.detail.appendChild(heading);

    const actionRow = element("div", "analysis-detail-actions");
    const confirm = element("button", "primary", "Подтвердить событие");
    confirm.type = "button";
    confirm.addEventListener("click", () => void updateEvent({ action: "confirm_event" }));
    const reanalyze = element("button", "", "Повторить анализ");
    reanalyze.type = "button";
    reanalyze.addEventListener("click", () => void reanalyzeEvent());
    const addUnknown = element("button", "", "+ Неизвестный локомотив");
    addUnknown.type = "button";
    addUnknown.addEventListener("click", () => void updateEvent({ action: "add_identity", recognized_number: "—" }));
    actionRow.append(confirm, reanalyze, addUnknown);
    refs.detail.appendChild(actionRow);

    const identitiesTitle = element("h4", "analysis-section-title", "Распознанные локомотивы");
    refs.detail.appendChild(identitiesTitle);
    for (const identity of event.locomotive_identities || []) {
      refs.detail.appendChild(identityEditor(identity));
    }

    refs.detail.appendChild(fragmentPanel(event));
    refs.detail.appendChild(explanationPanel(event));
    refs.detail.appendChild(evidencePanel(event));
    const notices = [...(event.warnings || []), ...(event.errors || [])];
    if (notices.length) {
      const panel = element("section", "analysis-detail-section analysis-warning-panel");
      panel.appendChild(element("h4", "analysis-section-title", "Требует внимания"));
      const list = element("ul", "analysis-warning-list");
      for (const notice of [...new Set(notices)]) list.appendChild(element("li", "", notice));
      panel.appendChild(list);
      refs.detail.appendChild(panel);
    }
  }

  function identityEditor(identity) {
    const card = element("section", "analysis-identity-card");
    const head = element("div", "analysis-identity-head");
    head.append(
      element("strong", "", identity.recognized_number || "—"),
      element("span", "analysis-confidence", `${fmtPercent(identity.assembly_confidence)} сборка`)
    );
    card.appendChild(head);
    const grid = element("div", "analysis-edit-grid");
    const number = inputField("Полный номер", identity.recognized_number || "", "text");
    const series = inputField("Серия", identity.recognized_series || "", "text");
    const sections = inputField("Секции", identity.section_count ?? "", "number");
    sections.input.min = "1";
    sections.input.max = "9";
    grid.append(number.label, series.label, sections.label);
    card.appendChild(grid);
    const meta = element(
      "div",
      "analysis-identity-meta",
      `${identity.fragment_track_ids?.length || 0} фрагм. · OCR ${fmtPercent(identity.ocr_confidence)} · ${identity.section_source || "unknown"}`
    );
    card.appendChild(meta);
    const actions = element("div", "analysis-inline-actions");
    const save = element("button", "primary", "Сохранить");
    save.type = "button";
    save.addEventListener("click", () =>
      void updateEvent({
        action: "update_identity",
        identity_id: identity.id,
        values: {
          recognized_number: number.input.value.trim() || "—",
          recognized_series: series.input.value.trim() || null,
          section_count: sections.input.value ? Number(sections.input.value) : null,
          manually_confirmed: true,
        },
      })
    );
    const remove = element("button", "analysis-danger", "Удалить ложный");
    remove.type = "button";
    remove.addEventListener("click", () => void updateEvent({ action: "delete_identity", identity_id: identity.id }));
    actions.append(save, remove);
    if ((identity.fragment_track_ids || []).length >= 2) {
      const split = element("button", "", "Разделить");
      split.type = "button";
      split.addEventListener("click", () =>
        void splitIdentity(identity.id, identity.fragment_track_ids.map((trackId) => [trackId]))
      );
      actions.appendChild(split);
    }
    card.appendChild(actions);
    return card;
  }

  function inputField(title, value, type) {
    const label = element("label", "analysis-field");
    label.appendChild(element("span", "", title));
    const input = document.createElement("input");
    input.type = type;
    input.value = String(value ?? "");
    label.appendChild(input);
    return { label, input };
  }

  function fragmentPanel(event) {
    const panel = element("section", "analysis-detail-section");
    panel.appendChild(element("h4", "analysis-section-title", "Номерные фрагменты"));
    const list = element("div", "analysis-fragment-list");
    for (const track of event.fragment_tracks || []) {
      const label = element("label", "analysis-fragment-row");
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.value = track.id;
      checkbox.dataset.fragmentSelect = "true";
      const text = element("span", "analysis-fragment-text");
      text.append(
        element("strong", "", track.best_text || "—"),
        element("small", "", `${track.fragment_type} · ${track.observation_count || track.observations?.length || 0} кадр. · ${fmtPercent(track.confidence)}${track.context_applied ? " · контекст серии" : ""}`)
      );
      if (track.best_crop_asset_id) {
        const image = document.createElement("img");
        image.src = `/api/analysis/assets/${track.best_crop_asset_id}`;
        image.alt = track.best_text || "Кроп фрагмента";
        label.append(image, checkbox, text);
      } else {
        label.append(checkbox, text);
      }
      list.appendChild(label);
    }
    panel.appendChild(list);
    const merge = element("button", "primary analysis-merge-button", "Объединить выбранные фрагменты");
    merge.type = "button";
    merge.disabled = !(event.fragment_tracks || []).length;
    merge.addEventListener("click", () => {
      const selected = Array.from(panel.querySelectorAll("[data-fragment-select]:checked")).map((input) => input.value);
      if (!selected.length) {
        showMessage("Выберите хотя бы один номерной фрагмент.", "warning");
        return;
      }
      void mergeFragments(selected);
    });
    panel.appendChild(merge);
    return panel;
  }

  function explanationPanel(event) {
    const panel = element("section", "analysis-detail-section analysis-explanation");
    panel.appendChild(element("h4", "analysis-section-title", "Как собран номер"));
    const primary = eventPrimaryIdentity(event);
    const tracks = new Map((event.fragment_tracks || []).map((track) => [String(track.id), track]));
    const fragments = (primary?.fragment_track_ids || []).map((id) => tracks.get(String(id))?.best_text || "—");
    panel.appendChild(
      element(
        "div",
        "analysis-equation",
        fragments.length ? `${fragments.map((value) => `[${value}]`).join(" + ")} → ${primary?.recognized_number || "—"}` : "Номерные зоны не найдены → [—]"
      )
    );
    const reasons = element("ul", "analysis-reason-list");
    reasons.append(
      element("li", "", `Подтверждающих наблюдений: ${primary?.observation_count || 0}`),
      element("li", "", `OCR confidence: ${fmtPercent(primary?.ocr_confidence)}`),
      element("li", "", `Assembly confidence: ${fmtPercent(primary?.assembly_confidence)}`),
      element("li", "", primary?.section_count == null ? "Секционность неизвестна" : `Секций: ${primary.section_count} (${primary.section_source})`)
    );
    panel.appendChild(reasons);
    return panel;
  }

  function evidencePanel(event) {
    const panel = element("section", "analysis-detail-section");
    panel.appendChild(element("h4", "analysis-section-title", "Галерея доказательств"));
    const gallery = element("div", "analysis-evidence-gallery");
    const assetIds = [
      event.best_frame_asset_id,
      ...(event.locomotive_identities || []).flatMap((identity) => identity.best_crop_asset_ids || []),
      ...(event.backup_frame_asset_ids || []),
    ].filter(Boolean);
    for (const assetId of [...new Set(assetIds)].slice(0, 8)) {
      const image = document.createElement("img");
      image.src = `/api/analysis/assets/${assetId}`;
      image.alt = "Доказательный кадр анализа";
      gallery.appendChild(image);
    }
    if (!assetIds.length) gallery.appendChild(element("div", "analysis-list-empty", "Доказательные кадры ещё не сохранены."));
    panel.appendChild(gallery);
    return panel;
  }

  async function updateEvent(payload) {
    if (!state.selectedEventId) return;
    try {
      state.selectedEvent = await analysisApi.patchEvent(state.selectedEventId, payload);
      replaceSelectedEvent();
      showMessage("Ручное изменение сохранено только в аналитическом событии.", "success");
      render();
    } catch (error) {
      showMessage(`Не удалось сохранить изменение: ${error.message}`, "error");
    }
  }

  async function mergeFragments(trackIds) {
    if (!state.selectedEventId) return;
    try {
      state.selectedEvent = await analysisApi.mergeFragments(state.selectedEventId, trackIds);
      replaceSelectedEvent();
      showMessage("Фрагменты объединены и защищены от автоматической перезаписи.", "success");
      render();
    } catch (error) {
      showMessage(`Не удалось объединить фрагменты: ${error.message}`, "error");
    }
  }

  async function splitIdentity(identityId, groups) {
    if (!state.selectedEventId) return;
    try {
      state.selectedEvent = await analysisApi.splitIdentity(state.selectedEventId, identityId, groups);
      replaceSelectedEvent();
      showMessage("Элемент схемы разделён.", "success");
      render();
    } catch (error) {
      showMessage(`Не удалось разделить элемент: ${error.message}`, "error");
    }
  }

  async function reanalyzeEvent() {
    if (!state.selectedEventId) return;
    const eventId = state.selectedEventId;
    const event = state.selectedEvent || state.events.find((item) => item.id === eventId);
    const fileId = event?.source_file_id;
    if (!fileId) return;
    syncSettingsFromUi();
    try {
      state.job = await analysisApi.reanalyze(eventId, state.settings);
      pendingReanalysis = { eventId, fileId };
      state.events = state.events.filter((item) => item.source_file_id !== fileId);
      state.selectedEventId = null;
      state.selectedEvent = null;
      state.selectedFileId = fileId;
      showMessage("Повторный анализ запущен только для выбранного файла. Старый результат заменяется.", "info");
      render();
      schedulePoll(100);
    } catch (error) {
      showMessage(`Не удалось запустить повторный анализ: ${error.message}`, "error");
    }
  }

  function replaceSelectedEvent() {
    const index = state.events.findIndex((event) => event.id === state.selectedEventId);
    if (index >= 0) state.events[index] = state.selectedEvent;
  }

  refs.uploadPhoto.addEventListener("click", () => refs.photoInput.click());
  refs.uploadVideo.addEventListener("click", () => refs.videoInput.click());
  refs.photoInput.addEventListener("change", () => void upload(Array.from(refs.photoInput.files || [])));
  refs.videoInput.addEventListener("change", () => void upload(Array.from(refs.videoInput.files || [])));
  refs.start.addEventListener("click", () => void start());
  refs.stop.addEventListener("click", () => void stop());
  refs.clearSession.addEventListener("click", async () => {
    if (!state.sessionId) return;
    if (!window.confirm("Очистить текущую сессию анализа? Все файлы, события и доказательные кадры будут удалены без возможности восстановления.")) return;
    state.busy = true;
    renderActions();
    try {
      state.session = await analysisApi.clearSession(state.sessionId);
      state.files = [];
      state.events = [];
      state.selectedFileId = null;
      state.selectedEventId = null;
      state.selectedEvent = null;
      state.job = null;
      persistAnalysisState(state);
      showMessage("Текущая сессия анализа очищена.", "success");
    } catch (error) {
      showMessage(`Не удалось очистить сессию: ${error.message}`, "error");
    } finally {
      state.busy = false;
      render();
    }
  });
  refs.search.addEventListener("input", () => {
    state.filters.search = refs.search.value;
    renderEvents();
  });
  refs.filter.addEventListener("change", () => {
    state.filters.status = refs.filter.value;
    renderEvents();
  });
  refs.sort.addEventListener("change", () => {
    state.filters.sort = refs.sort.value;
    renderEvents();
  });
  analysisApp.querySelectorAll("[data-analysis-setting]").forEach((input) => {
    input.addEventListener("change", syncSettingsFromUi);
  });
  for (const [checkbox, key] of [
    [refs.overlayTrain, "showTrain"],
    [refs.overlayNumber, "showNumber"],
    [refs.overlaySegmentation, "showSegmentation"],
    [refs.overlayOcr, "showOcr"],
  ]) {
    checkbox.addEventListener("change", () => {
      state.viewer[key] = checkbox.checked;
      renderViewer();
    });
  }
  for (const [button, format] of [
    [refs.exportJson, "json"],
    [refs.exportCsv, "csv"],
    [refs.exportZip, "zip"],
  ]) {
    button.addEventListener("click", () => {
      if (state.sessionId) window.location.href = `/api/analysis/sessions/${state.sessionId}/export?format=${format}`;
    });
  }

  async function initialize() {
    syncSettingsToUi();
    try {
      const [health] = await Promise.all([analysisApi.health(), ensureSession()]);
      state.health = health;
      const eventsPayload = await analysisApi.listEvents(state.sessionId);
      state.events = eventsPayload.events || [];
      render();
      if (health.ocr?.error) showMessage(`PaddleOCR: ${health.ocr.error}`, "warning");
    } catch (error) {
      showMessage(`Режим анализа не инициализирован: ${error.message}`, "error");
      render();
    }
    setMode(state.mode);
  }

  void initialize();
}
