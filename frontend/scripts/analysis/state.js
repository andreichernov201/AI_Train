const STORAGE_KEY = "ai-train-analysis-state-v1";

const DEFAULT_SETTINGS = Object.freeze({
  frame_interval_sec: 1,
  detect_confidence: 0.25,
  ocr_confidence: 0.55,
  event_timeout_sec: 2.5,
  max_ocr_candidates: 3,
  best_frame_count: 3,
  preprocessing: "adaptive",
});

export function createAnalysisState() {
  let saved = {};
  try {
    saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
  } catch {
    saved = {};
  }
  return {
    mode: saved.mode === "analysis" ? "analysis" : "annotation",
    sessionId: typeof saved.sessionId === "string" ? saved.sessionId : null,
    session: null,
    files: [],
    events: [],
    selectedFileId: null,
    selectedEventId: null,
    selectedEvent: null,
    job: null,
    health: null,
    busy: false,
    settings: { ...DEFAULT_SETTINGS, ...(saved.settings || {}) },
    filters: {
      search: "",
      status: "all",
      sort: "upload",
    },
    viewer: {
      showTrain: true,
      showNumber: true,
      showSegmentation: true,
      showOcr: true,
      highlightedTrackIds: [],
    },
  };
}

export function persistAnalysisState(state) {
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({
      mode: state.mode,
      sessionId: state.sessionId,
      settings: state.settings,
    })
  );
}

export function eventPrimaryIdentity(event) {
  const identities = Array.isArray(event?.locomotive_identities)
    ? event.locomotive_identities
    : [];
  return [...identities].sort(
    (a, b) =>
      Number(b.manually_confirmed) - Number(a.manually_confirmed) ||
      Number(b.assembly_confidence || 0) - Number(a.assembly_confidence || 0)
  )[0] || null;
}

export function filteredEvents(state) {
  const query = state.filters.search.trim().toLocaleUpperCase("ru-RU");
  const fileNames = new Map(state.files.map((file) => [file.id, file.original_name]));
  const fileOrder = new Map(state.files.map((file, index) => [file.id, index]));
  const uniqueEvents = Array.from(
    new Map((state.events || []).map((event) => [event.id, event])).values()
  );
  let events = uniqueEvents.filter((event) => {
    if (state.filters.status !== "all" && event.status !== state.filters.status) return false;
    if (!query) return true;
    const haystack = [
      event.id,
      fileNames.get(event.source_file_id),
      ...(event.locomotive_identities || []).flatMap((identity) => [
        identity.recognized_number,
        identity.recognized_series,
      ]),
      ...(event.fragment_tracks || []).map((track) => track.best_text),
    ]
      .filter(Boolean)
      .join(" ")
      .toLocaleUpperCase("ru-RU");
    return haystack.includes(query);
  });
  const confidence = (event) => Number(eventPrimaryIdentity(event)?.assembly_confidence || 0);
  if (state.filters.sort === "upload") {
    events = events.sort(
      (a, b) =>
        Number(fileOrder.get(a.source_file_id) ?? Number.MAX_SAFE_INTEGER) -
          Number(fileOrder.get(b.source_file_id) ?? Number.MAX_SAFE_INTEGER) ||
        Number(a.metadata?.start_timestamp_ms || 0) - Number(b.metadata?.start_timestamp_ms || 0) ||
        String(a.created_at || "").localeCompare(String(b.created_at || ""))
    );
  } else if (state.filters.sort === "confidence") {
    events = events.sort((a, b) => confidence(b) - confidence(a));
  } else if (state.filters.sort === "file") {
    events = events.sort((a, b) =>
      String(fileNames.get(a.source_file_id) || "").localeCompare(
        String(fileNames.get(b.source_file_id) || ""),
        "ru"
      )
    );
  } else if (state.filters.sort === "time") {
    events = events.sort((a, b) => String(b.started_at).localeCompare(String(a.started_at)));
  } else {
    const attention = (event) =>
      ["confirmed"].includes(event.status) ? 1 : event.status === "analysis_error" ? -1 : 0;
    events = events.sort((a, b) => attention(a) - attention(b) || confidence(a) - confidence(b));
  }
  return events;
}
