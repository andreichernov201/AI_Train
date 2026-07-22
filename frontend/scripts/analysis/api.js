async function request(url, options = {}) {
  const response = await fetch(url, options);
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();
  if (!response.ok) {
    const detail = payload?.detail ?? payload?.error ?? payload;
    const message = typeof detail === "string" ? detail : JSON.stringify(detail);
    throw new Error(message || `HTTP ${response.status}`);
  }
  return payload;
}

export const analysisApi = {
  health: () => request("/api/analysis/health"),
  listSessions: () => request("/api/analysis/sessions"),
  createSession: (settings) =>
    request("/api/analysis/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ settings }),
    }),
  getSession: (sessionId) => request(`/api/analysis/sessions/${sessionId}`),
  clearSession: (sessionId) =>
    request(`/api/analysis/sessions/${sessionId}/contents`, { method: "DELETE" }),
  uploadFiles: (sessionId, files) => {
    const form = new FormData();
    for (const file of files) form.append("files", file, file.name);
    return request(`/api/analysis/sessions/${sessionId}/files`, {
      method: "POST",
      body: form,
    });
  },
  start: (sessionId, settings) =>
    request(`/api/analysis/sessions/${sessionId}/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ settings }),
    }),
  getJob: (jobId) => request(`/api/analysis/jobs/${jobId}`),
  stopJob: (jobId) => request(`/api/analysis/jobs/${jobId}/stop`, { method: "POST" }),
  listEvents: (sessionId) => request(`/api/analysis/sessions/${sessionId}/events`),
  getEvent: (eventId) => request(`/api/analysis/events/${eventId}`),
  patchEvent: (eventId, payload) =>
    request(`/api/analysis/events/${eventId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  reanalyze: (eventId, settings) =>
    request(`/api/analysis/events/${eventId}/reanalyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ settings }),
    }),
  mergeFragments: (eventId, fragmentTrackIds, text = null) =>
    request(`/api/analysis/events/${eventId}/merge-fragments`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fragment_track_ids: fragmentTrackIds, text }),
    }),
  splitIdentity: (eventId, identityId, groups) =>
    request(`/api/analysis/events/${eventId}/split-identity`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ identity_id: identityId, groups }),
    }),
};
