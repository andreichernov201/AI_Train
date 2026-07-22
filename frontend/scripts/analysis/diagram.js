function percent(value) {
  return `${Math.round(Number(value || 0) * 100)}%`;
}

export function renderLocomotiveDiagram(root, event, onHighlight) {
  root.innerHTML = "";
  const identities = Array.isArray(event?.locomotive_identities)
    ? event.locomotive_identities
    : [];
  if (!identities.length) {
    const empty = document.createElement("div");
    empty.className = "analysis-diagram-empty";
    empty.textContent = "Схема появится после сборки номерных фрагментов.";
    root.appendChild(empty);
    return;
  }
  for (const identity of identities) {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "analysis-locomotive";
    item.dataset.identityId = identity.id;
    const sections = Number.isFinite(Number(identity.section_count))
      ? Math.max(1, Math.min(4, Number(identity.section_count)))
      : 1;
    const visual = document.createElement("span");
    visual.className = "analysis-locomotive-visual";
    visual.dataset.unknownSections = identity.section_count == null ? "true" : "false";
    for (let index = 0; index < sections; index += 1) {
      const section = document.createElement("span");
      section.className = "analysis-locomotive-section";
      visual.appendChild(section);
    }
    const text = document.createElement("span");
    text.className = "analysis-locomotive-text";
    const title = document.createElement("strong");
    title.textContent = identity.recognized_number || "—";
    const meta = document.createElement("small");
    const sectionText = identity.section_count == null
      ? "секционность неизвестна"
      : `${identity.section_count} секц.`;
    meta.textContent = `${sectionText} · ${percent(identity.assembly_confidence)}`;
    text.append(title, meta);
    item.append(visual, text);
    item.addEventListener("mouseenter", () => onHighlight(identity.fragment_track_ids || []));
    item.addEventListener("mouseleave", () => onHighlight([]));
    root.appendChild(item);
  }
}
