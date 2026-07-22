function percent(value) {
  return `${Math.round(Number(value || 0) * 100)}%`;
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = String(text);
  return node;
}

function identityStatus(identity) {
  const status = identity.manually_confirmed ? "manual" : identity.status || "low_confidence";
  const statuses = {
    manual: { label: "Подтверждено вручную", tone: "success" },
    confirmed: { label: "Подтверждено", tone: "success" },
    partial_number: { label: "Номер неполный", tone: "warning" },
    number_not_detected: { label: "Номер не найден", tone: "danger" },
    low_confidence: { label: "Распознано", tone: "warning" },
  };
  return statuses[status] || { label: "Нужна проверка", tone: "warning" };
}

function sectionCount(identity) {
  if (!Number.isFinite(Number(identity.section_count))) return null;
  return Math.max(1, Math.min(8, Math.round(Number(identity.section_count))));
}

export function renderLocomotiveDiagram(root, event, onHighlight) {
  root.innerHTML = "";
  const identities = Array.isArray(event?.locomotive_identities)
    ? event.locomotive_identities
    : [];

  if (!identities.length) {
    root.appendChild(
      element(
        "div",
        "analysis-diagram-empty",
        "Локомотивы появятся здесь после распознавания номера. Если номер не найден, результат можно добавить вручную справа."
      )
    );
    return;
  }

  const summary = element("div", "analysis-diagram-summary");
  summary.append(
    element("strong", "", `Локомотивов: ${identities.length}`),
    element("span", "", "Наведите на карточку — связанные номера подсветятся на кадре")
  );

  const track = element("div", "analysis-locomotive-track");
  identities.forEach((identity, identityIndex) => {
    const status = identityStatus(identity);
    const sections = sectionCount(identity);
    const doubleSided = sections === 1;
    const item = element(
      "button",
      `analysis-locomotive analysis-locomotive--${status.tone}`
    );
    item.type = "button";
    item.dataset.identityId = identity.id;
    item.setAttribute(
      "aria-label",
      `Локомотив ${identityIndex + 1}: ${identity.recognized_number || "номер не распознан"}`
    );

    const head = element("span", "analysis-locomotive-card-head");
    head.append(
      element("span", "analysis-locomotive-index", `Локомотив ${identityIndex + 1}`),
      element(
        "span",
        `analysis-locomotive-status analysis-locomotive-status--${status.tone}`,
        status.label
      )
    );

    const visual = element("span", "analysis-locomotive-visual");
    visual.dataset.unknownSections = String(sections == null);
    visual.dataset.doubleSided = String(doubleSided);
    visual.setAttribute(
      "aria-label",
      doubleSided
        ? "Одна двусторонняя секция: кабина с каждой стороны"
        : sections == null ? "Корпус локомотива: секционность неизвестна" : `${sections} секции, кабины на внешних концах`
    );
    const body = element("span", "analysis-locomotive-body analysis-locomotive-sections");
    const visibleSections = sections || 1;
    body.style.setProperty("--section-count", String(visibleSections));
    for (let index = 0; index < visibleSections; index += 1) {
      const section = element("span", "analysis-locomotive-section");
      const hasLeftCab = index === 0;
    body.dataset.sectionCount = String(visibleSections);
      const hasRightCab = index === visibleSections - 1;
      section.classList.toggle("has-left-cab", hasLeftCab);
      section.classList.toggle("has-right-cab", hasRightCab);
      const equipment = element("span", "analysis-locomotive-equipment");
      equipment.setAttribute("aria-hidden", "true");
      if (hasLeftCab) {
        const leftCab = element("span", "analysis-locomotive-cab analysis-locomotive-cab--left");
        leftCab.setAttribute("aria-hidden", "true");
        section.appendChild(leftCab);
      }
      section.appendChild(equipment);
      if (hasRightCab) {
        const rightCab = element("span", "analysis-locomotive-cab analysis-locomotive-cab--right");
        rightCab.setAttribute("aria-hidden", "true");
        section.appendChild(rightCab);
      }
      body.appendChild(section);
    }
    visual.appendChild(body);

    const text = element("span", "analysis-locomotive-text");
    text.append(
      element("strong", "", identity.recognized_number || "Номер не распознан"),
      element(
        "span",
        "analysis-locomotive-series",
        identity.recognized_series || "Серия не определена"
      )
    );

    const facts = element("span", "analysis-locomotive-facts");
    facts.append(
      element("span", "", sections == null ? "Секции: неизвестно" : doubleSided ? "1 секция · двусторонняя" : `Секций: ${sections}`),
      element("span", "", `Уверенность: ${percent(identity.assembly_confidence)}`)
    );

    const trackIds = identity.fragment_track_ids || [];
    const hint = element(
      "span",
      "analysis-locomotive-hint",
      trackIds.length
        ? `Связанных фрагментов: ${trackIds.length}`
        : "Связанных номерных фрагментов нет"
    );

    item.append(head, visual, text, facts, hint);
    const highlight = () => onHighlight(trackIds);
    item.addEventListener("mouseenter", highlight);
    item.addEventListener("mouseleave", () => onHighlight([]));
    item.addEventListener("focus", highlight);
    item.addEventListener("blur", () => onHighlight([]));
    track.appendChild(item);
  });

  root.append(summary, track);
}
