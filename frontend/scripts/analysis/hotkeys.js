export const ANALYSIS_HOTKEYS = Object.freeze([
  { keys: "Ctrl+Enter", description: "Запустить анализ всех загруженных файлов" },
  { keys: "Esc", description: "Остановить текущий анализ" },
  { keys: "← / →", description: "Предыдущее / следующее событие" },
  { keys: "/", description: "Перейти к поиску по событиям" },
  { keys: "Shift+R", description: "Повторить анализ выбранного файла" },
  { keys: "Shift+C", description: "Подтвердить выбранное событие" },
  { keys: "1 / 2 / 3 / 4", description: "Показать или скрыть train / number / сегментацию / OCR" },
  { keys: "?", description: "Показать или скрыть эту подсказку" },
]);

export function isAnalysisTypingTarget(target) {
  return (
    target instanceof HTMLInputElement ||
    target instanceof HTMLTextAreaElement ||
    target instanceof HTMLSelectElement ||
    target?.isContentEditable
  );
}

export function analysisHotkeyAction(event) {
  const mod = event.ctrlKey || event.metaKey;
  if (mod && !event.altKey && ["Enter", "NumpadEnter"].includes(event.code)) return "start";
  if (mod || event.altKey) return null;
  if (event.code === "Escape") return "stop";
  if (event.code === "ArrowRight" && !event.shiftKey) return "next-event";
  if (event.code === "ArrowLeft" && !event.shiftKey) return "previous-event";
  if (event.code === "KeyR" && event.shiftKey) return "reanalyze";
  if (event.code === "KeyC" && event.shiftKey) return "confirm";
  if (event.code === "Slash") return event.shiftKey ? "help" : "search";
  if (!event.shiftKey && event.code === "Digit1") return "toggle-train";
  if (!event.shiftKey && event.code === "Digit2") return "toggle-number";
  if (!event.shiftKey && event.code === "Digit3") return "toggle-segmentation";
  if (!event.shiftKey && event.code === "Digit4") return "toggle-ocr";
  return null;
}
