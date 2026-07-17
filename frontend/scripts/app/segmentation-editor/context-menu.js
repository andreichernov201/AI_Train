/**
 * Расширяемое контекстное меню маски. Компонент не знает о структуре
 * аннотаций: команды получают устойчивый контекст (imageId/detId).
 *
 * @param {{
 *   root: HTMLElement,
 *   commands: Array<{
 *     id: string,
 *     label: string,
 *     icon: string,
 *     isAvailable?: (context:any) => boolean,
 *     handler: (context:any) => void|Promise<void>,
 *   }>,
 * }} options
 */
export function createMaskContextMenu({ root, commands }) {
  let activeContext = null;

  function close() {
    activeContext = null;
    root.hidden = true;
    root.replaceChildren();
  }

  function isOpen() {
    return !root.hidden;
  }

  function positionAt(clientX, clientY) {
    const gap = 6;
    const edge = 8;
    root.style.left = `${clientX + gap}px`;
    root.style.top = `${clientY + gap}px`;
    const rect = root.getBoundingClientRect();
    const left = Math.max(
      edge,
      Math.min(clientX + gap, window.innerWidth - rect.width - edge)
    );
    const top = Math.max(
      edge,
      Math.min(clientY + gap, window.innerHeight - rect.height - edge)
    );
    root.style.left = `${left}px`;
    root.style.top = `${top}px`;
  }

  function open({ clientX, clientY, context }) {
    activeContext = { ...context };
    root.replaceChildren();
    for (const command of commands) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "mask-context-menu-item";
      button.dataset.command = command.id;
      button.setAttribute("role", "menuitem");
      button.disabled =
        typeof command.isAvailable === "function"
          ? !command.isAvailable(activeContext)
          : false;

      const icon = document.createElement("span");
      icon.className = "mask-context-menu-icon";
      icon.setAttribute("aria-hidden", "true");
      icon.textContent = command.icon;

      const label = document.createElement("span");
      label.textContent = command.label;
      button.append(icon, label);
      button.addEventListener("click", () => {
        if (button.disabled || !activeContext) return;
        const ctx = { ...activeContext };
        close();
        void command.handler(ctx);
      });
      root.appendChild(button);
    }
    root.hidden = false;
    positionAt(clientX, clientY);
    root.querySelector("button:not(:disabled)")?.focus({ preventScroll: true });
  }

  root.addEventListener("pointerdown", (event) => event.stopPropagation());
  document.addEventListener("pointerdown", (event) => {
    if (!isOpen() || root.contains(/** @type {Node} */ (event.target))) return;
    close();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape" || !isOpen()) return;
    close();
    event.preventDefault();
  });

  return { open, close, isOpen };
}
