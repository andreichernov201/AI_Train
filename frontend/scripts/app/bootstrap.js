import { startApp } from "./app.js?v=20260701-export-annot";

/**
 * @param {import("./dom-refs.js").AppDomRefs} refs
 */
export function bootstrap(refs) {
  startApp(refs);
}
