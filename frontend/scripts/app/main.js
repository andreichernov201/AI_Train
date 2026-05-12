import { collectDomRefs } from "./dom-refs.js?v=20260512-bootstrap-yaml-range";
import { bootstrap } from "./bootstrap.js?v=20260512-bootstrap-yaml-range";

const refs = collectDomRefs();
if (refs) {
  bootstrap(refs);
}
