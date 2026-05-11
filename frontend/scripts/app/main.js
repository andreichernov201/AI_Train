import { collectDomRefs } from "./dom-refs.js";
import { bootstrap } from "./bootstrap.js";

const refs = collectDomRefs();
if (refs) {
  bootstrap(refs);
}
