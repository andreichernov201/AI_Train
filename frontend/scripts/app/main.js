import { collectDomRefs } from "./dom-refs.js?v=20260701-export-annot";
import { bootstrap } from "./bootstrap.js?v=20260701-export-annot";
import { startAnalysisApp } from "../analysis/app.js?v=20260721-analysis";

const refs = collectDomRefs();
if (refs) {
  bootstrap(refs);
}
startAnalysisApp();
