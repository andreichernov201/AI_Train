from __future__ import annotations

import re
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

import server


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_COLORS_JS = (ROOT / "frontend" / "scripts" / "analysis" / "colors.js").read_text(encoding="utf-8")
VIEWER_JS = (ROOT / "frontend" / "scripts" / "analysis" / "viewer.js").read_text(encoding="utf-8")
VIEWER_ZOOM_JS = (ROOT / "frontend" / "scripts" / "analysis" / "viewer-zoom.js").read_text(encoding="utf-8")
VIEWER_CSS = (ROOT / "frontend" / "analysis-viewer.css").read_text(encoding="utf-8")
STATE_JS = (ROOT / "frontend" / "scripts" / "analysis" / "state.js").read_text(encoding="utf-8")
INDEX = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
APP_JS = (ROOT / "frontend" / "scripts" / "analysis" / "app.js").read_text(encoding="utf-8")
HOTKEYS_JS = (ROOT / "frontend" / "scripts" / "analysis" / "hotkeys.js").read_text(encoding="utf-8")


class AnalysisFrontendIntegrationTests(unittest.TestCase):
    def test_analysis_dom_ids_are_unique(self) -> None:
        element_ids = re.findall(r'\bid="([^"]+)"', INDEX)
        self.assertEqual(len(element_ids), len(set(element_ids)))
        self.assertIn("analysis-app", element_ids)

    def test_analysis_script_references_existing_elements(self) -> None:
        element_ids = set(re.findall(r'\bid="([^"]+)"', INDEX))
        required = set(re.findall(r'getElementById\("([^"]+)"\)', APP_JS))
        # These two intentionally support the original markup through CSS fallbacks.
        required -= {"annotation-app", "annotation-header-progress"}
        self.assertEqual(sorted(required - element_ids), [])

    def test_page_modes_have_explicit_isolation_rules(self) -> None:
        analysis_css = (ROOT / "frontend" / "analysis.css").read_text(encoding="utf-8")
        self.assertIn(".analysis-app[hidden]", analysis_css)
        self.assertIn(".app-columns[hidden]", analysis_css)
        self.assertIn(".app-header-progress[hidden]", analysis_css)

    def test_analysis_reuses_annotation_palette(self) -> None:
        self.assertIn('from "../app/colors.js"', ANALYSIS_COLORS_JS)
        self.assertIn("DETECTION_CLASS_NAMES", ANALYSIS_COLORS_JS)
        self.assertIn("SEGMENTATION_CLASS_NAMES", ANALYSIS_COLORS_JS)
        self.assertNotIn("const SEGMENT_COLORS", VIEWER_JS)

    def test_event_feed_defaults_to_upload_order_without_duplicate_ids(self) -> None:
        self.assertIn('sort: "upload"', STATE_JS)
        self.assertIn("uniqueEvents", STATE_JS)
        self.assertIn("fileOrder", STATE_JS)
        self.assertIn('<option value="upload">', INDEX)

    def test_locomotive_diagram_explains_status_sections_and_confidence(self) -> None:
        diagram = (ROOT / "frontend" / "scripts" / "analysis" / "locomotive-diagram.js").read_text(encoding="utf-8")
        for label in ("Локомотивов:", "Секций:", "Уверенность:", "Распознано"):
            with self.subTest(label=label):
                self.assertIn(label, diagram)
        self.assertIn('from "./locomotive-diagram.js"', APP_JS)
        for class_name in (
            "analysis-locomotive-body",
            "analysis-locomotive-cab--left",
            "analysis-locomotive-cab--right",
            "analysis-locomotive-sections",
        ):
            with self.subTest(class_name=class_name):
                self.assertIn(class_name, diagram)
        diagram_css = (ROOT / "frontend" / "locomotive-diagram.css").read_text(encoding="utf-8")
        self.assertNotIn("Низкая уверенность", APP_JS)
        self.assertNotIn("Низкая уверенность", INDEX)
        self.assertIn('low_confidence: "Распознано"', APP_JS)
        self.assertIn("min-width: 142px", diagram_css)
        self.assertIn(".analysis-locomotive-section:only-child::after", diagram_css)
        self.assertIn("body.dataset.sectionCount", diagram)
        self.assertIn('data-section-count="2"', diagram_css)
        self.assertIn('<option value="low_confidence">Распознано</option>', INDEX)
        self.assertIn("Одна двусторонняя секция: кабина с каждой стороны", diagram)
        self.assertIn("1 секция · двусторонняя", diagram)
        self.assertIn("analysis-locomotive-equipment", diagram)
        self.assertIn("has-left-cab", diagram)
        self.assertIn("has-right-cab", diagram)
        self.assertNotIn('"К1"', diagram)
        self.assertNotIn('"К2"', diagram)
        self.assertNotIn("analysis-locomotive-section-number", diagram)
        self.assertIn('data-double-sided', diagram_css)
        self.assertIn("--section-count", diagram_css)

    def test_missing_number_event_has_red_left_marker(self) -> None:
        analysis_css = (ROOT / "frontend" / "analysis.css").read_text(encoding="utf-8")
        self.assertIn('event.status === "number_not_recognized"', APP_JS)
        self.assertIn('button.classList.toggle("is-number-missing", numberMissing)', APP_JS)
        self.assertIn(".analysis-event-item.is-number-missing", analysis_css)
        self.assertIn("inset 3px 0 0 #fb7185", analysis_css)

    def test_analysis_hotkeys_are_scoped_and_documented(self) -> None:
        self.assertIn('from "./hotkeys.js"', APP_JS)
        self.assertIn("isAnalysisTypingTarget", APP_JS)
        for action in ("start", "stop", "next-event", "previous-event", "reanalyze", "confirm"):
            with self.subTest(action=action):
                self.assertIn(f'"{action}"', HOTKEYS_JS)
        for code in ("ArrowLeft", "ArrowRight", "KeyC"):
            with self.subTest(code=code):
                self.assertIn(code, HOTKEYS_JS)
        self.assertIn('id="analysis-hotkeys"', INDEX)
        self.assertIn("Подтвердить выбранное событие", INDEX)

    def test_analysis_viewer_fits_and_zooms_without_cropping(self) -> None:
        analysis_css = (ROOT / "frontend" / "analysis.css").read_text(encoding="utf-8")
        self.assertIn('from "./viewer-zoom.js"', VIEWER_JS)
        self.assertIn("containGeometry", VIEWER_ZOOM_JS)
        self.assertIn("offsetX", VIEWER_JS)
        self.assertIn("offsetY", VIEWER_JS)
        for event_name in ("wheel", "pointerdown", "pointermove", "dblclick"):
            with self.subTest(event_name=event_name):
                self.assertIn(f'addEventListener("{event_name}"', VIEWER_JS)
        self.assertIn("object-fit: contain", VIEWER_CSS)
        self.assertIn("width: 100%", VIEWER_CSS)
        self.assertIn("height: 100%", VIEWER_CSS)
        self.assertIn("--analysis-viewer-height", analysis_css)
        self.assertIn("двойной клик — сброс", INDEX)
    def test_analysis_viewer_hidden_media_never_overlays_active_frame(self) -> None:
        self.assertIn(".analysis-viewer-surface > [hidden]", VIEWER_CSS)
        hidden_rule = re.search(r"\.analysis-viewer-surface\s*>\s*\[hidden\]\s*\{([^}]*)\}", VIEWER_CSS)
        self.assertIsNotNone(hidden_rule)
        self.assertIn("display: none !important", hidden_rule.group(1))

    def test_entrypoint_and_static_modules_are_served(self) -> None:
        main_js = (ROOT / "frontend" / "scripts" / "app" / "main.js").read_text(encoding="utf-8")
        self.assertIn("startAnalysisApp", main_js)
        client = TestClient(server.app)
        for path in (
            "/",
            "/static/analysis.css",
            "/static/locomotive-diagram.css",
            "/static/analysis-hotkeys.css",
            "/static/analysis-viewer.css",
            "/static/scripts/analysis/app.js",
            "/static/scripts/analysis/api.js",
            "/static/scripts/analysis/viewer.js",
            "/static/scripts/analysis/viewer-zoom.js",
            "/static/scripts/analysis/colors.js",
            "/static/scripts/analysis/diagram.js",
            "/static/scripts/analysis/locomotive-diagram.js",
            "/static/scripts/analysis/hotkeys.js",
            "/static/scripts/analysis/state.js",
        ):
            with self.subTest(path=path):
                self.assertEqual(client.get(path).status_code, 200)


if __name__ == "__main__":
    unittest.main()
