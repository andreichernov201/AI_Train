from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path

from analysis.detection import link_numbers_to_trains, suppress_duplicate_trains
from analysis.event_association import EventAssociator
from analysis.jobs import AnalysisJobManager
from analysis.export import export_csv, export_json, export_zip
from analysis.manual import merge_fragments_payload, patch_event_payload, split_identity_payload
from analysis.numbers.fragment_assembler import NumberFragmentAssembler
from analysis.numbers.fragment_classifier import classify_fragment
from analysis.numbers.fragment_tracker import NumberFragmentTracker
from analysis.ocr.fusion import OCRFusionService
from analysis.ocr.normalization import normalize_ocr_text
from analysis.series_reference import SeriesReference
from analysis.segmentation import link_segments_to_trains
from analysis.storage import AnalysisStorage


REFERENCE = SeriesReference()


def track(track_id: str, text: str, kind: str, box: list[float], confidence: float = 0.9) -> dict:
    return {
        "id": track_id,
        "best_text": text,
        "fragment_type": kind,
        "relative_position": box,
        "confidence": confidence,
        "observation_count": 3,
    }


def observation(observation_id: str, text: str, frame: int, confidence: float = 0.85) -> dict:
    return {
        "id": observation_id,
        "source_file_id": "file_1",
        "frame_index": frame,
        "timestamp_ms": frame * 1000,
        "normalized_text": text,
        "fragment_type": classify_fragment(text, REFERENCE),
        "relative_bbox": [0.1, 0.1, 0.25, 0.2],
        "ocr_confidence": confidence,
        "quality_score": 0.82,
    }


class AnalysisNumberTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assembler = NumberFragmentAssembler(REFERENCE)

    def test_01_normalizes_latin_series_alias(self) -> None:
        result = normalize_ocr_text("EP20", REFERENCE)
        self.assertEqual(result.normalized_text, "ЭП20")
        self.assertEqual(result.series_part, "ЭП20")

    def test_02_normalizes_full_number_without_separator(self) -> None:
        result = normalize_ocr_text("EP20 077", REFERENCE)
        self.assertEqual(result.normalized_text, "ЭП20-077")
        self.assertEqual(result.serial_part, "077")

    def test_03_context_restores_blurred_vl80(self) -> None:
        result = normalize_ocr_text("ВЛВО", REFERENCE)
        self.assertEqual(result.normalized_text, "ВЛ80")
        self.assertTrue(result.context_applied)

    def test_04_context_restores_full_vl80_number(self) -> None:
        result = normalize_ocr_text("VL8O 123", REFERENCE)
        self.assertEqual(result.normalized_text, "ВЛ80-123")
        self.assertEqual(result.serial_part, "123")

    def test_05_context_distinguishes_vl80s(self) -> None:
        result = normalize_ocr_text("ВЛ8ОС 123", REFERENCE)
        self.assertEqual(result.normalized_text, "ВЛ80С-123")

    def test_06_context_handles_2es6(self) -> None:
        result = normalize_ocr_text("2ES6-077", REFERENCE)
        self.assertEqual(result.normalized_text, "2ЭС6-077")

    def test_07_normalized_ocr_uses_only_russian_letters_and_digits(self) -> None:
        result = normalize_ocr_text("ABZ$-12", REFERENCE)
        self.assertRegex(result.normalized_text or "", r"^[А-Я0-9-]+$")
        self.assertNotRegex(result.normalized_text or "", r"[A-Z]")

    def test_08_context_catalog_covers_many_series(self) -> None:
        self.assertGreaterEqual(len(REFERENCE.known_series_names), 140)
        required_series = {
            "ЧС1",
            "ЧС3",
            "ЧС7",
            "ЭП10",
            "Э5К",
            "ВЛ22",
            "3ЭС8",
            "3ЭС10",
            "2ТЭ70",
            "ТГ16М",
            "ТГ21",
            "ТГ22",
            "ТГМ23",
            "ЧМЭ3М",
        }
        self.assertFalse(required_series - set(REFERENCE.known_series_names))
        self.assertEqual(REFERENCE.match_known_series("ТЭП70БС-123"), "ТЭП70БС")

    def test_09_real_video_vl80_ocr_errors_use_context(self) -> None:
        examples = {"BN804662": "ВЛ80-4662", "B180475": "ВЛ80-475", "8180475": "ВЛ80-475"}
        for raw_text, expected in examples.items():
            with self.subTest(raw_text=raw_text):
                self.assertEqual(normalize_ocr_text(raw_text, REFERENCE).normalized_text, expected)

    def test_09_context_restores_chs_series_from_digit_only_ocr(self) -> None:
        examples = {"406": "ЧС6", "407": "ЧС7", "408": "ЧС8", "40200": "ЧС200"}
        for raw_text, expected in examples.items():
            with self.subTest(raw_text=raw_text):
                result = normalize_ocr_text(raw_text, REFERENCE)
                self.assertEqual(result.normalized_text, expected)
                self.assertEqual(result.series_part, expected)
                self.assertTrue(result.context_applied)
    def test_09_classifies_fragment_types(self) -> None:
        self.assertEqual(classify_fragment("ЭП20", REFERENCE), "series_fragment")
        self.assertEqual(classify_fragment("077", REFERENCE), "serial_fragment")
    def test_09_context_restores_ep1_from_zp_ocr_error(self) -> None:
        examples = {"ЗП": "ЭП1", "ЗП1": "ЭП1", "ЗП20": "ЭП20", "ЗП1 234": "ЭП1-234"}
        for raw_text, expected in examples.items():
            with self.subTest(raw_text=raw_text):
                result = normalize_ocr_text(raw_text, REFERENCE)
                self.assertEqual(result.normalized_text, expected)
                self.assertTrue(result.context_applied)

        self.assertEqual(classify_fragment("ЭП20-077", REFERENCE), "full_identifier")
        self.assertEqual(classify_fragment("ВN8-04662", REFERENCE), "mixed_fragment")

    def test_04_assembles_horizontal_fragments(self) -> None:
        identities, _, _ = self.assembler.assemble(
            "event_1",
            [
                track("series", "ЭП20", "series_fragment", [0.10, 0.20, 0.30, 0.30]),
                track("serial", "077", "serial_fragment", [0.33, 0.20, 0.48, 0.30]),
            ],
        )
        self.assertEqual(identities[0]["recognized_number"], "ЭП20-077")

    def test_05_assembles_vertical_fragments(self) -> None:
        identities, _, _ = self.assembler.assemble(
            "event_1",
            [
                track("series", "ЭП20", "series_fragment", [0.20, 0.12, 0.38, 0.22]),
                track("serial", "077", "serial_fragment", [0.21, 0.26, 0.37, 0.36]),
            ],
        )
        self.assertEqual(identities[0]["recognized_number"], "ЭП20-077")

    def test_06_rejects_geometrically_distant_pair(self) -> None:
        identities, _, _ = self.assembler.assemble(
            "event_1",
            [
                track("series", "ЭП20", "series_fragment", [0.02, 0.02, 0.12, 0.08]),
                track("serial", "077", "serial_fragment", [0.86, 0.86, 0.98, 0.94]),
            ],
        )
        self.assertEqual({item["recognized_number"] for item in identities}, {"ЭП20-—", "—-077"})

    def test_07_keeps_two_locomotive_pairs_separate(self) -> None:
        identities, _, _ = self.assembler.assemble(
            "event_1",
            [
                track("s1", "ЭП20", "series_fragment", [0.05, 0.15, 0.20, 0.24]),
                track("n1", "077", "serial_fragment", [0.22, 0.15, 0.34, 0.24]),
                track("s2", "ВЛ80С", "series_fragment", [0.58, 0.65, 0.75, 0.74]),
                track("n2", "123", "serial_fragment", [0.77, 0.65, 0.89, 0.74]),
            ],
        )
        self.assertEqual({item["recognized_number"] for item in identities}, {"ЭП20-077", "ВЛ80С-123"})

    def test_08_deduplicates_repeated_full_identifier(self) -> None:
        identities, _, _ = self.assembler.assemble(
            "event_1",
            [
                track("one", "ЭП20-077", "full_identifier", [0.1, 0.1, 0.3, 0.2]),
                track("two", "ЭП20-077", "full_identifier", [0.1, 0.1, 0.3, 0.2]),
            ],
        )
        self.assertEqual(len(identities), 1)
        self.assertEqual(set(identities[0]["fragment_track_ids"]), {"one", "two"})

    def test_09_returns_partial_series(self) -> None:
        identities, _, _ = self.assembler.assemble(
            "event_1", [track("series", "ЭП20", "series_fragment", [0.1, 0.1, 0.3, 0.2])]
        )
        self.assertEqual(identities[0]["recognized_number"], "ЭП20-—")
        self.assertEqual(identities[0]["status"], "partial_number")

    def test_10_single_weak_full_identifier_is_not_auto_confirmed(self) -> None:
        identities, _, _ = self.assembler.assemble(
            "event_1",
            [track("weak", "ВЛ80-123", "full_identifier", [0.1, 0.1, 0.3, 0.2], confidence=0.48)],
        )
        self.assertEqual(identities[0]["recognized_number"], "ВЛ80-123")
        self.assertEqual(identities[0]["status"], "low_confidence")

    def test_10_returns_explicit_no_number_result(self) -> None:
        identities, _, _ = self.assembler.assemble("event_1", [])
        self.assertEqual(identities[0]["status"], "number_not_detected")

    def test_11_tracker_fuses_repeated_observations(self) -> None:
        tracker = NumberFragmentTracker("event_1")
        first = tracker.update(observation("one", "077", 1))
        second = tracker.update(observation("two", "077", 2))
        self.assertEqual(first, second)
        self.assertEqual(tracker.to_dicts()[0]["observation_count"], 2)

    def test_12_tracker_separates_distant_same_type(self) -> None:
        tracker = NumberFragmentTracker("event_1")
        left = observation("left", "077", 1)
        right = observation("right", "123", 1)
        right["relative_bbox"] = [0.75, 0.70, 0.90, 0.80]
        self.assertNotEqual(tracker.update(left), tracker.update(right))

    def test_13_stable_ocr_wins_over_single_outlier(self) -> None:
        rows = [observation(f"good_{index}", "ЭП20", index, 0.82) for index in range(3)]
        rows.append(observation("outlier", "ЭП2О", 4, 0.99))
        self.assertEqual(OCRFusionService().choose(rows)["text"], "ЭП20")

    def test_14_manual_lock_has_absolute_priority(self) -> None:
        chosen = OCRFusionService().choose([observation("one", "123", 1)], locked_text="077")
        self.assertEqual(chosen, {"text": "077", "score": 1.0, "source": "manual", "locked": True})

    def test_15_links_number_to_containing_train(self) -> None:
        trains = [{"box": [0, 0, 500, 300]}, {"box": [600, 0, 1000, 300]}]
        linked, warnings = link_numbers_to_trains(trains, [{"box": [650, 100, 760, 150]}])
        self.assertFalse(warnings)
        self.assertEqual(len(linked[0]), 0)
        self.assertEqual(len(linked[1]), 1)

    def test_15b_links_segment_to_containing_train(self) -> None:
        trains = [{"box": [0, 0, 500, 300]}, {"box": [600, 0, 1000, 300]}]
        segment = {
            "cls_name": "body",
            "box": [640, 40, 930, 260],
            "segment": [[640, 40], [930, 40], [930, 260], [640, 260]],
        }
        linked = link_segments_to_trains(trains, [segment])
        self.assertEqual(linked[0], [])
        self.assertEqual(linked[1][0]["cls_name"], "body")

    def test_16_suppresses_nested_duplicate_train(self) -> None:
        trains = [
            {"box": [17.9, 145.8, 483.3, 688.8], "conf": 0.91},
            {"box": [32.9, 227.5, 374.4, 685.4], "conf": 0.72},
            {"box": [620, 120, 980, 680], "conf": 0.88},
        ]
        kept = suppress_duplicate_trains(trains)
        self.assertEqual(len(kept), 2)
        self.assertIn(trains[0], kept)
        self.assertIn(trains[2], kept)

    def test_17_associates_same_train_across_frames(self) -> None:
        associator = EventAssociator(timeout_ms=2500)
        one = associator.associate([{"box": [0, 0, 200, 100]}], 0, 0)[0]
        two = associator.associate([{"box": [5, 1, 205, 101]}], 1000, 1)[0]
        self.assertEqual(one, two)

    def test_18_expires_event_after_timeout(self) -> None:
        associator = EventAssociator(timeout_ms=1000)
        event_id = associator.associate([{"box": [0, 0, 200, 100]}], 0, 0)[0]
        self.assertEqual(associator.expire(1001), [event_id])

    def test_18_estimates_sections_from_series_rules(self) -> None:
        self.assertEqual(REFERENCE.section_estimate("ВЛ80С")[0], 2)
        self.assertEqual(REFERENCE.section_estimate("3ЭС5К")[0], 3)
        self.assertEqual(REFERENCE.section_estimate("ЧС6")[0], 2)
        self.assertEqual(REFERENCE.section_estimate("ЧС7")[0], 2)
        self.assertEqual(REFERENCE.section_estimate("ЧС8")[0], 2)
        self.assertEqual(REFERENCE.section_estimate("ТГ21")[0], 2)
        self.assertEqual(REFERENCE.section_estimate("ТГ22")[0], 1)
        self.assertEqual(REFERENCE.section_estimate("ЧС200")[0], 2)
        self.assertIsNone(REFERENCE.section_estimate("НЕИЗВЕСТНО")[0])

    def test_19_manual_merge_and_split_are_locked(self) -> None:
        event = {
            "id": "event_1",
            "fragment_tracks": [
                track("series", "ЭП20", "series_fragment", [0.1, 0.1, 0.3, 0.2]),
                track("serial", "077", "serial_fragment", [0.3, 0.1, 0.5, 0.2]),
            ],
            "locomotive_identities": [],
        }
        merged, _ = merge_fragments_payload(event, ["series", "serial"], REFERENCE)
        identity = merged["locomotive_identities"][0]
        self.assertEqual(identity["recognized_number"], "ЭП20-077")
        self.assertTrue(identity["locked"])
        split, _ = split_identity_payload(merged, identity["id"], [["series"], ["serial"]], REFERENCE)
        self.assertEqual(len(split["locomotive_identities"]), 2)

    def test_20_manual_confirmation_locks_model_result(self) -> None:
        event = {"id": "event_1", "status": "low_confidence", "locomotive_identities": [{"id": "identity_1"}]}
        updated, _ = patch_event_payload(event, {"action": "confirm_event"}, REFERENCE)
        self.assertEqual(updated["status"], "confirmed")
        self.assertTrue(updated["locomotive_identities"][0]["locked"])


class AnalysisStorageExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.storage = AnalysisStorage(Path(self.temporary.name))
        self.session = self.storage.create_session()
        self.file, target = self.storage.create_file_record(
            self.session["id"], "sample.jpg", "image/jpeg", ".jpg", "image"
        )
        target.write_bytes(b"image-data")
        self.storage.finalize_file(self.file["id"], target.stat().st_size)
        self.event = {
            "id": "event_export",
            "session_id": self.session["id"],
            "source_file_id": self.file["id"],
            "started_at": "2026-07-21T10:00:00+00:00",
            "status": "confirmed",
            "fragment_tracks": [track("series", "ЭП20", "series_fragment", [0.1, 0.1, 0.2, 0.2])],
            "locomotive_identities": [
                {
                    "id": "identity_export",
                    "recognized_number": "ЭП20-077",
                    "recognized_series": "ЭП20",
                    "serial_number": "077",
                    "fragment_track_ids": ["series"],
                    "section_count": 1,
                    "ocr_confidence": 0.91,
                    "assembly_confidence": 0.94,
                    "status": "confirmed",
                    "warnings": [],
                    "manually_confirmed": False,
                }
            ],
            "metadata": {"frame_count": 4},
        }
        self.storage.save_event(self.event)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_21_json_export_preserves_unicode_and_structure(self) -> None:
        payload = json.loads(export_json(self.storage, self.session["id"]).decode("utf-8"))
        self.assertEqual(payload["events"][0]["locomotive_identities"][0]["recognized_number"], "ЭП20-077")

    def test_22_csv_export_has_excel_utf8_bom(self) -> None:
        content = export_csv(self.storage, self.session["id"])
        self.assertTrue(content.startswith(b"\xef\xbb\xbf"))
        self.assertIn("ЭП20-077", content.decode("utf-8-sig"))

    def test_23_zip_export_contains_one_manifest_and_evidence(self) -> None:
        asset_id = self.storage.save_asset(
            event_id=self.event["id"],
            file_id=self.file["id"],
            kind="best_frame",
            data=b"jpeg-evidence",
            suffix=".jpg",
            mime_type="image/jpeg",
        )
        with zipfile.ZipFile(io.BytesIO(export_zip(self.storage, self.session["id"]))) as archive:
            names = archive.namelist()
            self.assertEqual(names.count("manifest.json"), 1)
            self.assertIn("results.csv", names)
            self.assertTrue(any(asset_id in name for name in names))

    def test_24_upload_filename_is_sanitized(self) -> None:
        record, path = self.storage.create_file_record(
            self.session["id"], "../../outside.png", "image/png", ".png", "image"
        )
        self.assertEqual(record["original_name"], "outside.png")
        self.assertTrue(str(path).startswith(str(Path(self.temporary.name).resolve())))

    def test_25_clear_session_removes_content_but_keeps_settings(self) -> None:
        asset_id = self.storage.save_asset(
            event_id=self.event["id"],
            file_id=self.file["id"],
            kind="best_frame",
            data=b"evidence",
            suffix=".jpg",
            mime_type="image/jpeg",
        )
        asset_path = self.storage.get_asset(asset_id)[1]
        upload_path = self.storage.file_path(self.file["id"])
        cleared = self.storage.clear_session(self.session["id"])
        self.assertEqual(cleared["id"], self.session["id"])
        self.assertEqual(cleared["settings"], self.session["settings"])
        self.assertEqual(self.storage.list_files(self.session["id"]), [])
        self.assertEqual(self.storage.list_events(self.session["id"]), [])
        self.assertFalse(asset_path.exists())
        self.assertFalse(upload_path.exists())

    def test_26_clear_session_rejects_active_job(self) -> None:
        job = self.storage.create_job(self.session["id"])
        with self.assertRaises(RuntimeError):
            self.storage.clear_session(self.session["id"])
        self.assertIsNotNone(self.storage.get_job(job["id"]))
        self.assertIsNotNone(self.storage.get_file(self.file["id"]))
    def test_27_clear_analysis_results_keeps_uploaded_file(self) -> None:
        asset_id = self.storage.save_asset(
            event_id=self.event["id"],
            file_id=self.file["id"],
            kind="best_frame",
            data=b"evidence",
            suffix=".jpg",
            mime_type="image/jpeg",
        )
        asset_path = self.storage.get_asset(asset_id)[1]
        upload_path = self.storage.file_path(self.file["id"])

        deleted = self.storage.clear_analysis_results(self.session["id"])

        self.assertEqual(deleted, 1)
        self.assertEqual(self.storage.list_events(self.session["id"]), [])
        self.assertEqual(self.storage.list_files(self.session["id"])[0]["status"], "ready")
        self.assertTrue(upload_path.exists())
        self.assertFalse(asset_path.exists())

    def test_28_events_follow_upload_order(self) -> None:
        second_file, second_path = self.storage.create_file_record(
            self.session["id"], "second.jpg", "image/jpeg", ".jpg", "image"
        )
        second_path.write_bytes(b"second-image")
        self.storage.finalize_file(second_file["id"], second_path.stat().st_size)
        self.storage.clear_analysis_results(self.session["id"])

        second_event = {
            **self.event,
            "id": "event_second",
            "source_file_id": second_file["id"],
            "started_at": "2026-07-21T09:00:00+00:00",
        }
        first_event = {
            **self.event,
            "id": "event_first",
            "started_at": "2026-07-21T11:00:00+00:00",
        }
        self.storage.save_event(second_event)
        self.storage.save_event(first_event)

        ordered = self.storage.list_events(self.session["id"])
        self.assertEqual(
            [event["source_file_id"] for event in ordered],
            [self.file["id"], second_file["id"]],
        )

    def test_29_regular_start_replaces_previous_results(self) -> None:
        completed = threading.Event()

        class RecordingPipeline:
            calls: list[dict] = []

            def run_session(inner_self, **kwargs) -> None:
                inner_self.calls.append(kwargs)
                completed.set()

        pipeline = RecordingPipeline()
        manager = AnalysisJobManager(self.storage, pipeline)
        try:
            manager.start(self.session["id"])
            self.assertEqual(self.storage.list_events(self.session["id"]), [])
            self.assertTrue(completed.wait(2))
            self.assertIsNone(pipeline.calls[0]["target_file_id"])
        finally:
            manager._executor.shutdown(wait=True)

    def test_30_reanalysis_replaces_only_selected_file(self) -> None:
        second_file, second_path = self.storage.create_file_record(
            self.session["id"], "second.jpg", "image/jpeg", ".jpg", "image"
        )
        second_path.write_bytes(b"second-image")
        self.storage.finalize_file(second_file["id"], second_path.stat().st_size)
        second_event = {
            **self.event,
            "id": "event_second",
            "source_file_id": second_file["id"],
        }
        self.storage.save_event(second_event)
        completed = threading.Event()

        class RecordingPipeline:
            calls: list[dict] = []

            def run_session(inner_self, **kwargs) -> None:
                inner_self.calls.append(kwargs)
                completed.set()

        pipeline = RecordingPipeline()
        manager = AnalysisJobManager(self.storage, pipeline)
        try:
            manager.start(self.session["id"], reanalyze_event_id=self.event["id"])
            remaining = self.storage.list_events(self.session["id"])
            self.assertEqual([event["id"] for event in remaining], ["event_second"])
            self.assertTrue(completed.wait(2))
            self.assertEqual(pipeline.calls[0]["target_file_id"], self.file["id"])
            self.assertEqual(pipeline.calls[0]["reanalyze_event_id"], self.event["id"])
        finally:
            manager._executor.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
