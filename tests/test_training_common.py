from __future__ import annotations

import os
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from tools.training_common import build_run_name, resolve_dataset_yaml


class TrainingCommonTests(unittest.TestCase):
    def test_zip_is_extracted_and_old_yaml_becomes_portable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "export.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr(
                    "data.yaml",
                    "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames: [train, number]\n",
                )
                zf.writestr("images/train/001.png", b"image")
                zf.writestr("labels/train/001.txt", b"0 0.5 0.5 0.2 0.2\n")

            yaml_path = resolve_dataset_yaml(root, "detect", archive)
            self.assertEqual(yaml_path.name, "data.local.yaml")
            text = yaml_path.read_text(encoding="utf-8")
            self.assertNotIn("path:", text)
            self.assertNotIn("test:", text)
            self.assertTrue((yaml_path.parent / "images/train/001.png").is_file())

    def test_zip_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("../escape.txt", "no")
                zf.writestr("data.yaml", "train: images/train\nval: images/val\nnames: [x]\n")

            with self.assertRaises(ValueError):
                resolve_dataset_yaml(root, "segment", archive)
            self.assertFalse((root / "escape.txt").exists())

    def test_newest_data_yaml_is_selected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset" / "detection"
            older = dataset / "data.yaml"
            newer = dataset / "imports" / "fresh" / "data.yaml"
            newer.parent.mkdir(parents=True)
            older.parent.mkdir(parents=True, exist_ok=True)
            older.write_text("train: images/train\nval: images/val\nnames: [old]\n")
            newer.write_text("train: images/train\nval: images/val\nnames: [new]\n")
            now = time.time()
            os.utime(older, (now - 10, now - 10))
            os.utime(newer, (now, now))

            self.assertEqual(resolve_dataset_yaml(root, "detect"), newer.resolve())

    def test_run_name_is_timestamped_and_sanitized(self) -> None:
        name = build_run_name("yolo11m.pt", "мой запуск")
        self.assertRegex(name, r"^\d{8}_\d{6}_мой_запуск$")


class ServerModelResolutionTests(unittest.TestCase):
    def test_detection_uses_latest_best_pt(self) -> None:
        import server

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"AI_TRAIN_MODEL_PATH": ""}
        ):
            runs = Path(tmp) / "detect"
            old = runs / "old" / "weights" / "best.pt"
            new = runs / "new" / "weights" / "best.pt"
            old.parent.mkdir(parents=True)
            new.parent.mkdir(parents=True)
            old.write_bytes(b"old")
            new.write_bytes(b"new")
            now = time.time()
            os.utime(old, (now - 10, now - 10))
            os.utime(new, (now, now))

            with patch.object(server, "RUNS_DETECT_DIR", str(runs)), patch.object(
                server, "DEFAULT_DETECT_MODEL_PATH", str(runs / "weights" / "best.pt")
            ):
                self.assertEqual(server.resolve_detection_model_path(), str(new.resolve()))


if __name__ == "__main__":
    unittest.main()
