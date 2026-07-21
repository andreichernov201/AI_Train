from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

import server


class ServerSegmentationTests(unittest.TestCase):
    def test_disconnected_mask_keeps_largest_component_without_joining_lines(self) -> None:
        mask = np.zeros((1, 10, 10), dtype=np.float32)
        mask[0, 2:8, 1:6] = 1
        mask[0, 0:2, 8:10] = 1
        result = SimpleNamespace(
            names={0: "body"},
            boxes=SimpleNamespace(
                xyxy=np.asarray([[0, 0, 200, 100]], dtype=np.float32),
                conf=np.asarray([0.9], dtype=np.float32),
                cls=np.asarray([0], dtype=np.float32),
            ),
            masks=SimpleNamespace(data=mask, orig_shape=(100, 200)),
        )

        detections = server.yolo_result_to_detections(
            result,
            include_segments=True,
            allowed_class_names=server.SEGMENTATION_CLASS_NAMES,
        )

        self.assertEqual(len(detections), 1)
        polygon = np.asarray(detections[0]["segment"])
        self.assertGreaterEqual(len(polygon), 4)
        self.assertLessEqual(float(polygon[:, 0].max()), 100.0)
        self.assertGreaterEqual(float(polygon[:, 1].min()), 20.0)

    def test_empty_mask_is_not_returned_as_segmentation(self) -> None:
        result = SimpleNamespace(
            names={0: "body"},
            boxes=SimpleNamespace(
                xyxy=np.asarray([[0, 0, 10, 10]], dtype=np.float32),
                conf=np.asarray([0.9], dtype=np.float32),
                cls=np.asarray([0], dtype=np.float32),
            ),
            masks=SimpleNamespace(
                data=np.zeros((1, 10, 10), dtype=np.float32),
                orig_shape=(10, 10),
            ),
        )

        self.assertEqual(
            server.yolo_result_to_detections(result, include_segments=True),
            [],
        )

    def test_same_class_duplicate_masks_keep_highest_confidence(self) -> None:
        masks = np.zeros((2, 12, 12), dtype=np.float32)
        masks[0, 2:10, 2:10] = 1
        masks[1, 2:10, 3:11] = 1
        result = SimpleNamespace(
            names={0: "body"},
            boxes=SimpleNamespace(
                xyxy=np.asarray(
                    [[2, 2, 10, 10], [3, 2, 11, 10]],
                    dtype=np.float32,
                ),
                conf=np.asarray([0.55, 0.92], dtype=np.float32),
                cls=np.asarray([0, 0], dtype=np.float32),
            ),
            masks=SimpleNamespace(data=masks, orig_shape=(12, 12)),
        )

        detections = server.yolo_result_to_detections(
            result,
            include_segments=True,
        )

        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0]["conf"], 0.92, places=5)

    def test_nested_same_class_mask_is_treated_as_duplicate(self) -> None:
        masks = np.zeros((2, 12, 12), dtype=np.float32)
        masks[0, 1:11, 1:11] = 1
        masks[1, 3:9, 3:9] = 1
        result = SimpleNamespace(
            names={2: "axlebox"},
            boxes=SimpleNamespace(
                xyxy=np.asarray(
                    [[1, 1, 11, 11], [3, 3, 9, 9]],
                    dtype=np.float32,
                ),
                conf=np.asarray([0.88, 0.7], dtype=np.float32),
                cls=np.asarray([2, 2], dtype=np.float32),
            ),
            masks=SimpleNamespace(data=masks, orig_shape=(12, 12)),
        )

        detections = server.yolo_result_to_detections(
            result,
            include_segments=True,
        )

        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0]["conf"], 0.88, places=5)

    def test_neighboring_masks_of_same_class_are_preserved(self) -> None:
        masks = np.zeros((2, 12, 12), dtype=np.float32)
        masks[0, 2:10, 1:5] = 1
        masks[1, 2:10, 7:11] = 1
        result = SimpleNamespace(
            names={2: "axlebox"},
            boxes=SimpleNamespace(
                xyxy=np.asarray(
                    [[1, 2, 5, 10], [7, 2, 11, 10]],
                    dtype=np.float32,
                ),
                conf=np.asarray([0.9, 0.85], dtype=np.float32),
                cls=np.asarray([2, 2], dtype=np.float32),
            ),
            masks=SimpleNamespace(data=masks, orig_shape=(12, 12)),
        )

        detections = server.yolo_result_to_detections(
            result,
            include_segments=True,
        )

        self.assertEqual(len(detections), 2)

    def test_overlapping_masks_of_different_classes_are_preserved(self) -> None:
        masks = np.zeros((2, 12, 12), dtype=np.float32)
        masks[:, 2:10, 2:10] = 1
        result = SimpleNamespace(
            names={0: "body", 1: "bogie"},
            boxes=SimpleNamespace(
                xyxy=np.asarray(
                    [[2, 2, 10, 10], [2, 2, 10, 10]],
                    dtype=np.float32,
                ),
                conf=np.asarray([0.9, 0.8], dtype=np.float32),
                cls=np.asarray([0, 1], dtype=np.float32),
            ),
            masks=SimpleNamespace(data=masks, orig_shape=(12, 12)),
        )

        detections = server.yolo_result_to_detections(
            result,
            include_segments=True,
        )

        self.assertEqual(len(detections), 2)
        self.assertEqual(
            {row["cls_name"] for row in detections},
            {"body", "bogie"},
        )

    def test_full_image_uses_trained_resolution_and_f1_threshold(self) -> None:
        model = MagicMock(return_value=[SimpleNamespace()])
        converted = [{"id": 0, "segment": [[0, 0], [1, 0], [1, 1]]}]
        with patch.object(
            server,
            "get_segmentation_model",
            return_value=model,
        ), patch.object(
            server,
            "yolo_result_to_detections",
            return_value=converted,
        ) as converter, patch.dict(
            os.environ,
            {"AI_TRAIN_SEG_CONF": "", "AI_TRAIN_SEG_IMGSZ": ""},
        ):
            image = Image.new("RGB", (1920, 1080))
            result = server.segment_full_image(image)

        self.assertEqual(result, converted)
        model.assert_called_once_with(
            image,
            verbose=False,
            retina_masks=True,
            imgsz=1280,
            conf=0.39,
        )
        converter.assert_called_once()

    def test_inference_settings_can_be_overridden(self) -> None:
        with patch.dict(
            os.environ,
            {"AI_TRAIN_SEG_CONF": "0.55", "AI_TRAIN_SEG_IMGSZ": "1536"},
        ):
            self.assertEqual(server._segmentation_confidence(), 0.55)
            self.assertEqual(server._segmentation_image_size(), 1536)


if __name__ == "__main__":
    unittest.main()
