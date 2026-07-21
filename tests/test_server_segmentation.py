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
