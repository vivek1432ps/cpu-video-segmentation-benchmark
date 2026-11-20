import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx

from benchmark import HEIGHT, WIDTH, benchmark, generate, preprocess, scene


class VideoBenchmarkTests(unittest.TestCase):
    def test_fixture_is_deterministic(self):
        frame, mask = scene(6)
        self.assertEqual(frame.shape, (HEIGHT, WIDTH, 3))
        np.testing.assert_array_equal(frame, scene(6)[0])
        self.assertFalse(np.array_equal(mask, scene(7)[1]))
        self.assertGreater(mask.sum(), 0)
        self.assertLess(mask.sum(), mask.size)

    def test_preprocess_is_normalized_bgr_nchw(self):
        sample = np.array([[[0, 51, 255]]], dtype=np.uint8)
        result = preprocess(sample)
        self.assertEqual(result.shape, (1, 3, 1, 1))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result.flatten(), [0, 0.2, 1.0])

    def test_video_and_backends(self):
        with tempfile.TemporaryDirectory() as work:
            video, model = generate(Path(work), 6)
            onnx.checker.check_model(onnx.load(model))
            for backend in ("ort", "opencv"):
                with self.subTest(backend=backend):
                    result = benchmark(video, model, backend, 6, 2)
                    self.assertEqual(result["frames"], 4)
                    self.assertEqual(result["mask_iou"], 1.0)
                    self.assertGreater(result["pipeline_fps"], 0)
                    for latency in result["latency_ms"].values():
                        self.assertLessEqual(latency["p50"], latency["p95"])
            with self.assertRaisesRegex(RuntimeError, "Unexpected end of video"):
                benchmark(video, model, "ort", 7, 2)


if __name__ == "__main__":
    unittest.main()
