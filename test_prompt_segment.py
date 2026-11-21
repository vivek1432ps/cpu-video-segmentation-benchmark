import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
try:
    from PIL import Image
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("Pillow is installed with the optional YOLOE requirements") from exc

from prompt_segment import Segmenter, load_image, main, parse_prompts, save_result


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values

    def tolist(self):
        return self.values.tolist()


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (5, 3), (120, 20, 9))
        self.model = Mock()
        self.model.get_text_pe.return_value = 'embedding'
        self.result = SimpleNamespace(
            masks=SimpleNamespace(data=Tensor([np.ones((3, 5))])),
            boxes=SimpleNamespace(cls=Tensor([0]), conf=Tensor([0.91])),
            names={0: "person"},
            plot=Mock(return_value=np.full((3, 5, 3), (21, 57, 122), dtype=np.uint8)),
        )
        self.model.predict.return_value = [self.result]
        self.segmenter = Segmenter(model_factory=lambda _checkpoint: self.model)

    def test_parse_and_validate(self):
        self.assertEqual(parse_prompts(" woman,traffic light,, woman "), ["woman", "traffic light"])
        for invalid in (" , ", "a\nb", "a" * 81, ",".join(map(str, range(21)))):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_prompts(invalid)

    def test_masks_cpu_output_and_prompt_switching(self):
        overlay, items, details = self.segmenter.run(self.image, "person")
        self.segmenter.run(self.image, "person , person")
        self.model.set_classes.assert_called_once_with(["person"], "embedding")
        self.assertEqual(overlay.getpixel((0, 0)), (122, 57, 21))
        np.testing.assert_array_equal(items[0][0], np.ones((3, 5), dtype=bool))
        self.assertEqual(items[0][1:], ("person", 0.91))
        self.assertEqual(details["prompts"], ["person"])
        kwargs = self.model.predict.call_args.kwargs
        self.assertEqual(kwargs["device"], "cpu")
        self.assertTrue(kwargs["retina_masks"])
        self.assertEqual(kwargs["source"][0, 0].tolist(), [9, 20, 120])
        self.segmenter.run(self.image, "bus")
        self.model.set_classes.assert_called_with(["bus"], "embedding")

    def test_empty_prediction_and_export(self):
        self.result.masks = None
        result = self.segmenter.run(self.image, "bus")
        self.assertEqual(result[0].tobytes(), self.image.tobytes())
        self.assertEqual(result[1], [])
        with TemporaryDirectory() as work:
            output = Path(work) / "out"
            save_result(output, *result)
            data = json.loads((output / "result.json").read_text())
            self.assertEqual(data["instances"], [])
            self.assertEqual(data["width"], 5)
            with self.assertRaises(FileExistsError):
                save_result(output, *result)

    def test_binary_mask_export(self):
        with TemporaryDirectory() as work:
            output = Path(work) / "test"
            save_result(output, *self.segmenter.run(self.image, "person"))
            with Image.open(output / "mask-001.png") as mask:
                self.assertEqual(mask.mode, "L")
                self.assertEqual(mask.size, (5, 3))
                self.assertEqual(mask.getpixel((0, 0)), 255)

    def test_load_local_image_with_pixel_limit(self):
        with TemporaryDirectory() as work:
            image_path = Path(work) / "source.png"
            self.image.save(image_path)
            self.assertEqual(load_image(image_path).mode, "RGB")
            with patch("prompt_segment.MAX_PIXELS", 9), self.assertRaisesRegex(ValueError, "resize"):
                load_image(image_path)

    def test_cli_never_overwrites_existing_output(self):
        with TemporaryDirectory() as work, patch("prompt_segment.Segmenter") as ctor:
            with self.assertRaises(SystemExit) as caught:
                main(["image.png", "--prompts", "person", "--out", work])
            self.assertEqual(caught.exception.code, 1)
            ctor.assert_not_called()

    def test_prompt_free_checkpoint_not_supported(self):
        with self.assertRaisesRegex(ValueError, "official YOLOE-26"):
            Segmenter("yoloe-26n-seg-pf.pt", model_factory=lambda _checkpoint: self.model)


if __name__ == "__main__":
    unittest.main()
