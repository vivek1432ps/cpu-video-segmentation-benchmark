"""Segment local images using YOLOE-26 text prompts, on CPU by default."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageOps

CHECKPOINTS = tuple(f"yoloe-26{s}-seg.pt" for s in "nsmlx")
IMAGE_TYPES = ("JPEG", "PNG", "WEBP", "BMP", "TIFF")
MAX_PIXELS = 16_000_000
MASK_PIXEL_BUDGET = 32_000_000


def parse_prompts(text):
    """Trim and deduplicate comma-separated category names, preserving order."""
    names = list(dict.fromkeys(part.strip() for part in text.split(",") if part.strip()))
    if not names or len(names) > 20 or any(
        len(name) > 80 or not name.isprintable() for name in names
    ):
        raise ValueError("Enter 1–20 comma-separated category names (max 80 characters each).")
    return names


def load_image(path):
    """Decode a local image as orientation-corrected RGB."""
    with Image.open(Path(path).expanduser(), formats=IMAGE_TYPES) as source:
        if source.width * source.height > MAX_PIXELS:
            raise ValueError("Input exceeds 16 megapixels; resize it first.")
        return ImageOps.exif_transpose(source).convert("RGB")


def save_result(directory, overlay, instances, metadata):
    """Write an overlay plus full-resolution, 0/255 per-instance masks and metadata."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    overlay.save(directory / "overlay.png")
    items = []
    for index, (mask, name, score) in enumerate(instances, 1):
        filename = f"mask-{index:03d}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(directory / filename)
        items.append({"label": name, "confidence": score, "mask": filename})
    report = {**metadata, "width": overlay.width, "height": overlay.height,
              "overlay": "overlay.png", "instances": items}
    (directory / "result.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


class Segmenter:
    """Keep one model in memory; refresh text embeddings when prompts change."""

    def __init__(self, checkpoint=CHECKPOINTS[0], device="cpu", model_factory=None):
        if checkpoint not in CHECKPOINTS:
            raise ValueError("Choose an official YOLOE-26 text-prompt checkpoint.")
        if model_factory is None:
            try:
                from ultralytics import YOLOE
            except ImportError as exc:
                raise RuntimeError("Install requirements-yoloe.txt in the YOLOE environment.") from exc
            model_factory = YOLOE
        self.model = model_factory(checkpoint)
        self.checkpoint, self.device = checkpoint, device
        self.names = None

    def run(self, image, text, conf=0.25, imgsz=640):
        names = parse_prompts(text)
        if not 0 < conf <= 1:
            raise ValueError("Confidence must be greater than 0 and at most 1.")
        if not 128 <= imgsz <= 1280 or imgsz % 32 != 0:
            raise ValueError("Inference size must be a multiple of 32 from 128 to 1280.")
        if names != self.names:
            # Official YOLOE text-prompt API also supports explicit embeddings.
            self.model.set_classes(names, self.model.get_text_pe(names))
            self.names = names
        bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])
        # Full-resolution masks can be much larger than the model input.
        max_instances = max(1, min(100, MASK_PIXEL_BUDGET // (image.width * image.height)))
        start = perf_counter()
        result = self.model.predict(source=bgr, device=self.device, conf=conf,
                                    imgsz=imgsz, retina_masks=True, verbose=False,
                                    max_det=max_instances)[0]
        elapsed = (perf_counter() - start) * 1000
        instances = []
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            ids = result.boxes.cls.cpu().tolist()
            scores = result.boxes.conf.cpu().tolist()
            if not (len(masks) == len(ids) == len(scores)):
                raise RuntimeError("The model returned mismatched mask and detection counts.")
            for mask, key, score in zip(masks, ids, scores):
                if mask.shape != (image.height, image.width):
                    raise RuntimeError("The model did not return masks at the input resolution.")
                instances.append((mask > 0.5, result.names[int(key)], round(float(score), 5)))
        overlay = Image.fromarray(np.ascontiguousarray(result.plot(boxes=False)[:, :, ::-1])) if instances else image.copy()
        return overlay, instances, {"checkpoint": self.checkpoint, "device": str(self.device),
                                    "prompts": names, "confidence_threshold": conf,
                                    "image_size": imgsz, "max_instances": max_instances,
                                    "predict_call_ms": round(elapsed, 2)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="local JPEG, PNG, WebP, BMP or TIFF")
    parser.add_argument("--prompts", required=True, help='quoted list, e.g. "person, bus"')
    parser.add_argument("--out", required=True, type=Path, help="new directory for overlay, masks and JSON")
    parser.add_argument("--model", choices=CHECKPOINTS, default=CHECKPOINTS[0])
    parser.add_argument("--device", default="cpu", help='CPU by default; e.g. "0" for first CUDA GPU')
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args(argv)
    try:
        parse_prompts(args.prompts)
        if args.out.exists():
            raise ValueError("Choose a new output directory (existing results are not overwritten).")
        image = load_image(args.image)
        segmenter = Segmenter(args.model, args.device)
        overlay, instances, metadata = segmenter.run(image, args.prompts, args.conf, args.imgsz)
        save_result(args.out, overlay, instances, metadata)
    except (OSError, ValueError, RuntimeError, ImportError, Image.DecompressionBombError) as exc:
        parser.exit(1, f"Segmentation failed: {exc}\n")
    print(f"{len(instances)} instance masks saved to {args.out} ({metadata['predict_call_ms']:.0f} ms prediction call).")


if __name__ == "__main__":
    main()
