"""Small CPU video-segmentation benchmark with a synthetic oracle."""
import argparse
import json
import platform
from pathlib import Path
from time import perf_counter_ns

import cv2
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

WIDTH, HEIGHT = 640, 360
STAGES = ("decode", "preprocess", "inference", "postprocess", "pipeline")


def scene(index):
    """Return a noisy BGR frame and a mask defined by scene geometry."""
    rng = np.random.default_rng(index)
    bgr = rng.integers(10, 56, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.circle(mask, (40 + (7 * index) % (WIDTH - 80),
                      40 + (5 * index) % (HEIGHT - 80)), 26, 1, -1)
    cv2.rectangle(mask, (20 + (3 * index) % (WIDTH - 90), HEIGHT - 85),
                  (70 + (3 * index) % (WIDTH - 90), HEIGHT - 58), 1, -1)
    bgr[..., 2] += mask * 180
    return bgr, mask.astype(bool)


def generate(folder, frames):
    folder.mkdir(parents=True, exist_ok=True)
    video = folder / "synthetic.avi"
    writer = cv2.VideoWriter(str(video), cv2.CAP_FFMPEG,
                             cv2.VideoWriter_fourcc(*"FFV1"), 30, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the FFV1 writer")
    try:
        for index in range(frames):
            writer.write(scene(index)[0])
    finally:
        writer.release()

    # Hand-designed linear classifier for synthetic red shapes, not trained.
    weights = numpy_helper.from_array(np.array([-0.5, -0.5, 1], dtype=np.float32)
                                     .reshape(1, 3, 1, 1), "weights")
    bias = numpy_helper.from_array(np.array([-0.4], dtype=np.float32), "bias")
    graph = helper.make_graph(
        [helper.make_node("Conv", ["frame", "weights", "bias"], ["logits"],
                          kernel_shape=[1, 1])],
        "synthetic_chroma_segmenter",
        [helper.make_tensor_value_info("frame", TensorProto.FLOAT,
                                       [1, 3, HEIGHT, WIDTH])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT,
                                       [1, 1, HEIGHT, WIDTH])],
        [weights, bias],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)],
                              ir_version=9)
    onnx.checker.check_model(model)
    model_path = folder / "synthetic.onnx"
    onnx.save(model, model_path)
    return video, model_path


def preprocess(frame):
    return np.ascontiguousarray(frame.transpose(2, 0, 1)[None],
                                dtype=np.float32) / np.float32(255)


def make_predictor(model, backend):
    if backend == "ort":
        config = ort.SessionOptions()
        config.intra_op_num_threads = 1
        config.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(str(model), config,
                                       providers=["CPUExecutionProvider"])
        return lambda tensor: session.run(["logits"], {"frame": tensor})[0]
    if backend == "opencv":
        net = cv2.dnn.readNetFromONNX(str(model))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        def predict(tensor):
            net.setInput(tensor)
            return net.forward()

        return predict
    raise ValueError(f"Unknown backend: {backend}")


def benchmark(video, model, backend, frames, warmup):
    cv2.setNumThreads(1)  # FFmpeg can still manage its decoder independently.
    cap = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open the generated video")
    times = {stage: [] for stage in STAGES}
    intersections = unions = 0
    try:
        predict = make_predictor(model, backend)
        for index in range(frames):
            start = perf_counter_ns()
            ok, frame = cap.read()
            decoded = perf_counter_ns()
            if not ok:
                raise RuntimeError(f"Unexpected end of video at frame {index}")
            tensor = preprocess(frame)
            prepared = perf_counter_ns()
            logits = predict(tensor)
            inferred = perf_counter_ns()
            mask = logits[0, 0] > 0
            finished = perf_counter_ns()
            if index < warmup:
                continue
            durations = (decoded - start, prepared - decoded, inferred - prepared,
                         finished - inferred, finished - start)
            for stage, duration in zip(STAGES, durations):
                times[stage].append(duration)
            # Oracle regeneration and metric computation are not timed.
            truth = scene(index)[1]
            intersections += np.count_nonzero(mask & truth)
            unions += np.count_nonzero(mask | truth)
    finally:
        cap.release()
    return {
        "backend": backend,
        "frames": frames - warmup,
        "warmup_frames": warmup,
        "pipeline_fps": (frames - warmup) * 1e9 / sum(times["pipeline"]),
        "mask_iou": intersections / unions if unions else 1.0,
        "latency_ms": {
            stage: {"p50": float(np.percentile(values, 50) / 1e6),
                    "p95": float(np.percentile(values, 95) / 1e6)}
            for stage, values in times.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--backend", choices=("ort", "opencv", "both"), default="both")
    args = parser.parse_args()
    if not 0 <= args.warmup < args.frames:
        parser.error("require 0 <= warmup < frames")
    video, model = generate(args.out, args.frames)
    backends = ("ort", "opencv") if args.backend == "both" else (args.backend,)
    report = {
        "input": {"width": WIDTH, "height": HEIGHT, "codec": "FFV1",
                  "source": "deterministic synthetic clips", "nominal_video_fps": 30},
        "python": platform.python_version(),
        "os": platform.platform(),
        "versions": {"numpy": np.__version__, "opencv": cv2.__version__,
                     "onnx": onnx.__version__, "onnxruntime": ort.__version__},
        "results": [benchmark(video, model, b, args.frames, args.warmup)
                    for b in backends],
    }
    output = json.dumps(report, indent=2) + "\n"
    (args.out / "report.json").write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
