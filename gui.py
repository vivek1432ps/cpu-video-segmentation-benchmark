"""An offline browser GUI for the CPU ONNX segmentation demo."""
import argparse
import base64
import html
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from urllib.parse import parse_qs, urlsplit
import webbrowser

import cv2
import numpy as np

from benchmark import (HEIGHT, WIDTH, benchmark, generate, make_predictor,
                       preprocess, scene, write_model)

FRAME_MAX = 119
SAMPLE_FRAMES, SAMPLE_WARMUP = 24, 4
BACKENDS = {"ort": "ONNX Runtime CPU", "opencv": "OpenCV DNN CPU"}
STYLE = """*{box-sizing:border-box}body{margin:0;background:#080712;color:#eef4fa;
font:16px/1.5 system-ui,sans-serif}main{width:min(1080px,calc(100% - 40px));
margin:46px auto 36px}h1{font-size:3.5rem;font-size:clamp(2.4rem,5.2vw,4.1rem);line-height:1.02;
letter-spacing:-.055em;margin:14px 0 14px}h2{font-size:1rem;margin:0 0 12px}p{margin:0}
.tag,.eyebrow{font-size:.75rem;font-weight:750;letter-spacing:.09em;text-transform:uppercase;
color:#35f7ff}.tag{display:inline-block;padding:5px 10px;border:1px solid #237782;border-radius:20px}
.lead,.note,figcaption,.control-note{color:#c0bbd6}.lead{max-width:680px;font-size:1.05rem}
form{margin:28px 0 20px;padding:18px 20px;display:flex;align-items:end;gap:16px;flex-wrap:wrap;
background:#141026;border:1px solid #372047;border-radius:14px}
label{display:block;margin-bottom:5px;font-weight:650;font-size:.87rem}select,input,button{
font:inherit;padding:9px 11px;border-radius:7px;border:1px solid #73598c;min-height:44px}
select,input{background:#110a22;color:#eef4fa}input{width:100px}button{background:#35f7ff;
color:#082129;font-weight:750;border:1px solid #35f7ff;cursor:pointer}button:hover{background:#9bfcff}
:focus-visible{outline:3px solid #ff61c4;outline-offset:3px}.control-note{font-size:.78rem;
margin-left:auto;align-self:center}.stats,.panels{display:grid;gap:13px}.stats{
grid-template-columns:repeat(3,1fr);margin-bottom:18px}.card,figure{background:#141026;
border:1px solid #372047;border-radius:12px}.card{padding:13px 16px}.metric{
display:block;font-size:1.63rem;font-weight:740;letter-spacing:-.04em;margin:3px 0 0}
.panels{grid-template-columns:repeat(2,1fr)}figure{margin:0;overflow:hidden;padding:13px}
img{width:100%;height:auto;display:block;border-radius:6px}figcaption{margin-top:9px;font-size:.84rem}
.note{margin-top:18px;font-size:.84rem}.note strong{color:#eef4fa}.error{color:#ffbe9c;margin:20px 0}

body{background:radial-gradient(ellipse at 78% -15%,#44083e 0,transparent 45%),#080712}
header{position:relative;border-left:4px solid #ff38be;padding-left:20px}
h1{text-shadow:0 0 22px #35f7ff43}section.sample{padding:16px 18px;background:#160b25;
border:1px solid #8a2c80;border-radius:12px;box-shadow:0 0 28px #ff38be24;margin:20px 0}
.sample form{background:none;border:0;padding:0;margin:0 0 11px;align-items:center}
.sample button{background:#ff38be;border-color:#ff38be;color:#22051e;box-shadow:0 0 14px #ff38be50}
.sample button:hover{background:#ff8bd3}.sample p{font-size:.83rem;color:#c0bbd6}
.compare{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:12px 0}
.compare .card:first-child{border-color:#237782}.compare .card:last-child{border-color:#81347e}
.compare .card:last-child .eyebrow{color:#ff79d0}.label{color:#c0bbd6;font-size:.83rem}
@media(max-width:650px){main{margin:26px auto}.control-note{margin-left:0}.panels{grid-template-columns:1fr}
.stats{gap:7px}.card{padding:10px}.metric{font-size:1.05rem}.eyebrow{font-size:.65rem}}"""

def parse_selection(query):
    fields = parse_qs(query, keep_blank_values=True, max_num_fields=6)
    if set(fields) - {"backend", "frame"} or any(len(v) != 1 for v in fields.values()):
        raise ValueError("Choose one backend and one frame number.")
    backend = fields.get("backend", ["ort"])[0]
    frame = fields.get("frame", ["24"])[0]
    if backend not in BACKENDS or not re.fullmatch(r"[0-9]{1,3}", frame) or int(frame) > FRAME_MAX:
        raise ValueError(f"Choose a listed backend and a frame from 0 to {FRAME_MAX}.")
    return backend, int(frame)


def png_uri(image):
    ok, compressed = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Could not encode the preview image.")
    return "data:image/png;base64," + base64.b64encode(compressed).decode("ascii")


class Demo:
    def __init__(self, model):
        cv2.setNumThreads(1)
        self.predictors = {backend: make_predictor(model, backend) for backend in BACKENDS}
        warmup = preprocess(scene(0)[0])
        for predict in self.predictors.values():
            predict(warmup)

    def run(self, backend, index):
        frame, truth = scene(index)
        tensor = preprocess(frame)
        start = perf_counter_ns()
        logits = self.predictors[backend](tensor)
        latency = (perf_counter_ns() - start) / 1e6
        mask = logits[0, 0] > 0
        intersection = np.count_nonzero(mask & truth)
        union = np.count_nonzero(mask | truth)
        mask_image = np.full_like(frame, (34, 24, 15))
        mask_image[mask] = (188, 236, 114)
        return latency, intersection / union if union else 1.0, png_uri(frame), png_uri(mask_image)

    def sample_video(self):
        """Run both engines in sequence on one generated, identical local clip."""
        with TemporaryDirectory(prefix="segmentation-sample-") as work:
            video, model = generate(Path(work), SAMPLE_FRAMES)
            return [benchmark(video, model, backend, SAMPLE_FRAMES, SAMPLE_WARMUP)
                    for backend in BACKENDS]


def render_sample(backend, frame, results):
    comparison = ""
    if results is not None:
        cards = []
        for result in results:
            pipeline = result["latency_ms"]["pipeline"]
            inference = result["latency_ms"]["inference"]
            cards.append(f'<div class="card"><span class="eyebrow">{BACKENDS[result["backend"]]}</span>'
                         f'<span class="metric">{pipeline["p50"]:.2f} ms</span>'
                         f'<p>Pipeline p50 · p95 {pipeline["p95"]:.2f} ms</p>'
                         f'<p>Inference p50 {inference["p50"]:.2f} ms · Mask IoU {result["mask_iou"]:.3f}</p></div>')
        comparison = ('<div class="compare" aria-label="Video benchmark results">' + "".join(cards)
                      + '</div><p id="result-note">Same FFV1 clip, 20 measured frames per engine after 4 warm-up frames. '
                        'Sequential runs; timings vary, and this is not a general backend ranking.</p>')
    return f"""<section class="sample" aria-labelledby="sample-heading">
<h2 id="sample-heading">Sample video · CPU head-to-head</h2>
<form method="post" action="/sample?backend={backend}&amp;frame={frame}">
<button type="submit">Run sample video</button><p>Both engines · 24 synthetic frames · 640 × 360</p></form>
<p>Pipeline time is read/decode through mask threshold. Inference time covers only the model call.</p>{comparison}</section>"""


def render(demo, backend="ort", frame=24, error="", comparison=None):
    options = "".join(f'<option value="{key}"{" selected" if backend == key else ""}>'
                      f'{name}</option>' for key, name in BACKENDS.items())
    content = ""
    if error:
        content = f'<p class="error" role="alert">{html.escape(error)}</p>'
    else:
        latency, iou, source, mask = demo.run(backend, frame)
        content = f"""
<section class="stats" aria-label="Selected frame measurements">
<div class="card"><span class="eyebrow">Inference</span><span class="metric">{latency:.3f} ms</span></div>
<div class="card"><span class="eyebrow">Mask IoU</span><span class="metric">{iou:.3f}</span></div>
<div class="card"><span class="eyebrow">Input size</span><span class="metric">{WIDTH} × {HEIGHT}</span></div>
</section>
<section class="panels" aria-label="Segmentation images">
<figure><h2>01 · Synthetic input</h2><img src="{source}" width="{WIDTH}" height="{HEIGHT}"
alt="Red circle and rectangle on a noisy background"><figcaption>Frame {frame:03d} · Generated on this machine</figcaption></figure>
<figure><h2>02 · Predicted mask</h2><img src="{mask}" width="{WIDTH}" height="{HEIGHT}"
alt="Green predicted foreground shapes on a dark background"><figcaption>Positive logits · {BACKENDS[backend]}</figcaption></figure>
</section>"""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CPU Segmentation Lab</title><style>{STYLE}</style></head>
<body><main>
<header><span class="tag">Local demo · CPU only</span><h1>CPU Segmentation Lab</h1>
<p class="lead">Inspect the same tiny ONNX model on two CPU backends. Select a synthetic frame to run inference and see the predicted foreground.</p></header>
{render_sample(backend, frame, comparison)}
<form action="/" method="get"><div><label for="backend">Inference backend</label>
<select id="backend" name="backend">{options}</select></div><div><label for="frame">Frame</label>
<input id="frame" name="frame" type="number" min="0" max="{FRAME_MAX}" value="{frame}" required>
</div><button type="submit">Run inference</button><p class="control-note">Frames 0–{FRAME_MAX} · No uploads</p></form>
{content}
<p class="note"><strong>What is measured:</strong> One warmed inference call on an in-memory frame.
Does not include video decoding, image rendering, or browser time. IoU compares the prediction with the synthetic scene mask.
This is not a trained model or a real-world accuracy claim.</p>
</main></body></html>"""


def make_handler(demo):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            target = urlsplit(self.path)
            if target.path != "/sample":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                backend, frame = parse_selection(target.query)
            except ValueError as exc:
                self.send_page(HTTPStatus.BAD_REQUEST, render(demo, error=str(exc)))
                return
            self.send_page(HTTPStatus.OK, render(demo, backend, frame, comparison=demo.sample_video()))

        def do_GET(self):
            target = urlsplit(self.path)
            if target.path != "/":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                backend, frame = parse_selection(target.query)
            except ValueError as exc:
                status, page = HTTPStatus.BAD_REQUEST, render(demo, error=str(exc))
            else:
                status, page = HTTPStatus.OK, render(demo, backend, frame)
            self.send_page(status, page)

        def send_page(self, status, page):
            body = page.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy",
                             "default-src 'none'; style-src 'unsafe-inline'; img-src data:; form-action 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765,
                        help="loopback port; use 0 for an automatically selected port")
    parser.add_argument("--open", action="store_true", help="open your default browser")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    with TemporaryDirectory(prefix="segmentation-lab-") as work:
        model = write_model(Path(work) / "synthetic.onnx")
        demo = Demo(model)
        try:
            server = HTTPServer(("127.0.0.1", args.port), make_handler(demo))
        except OSError as exc:
            parser.error(f"could not bind loopback port {args.port}: {exc}; try --port 0")
        with server:
            url = f"http://127.0.0.1:{server.server_port}/"
            print(f"Open {url}  (Ctrl+C to stop)", flush=True)
            if args.open:
                webbrowser.open(url)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
