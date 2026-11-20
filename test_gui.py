import re
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from unittest import TestCase
from urllib.error import HTTPError
from urllib.request import build_opener, ProxyHandler

from benchmark import write_model
from gui import Demo, HTTPServer, make_handler, parse_selection


class GuiTests(TestCase):
    @classmethod
    def setUpClass(cls):
        directory = TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        demo = Demo(write_model(Path(directory.name) / "synthetic.onnx"))
        cls.server = HTTPServer(("127.0.0.1", 0), make_handler(demo))
        cls.addClassCleanup(cls.server.server_close)
        worker = Thread(target=cls.server.serve_forever, daemon=True)
        worker.start()
        cls.addClassCleanup(worker.join)
        cls.addClassCleanup(cls.server.shutdown)
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"
        cls.client = build_opener(ProxyHandler({}))

    def test_selection(self):
        self.assertEqual(parse_selection(""), ("ort", 24))
        for query in ("backend=gpu", "frame=120", "frame=-1", "frame=1&frame=3", "upload=x"):
            with self.subTest(query=query), self.assertRaises(ValueError):
                parse_selection(query)

    def test_http_backend_and_frame(self):
        sources = []
        for backend, frame, name in (("ort", 24, "ONNX Runtime CPU"),
                                     ("opencv", 91, "OpenCV DNN CPU")):
            with self.subTest(backend=backend):
                with self.client.open(f"{self.url}/?backend={backend}&frame={frame}", timeout=4) as response:
                    self.assertEqual(response.status, 200)
                    self.assertEqual(response.headers["Cache-Control"], "no-store")
                    self.assertIn("default-src 'none'", response.headers["Content-Security-Policy"])
                    page = response.read().decode()
                self.assertIn(f"Frame {frame:03d}", page)
                self.assertIn(f"Positive logits · {name}", page)
                self.assertIn(">1.000<", page)
                self.assertRegex(page, r"\d+\.\d{3} ms")
                images = re.findall(r'data:image/png;base64,([^"]+)', page)
                self.assertEqual(len(images), 2)
                sources.append(images[0])
        self.assertNotEqual(*sources)

    def test_invalid_requests(self):
        for suffix, status in (("/?frame=300", 400), ("/missing", 404)):
            with self.subTest(suffix=suffix), self.assertRaises(HTTPError) as caught:
                self.client.open(self.url + suffix, timeout=4)
            self.assertEqual(caught.exception.code, status)
            caught.exception.close()
