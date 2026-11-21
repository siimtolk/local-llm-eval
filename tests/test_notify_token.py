"""Offline test: notify.post_json sends the bridge bearer token from [notify] token_file.
Run from the repo root: python3 -m unittest tests.test_notify_token -v"""
import http.server
import sys
import tempfile
import threading
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import notify  # noqa: E402

TOKEN = "test-token-123"


class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        ok = self.headers.get("Authorization") == "Bearer " + TOKEN
        self.send_response(200 if ok else 401)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *a):
        pass


class TokenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = "http://127.0.0.1:%d/send" % cls.srv.server_port

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def cfg(self, tf):
        return mock.patch.object(notify.config, "get", lambda s, k, d=None: tf if k == "token_file" else d)

    def test_with_token(self):
        with tempfile.NamedTemporaryFile("w") as f, self.cfg(f.name):
            f.write(TOKEN + "\n")
            f.flush()
            self.assertEqual(notify.post_json(self.url, {}), {})

    def test_without_token_401(self):
        with self.cfg(""), self.assertRaises(urllib.error.HTTPError) as e:
            notify.post_json(self.url, {})
        self.assertEqual(e.exception.code, 401)

    def test_unreadable(self):
        with self.cfg("/nonexistent/tok"), self.assertRaises(OSError) as e:
            notify.post_json(self.url, {})
        self.assertEqual(str(e.exception), "notify token_file unreadable")


if __name__ == "__main__":
    unittest.main()
