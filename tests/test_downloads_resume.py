"""Resume behaviour of download_file, exercised against a local server."""
import os, threading, http.server, functools, tempfile
import pytest
from utils.downloads import download_file

PAYLOAD = bytes(range(256)) * 12000  # 3 MB, several download chunks
# iter_content never yields a partial chunk, so a break inside the first
# chunk leaves nothing on disk to resume from. Real files are far larger
# than CHUNK_SIZE, so the payload has to be too for the test to mean anything.


class Flaky(http.server.BaseHTTPRequestHandler):
    cut_after = 1_500_000
    served = []

    def do_GET(self):
        start = 0
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
        body = PAYLOAD[start:]
        self.send_response(206 if rng else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        # drop the connection partway on the first attempt only
        limit = self.cut_after if not Flaky.served else len(body)
        Flaky.served.append(start)
        try:
            self.wfile.write(body[:limit])
        except Exception:
            pass

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    Flaky.served = []
    srv = http.server.HTTPServer(("127.0.0.1", 0), Flaky)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/f"
    srv.shutdown()


def test_resume_continues_from_the_partial(server, tmp_path):
    dest = str(tmp_path / "f.bin")
    ok = download_file(server, dest, threading.Event(), lambda d, t: None,
                       resume=True, tries=4)
    assert ok
    assert open(dest, "rb").read() == PAYLOAD
    # the retry asked for the bytes it was missing rather than starting over
    assert Flaky.served[0] == 0
    assert Flaky.served[1] > 0, "second attempt restarted instead of resuming"


def test_without_resume_a_broken_transfer_raises_and_leaves_nothing(server, tmp_path):
    dest = str(tmp_path / "f.bin")
    with pytest.raises(Exception):
        download_file(server, dest, threading.Event(), lambda d, t: None)
    assert not os.path.exists(dest)
    assert not os.path.exists(dest + ".part")
