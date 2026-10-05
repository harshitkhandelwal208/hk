"""The official OpenAI Python client works against `hk serve`.

The server itself is tested end to end in Zig (tests/server_tests.zig). This is the one check that
needs the client library, so it is skipped when `openai` is not installed.
"""
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest

openai = pytest.importorskip("openai")

ROOT = Path(__file__).resolve().parent.parent
HK = ROOT / "zig-out" / "bin" / "hk"
TINY = ROOT / "zig-out" / "bin" / "hk-tiny-model"
pytestmark = pytest.mark.skipif(not (HK.exists() and TINY.exists()), reason="run `zig build` first")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    d = tmp_path_factory.mktemp("openai")
    gguf, hk_file = d / "tiny.gguf", d / "tiny.hk"
    subprocess.run([str(TINY), str(gguf), "--chat"], check=True)
    subprocess.run([str(HK), "convert-gguf", str(gguf), str(hk_file)], check=True, capture_output=True)
    port = free_port()
    proc = subprocess.Popen([str(HK), "serve", str(hk_file), "--port", str(port), "--threads", "2"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            urllib.request.urlopen(url + "/health", timeout=1).read()
            break
        except Exception:
            time.sleep(0.1)
    yield url
    proc.terminate()
    proc.wait(5)


def test_official_openai_client(server):
    c = openai.OpenAI(base_url=server + "/v1", api_key="none")
    r = c.chat.completions.create(model="tiny", messages=[{"role": "user", "content": "hello"}], temperature=0, max_tokens=8)
    assert r.choices[0].message.content is not None and r.usage.completion_tokens == 8
    chunks = list(c.chat.completions.create(model="tiny", messages=[{"role": "user", "content": "hello"}],
                                            temperature=0, max_tokens=8, stream=True))
    assert chunks[-1].choices[0].finish_reason in ("length", "stop")
    comp = c.completions.create(model="tiny", prompt="hello", max_tokens=5, temperature=0)
    assert comp.choices[0].text is not None
