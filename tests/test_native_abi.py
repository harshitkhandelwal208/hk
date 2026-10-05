"""The native engine through the C ABI, as the Python package uses it."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
HK = ROOT / "zig-out" / "bin" / "hk"
TINY = ROOT / "zig-out" / "bin" / "hk-tiny-model"
PROBE = ROOT / "zig-out" / "bin" / "hk-probe"

try:
    from hk import native
except Exception as e:  # pragma: no cover
    native = None
    _why = str(e)

pytestmark = pytest.mark.skipif(
    native is None or not native.is_native_available() or not hasattr(native._LIB, "hk_engine_forward_tokens"),
    reason="needs a freshly built libhk: zig build -Doptimize=ReleaseFast",
)


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    d = tmp_path_factory.mktemp("abi")
    g = d / "tiny.gguf"
    subprocess.run([str(TINY), str(g)], check=True)
    h = d / "tiny.hk"
    r = subprocess.run([str(HK), "convert-gguf", str(g), str(h)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return str(h)


def probe_logits(model, ids, tmp_path):
    out = tmp_path / "ref.f32"
    r = subprocess.run([str(PROBE), model, ",".join(map(str, ids)), str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return np.fromfile(out, dtype="<f4")


def test_forward_matches_the_probe_tool(model, tmp_path):
    ids = [3, 17, 5, 5, 29, 0, 11, 8]
    with native.NativeHKEngine(model) as eng:
        got = eng.forward(ids)
    want = probe_logits(model, ids, tmp_path)
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_batched_prefill_equals_token_by_token(model):
    ids = [3, 17, 5, 5, 29, 0, 11, 8, 40, 41]
    with native.NativeHKEngine(model) as eng:
        batched = eng.forward(ids)
        eng.reset_cache()
        step = None
        for pos, t in enumerate(ids):
            step = eng.forward_step(t, pos)
    np.testing.assert_allclose(batched, step, rtol=1e-3, atol=1e-3)
    assert batched.argmax() == step.argmax()


def test_reset_cache_forgets_the_conversation(model):
    with native.NativeHKEngine(model) as eng:
        first = eng.forward([3, 17, 5])
        eng.reset_cache()
        again = eng.forward([3, 17, 5])
    np.testing.assert_array_equal(first, again)


def test_failures_are_reported_not_guessed(model, tmp_path):
    with native.NativeHKEngine(model) as eng:
        with pytest.raises(ValueError, match="outside the vocabulary"):
            eng.forward([eng.vocab_size])
        with pytest.raises(ValueError, match="at least one token"):
            eng.forward([])
        with pytest.raises(RuntimeError, match="context window is full"):
            eng.forward([1], pos=eng.context_size)
    junk = tmp_path / "junk.hk"
    junk.write_bytes(b"not a model" * 100)
    with pytest.raises(RuntimeError, match="Failed to load native engine"):
        native.NativeHKEngine(str(junk))
    with pytest.raises(RuntimeError, match="Failed to load native engine"):
        native.NativeHKEngine(str(tmp_path / "missing.hk"))


def test_closed_engine_refuses_work(model):
    eng = native.NativeHKEngine(model)
    eng.close()
    eng.close()  # closing twice is harmless
    with pytest.raises(RuntimeError, match="not loaded"):
        eng.forward([1])
