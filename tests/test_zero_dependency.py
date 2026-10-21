"""`import hk` and the native layer must work with neither NumPy nor PyTorch installed (plan section 1.6).

Runs a child interpreter that refuses to import them, so the test is meaningful on a developer machine that has both.
"""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

CHILD = textwrap.dedent(
    """
    import json, sys
    BLOCKED = ("torch", "numpy", "transformers", "safetensors", "gguf", "jax", "flax")

    class Block:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ModuleNotFoundError(f"No module named '{name}'", name=name)

    sys.meta_path.insert(0, Block())
    sys.path.insert(0, sys.argv[1])
    import hk, hk.cli
    out = {"version": hk.__version__, "native": hk.native.is_native_available(),
           "config": hk.HKConfig.__name__, "tokenizer": hk.NativeHKTokenizer.__name__,
           "engine": hk.NativeHKEngine.__name__}
    for feature in ("HKForCausalLM", "save_file", "HKTrainer"):
        try:
            getattr(hk, feature)
            out[feature] = "loaded"
        except ImportError as e:
            out[feature] = str(e)
    try:
        hk.native.native_dot([1.0], [1.0])
        out["numpy_call"] = "no error"
    except ImportError as e:
        out["numpy_call"] = str(e)
    out["heavy_imported"] = sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED)
    print(json.dumps(out))
    """
)


def run_child():
    r = subprocess.run([sys.executable, "-I", "-c", CHILD, str(ROOT / "python")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_import_hk_needs_nothing():
    out = run_child()
    assert out["version"]
    assert out["config"] == "HKConfig" and out["tokenizer"] == "NativeHKTokenizer" and out["engine"] == "NativeHKEngine"
    assert out["heavy_imported"] == []


def test_optional_features_explain_what_to_install():
    out = run_child()
    for feature in ("HKForCausalLM", "save_file", "HKTrainer"):
        assert "pip install 'hknt[" in out[feature], (feature, out[feature])
    assert "pip install 'hknt[numpy]'" in out["numpy_call"]


def test_pyproject_declares_no_required_dependencies():
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["dependencies"] == []
    assert {"torch", "numpy"} <= set(project["optional-dependencies"])
