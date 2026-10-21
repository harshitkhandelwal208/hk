#!/usr/bin/env python3
"""Install a built hknt wheel into a clean virtual environment and check that it works with nothing else installed.

    python tests/wheel_smoke.py dist/hknt-2.0.0-py3-none-manylinux2014_x86_64.whl

What it proves, on the machine it runs on:
  * `pip install` of the wheel pulls in no other package (the venv holds pip and hknt only);
  * the `hk` command exists on PATH and runs the bundled native executable, with no download step;
  * `import hk` finds the bundled native library and imports neither torch nor numpy;
  * a feature that does need numpy or torch fails with advice to install the extra, not with a bare traceback.
It cannot run a wheel built for another platform; the release workflow runs it on a runner of each platform.
"""
import glob
import json
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

PROBE = r"""
import json, sys
import hk
out = {"version": hk.__version__, "native": hk.native.is_native_available(),
       "torch_imported": "torch" in sys.modules, "numpy_imported": "numpy" in sys.modules}
try:
    hk.HKForCausalLM
    out["torch_feature"] = "no error"
except ImportError as e:
    out["torch_feature"] = str(e)
out["torch_imported_after"] = "torch" in sys.modules
print(json.dumps(out))
"""


def run(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)


def main() -> int:
    # PowerShell does not expand `dist/*.whl`, so the pattern is expanded here.
    matches = sorted(glob.glob(sys.argv[1]))
    if len(matches) != 1:
        return fail(f"expected exactly one wheel matching {sys.argv[1]!r}, found {matches}")
    wheel = Path(matches[0]).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        env_dir = Path(tmp) / "venv"
        builder = venv.EnvBuilder(with_pip=True)
        builder.create(env_dir)
        ctx = builder.ensure_directories(env_dir)
        py = Path(ctx.env_exe)
        bindir = Path(ctx.bin_path)

        r = run([py, "-m", "pip", "install", "--no-index", "--disable-pip-version-check", wheel])
        if r.returncode:
            print(r.stdout, r.stderr)
            return fail("pip install of the wheel failed (with --no-index: it must need no other package)")
        listing = run([py, "-m", "pip", "list", "--format=json"]).stdout
        installed = sorted(p["name"].lower() for p in json.loads(listing))
        extra = [n for n in installed if n not in ("pip", "setuptools", "wheel", "hknt")]
        if extra:
            return fail(f"installing hknt also installed {extra}")

        exe = bindir / ("hk.exe" if sys.platform == "win32" else "hk")
        r = run([exe, "--help"])
        if r.returncode != 0 or "HK CLI v" not in r.stdout + r.stderr:
            print(r.stdout, r.stderr)
            return fail("the installed `hk` command did not run the native executable")
        print("hk command:", (r.stdout + r.stderr).splitlines()[0])

        # Not from the checkout: that would import python/hk instead of the installed package.
        r = run([py, "-I", "-c", PROBE], cwd=tmp)
        if r.returncode:
            print(r.stdout, r.stderr)
            return fail("import hk failed in the clean environment")
        info = json.loads(r.stdout.strip().splitlines()[-1])
        print("import hk:", info)
        if not info["native"]:
            return fail("hk could not find its bundled native library")
        if info["torch_imported"] or info["numpy_imported"] or info["torch_imported_after"]:
            return fail("hk imported torch or numpy")
        if "hknt[" not in info["torch_feature"]:
            return fail("a torch-only feature did not tell the user which extra to install")
    print("wheel smoke test: ok")
    return 0


def fail(msg: str) -> int:
    print("wheel smoke test FAILED:", msg)
    return 1


if __name__ == "__main__":
    sys.exit(main())
