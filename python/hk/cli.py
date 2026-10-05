#!/usr/bin/env python3
"""The `hk` command of the Python package.

The command line is implemented once, in the native `hk` executable (Zig). This entry point only
finds that executable and runs it, so `pip install hknt` and a source build behave the same and
nothing is implemented twice. The one command that is not native is `gui`, which opens the Tk
model editor and so has to run in Python.
"""

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional


def _candidate_names() -> list:
    """File names the native executable can have: a plain build, or a release asset such as
    `hk-x86_64-linux` (which is how the release workflow names the binaries it bundles)."""
    machine = platform.machine().lower()
    arch = "aarch64" if ("arm" in machine or "aarch64" in machine) else "x86_64"
    system = platform.system().lower()
    os_name = "windows" if system == "windows" else ("macos" if system == "darwin" else "linux")
    suffix = ".exe" if os_name == "windows" else ""
    return [f"hk-{arch}-{os_name}{suffix}", "hk.exe", "hk"]


def _find_native_cli() -> Optional[str]:
    """Locates the compiled native `hk` executable."""
    here = Path(__file__).resolve().parent
    search_dirs = [
        here,
        here / "bin",
        here.parent.parent / "zig-out" / "bin",
        Path(os.getcwd()) / "zig-out" / "bin",
    ]
    env_dir = os.environ.get("HK_BIN_DIR")
    if env_dir:
        search_dirs.insert(0, Path(env_dir))
    for d in search_dirs:
        for name in _candidate_names():
            candidate = d / name
            if not candidate.is_file():
                continue
            if not os.access(candidate, os.X_OK):
                # Wheels do not keep the executable bit, so restore it on the bundled binary.
                try:
                    os.chmod(candidate, candidate.stat().st_mode | 0o111)
                except OSError:
                    continue
            if os.access(candidate, os.X_OK):
                return str(candidate)
    # A different program called `hk` earlier on PATH is not ours, so PATH is not searched.
    return None


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] == "gui":
        from hk.gui import launch_gui

        launch_gui(argv[1] if len(argv) > 1 else None)
        return

    native = _find_native_cli()
    if native is None:
        sys.stderr.write(
            "hk: the native `hk` executable was not found next to this package.\n"
            "    Download it from https://github.com/harshitkhandelwal208/hk/releases,\n"
            "    or build it with `zig build -Doptimize=ReleaseFast` and set HK_BIN_DIR=zig-out/bin.\n"
        )
        sys.exit(127)
    sys.exit(subprocess.call([native] + argv))


if __name__ == "__main__":
    main()
