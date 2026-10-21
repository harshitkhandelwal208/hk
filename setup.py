"""Packaging hook: a wheel that carries the native library and the `hk` executable is specific to one platform.

Without this, setuptools tags a wheel that contains only Python files `py3-none-any`, and pip installs the same wheel
everywhere. hk's wheels hold compiled code, so they are tagged `py3-none-<platform>` (the code is a plain shared library
loaded with ctypes, so it does not depend on the Python version). When `python/hk/` holds no native files (a source
distribution, or an editable install from a checkout) the result is still a normal pure wheel.
"""

from pathlib import Path

from setuptools import Distribution, setup

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:  # setuptools older than 70.1 keeps it in the separate `wheel` package
    from wheel.bdist_wheel import bdist_wheel

PACKAGE = Path(__file__).parent / "python" / "hk"


def _has_native_files() -> bool:
    return any(PACKAGE.glob("libhk*")) or any(PACKAGE.glob("hk.dll")) or (PACKAGE / "bin").is_dir()


class NativeDistribution(Distribution):
    """Tells setuptools the package holds compiled code, so files go in the platform-specific part of the wheel."""

    def has_ext_modules(self):
        return _has_native_files()


class PlatformWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        if _has_native_files():
            self.root_is_pure = False

    def get_tag(self):
        python, abi, plat = super().get_tag()
        if _has_native_files():
            return "py3", "none", plat
        return python, abi, plat


setup(cmdclass={"bdist_wheel": PlatformWheel}, distclass=NativeDistribution)
