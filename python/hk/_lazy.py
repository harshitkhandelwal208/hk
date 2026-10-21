"""Optional third-party modules, imported on first use.

hk installs with no dependencies. NumPy and PyTorch are optional: the command line, the native engine, tokenizers,
the downloader and the converters never touch them. A module that only needs them for some functions binds
`np = lazy("numpy")` instead of importing it, so `import hk` stays cheap and the error for a missing package appears
where the package is first used, saying how to install it.
"""

import importlib
from types import ModuleType
from typing import Any

_EXTRA = {"numpy": "numpy", "torch": "torch"}


class MissingOptionalDependency(ImportError):
    """A feature needs a package that is not installed."""


class _Lazy:
    __slots__ = ("_name", "_mod")

    def __init__(self, name: str) -> None:
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_mod", None)

    def _load(self) -> ModuleType:
        mod = object.__getattribute__(self, "_mod")
        if mod is None:
            name = object.__getattribute__(self, "_name")
            try:
                mod = importlib.import_module(name)
            except ImportError as e:
                extra = _EXTRA.get(name.split(".")[0], name)
                raise MissingOptionalDependency(
                    f"this hk feature needs `{name}`, which is not installed. Install it with: pip install 'hknt[{extra}]'"
                ) from e
            object.__setattr__(self, "_mod", mod)
        return mod

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._load(), attr)

    def __repr__(self) -> str:
        return f"<lazy module {object.__getattribute__(self, '_name')!r}>"


def lazy(name: str) -> Any:
    """A stand-in for `import name` that imports the module on first attribute access."""
    return _Lazy(name)
