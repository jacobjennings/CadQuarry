"""
Thin progress-bar helper around tqdm.

Keeps tqdm an optional convenience: if it isn't installed, ``progress_bar``
returns a no-op stand-in with the same minimal surface (update/close/write/
set_postfix and context-manager support) so call sites never need to branch.
"""
from __future__ import annotations


class _NullBar:
    def update(self, n: int = 1) -> None:  # noqa: D401
        pass

    def set_postfix(self, *args, **kwargs) -> None:
        pass

    def write(self, msg: str = "", *args, **kwargs) -> None:
        print(msg)

    def close(self) -> None:
        pass

    def __enter__(self) -> "_NullBar":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def progress_bar(total: int | None = None, desc: str = "", unit: str = "it", leave: bool = True):
    """Return a tqdm bar, or a no-op stand-in if tqdm isn't installed."""
    try:
        from tqdm import tqdm
    except ImportError:
        return _NullBar()
    # disable=None makes tqdm auto-silence when stderr isn't a TTY, so redirected
    # build logs stay clean while interactive runs still get a live bar.
    return tqdm(
        total=total, desc=desc, unit=unit, leave=leave,
        dynamic_ncols=True, disable=None,
    )
