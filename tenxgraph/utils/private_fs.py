"""Create files and directories that only the owner can read.

The modes are passed at creation, so there is no window where a file is world-readable. The
process umask can only make them stricter. Existing files and directories keep their mode.
On Windows the mode arguments have little effect and nothing is enforced.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path


DIR_MODE = 0o700
FILE_MODE = 0o600


def mkdir_private(path: Path) -> None:
    """Create ``path`` and any missing parents with mode 0o700 (``Path.mkdir`` only applies
    the mode to the last level)."""
    missing: list[Path] = []
    current = path
    while not current.exists() and current != current.parent:
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir(mode=DIR_MODE, exist_ok=True)


def write_private(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path``, creating the file with mode 0o600 if it does not exist."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, FILE_MODE)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)


def write_private_atomic(path: Path, data: bytes) -> None:
    """Replace ``path`` with ``data`` atomically, keeping mode 0o600."""
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:  # mkstemp creates the file with mode 0o600
            fh.write(data)
        Path(tmp_name).replace(path)
    except BaseException:
        with contextlib.suppress(OSError):
            Path(tmp_name).unlink()
        raise


def touch_private(path: Path) -> None:
    """Create an empty file with mode 0o600 if it does not exist."""
    with contextlib.suppress(FileExistsError):
        os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, FILE_MODE))
