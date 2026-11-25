"""File system steps shared by the stores that are directories of arrays."""

from __future__ import annotations

import errno
import os
import shutil
import uuid
from pathlib import Path


def scratch_beside(path: Path, what: str) -> Path:
    """A directory name beside ``path`` that no other writer uses."""
    return path.with_name(f"{path.name}.{what}-{uuid.uuid4().hex}")


def replace_directory(new: Path, path: Path) -> None:
    """Put the directory ``new`` at ``path``, removing what is there.

    What was at ``path`` is moved aside before ``new`` is moved in, so
    ``path`` never holds a mix of the two; for the instant between the two
    moves it does not exist. Several processes may replace one path at
    once: each of them succeeds in turn and the last one stays.
    """
    while True:
        try:
            os.rename(new, path)  # succeeds onto nothing or an empty directory
            return
        except OSError as error:
            if error.errno not in (errno.ENOTEMPTY, errno.EEXIST, errno.ENOTDIR):
                raise
        old = scratch_beside(path, "old")
        try:
            os.rename(path, old)
        except FileNotFoundError:
            continue  # another process moved it aside first
        if old.is_dir() and not old.is_symlink():
            shutil.rmtree(old, ignore_errors=True)
        else:
            old.unlink()
