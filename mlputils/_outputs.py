"""Paths for generated workflow files."""
from contextlib import contextmanager
from pathlib import Path
import os


def output_path(folder, filename):
    directory = Path(folder).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


@contextmanager
def in_output_folder(folder):
    """Contain third-party writers that only accept filenames."""
    directory = Path(folder).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    try:
        os.chdir(directory)
        yield
    finally:
        os.chdir(previous)
