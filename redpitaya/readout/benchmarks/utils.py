"""Utility functions for benchmarks.
"""

from contextlib import contextmanager
from io import TextIOWrapper
import time
from typing import Optional


@contextmanager
def timer(name: Optional[str]=None, fid: Optional[TextIOWrapper]=None):
    """Context manager for timing code.
    Message is comma-separated: name, duration.

    Args:
        name: The name of code timed. Defaults to None.
        fid: The file handler. Set to None to print to stdout. Defaults to None.
    """
    start = time.perf_counter()
    yield None
    end = time.perf_counter()
    duration = end - start
    message = f"{name}, {duration:.3g}"
    if fid is None:
        print(message)
    else:
        fid.write(message + "\n")
