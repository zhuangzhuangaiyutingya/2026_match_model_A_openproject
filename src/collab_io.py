# -*- coding: utf-8 -*-
"""Small dependency-free helpers for safe collaboration-file writes."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


class FileLock:
    """A small cross-platform lock-file guard for one shared result file."""

    def __init__(self, target: Path, timeout: float = 300.0, poll: float = 0.2):
        self.target = Path(target)
        self.lock_path = Path(str(self.target) + ".lock")
        self.timeout = timeout
        self.poll = poll
        self._held = False

    def acquire(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout
        payload = f"pid={os.getpid()}\nstarted={time.time():.3f}\n"
        while True:
            try:
                fd = os.open(
                    str(self.lock_path),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    stream.write(payload)
                self._held = True
                return
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out waiting for {self.lock_path}; "
                        "if no other run is active, remove the stale .lock file"
                    )
                time.sleep(self.poll)

    def release(self) -> None:
        if self._held:
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass
            self._held = False

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def atomic_write_text(path: Path, text: str) -> None:
    """Write text through a sibling temporary file and replace atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(str(tmp), str(path))
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, value, *, indent: int = 1) -> None:
    atomic_write_text(
        Path(path),
        json.dumps(value, ensure_ascii=False, indent=indent),
    )
