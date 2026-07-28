"""Atomic small-file writes for the user-editable YAML state files.

Every `save()` in this app was a plain in-place `write_text`; a crash or a
Windows-Update power cut mid-write leaves a truncated file, and the tolerant
loaders then read it as EMPTY — silent total loss of the load-time history,
cookbook, defaults or registry (review 2026-07-29; invariant 17 documents that
surprise reboots are a live path here). Write-to-temp + `os.replace` is atomic
on NTFS: the reader sees the old file or the new one, never a torn one.
"""
from __future__ import annotations

import os
from pathlib import Path


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
