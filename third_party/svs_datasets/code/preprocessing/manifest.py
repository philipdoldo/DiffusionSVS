"""Canonical manifest parsing and JSONL serialization helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from core.types import CanonicalExample


def write_jsonl_manifest(
    examples: Iterable[CanonicalExample], path: str | Path
) -> None:
    """Write canonical examples as JSONL."""
    destination = Path(path)
    with destination.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(
                json.dumps(example.to_dict(), ensure_ascii=True, sort_keys=True)
            )
            handle.write("\n")


def load_jsonl_manifest(path: str | Path) -> tuple[CanonicalExample, ...]:
    """Load canonical examples from a JSONL manifest."""
    source = Path(path)
    examples: list[CanonicalExample] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"manifest line {line_number} must decode to an object, got "
                    f"{type(payload).__name__}"
                )
            examples.append(CanonicalExample.from_dict(payload))
    return tuple(examples)


__all__ = [
    "load_jsonl_manifest",
    "write_jsonl_manifest",
]
