from __future__ import annotations

try:
    from enum import StrEnum as StrEnum
except ImportError:  # pragma: no cover
    from enum import Enum

    class StrEnum(str, Enum):
        """Backport-compatible stand-in for enum.StrEnum."""

