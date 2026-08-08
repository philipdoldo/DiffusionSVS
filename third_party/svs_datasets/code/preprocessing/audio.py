"""Lightweight audio container helpers for best-effort metadata extraction during preprocessing."""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

AUDIO_FILE_SUFFIXES = (".flac", ".wav", ".wave")
_AUDIO_CONTAINER_DIR_CANDIDATES = {
    "wav": ("wav", "flac", "wavs"),
    "wavs": ("wavs", "flac", "wav"),
    "flac": ("flac", "wavs", "wav"),
}


@dataclass(frozen=True, slots=True)
class AudioMetadata:
    """Best-effort audio metadata used during preprocessing."""

    path: str
    sample_rate: int | None
    num_samples: int | None
    num_channels: int | None
    container: str


@dataclass(frozen=True, slots=True)
class _WavContainer:
    path: str
    sample_rate: int
    num_samples: int
    num_channels: int
    block_align: int
    bits_per_sample: int
    audio_format_tag: int
    data_byte_count: int


@dataclass(frozen=True, slots=True)
class _FlacContainer:
    path: str
    sample_rate: int
    num_samples: int
    num_channels: int
    bits_per_sample: int


def resolve_audio_path(path: str | Path) -> Path:
    """Resolve a best-effort on-disk audio path across WAV/FLAC layout variants.

    This handles the repository's current corpus mirrors, where:
    - some datasets renamed `wav/` or `wavs/` directories to `flac/`
    - some datasets converted files in place from `.wav` to `.flac`
    - some corpora still legitimately remain WAV-backed
    """
    path_obj = Path(path)
    if path_obj.exists():
        return path_obj

    suffix = path_obj.suffix.lower()
    if suffix and suffix not in AUDIO_FILE_SUFFIXES:
        return path_obj

    if suffix:
        stem = path_obj.stem
        preferred_suffixes = (suffix,) + tuple(
            candidate for candidate in AUDIO_FILE_SUFFIXES if candidate != suffix
        )
    else:
        stem = path_obj.name
        preferred_suffixes = AUDIO_FILE_SUFFIXES

    candidate_directories = [path_obj.parent]
    parent_name = path_obj.parent.name.lower()
    if parent_name in _AUDIO_CONTAINER_DIR_CANDIDATES:
        parent_root = path_obj.parent.parent
        for candidate_name in _AUDIO_CONTAINER_DIR_CANDIDATES[parent_name]:
            candidate_directory = parent_root / candidate_name
            if candidate_directory not in candidate_directories:
                candidate_directories.append(candidate_directory)

    for directory in candidate_directories:
        for candidate_suffix in preferred_suffixes:
            candidate_path = directory / f"{stem}{candidate_suffix}"
            if candidate_path.exists():
                return candidate_path

    return path_obj


def _read_wav_container(audio_path: Path) -> _WavContainer:
    with audio_path.open("rb") as handle:
        riff_header = handle.read(12)
        if (
            len(riff_header) != 12
            or riff_header[:4] != b"RIFF"
            or riff_header[8:12] != b"WAVE"
        ):
            raise ValueError(f"{audio_path} does not look like a RIFF/WAVE file")

        sample_rate: int | None = None
        num_channels: int | None = None
        block_align: int | None = None
        bits_per_sample: int | None = None
        audio_format_tag: int | None = None
        data_byte_count: int | None = None

        while True:
            chunk_header = handle.read(8)
            if not chunk_header:
                break
            if len(chunk_header) != 8:
                raise ValueError(f"{audio_path} ended inside a WAV chunk header")

            chunk_id = chunk_header[:4]
            chunk_size = int.from_bytes(
                chunk_header[4:8], byteorder="little", signed=False
            )
            if chunk_id == b"fmt ":
                chunk_payload = handle.read(chunk_size)
                if len(chunk_payload) != chunk_size:
                    raise ValueError(f"{audio_path} ended inside WAV chunk {chunk_id!r}")
                if chunk_size < 16:
                    raise ValueError(f"{audio_path} has a truncated fmt chunk")
                audio_format_tag = int.from_bytes(
                    chunk_payload[0:2], byteorder="little", signed=False
                )
                num_channels = int.from_bytes(
                    chunk_payload[2:4], byteorder="little", signed=False
                )
                sample_rate = int.from_bytes(
                    chunk_payload[4:8], byteorder="little", signed=False
                )
                block_align = int.from_bytes(
                    chunk_payload[12:14], byteorder="little", signed=False
                )
                bits_per_sample = int.from_bytes(
                    chunk_payload[14:16],
                    byteorder="little",
                    signed=False,
                )

                if audio_format_tag == 0xFFFE:
                    if chunk_size < 40:
                        raise ValueError(
                            f"{audio_path} has a truncated extensible fmt chunk"
                        )
                    audio_format_tag = int.from_bytes(
                        chunk_payload[24:26],
                        byteorder="little",
                        signed=False,
                    )
            elif chunk_id == b"data":
                data_byte_count = chunk_size
                handle.seek(chunk_size, 1)
            else:
                handle.seek(chunk_size, 1)

            if chunk_size % 2 == 1:
                handle.seek(1, 1)

        if sample_rate is None or num_channels is None or block_align is None:
            raise ValueError(f"{audio_path} is missing required WAV fmt metadata")
        if bits_per_sample is None or audio_format_tag is None:
            raise ValueError(f"{audio_path} is missing bits_per_sample or format tag")
        if data_byte_count is None:
            raise ValueError(f"{audio_path} is missing a WAV data chunk")
        if block_align <= 0:
            raise ValueError(f"{audio_path} has invalid WAV block_align {block_align}")

        num_samples = data_byte_count // block_align
        return _WavContainer(
            path=str(audio_path),
            sample_rate=sample_rate,
            num_samples=num_samples,
            num_channels=num_channels,
            block_align=block_align,
            bits_per_sample=bits_per_sample,
            audio_format_tag=audio_format_tag,
            data_byte_count=data_byte_count,
        )


def _read_wav_riff(audio_path: Path) -> AudioMetadata:
    container = _read_wav_container(audio_path)
    return AudioMetadata(
        path=container.path,
        sample_rate=container.sample_rate,
        num_samples=container.num_samples,
        num_channels=container.num_channels,
        container="wav",
    )


def _read_flac_container(audio_path: Path) -> _FlacContainer:
    with audio_path.open("rb") as handle:
        magic = handle.read(4)
        if magic != b"fLaC":
            raise ValueError(f"{audio_path} does not start with the FLAC magic header")

        while True:
            header = handle.read(4)
            if len(header) != 4:
                raise ValueError(
                    f"{audio_path} ended before a FLAC STREAMINFO block was found"
                )

            is_last_block = bool(header[0] & 0x80)
            block_type = header[0] & 0x7F
            block_length = int.from_bytes(header[1:4], byteorder="big", signed=False)
            block_payload = handle.read(block_length)
            if len(block_payload) != block_length:
                raise ValueError(f"{audio_path} ended inside a FLAC metadata block")

            if block_type == 0:
                if block_length != 34:
                    raise ValueError(
                        f"{audio_path} has an unexpected FLAC STREAMINFO length {block_length}"
                    )
                packed = int.from_bytes(
                    block_payload[10:18], byteorder="big", signed=False
                )
                sample_rate = packed >> 44
                num_channels = ((packed >> 41) & 0x7) + 1
                bits_per_sample = ((packed >> 36) & 0x1F) + 1
                total_samples = packed & ((1 << 36) - 1)
                return _FlacContainer(
                    path=str(audio_path),
                    sample_rate=sample_rate,
                    num_samples=total_samples,
                    num_channels=num_channels,
                    bits_per_sample=bits_per_sample,
                )

            if is_last_block:
                break

    raise ValueError(f"{audio_path} did not contain a FLAC STREAMINFO block")


def _read_flac_streaminfo(audio_path: Path) -> AudioMetadata:
    container = _read_flac_container(audio_path)
    return AudioMetadata(
        path=container.path,
        sample_rate=container.sample_rate,
        num_samples=container.num_samples,
        num_channels=container.num_channels,
        container="flac",
    )


def read_audio_metadata(path: str | Path, *, resolve: bool = True) -> AudioMetadata:
    """Read lightweight container metadata.

    Currently supported:
    - WAV via the Python standard library
    - FLAC via direct STREAMINFO parsing

    Unsupported containers return `None` for fields that require container-specific
    decoding.
    """
    audio_path = resolve_audio_path(path) if resolve else Path(path)
    suffix = audio_path.suffix.lower()

    if suffix in {".wav", ".wave"}:
        try:
            with wave.open(str(audio_path), "rb") as handle:
                return AudioMetadata(
                    path=str(audio_path),
                    sample_rate=handle.getframerate(),
                    num_samples=handle.getnframes(),
                    num_channels=handle.getnchannels(),
                    container="wav",
                )
        except wave.Error:
            return _read_wav_riff(audio_path)
    if suffix == ".flac":
        return _read_flac_streaminfo(audio_path)

    return AudioMetadata(
        path=str(audio_path),
        sample_rate=None,
        num_samples=None,
        num_channels=None,
        container=suffix.lstrip(".") or "unknown",
    )

__all__ = [
    "AUDIO_FILE_SUFFIXES",
    "AudioMetadata",
    "read_audio_metadata",
    "resolve_audio_path",
]
