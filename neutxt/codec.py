"""NEUTXT container format: read/write .neutxt files with AV2 framing."""
from __future__ import annotations

import io
import json
import struct
import subprocess
import wave
import zlib
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Tuple

try:
    import zstandard as zstd
    _HAVE_ZSTD = True
except ImportError:
    zstd = None
    _HAVE_ZSTD = False


# Record: type (1B) | ts_ms (int32 LE) | payload_len (uint32 LE) | payload
_REC_HDR = struct.Struct("<c i I")

# Video payload header: gh (u16) | gw (u16) | packed_dtype (u8) | code_bits (u8)
_V_HDR = struct.Struct("<H H B B")

# AV2 packet header: flags (u8) | codec_id (u8) | raw_len (u16) | comp_len (u32) | crc32 (u32)
_V2_PKT_HDR = struct.Struct("<B B H I I")


@dataclass
class VideoRecord:
    ts_ms: int
    gh: int
    gw: int
    packed_dtype: int
    code_bits: int
    payload: bytes


@dataclass
class AudioRecord:
    ts_ms: int
    payload: bytes


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------

def compress_bytes(data: bytes, codec: str = "zlib", level: int = 6) -> bytes:
    c = codec.lower()
    if c == "zstd":
        if not _HAVE_ZSTD:
            raise RuntimeError("zstandard not installed — pip install zstandard")
        return zstd.ZstdCompressor(level=level).compress(data)
    if c == "zlib":
        return zlib.compress(data, level=level)
    raise ValueError(f"Unsupported codec={codec}")


def decompress_bytes(data: bytes, codec: str = "zlib") -> bytes:
    c = codec.lower()
    if c == "zstd":
        if not _HAVE_ZSTD:
            raise RuntimeError("zstandard not installed — pip install zstandard")
        try:
            return zstd.ZstdDecompressor().decompress(data)
        except Exception:
            return zlib.decompress(data)
    if c == "zlib":
        try:
            return zlib.decompress(data)
        except zlib.error:
            if not _HAVE_ZSTD:
                raise
            return zstd.ZstdDecompressor().decompress(data)
    if c == "auto":
        if _HAVE_ZSTD:
            try:
                return zstd.ZstdDecompressor().decompress(data)
            except Exception:
                pass
        return zlib.decompress(data)
    raise ValueError(f"Unsupported codec={codec}")


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class NeutxtWriter:
    def __init__(self, path: str, magic: str, meta: Dict[str, Any]):
        self.path = path
        self.f = open(path, "wb")
        self._write_header(magic, meta)

    def _write_header(self, magic: str, meta: Dict[str, Any]):
        magic_line = (magic.rstrip("\n") + "\n").encode("utf-8")
        meta_line = (json.dumps(meta, ensure_ascii=False) + "\n").encode("utf-8")
        self.f.write(magic_line)
        self.f.write(meta_line)
        self.f.flush()

    def write_audio(self, ts_ms: int, compressed_pcm: bytes):
        self.f.write(_REC_HDR.pack(b"A", int(ts_ms), len(compressed_pcm)))
        self.f.write(compressed_pcm)

    def write_video(self, ts_ms: int, gh: int, gw: int,
                    packed_dtype: int, code_bits: int, packed_codes: bytes):
        payload = _V_HDR.pack(gh, gw, packed_dtype, code_bits) + packed_codes
        self.f.write(_REC_HDR.pack(b"V", int(ts_ms), len(payload)))
        self.f.write(payload)

    def close(self):
        try:
            self.f.flush()
        finally:
            self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def read_header(path: str) -> Tuple[str, Dict[str, Any]]:
    with open(path, "rb") as f:
        magic = f.readline().decode("utf-8", errors="replace").rstrip("\n")
        meta = json.loads(f.readline().decode("utf-8", errors="replace"))
    return magic, meta


def iter_records(path: str) -> Iterator[Tuple[bytes, int, bytes]]:
    with open(path, "rb") as f:
        f.readline()  # magic
        f.readline()  # meta
        while True:
            hdr = f.read(_REC_HDR.size)
            if not hdr or len(hdr) < _REC_HDR.size:
                break
            rtype, ts_ms, n = _REC_HDR.unpack(hdr)
            payload = f.read(n)
            if len(payload) != n:
                break
            yield rtype, ts_ms, payload


def iter_video_records(path: str) -> Iterator[VideoRecord]:
    for rtype, ts_ms, payload in iter_records(path):
        if rtype != b"V":
            continue
        if len(payload) < _V_HDR.size:
            continue
        gh, gw, packed_dtype, code_bits = _V_HDR.unpack(payload[:_V_HDR.size])
        yield VideoRecord(ts_ms=ts_ms, gh=gh, gw=gw,
                          packed_dtype=packed_dtype, code_bits=code_bits,
                          payload=payload[_V_HDR.size:])


def iter_audio_records(path: str) -> Iterator[AudioRecord]:
    for rtype, ts_ms, payload in iter_records(path):
        if rtype != b"A":
            continue
        yield AudioRecord(ts_ms=ts_ms, payload=payload)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def encode_audio_to_records(
    media_path: str,
    sr: int = 24000,
    chunk_seconds: float = 10.0,
    channels: int = 1,
    codec: str = "zlib",
    level: int = 6,
) -> Iterator[Tuple[int, bytes]]:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", media_path, "-vn",
        "-ac", str(channels), "-ar", str(sr),
        "-f", "s16le", "pipe:1",
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert p.stdout is not None

    chunk_bytes = int(round(chunk_seconds * sr * channels * 2))
    ts_ms = 0
    while True:
        buf = p.stdout.read(chunk_bytes)
        if not buf:
            break
        yield ts_ms, compress_bytes(buf, codec=codec, level=level)
        ts_ms += int(round(chunk_seconds * 1000))
    p.stdout.close()
    p.wait()


def decode_audio_records_to_wav_bytes(
    path: str, sr: int, channels: int = 1, codec: str = "zlib",
) -> bytes:
    pcm_parts = []
    for rec in iter_audio_records(path):
        pcm_parts.append(decompress_bytes(rec.payload, codec=codec))
    pcm = b"".join(pcm_parts)
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)
    return bio.getvalue()
