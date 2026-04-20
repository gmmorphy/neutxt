"""NEUTXT text format: encode/decode neural media as pure UTF-8 strings.

This is the patent-aligned format — every byte is printable UTF-8 text,
suitable for pasting into LLM prompts, git repos, chat messages, etc.

Supports three modes: video-only, audio-only, audiovisual.

Format (audiovisual example):
    --- NEUTXT v2 ---
    mode: av
    video_model: MAGVIT2_256_L
    audio_model: ENCODEC_24K
    fps: 12.0
    resolution: 256x256
    video_code_bits: 18
    tokens_per_frame: 256
    keyint: 12
    audio_sr: 24000
    audio_bandwidth: 6.0
    audio_quantizers: 8
    audio_code_bits: 10
    audio_chunk_seconds: 1.0
    compression: zstd
    video_frames: 36
    audio_chunks: 3
    checksum: a1b2c3d4
    ---
    K:<base85 payload>    ← video keyframe
    D:<base85 payload>    ← video delta
    A:<base85 payload>    ← audio chunk
    ...

Frame prefixes: K = video keyframe, D = video delta (XOR), A = audio chunk.
Payload = base85(compress(bitpack(codes)))
"""
from __future__ import annotations

import base64
import zlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


try:
    import zstandard as zstd
    _HAVE_ZSTD = True
except ImportError:
    zstd = None
    _HAVE_ZSTD = False

HEADER_START = "--- NEUTXT v2 ---"
HEADER_END = "---"


@dataclass
class TextNeutxtHeader:
    # mode: "v" (video-only), "a" (audio-only), "av" (audiovisual)
    mode: str = "v"

    # Video fields (used when mode in {"v", "av"})
    video_model: str = "MAGVIT2_256_L"
    fps: float = 12.0
    resolution: str = "256x256"
    video_code_bits: int = 18
    tokens_per_frame: int = 256
    keyint: int = 12
    video_frames: int = 0

    # Audio fields (used when mode in {"a", "av"})
    audio_model: str = "ENCODEC_24K"
    audio_sr: int = 24000
    audio_bandwidth: float = 6.0
    audio_quantizers: int = 8
    audio_code_bits: int = 10
    audio_chunk_seconds: float = 1.0
    audio_chunks: int = 0

    # Common
    compression: str = "zstd"
    checksum: str = ""

    # Legacy field kept for back-compat with v-only decoder
    frames: int = 0
    code_bits: int = 18
    model: str = "MAGVIT2_256_L"


@dataclass
class TextNeutxtFile:
    header: TextNeutxtHeader
    frame_lines: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Bit packing
# ---------------------------------------------------------------------------

def bitpack18(codes: np.ndarray) -> bytes:
    """Pack uint32 codes (18-bit values) into a dense byte stream."""
    out = bytearray()
    acc = 0
    bits = 0
    mask = (1 << 18) - 1
    for v in codes.ravel():
        acc |= (int(v) & mask) << bits
        bits += 18
        while bits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            bits -= 8
    if bits:
        out.append(acc & 0xFF)
    return bytes(out)


def bitunpack18(packed: bytes, n_tokens: int) -> np.ndarray:
    """Unpack dense byte stream back to uint32 codes."""
    out = np.empty((n_tokens,), dtype=np.uint32)
    acc = 0
    bits = 0
    idx = 0
    for b in packed:
        acc |= int(b) << bits
        bits += 8
        while bits >= 18 and idx < n_tokens:
            out[idx] = acc & ((1 << 18) - 1)
            acc >>= 18
            bits -= 18
            idx += 1
        if idx >= n_tokens:
            break
    if idx != n_tokens:
        raise RuntimeError(f"bitunpack18: expected {n_tokens}, got {idx}")
    return out


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------

def _compress(data: bytes, codec: str = "zstd", level: int = 9) -> bytes:
    if codec == "zstd" and _HAVE_ZSTD:
        return zstd.ZstdCompressor(level=level).compress(data)
    return zlib.compress(data, level=min(level, 9))


def _decompress(data: bytes, codec: str = "zstd") -> bytes:
    if codec == "zstd" and _HAVE_ZSTD:
        try:
            return zstd.ZstdDecompressor().decompress(data)
        except Exception:
            return zlib.decompress(data)
    try:
        return zlib.decompress(data)
    except zlib.error:
        if _HAVE_ZSTD:
            return zstd.ZstdDecompressor().decompress(data)
        raise


# ---------------------------------------------------------------------------
# XOR delta
# ---------------------------------------------------------------------------

def _xor_bytes(a: bytes, b: bytes) -> bytes:
    ba = np.frombuffer(a, dtype=np.uint8)
    bb = np.frombuffer(b, dtype=np.uint8)
    return bytes(np.bitwise_xor(ba, bb))


# ---------------------------------------------------------------------------
# Encode: video-only (backward-compatible)
# ---------------------------------------------------------------------------

def encode_frames_to_text(
    all_codes: list[np.ndarray],
    fps: float = 12.0,
    keyint: int = 12,
    model: str = "MAGVIT2_256_L",
    compression: str = "zstd",
    level: int = 9,
) -> str:
    """Encode video-only frames into a NEUTXT v2 text string."""
    return encode_av_to_text(
        video_codes=all_codes,
        audio_codes=None,
        fps=fps,
        keyint=keyint,
        video_model=model,
        compression=compression,
        level=level,
    )


# ---------------------------------------------------------------------------
# Encode: unified audio + video
# ---------------------------------------------------------------------------

def _bitpack10(codes: np.ndarray) -> bytes:
    """Pack uint16 codes (10-bit values). Duplicated here to avoid circular import."""
    out = bytearray()
    acc = 0
    bits = 0
    mask = (1 << 10) - 1
    for v in codes.ravel():
        acc |= (int(v) & mask) << bits
        bits += 10
        while bits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            bits -= 8
    if bits:
        out.append(acc & 0xFF)
    return bytes(out)


def _bitunpack10(packed: bytes, n_codes: int) -> np.ndarray:
    out = np.empty((n_codes,), dtype=np.uint16)
    acc = 0
    bits = 0
    idx = 0
    for b in packed:
        acc |= int(b) << bits
        bits += 8
        while bits >= 10 and idx < n_codes:
            out[idx] = acc & ((1 << 10) - 1)
            acc >>= 10
            bits -= 10
            idx += 1
        if idx >= n_codes:
            break
    if idx != n_codes:
        raise RuntimeError(f"bitunpack10: expected {n_codes}, got {idx}")
    return out


def encode_av_to_text(
    video_codes: list[np.ndarray] | None = None,
    audio_codes: list[np.ndarray] | None = None,
    fps: float = 12.0,
    keyint: int = 12,
    video_model: str = "MAGVIT2_256_L",
    audio_model: str = "ENCODEC_24K",
    audio_sr: int = 24000,
    audio_bandwidth: float = 6.0,
    audio_quantizers: int = 8,
    audio_chunk_seconds: float = 1.0,
    compression: str = "zstd",
    level: int = 9,
) -> str:
    """
    Encode audio and/or video into a NEUTXT v2 text string.

    Args:
        video_codes: list of uint32 arrays (256,) per frame, or None
        audio_codes: list of uint16 arrays (n_quantizers, timesteps) per chunk, or None
        fps: video frames per second
        keyint: video keyframe interval
        audio_sr: audio sample rate
        audio_bandwidth: EnCodec target bandwidth (kbps)
        audio_quantizers: number of residual quantizers (from bandwidth)
        audio_chunk_seconds: duration per audio chunk

    Returns:
        Complete NEUTXT v2 text string.
    """
    has_video = bool(video_codes)
    has_audio = bool(audio_codes)

    if not has_video and not has_audio:
        raise ValueError("Must provide video_codes or audio_codes (or both)")

    mode = "av" if has_video and has_audio else ("v" if has_video else "a")

    frame_lines: list[str] = []

    # --- Video frames (K: / D:) ---
    prev_packed: bytes | None = None
    if has_video:
        for i, codes in enumerate(video_codes or []):
            packed = bitpack18(codes)
            is_key = (i == 0) or (i % keyint == 0)
            if is_key or prev_packed is None:
                payload = packed
                prefix = "K"
            else:
                payload = _xor_bytes(prev_packed, packed)
                prefix = "D"
            prev_packed = packed
            compressed = _compress(payload, codec=compression, level=level)
            encoded = base64.b85encode(compressed).decode("ascii")
            frame_lines.append(f"{prefix}:{encoded}")

    # --- Audio chunks (A:) ---
    if has_audio:
        for codes in audio_codes or []:
            # codes shape: (n_quantizers, timesteps), uint16
            packed = _bitpack10(codes)
            compressed = _compress(packed, codec=compression, level=level)
            encoded = base64.b85encode(compressed).decode("ascii")
            # Embed timesteps in the line so decoder can unpack correctly
            timesteps = codes.shape[1] if codes.ndim == 2 else len(codes)
            frame_lines.append(f"A:{timesteps}:{encoded}")

    full_data = "\n".join(frame_lines).encode("utf-8")
    checksum = format(zlib.crc32(full_data) & 0xFFFFFFFF, "08x")

    header = TextNeutxtHeader(
        mode=mode,
        video_model=video_model,
        fps=fps,
        resolution="256x256",
        video_code_bits=18,
        tokens_per_frame=256,
        keyint=keyint,
        video_frames=len(video_codes) if has_video else 0,
        audio_model=audio_model,
        audio_sr=audio_sr,
        audio_bandwidth=audio_bandwidth,
        audio_quantizers=audio_quantizers,
        audio_code_bits=10,
        audio_chunk_seconds=audio_chunk_seconds,
        audio_chunks=len(audio_codes) if has_audio else 0,
        compression=compression,
        checksum=checksum,
        # Legacy fields for back-compat
        frames=len(video_codes) if has_video else 0,
        code_bits=18,
        model=video_model,
    )

    lines = [HEADER_START]
    # Only emit fields relevant to the mode, for a cleaner output
    if mode == "v":
        emit = ["mode", "video_model", "fps", "resolution", "video_code_bits",
                "tokens_per_frame", "keyint", "compression", "video_frames", "checksum"]
    elif mode == "a":
        emit = ["mode", "audio_model", "audio_sr", "audio_bandwidth",
                "audio_quantizers", "audio_code_bits", "audio_chunk_seconds",
                "compression", "audio_chunks", "checksum"]
    else:
        emit = ["mode", "video_model", "fps", "resolution", "video_code_bits",
                "tokens_per_frame", "keyint", "video_frames",
                "audio_model", "audio_sr", "audio_bandwidth", "audio_quantizers",
                "audio_code_bits", "audio_chunk_seconds", "audio_chunks",
                "compression", "checksum"]

    for k in emit:
        lines.append(f"{k}: {getattr(header, k)}")
    lines.append(HEADER_END)
    lines.extend(frame_lines)
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Decode: NEUTXT text string -> list of code arrays
# ---------------------------------------------------------------------------

def parse_text_neutxt(text: str) -> TextNeutxtFile:
    """Parse a NEUTXT v2 text string into header + frame lines."""
    lines = text.strip().split("\n")

    if not lines or lines[0].strip() != HEADER_START:
        raise ValueError(f"Not a NEUTXT v2 file (expected '{HEADER_START}')")

    header_dict = {}
    i = 1
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if line == HEADER_END:
            break
        if ":" in line:
            k, v = line.split(":", 1)
            header_dict[k.strip()] = v.strip()

    def _g(key, default):
        return header_dict.get(key, default)

    mode = _g("mode", "v")
    video_code_bits = int(_g("video_code_bits", _g("code_bits", "18")))
    video_frames = int(_g("video_frames", _g("frames", "0")))
    video_model = _g("video_model", _g("model", "MAGVIT2_256_L"))

    header = TextNeutxtHeader(
        mode=mode,
        video_model=video_model,
        fps=float(_g("fps", "12.0")),
        resolution=_g("resolution", "256x256"),
        video_code_bits=video_code_bits,
        tokens_per_frame=int(_g("tokens_per_frame", "256")),
        keyint=int(_g("keyint", "12")),
        video_frames=video_frames,
        audio_model=_g("audio_model", "ENCODEC_24K"),
        audio_sr=int(_g("audio_sr", "24000")),
        audio_bandwidth=float(_g("audio_bandwidth", "6.0")),
        audio_quantizers=int(_g("audio_quantizers", "8")),
        audio_code_bits=int(_g("audio_code_bits", "10")),
        audio_chunk_seconds=float(_g("audio_chunk_seconds", "1.0")),
        audio_chunks=int(_g("audio_chunks", "0")),
        compression=_g("compression", "zstd"),
        checksum=_g("checksum", ""),
        frames=video_frames,
        code_bits=video_code_bits,
        model=video_model,
    )

    frame_lines = []
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if line and (line.startswith("K:") or line.startswith("D:") or line.startswith("A:")):
            frame_lines.append(line)

    return TextNeutxtFile(header=header, frame_lines=frame_lines)


def decode_text_to_frames(text: str) -> tuple[TextNeutxtHeader, list[np.ndarray]]:
    """
    Decode video frames from a NEUTXT text string (ignores audio).

    Returns:
        (header, list of np.ndarray(uint32, shape=(256,)))
    """
    parsed = parse_text_neutxt(text)
    h = parsed.header

    all_codes: list[np.ndarray] = []
    prev_packed: bytes | None = None

    for line in parsed.frame_lines:
        if not (line.startswith("K:") or line.startswith("D:")):
            continue
        prefix = line[0]
        b85_data = line[2:]
        compressed = base64.b85decode(b85_data)
        payload = _decompress(compressed, codec=h.compression)

        if prefix == "K" or prev_packed is None:
            packed = payload
        else:
            packed = _xor_bytes(prev_packed, payload)

        prev_packed = packed
        codes = bitunpack18(packed, h.tokens_per_frame)
        all_codes.append(codes)

    return h, all_codes


def decode_text_to_audio_codes(text: str) -> tuple[TextNeutxtHeader, list[np.ndarray]]:
    """
    Decode audio chunks from a NEUTXT text string (ignores video).

    Returns:
        (header, list of np.ndarray(uint16, shape=(n_quantizers, timesteps)))
    """
    parsed = parse_text_neutxt(text)
    h = parsed.header

    all_audio: list[np.ndarray] = []
    for line in parsed.frame_lines:
        if not line.startswith("A:"):
            continue
        # Format: A:<timesteps>:<base85>
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, timesteps_str, b85_data = parts
        timesteps = int(timesteps_str)
        n_codes = h.audio_quantizers * timesteps

        compressed = base64.b85decode(b85_data)
        packed = _decompress(compressed, codec=h.compression)
        flat = _bitunpack10(packed, n_codes)
        codes = flat.reshape(h.audio_quantizers, timesteps)
        all_audio.append(codes)

    return h, all_audio


def decode_text_av(text: str) -> tuple[TextNeutxtHeader, list[np.ndarray], list[np.ndarray]]:
    """
    Decode both video and audio from a NEUTXT text string.

    Returns:
        (header, video_codes_list, audio_codes_list)
    """
    h, video_codes = decode_text_to_frames(text)
    _, audio_codes = decode_text_to_audio_codes(text)
    return h, video_codes, audio_codes


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------

def text_stats(text: str) -> dict:
    """Return size statistics for a NEUTXT text string."""
    parsed = parse_text_neutxt(text)
    h = parsed.header
    n_key = sum(1 for l in parsed.frame_lines if l.startswith("K:"))
    n_delta = sum(1 for l in parsed.frame_lines if l.startswith("D:"))
    n_audio = sum(1 for l in parsed.frame_lines if l.startswith("A:"))
    n_video = n_key + n_delta

    total_chars = len(text)
    video_duration = n_video / h.fps if h.fps > 0 and n_video > 0 else 0
    audio_duration = n_audio * h.audio_chunk_seconds if n_audio > 0 else 0
    duration = max(video_duration, audio_duration)

    return {
        "mode": h.mode,
        "video_frames": n_video,
        "keyframes": n_key,
        "delta_frames": n_delta,
        "audio_chunks": n_audio,
        "duration_sec": round(duration, 2),
        "total_chars": total_chars,
        "chars_per_second": round(total_chars / duration, 1) if duration > 0 else 0,
        "approx_llm_tokens": total_chars // 4,
        # Legacy field
        "frames": n_video,
    }
