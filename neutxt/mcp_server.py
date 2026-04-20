"""NEUTXT MCP server.

Exposes NEUTXT manipulation as MCP tools so any Claude Code / Claude desktop
client that adds this server gets native `neutxt_*` tools — no system prompt
needed, no format-teaching required per conversation.

Run:
    python -m neutxt mcp

Configure in Claude Code (~/.claude/mcp_servers.json or equivalent):
    {
      "neutxt": {
        "command": "python",
        "args": ["-m", "neutxt", "mcp"],
        "env": {
          "NEUTXT_VQ_CKPT": "/path/to/magvit2_256L.ckpt",
          "NEUTXT_VQ_CONFIG": "/path/to/imagenet_lfqgan_256_L.yaml"
        }
      }
    }

The VQ_CKPT/VQ_CONFIG env vars are only needed for tools that decode pixels
(neutxt_preview and future pixel-transform tools). Structural tools work
without them.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.types import Image

from neutxt.text_codec import (
    parse_text_neutxt, text_stats, encode_av_to_text,
    decode_text_to_frames, decode_text_to_audio_codes,
    HEADER_START,
)


mcp = FastMCP("neutxt")


# ---------------------------------------------------------------------------
# Lazy-loaded codecs
# ---------------------------------------------------------------------------

_tok = None
_enc_24 = None
_enc_48 = None


def _get_tokenizer():
    global _tok
    if _tok is not None:
        return _tok
    from neutxt.vq import pick_device, load_magvit2
    ckpt = os.environ.get("NEUTXT_VQ_CKPT")
    cfg = os.environ.get("NEUTXT_VQ_CONFIG")
    if not ckpt or not cfg:
        raise RuntimeError(
            "NEUTXT_VQ_CKPT and NEUTXT_VQ_CONFIG env vars must be set for "
            "tools that decode video. Add them to the MCP server config."
        )
    device = pick_device("auto")
    _tok = load_magvit2(ckpt, cfg, device=device, res=256)
    return _tok


def _get_encoder(sample_rate: int):
    global _enc_24, _enc_48
    if sample_rate == 24000 and _enc_24 is not None:
        return _enc_24
    if sample_rate == 48000 and _enc_48 is not None:
        return _enc_48
    from neutxt.vq import pick_device
    from neutxt.audio_codec import load_encodec
    device = pick_device("auto")
    enc = load_encodec(device, sample_rate=sample_rate, bandwidth=6.0)
    if sample_rate == 24000:
        _enc_24 = enc
    else:
        _enc_48 = enc
    return enc


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def _read(path: str) -> str:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"NEUTXT file not found: {p}")
    return p.read_text(encoding="utf-8")


def _write(path: str, text: str) -> str:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# Structural tools (text-only; no pixel decode needed)
# ---------------------------------------------------------------------------

@mcp.tool()
def neutxt_info(path: str) -> dict:
    """Return metadata and statistics about a NEUTXT file.

    Works without any neural models — pure text inspection.

    Args:
        path: Path to a .neutxt.txt file.
    """
    text = _read(path)
    stats = text_stats(text)
    parsed = parse_text_neutxt(text)
    h = parsed.header
    return {
        "path": str(Path(path).expanduser().resolve()),
        "size_chars": stats["total_chars"],
        "approx_llm_tokens": stats["approx_llm_tokens"],
        "mode": stats["mode"],
        "duration_sec": stats["duration_sec"],
        "video": {
            "frames": stats["video_frames"],
            "keyframes": stats["keyframes"],
            "delta_frames": stats["delta_frames"],
            "fps": h.fps,
            "resolution": h.resolution,
            "model": h.video_model,
            "keyint": h.keyint,
        } if stats["video_frames"] else None,
        "audio": {
            "chunks": stats["audio_chunks"],
            "sample_rate": h.audio_sr,
            "bandwidth_kbps": h.audio_bandwidth,
            "quantizers": h.audio_quantizers,
            "chunk_seconds": h.audio_chunk_seconds,
            "model": h.audio_model,
        } if stats["audio_chunks"] else None,
        "compression": h.compression,
        "checksum": h.checksum,
    }


@mcp.tool()
def neutxt_reverse(path: str, output_path: str) -> dict:
    """Reverse the temporal order of a NEUTXT file.

    Requires all video frames to be keyframes (K:). If any D: (delta) frames
    are present, reversing is unsafe — the function returns an error with
    guidance. Audio chunks (A:) are always safe to reverse.

    Args:
        path: Input .neutxt.txt path.
        output_path: Where to write the reversed file.
    """
    text = _read(path)
    parsed = parse_text_neutxt(text)
    h = parsed.header

    video_lines = [l for l in parsed.frame_lines if l.startswith(("K:", "D:"))]
    audio_lines = [l for l in parsed.frame_lines if l.startswith("A:")]

    has_deltas = any(l.startswith("D:") for l in video_lines)
    if has_deltas:
        return {
            "error": "Cannot reverse: file contains delta (D:) frames. "
                     "Re-encode the source with keyint=1 (all keyframes) first.",
            "delta_count": sum(1 for l in video_lines if l.startswith("D:")),
        }

    new_video = list(reversed(video_lines))
    new_audio = list(reversed(audio_lines))
    new_frame_lines = new_video + new_audio

    # Rebuild the text with the same header, just new body
    header_block = text.split("---\n", 2)
    # Preserve original header exactly, swap body
    out_lines = []
    in_body = False
    header_end_count = 0
    for line in text.split("\n"):
        if line.strip() == "---":
            header_end_count += 1
            out_lines.append(line)
            if header_end_count == 2:
                in_body = True
                out_lines.extend(new_frame_lines)
                break
        else:
            out_lines.append(line)
    out_text = "\n".join(out_lines) + "\n"
    written = _write(output_path, out_text)
    return {"output_path": written, "video_frames": len(new_video),
            "audio_chunks": len(new_audio)}


@mcp.tool()
def neutxt_strip_audio(path: str, output_path: str) -> dict:
    """Remove all audio (A:) lines from a NEUTXT file, keeping only video.

    Args:
        path: Input .neutxt.txt path.
        output_path: Where to write the video-only file.
    """
    text = _read(path)
    parsed = parse_text_neutxt(text)
    video_lines = [l for l in parsed.frame_lines if l.startswith(("K:", "D:"))]
    if not video_lines:
        return {"error": "No video lines present — nothing to keep."}
    return _rewrite_with_new_body(text, video_lines, output_path, new_mode="v")


@mcp.tool()
def neutxt_strip_video(path: str, output_path: str) -> dict:
    """Remove all video (K:/D:) lines from a NEUTXT file, keeping only audio.

    Args:
        path: Input .neutxt.txt path.
        output_path: Where to write the audio-only file.
    """
    text = _read(path)
    parsed = parse_text_neutxt(text)
    audio_lines = [l for l in parsed.frame_lines if l.startswith("A:")]
    if not audio_lines:
        return {"error": "No audio lines present — nothing to keep."}
    return _rewrite_with_new_body(text, audio_lines, output_path, new_mode="a")


@mcp.tool()
def neutxt_trim(path: str, start_sec: float, end_sec: float,
                output_path: str) -> dict:
    """Trim a NEUTXT file to [start_sec, end_sec].

    Video: keeps frames whose timestamp (i / fps) falls in the range.
    The first kept frame MUST be a keyframe (K:); if the range starts
    inside a delta group, the start is rounded up to the next keyframe.

    Audio: keeps A: chunks whose window overlaps the range.

    Args:
        path: Input .neutxt.txt path.
        start_sec: Range start (inclusive).
        end_sec: Range end (exclusive).
        output_path: Where to write the trimmed file.
    """
    if end_sec <= start_sec:
        return {"error": "end_sec must be greater than start_sec"}

    text = _read(path)
    parsed = parse_text_neutxt(text)
    h = parsed.header

    video_lines = [l for l in parsed.frame_lines if l.startswith(("K:", "D:"))]
    audio_lines = [l for l in parsed.frame_lines if l.startswith("A:")]

    kept_video: list[str] = []
    adjusted_start = start_sec
    if video_lines and h.fps > 0:
        start_idx = int(start_sec * h.fps)
        end_idx = int(end_sec * h.fps)
        # Find first keyframe at or after start_idx
        while start_idx < len(video_lines) and not video_lines[start_idx].startswith("K:"):
            start_idx += 1
        if start_idx >= len(video_lines):
            return {"error": "No keyframe found at or after start_sec; nothing to keep."}
        adjusted_start = start_idx / h.fps
        kept_video = video_lines[start_idx:min(end_idx, len(video_lines))]

    kept_audio: list[str] = []
    if audio_lines:
        chunk = h.audio_chunk_seconds or 1.0
        for i, line in enumerate(audio_lines):
            chunk_start = i * chunk
            chunk_end = (i + 1) * chunk
            if chunk_end > start_sec and chunk_start < end_sec:
                kept_audio.append(line)

    new_body = kept_video + kept_audio
    mode = "av" if kept_video and kept_audio else ("v" if kept_video else "a")
    result = _rewrite_with_new_body(text, new_body, output_path, new_mode=mode)
    result["adjusted_start_sec"] = round(adjusted_start, 3)
    result["kept_video"] = len(kept_video)
    result["kept_audio"] = len(kept_audio)
    return result


@mcp.tool()
def neutxt_concat(paths: list[str], output_path: str) -> dict:
    """Concatenate multiple NEUTXT files in order.

    All inputs must use the same video model, resolution, fps, and audio
    parameters. Mismatches return an error.

    Args:
        paths: List of input .neutxt.txt paths, in the order to concatenate.
        output_path: Where to write the concatenated file.
    """
    if len(paths) < 2:
        return {"error": "Need at least 2 input paths."}

    parsed_files = [parse_text_neutxt(_read(p)) for p in paths]
    base = parsed_files[0].header
    for i, pf in enumerate(parsed_files[1:], 1):
        h = pf.header
        if (h.video_model != base.video_model or h.fps != base.fps or
                h.resolution != base.resolution or
                h.audio_sr != base.audio_sr or
                h.audio_bandwidth != base.audio_bandwidth):
            return {"error": f"Input #{i} ({paths[i]}) has incompatible "
                             "parameters (model/fps/resolution/audio)."}

    merged = []
    for pf in parsed_files:
        merged.extend(pf.frame_lines)

    has_v = any(l.startswith(("K:", "D:")) for l in merged)
    has_a = any(l.startswith("A:") for l in merged)
    mode = "av" if has_v and has_a else ("v" if has_v else "a")
    return _rewrite_with_new_body(_read(paths[0]), merged, output_path, new_mode=mode)


# ---------------------------------------------------------------------------
# Preview tool (decodes pixels so Claude can see)
# ---------------------------------------------------------------------------

@mcp.tool(structured_output=False)
def neutxt_preview(path: str, start_sec: float = 0.0, end_sec: float = 2.0,
                    max_frames: int = 4):
    """Decode video frames from a range and return them as PNG images.

    Use this when you need to SEE the video content to answer a question or
    plan an edit. The images are returned as MCP image content — Claude's
    vision can read them directly.

    Args:
        path: Input .neutxt.txt path.
        start_sec: Start of preview window (default 0).
        end_sec: End of preview window (default 2).
        max_frames: Maximum number of frames to return (default 4, capped at 8).
    """
    from PIL import Image as PILImage
    from neutxt.vq import decode_packed_codes_to_frame

    max_frames = max(1, min(max_frames, 8))
    text = _read(path)
    header, all_codes = decode_text_to_frames(text)

    if not all_codes:
        return []

    fps = header.fps or 8.0
    start_idx = max(0, int(start_sec * fps))
    end_idx = min(len(all_codes), max(start_idx + 1, int(end_sec * fps)))

    if start_idx >= len(all_codes):
        return []

    # Sample up to max_frames evenly across the range
    n_in_range = end_idx - start_idx
    if n_in_range <= max_frames:
        sample_idx = list(range(start_idx, end_idx))
    else:
        step = n_in_range / max_frames
        sample_idx = [start_idx + int(i * step) for i in range(max_frames)]

    tok = _get_tokenizer()
    images: list[Image] = []
    for idx in sample_idx:
        frame = decode_packed_codes_to_frame(tok, all_codes[idx], 16, 16)
        buf = io.BytesIO()
        PILImage.fromarray(frame, "RGB").save(buf, format="PNG", optimize=True)
        images.append(Image(data=buf.getvalue(), format="png"))
    return images


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rewrite_with_new_body(original_text: str, new_body_lines: list[str],
                            output_path: str, new_mode: Optional[str] = None) -> dict:
    """Rebuild a NEUTXT file with a new body, updating counts and mode."""
    import zlib

    lines = original_text.split("\n")
    # Find header end
    header_end = None
    for i, line in enumerate(lines):
        if i > 0 and line.strip() == "---":
            header_end = i
            break
    if header_end is None:
        raise ValueError("Malformed NEUTXT: no closing '---' after header")

    header_lines = lines[:header_end + 1]

    # Recompute counts and checksum
    n_video = sum(1 for l in new_body_lines if l.startswith(("K:", "D:")))
    n_audio = sum(1 for l in new_body_lines if l.startswith("A:"))
    new_checksum = format(
        zlib.crc32("\n".join(new_body_lines).encode("utf-8")) & 0xFFFFFFFF, "08x")

    updated_header = []
    for line in header_lines:
        s = line.strip()
        if s.startswith("mode:") and new_mode is not None:
            updated_header.append(f"mode: {new_mode}")
        elif s.startswith("video_frames:") or s.startswith("frames:"):
            key = s.split(":", 1)[0]
            updated_header.append(f"{key}: {n_video}")
        elif s.startswith("audio_chunks:"):
            updated_header.append(f"audio_chunks: {n_audio}")
        elif s.startswith("checksum:"):
            updated_header.append(f"checksum: {new_checksum}")
        else:
            updated_header.append(line)

    out_text = "\n".join(updated_header + new_body_lines) + "\n"
    written = _write(output_path, out_text)
    return {"output_path": written, "mode": new_mode,
            "video_frames": n_video, "audio_chunks": n_audio}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    mcp.run()


if __name__ == "__main__":
    main()
