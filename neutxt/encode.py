"""Encode video/audio files to .neutxt format."""
from __future__ import annotations

import argparse
import os
import struct
import subprocess
import zlib
from typing import Any, Dict, Iterator

import numpy as np

from neutxt.codec import NeutxtWriter, compress_bytes, encode_audio_to_records
from neutxt.vq import pick_device, load_magvit2, encode_frame_to_packed_codes

try:
    import zstandard as zstd
    _HAVE_ZSTD = True
except ImportError:
    zstd = None
    _HAVE_ZSTD = False


def _bitpack18(codes_u32: np.ndarray) -> bytes:
    out = bytearray()
    acc = 0
    bits = 0
    mask = (1 << 18) - 1
    for v in codes_u32.ravel():
        acc |= (int(v) & mask) << bits
        bits += 18
        while bits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            bits -= 8
    if bits:
        out.append(acc & 0xFF)
    return bytes(out)


def _xor_bytes(a: bytes, b: bytes) -> bytes:
    return bytes(x ^ y for x, y in zip(a, b))


def _crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def _ffmpeg_frames_rgb(path: str, fps: float, width: int, height: int) -> Iterator[np.ndarray]:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", path,
        "-vf", f"fps={fps},scale={width}:{height}:flags=lanczos",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert p.stdout is not None
    frame_bytes = width * height * 3
    while True:
        buf = p.stdout.read(frame_bytes)
        if not buf or len(buf) != frame_bytes:
            break
        yield np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 3))
    p.stdout.close()
    p.wait()


def main():
    ap = argparse.ArgumentParser(description="Encode media to .neutxt")
    ap.add_argument("input", help="input media file (mp4, etc)")
    ap.add_argument("--mode", default="av", choices=["a", "v", "av"])
    ap.add_argument("--vq_ckpt", required=True)
    ap.add_argument("--vq_config", required=True, help="MAGVIT2 YAML config")
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--sr", type=int, default=24000)
    ap.add_argument("--video_codec", default="zstd", choices=["zlib", "zstd"])
    ap.add_argument("--audio_codec", default="zstd", choices=["zlib", "zstd"])
    ap.add_argument("--clevel", type=int, default=6, help="compression level")
    ap.add_argument("--keyint", type=int, default=30, help="keyframe interval")
    ap.add_argument("--achunk", type=float, default=10.0, help="audio chunk seconds")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.width != 256:
        raise ValueError("MAGVIT2_256_L requires --width 256")

    out_path = args.out or (os.path.splitext(args.input)[0] + ".neutxt")
    device = pick_device(args.device)
    tokenizer_id = "MAGVIT2_256_L"
    magic = f"NEUTXT_AV2|{tokenizer_id}"

    meta: Dict[str, Any] = {
        "container": "NEUTXT_AV2",
        "magic": magic,
        "source": os.path.basename(args.input),
        "audio": {
            "sr": args.sr,
            "chunk_seconds": args.achunk,
            "channels": 1,
            "codec": args.audio_codec,
            "clevel": args.clevel,
        },
        "video": {
            "fps": args.fps,
            "width": 256,
            "height": 256,
            "tokenizer": "MAGVIT2",
            "tokenizer_id": tokenizer_id,
            "packed_dtype": 4,
            "code_bits": 18,
            "video_codec": args.video_codec,
            "audio_codec": args.audio_codec,
            "clevel": args.clevel,
            "keyint": args.keyint,
        },
    }

    with NeutxtWriter(out_path, magic=magic, meta=meta) as w:
        tok = None
        if args.mode in ("v", "av"):
            print(f"Loading MAGVIT2 on {device} ...")
            tok = load_magvit2(args.vq_ckpt, args.vq_config, device=device, res=256)

        if args.mode in ("a", "av"):
            print("Encoding audio ...")
            for ts_ms, comp_pcm in encode_audio_to_records(
                    args.input, sr=args.sr, chunk_seconds=args.achunk,
                    channels=1, codec=args.audio_codec, level=args.clevel):
                w.write_audio(ts_ms, comp_pcm)

        if args.mode in ("v", "av") and tok is not None:
            print("Encoding video frames ...")
            prev_packed18: bytes | None = None

            for i, frame in enumerate(_ffmpeg_frames_rgb(
                    args.input, fps=args.fps, width=256, height=256)):
                ts_ms = int(round((i / args.fps) * 1000.0))
                codes_u32, gh, gw = encode_frame_to_packed_codes(tok, frame)

                packed18 = _bitpack18(codes_u32)
                is_key = (i == 0) or (i % args.keyint == 0)

                if is_key or prev_packed18 is None:
                    payload_raw = packed18
                    flags = 0x01  # keyframe
                else:
                    payload_raw = _xor_bytes(prev_packed18, packed18)
                    flags = 0x02  # delta

                prev_packed18 = packed18

                payload_comp = compress_bytes(payload_raw, codec=args.video_codec,
                                              level=args.clevel)
                codec_id = 1 if args.video_codec == "zstd" else 0
                checksum = _crc32(payload_comp)

                pkt_hdr = struct.pack("<B B H I I",
                                     flags, codec_id, len(payload_raw),
                                     len(payload_comp), checksum)
                av2_packet = pkt_hdr + payload_comp

                w.write_video(ts_ms, gh=gh, gw=gw, packed_dtype=4,
                              code_bits=tok.code_bits, packed_codes=av2_packet)

                if (i + 1) % 50 == 0:
                    print(f"  frame {i+1} encoded")

    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
