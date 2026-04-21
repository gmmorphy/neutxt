"""Play .neutxt files.

Both formats share the .neutxt extension and are distinguished by magic
bytes: text files start with '--- NEUTXT', binary files do not.

Text files are decoded to a standard .mp4 with ffmpeg and opened in the
system player. Binary files use the Tkinter + sounddevice player below.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import struct
import sys
import tempfile
import threading
import time
import queue
import zlib
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _is_text_neutxt(path: str) -> bool:
    """Return True if path points to a NEUTXT text-format file."""
    try:
        with open(path, "rb") as f:
            head = f.read(64)
        return head.startswith(b"--- NEUTXT")
    except OSError:
        return False


def _system_open(path: str):
    opener = {"darwin": "open", "linux": "xdg-open", "win32": "start"}.get(
        sys.platform, "open")
    subprocess.Popen([opener, path])


def _play_text_neutxt(input_path: str, keep: bool = False):
    """Decode a NEUTXT text file to MP4 and launch the system player."""
    from neutxt.mcp_server import _decode_to_mp4

    if keep:
        out = str(Path(input_path).with_suffix(".mp4"))
        result = _decode_to_mp4(input_path, out)
        print(f"Saved: {result['output_path']}")
    else:
        td = tempfile.mkdtemp(prefix="neutxt_play_")
        out = os.path.join(td, "preview.mp4")
        result = _decode_to_mp4(input_path, out)
        print(f"Decoded to: {result['output_path']}")

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"  duration={result['duration_sec']}s, "
          f"video={result['video_frames']} frames, "
          f"audio={result['audio_chunks']} chunks")
    _system_open(result["output_path"])

try:
    import zstandard as zstd
    _HAVE_ZSTD = True
except ImportError:
    zstd = None
    _HAVE_ZSTD = False

try:
    import sounddevice as sd
    _HAVE_SD = True
except ImportError:
    sd = None
    _HAVE_SD = False

# PIL + tkinter are only needed for the legacy binary-format player below.
# We lazy-import them inside main() so systems without _tkinter can still
# use the text-format path.
Image = ImageFilter = ImageTk = tk = None

from neutxt.codec import read_header, iter_video_records, decode_audio_records_to_wav_bytes
from neutxt.vq import pick_device, load_magvit2, decode_packed_codes_to_frame


def _bitunpack18(packed: bytes, n_tokens: int) -> np.ndarray:
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


def _xor_bytes(a: bytes, b: bytes) -> bytes:
    return bytes(x ^ y for x, y in zip(a, b))


def _crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


# AV2 packet header: flags(u8) codec_id(u8) raw_len(u16) comp_len(u32) crc32(u32)
_V2_PKT_HDR = struct.Struct("<B B H I I")
# Legacy AV2 without CRC (8 bytes)
_V2_PKT_HDR_LEGACY = struct.Struct("<B B H I")


class DecoderThread(threading.Thread):
    def __init__(self, tok, video_iter, frame_q, stop_evt):
        super().__init__(daemon=True)
        self.tok = tok
        self.video_iter = video_iter
        self.frame_q: queue.Queue[Tuple[int, np.ndarray]] = frame_q
        self.stop_evt: threading.Event = stop_evt
        self.err: Optional[Exception] = None

    def run(self):
        prev_packed18: bytes | None = None
        try:
            for rec in self.video_iter:
                if self.stop_evt.is_set():
                    break

                if rec.packed_dtype in (1, 2, 3):
                    dtype_map = {1: np.uint8, 2: np.uint16, 3: np.uint32}
                    packed = np.frombuffer(rec.payload, dtype=dtype_map[rec.packed_dtype])

                elif rec.packed_dtype == 4:
                    # Try new header (with CRC, 12 bytes) first, fall back to legacy (8 bytes)
                    if len(rec.payload) >= _V2_PKT_HDR.size:
                        flags, codec_id, raw_len, comp_len, crc = _V2_PKT_HDR.unpack(
                            rec.payload[:_V2_PKT_HDR.size])
                        hdr_size = _V2_PKT_HDR.size
                        payload_comp = rec.payload[hdr_size:hdr_size + comp_len]

                        actual_crc = _crc32(payload_comp)
                        if crc != 0 and actual_crc != crc:
                            print(f"WARNING: CRC mismatch at ts={rec.ts_ms}ms "
                                  f"(expected {crc:#x}, got {actual_crc:#x}), skipping")
                            continue
                    elif len(rec.payload) >= _V2_PKT_HDR_LEGACY.size:
                        flags, codec_id, raw_len, comp_len = _V2_PKT_HDR_LEGACY.unpack(
                            rec.payload[:_V2_PKT_HDR_LEGACY.size])
                        hdr_size = _V2_PKT_HDR_LEGACY.size
                        payload_comp = rec.payload[hdr_size:hdr_size + comp_len]
                    else:
                        raise RuntimeError("Bad AV2 packet (too small)")

                    if len(payload_comp) != comp_len:
                        raise RuntimeError("Bad AV2 packet (truncated)")

                    if codec_id == 1:
                        if not _HAVE_ZSTD:
                            raise RuntimeError("AV2 uses zstd — pip install zstandard")
                        payload_raw = zstd.ZstdDecompressor().decompress(payload_comp)
                    else:
                        payload_raw = zlib.decompress(payload_comp)

                    if flags & 0x01 or prev_packed18 is None:
                        packed18 = payload_raw
                    else:
                        packed18 = _xor_bytes(prev_packed18, payload_raw)

                    prev_packed18 = packed18
                    packed = _bitunpack18(packed18, rec.gh * rec.gw)
                else:
                    raise RuntimeError(f"Unsupported packed_dtype={rec.packed_dtype}")

                frame = decode_packed_codes_to_frame(self.tok, packed, rec.gh, rec.gw)
                try:
                    self.frame_q.put((rec.ts_ms, frame), timeout=0.5)
                except queue.Full:
                    pass
        except Exception as e:
            self.err = e


def _play_audio_sd(wav_bytes: bytes, sr: int):
    """Play WAV audio using sounddevice (cross-platform)."""
    import wave
    import io
    bio = io.BytesIO(wav_bytes)
    with wave.open(bio, "rb") as wf:
        channels = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
    pcm = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        pcm = pcm.reshape(-1, channels)
    sd.play(pcm, samplerate=sr, blocking=False)


def main():
    ap = argparse.ArgumentParser(description="Play .neutxt files")
    ap.add_argument("input", help=".neutxt file (text or binary)")
    ap.add_argument("--vq_ckpt",
                    help="MAGVIT2 checkpoint (required for legacy binary format)")
    ap.add_argument("--vq_config",
                    help="MAGVIT2 YAML config (required for legacy binary format)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--qsize", type=int, default=120)
    ap.add_argument("--scale", type=int, default=1, help="UI upscale factor")
    ap.add_argument("--sharpen", action="store_true")
    ap.add_argument("--sharpen_pct", type=int, default=80, help="unsharp mask percent")
    ap.add_argument("--noaudio", action="store_true")
    ap.add_argument("--novideo", action="store_true")
    ap.add_argument("--keep", action="store_true",
                    help="For text files: save the decoded .mp4 next to the input "
                         "instead of using a temp file")
    args = ap.parse_args()

    # Text-format path: decode to .mp4 and open in system player
    if _is_text_neutxt(args.input):
        # Pass MAGVIT2 paths via env so the shared decode helper can load them
        if args.vq_ckpt and not os.environ.get("NEUTXT_VQ_CKPT"):
            os.environ["NEUTXT_VQ_CKPT"] = args.vq_ckpt
        if args.vq_config and not os.environ.get("NEUTXT_VQ_CONFIG"):
            os.environ["NEUTXT_VQ_CONFIG"] = args.vq_config
        _play_text_neutxt(args.input, keep=args.keep)
        return

    # Legacy binary format requires MAGVIT2 paths
    if not (args.vq_ckpt and args.vq_config):
        ap.error("--vq_ckpt and --vq_config are required for legacy binary .neutxt files")

    global Image, ImageFilter, ImageTk, tk
    from PIL import Image, ImageFilter, ImageTk  # noqa: F811
    import tkinter as tk  # noqa: F811

    magic, meta = read_header(args.input)
    print("Magic:", magic)
    print("Meta:", meta)

    vid = meta.get("video", {})
    if vid.get("tokenizer") != "MAGVIT2":
        raise ValueError("This player expects MAGVIT2 files")
    if int(vid.get("width", 0)) != 256 or int(vid.get("height", 0)) != 256:
        raise ValueError("This player expects 256x256 MAGVIT2 files")

    audio_codec = meta.get("audio", {}).get("codec",
                    vid.get("audio_codec", "zlib"))

    device = pick_device(args.device)

    wav_bytes = None
    if not args.noaudio:
        sr = int(meta["audio"]["sr"])
        wav_bytes = decode_audio_records_to_wav_bytes(
            args.input, sr=sr,
            channels=int(meta["audio"].get("channels", 1)),
            codec=audio_codec)
        print(f"Decoded audio: {len(wav_bytes)/1024/1024:.2f} MB wav")

    tok = None
    if not args.novideo:
        print(f"Loading MAGVIT2 on {device} ...")
        tok = load_magvit2(args.vq_ckpt, args.vq_config, device=device, res=256)

    if wav_bytes is not None:
        if not _HAVE_SD:
            print("WARNING: sounddevice not installed — pip install sounddevice. "
                  "Audio will not play.")
        else:
            _play_audio_sd(wav_bytes, sr=int(meta["audio"]["sr"]))

    if args.novideo:
        print("Audio-only mode. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(0.25)
        except KeyboardInterrupt:
            pass
        return

    fps = float(vid.get("fps", 12.0))
    frame_period = 1.0 / fps
    frame_q: queue.Queue[Tuple[int, np.ndarray]] = queue.Queue(maxsize=args.qsize)
    stop_evt = threading.Event()

    video_iter = iter_video_records(args.input)
    dec = DecoderThread(tok, video_iter, frame_q, stop_evt)
    dec.start()

    start_wall = time.time()
    start_ts: int | None = None

    root = tk.Tk()
    root.title("NEUTXT Player")
    lbl = tk.Label(root)
    lbl.pack()

    state = {"last_imgtk": None}

    def tick():
        nonlocal start_ts

        if dec.err:
            stop_evt.set()
            print(f"Decoder error: {dec.err}")
            root.destroy()
            return

        target_elapsed = time.time() - start_wall

        while True:
            try:
                ts_ms, frame = frame_q.get_nowait()
            except queue.Empty:
                break

            if start_ts is None:
                start_ts = ts_ms

            vtime = (ts_ms - start_ts) / 1000.0
            if vtime <= target_elapsed + 0.02:
                im = Image.fromarray(frame, mode="RGB")
                if args.scale > 1:
                    im = im.resize(
                        (im.size[0] * args.scale, im.size[1] * args.scale),
                        resample=Image.LANCZOS)
                if args.sharpen:
                    im = im.filter(ImageFilter.UnsharpMask(
                        radius=1, percent=args.sharpen_pct, threshold=3))
                imgtk = ImageTk.PhotoImage(image=im)
                lbl.configure(image=imgtk)
                state["last_imgtk"] = imgtk
            else:
                frame_q.put((ts_ms, frame))
                break

        root.after(int(frame_period * 1000), tick)

    def on_close():
        stop_evt.set()
        if _HAVE_SD:
            sd.stop()
        try:
            root.destroy()
        except Exception:
            pass

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(0, tick)
    root.mainloop()

    stop_evt.set()
    dec.join(timeout=1.0)


if __name__ == "__main__":
    main()
