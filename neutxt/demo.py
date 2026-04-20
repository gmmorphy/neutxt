"""NEUTXT Demo: encode media to text, show stats, decode back.

Usage:
    # Video + audio (default)
    python -m neutxt.demo input.mp4 --vq_ckpt ckpt.pt --vq_config cfg.yaml

    # Video only
    python -m neutxt.demo input.mp4 --vq_ckpt ckpt.pt --vq_config cfg.yaml --mode v

    # Audio only (no MAGVIT2 needed)
    python -m neutxt.demo input.mp3 --mode a

    # Control duration and quality
    python -m neutxt.demo input.mp4 --vq_ckpt ckpt.pt --vq_config cfg.yaml \\
        --seconds 3 --fps 8 --keyint 8 --audio_bandwidth 6.0
"""
from __future__ import annotations

import argparse
import os
import subprocess
from typing import Iterator

import numpy as np

from neutxt.vq import pick_device, load_magvit2, encode_frame_to_packed_codes, decode_packed_codes_to_frame
from neutxt.text_codec import encode_av_to_text, decode_text_av, text_stats
from neutxt.audio_codec import (
    load_encodec, load_audio_mono_pcm, pcm_to_wav_bytes,
    encode_audio_to_codes, decode_codes_to_audio,
    BANDWIDTH_TO_QUANTIZERS,
)


def _ffmpeg_frames(path: str, fps: float, width: int, height: int,
                   max_seconds: float | None = None) -> Iterator[np.ndarray]:
    vf = f"fps={fps},scale={width}:{height}:flags=lanczos"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path]
    if max_seconds:
        cmd.extend(["-t", str(max_seconds)])
    cmd.extend(["-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])

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


def _save_gif(frames: list[np.ndarray], path: str, fps: float):
    from PIL import Image
    images = [Image.fromarray(f, "RGB") for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)


def _chunk_audio_codes(codes: np.ndarray, sample_rate: int,
                        chunk_seconds: float) -> list[np.ndarray]:
    """Split (n_q, T) audio codes into fixed-duration chunks.

    EnCodec operates at ~75 frames/sec for 24kHz, but the exact ratio depends
    on the model. We derive it from the code length and known PCM duration.
    """
    # We don't know EnCodec's internal downsampling ratio a priori; split
    # the time axis proportionally instead.
    n_q, total_T = codes.shape
    return [codes]  # single chunk is fine for most clips; format supports more


def _chunk_audio_by_time(codes: np.ndarray, pcm_len: int, sample_rate: int,
                          chunk_seconds: float) -> list[np.ndarray]:
    """Split codes into chunks of approximately chunk_seconds each."""
    n_q, total_T = codes.shape
    total_duration = pcm_len / sample_rate
    if total_duration <= chunk_seconds:
        return [codes]
    frames_per_sec = total_T / total_duration
    step = max(1, int(round(frames_per_sec * chunk_seconds)))
    chunks = []
    for start in range(0, total_T, step):
        chunks.append(codes[:, start:start + step].copy())
    return chunks


def main():
    ap = argparse.ArgumentParser(description="NEUTXT text-format demo")
    ap.add_argument("input", help="input media file")
    ap.add_argument("--mode", default="av", choices=["v", "a", "av"],
                    help="v=video only, a=audio only, av=both (default)")
    ap.add_argument("--vq_ckpt", help="MAGVIT2 checkpoint path (required for v/av)")
    ap.add_argument("--vq_config", help="MAGVIT2 YAML config (required for v/av)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--seconds", type=float, default=3.0, help="max seconds to encode")
    ap.add_argument("--keyint", type=int, default=8, help="video keyframe interval")
    ap.add_argument("--audio_sr", type=int, default=24000, choices=[24000, 48000])
    ap.add_argument("--audio_bandwidth", type=float, default=6.0,
                    choices=list(BANDWIDTH_TO_QUANTIZERS.keys()))
    ap.add_argument("--audio_chunk_seconds", type=float, default=1.0)
    ap.add_argument("--compression", default="zstd", choices=["zlib", "zstd"])
    ap.add_argument("--level", type=int, default=9)
    ap.add_argument("--out_dir", default=".", help="output directory")
    ap.add_argument("--save_text", action="store_true", help="save .neutxt text file")
    ap.add_argument("--print_text", action="store_true")
    args = ap.parse_args()

    if args.mode in ("v", "av") and not (args.vq_ckpt and args.vq_config):
        ap.error("--vq_ckpt and --vq_config are required for mode 'v' or 'av'")

    device = pick_device(args.device)
    basename = os.path.splitext(os.path.basename(args.input))[0]
    os.makedirs(args.out_dir, exist_ok=True)

    tok = None
    enc = None
    video_codes: list[np.ndarray] = []
    audio_codes_chunks: list[np.ndarray] = []
    pcm_len = 0

    # --- Step 1a: Encode video ---
    if args.mode in ("v", "av"):
        print(f"Loading MAGVIT2 on {device} ...")
        tok = load_magvit2(args.vq_ckpt, args.vq_config, device=device, res=256)

        print(f"Encoding video frames (max {args.seconds}s @ {args.fps}fps) ...")
        for i, frame in enumerate(_ffmpeg_frames(args.input, args.fps, 256, 256, args.seconds)):
            codes, _, _ = encode_frame_to_packed_codes(tok, frame)
            video_codes.append(codes)
            if (i + 1) % 10 == 0:
                print(f"  encoded {i+1} video frames")
        print(f"  total: {len(video_codes)} video frames")

    # --- Step 1b: Encode audio ---
    if args.mode in ("a", "av"):
        print(f"Loading EnCodec ({args.audio_sr} Hz, {args.audio_bandwidth} kbps) ...")
        enc = load_encodec(device, sample_rate=args.audio_sr, bandwidth=args.audio_bandwidth)

        print(f"Loading audio (max {args.seconds}s) ...")
        pcm = load_audio_mono_pcm(args.input, args.audio_sr, args.seconds)
        pcm_len = len(pcm)
        if pcm_len == 0:
            if args.mode == "a":
                raise RuntimeError(f"No audio stream found in {args.input}")
            print("  WARNING: no audio track found, skipping audio encoding")
        else:
            print(f"Encoding audio ({pcm_len/args.audio_sr:.2f}s) ...")
            codes = encode_audio_to_codes(enc, pcm)
            print(f"  audio codes shape: {codes.shape}")
            audio_codes_chunks = _chunk_audio_by_time(
                codes, pcm_len, args.audio_sr, args.audio_chunk_seconds)
            print(f"  split into {len(audio_codes_chunks)} chunks")

    # --- Step 2: Serialize to NEUTXT text ---
    print("Generating NEUTXT text ...")
    effective_mode = args.mode
    if args.mode == "av" and not audio_codes_chunks:
        effective_mode = "v"

    neutxt_text = encode_av_to_text(
        video_codes=video_codes if video_codes else None,
        audio_codes=audio_codes_chunks if audio_codes_chunks else None,
        fps=args.fps,
        keyint=args.keyint,
        audio_sr=args.audio_sr,
        audio_bandwidth=args.audio_bandwidth,
        audio_quantizers=BANDWIDTH_TO_QUANTIZERS.get(args.audio_bandwidth, 8),
        audio_chunk_seconds=args.audio_chunk_seconds,
        compression=args.compression,
        level=args.level,
    )

    stats = text_stats(neutxt_text)

    # --- Step 3: Show stats ---
    orig_size = os.path.getsize(args.input)
    text_size = len(neutxt_text.encode("utf-8"))

    print()
    print("=" * 60)
    print("  NEUTXT TEXT ENCODING RESULTS")
    print("=" * 60)
    print(f"  Source:            {os.path.basename(args.input)} ({orig_size/1024:.1f} KB)")
    print(f"  Mode:              {stats['mode']}")
    print(f"  Duration:          {stats['duration_sec']}s")
    if stats["video_frames"]:
        print(f"  Video frames:      {stats['video_frames']} "
              f"({stats['keyframes']} key + {stats['delta_frames']} delta) @ {args.fps}fps")
    if stats["audio_chunks"]:
        print(f"  Audio chunks:      {stats['audio_chunks']} "
              f"@ {args.audio_sr}Hz, {args.audio_bandwidth}kbps")
    print(f"  NEUTXT text size:  {text_size:,} characters ({text_size/1024:.1f} KB)")
    print(f"  ~LLM tokens:       ~{stats['approx_llm_tokens']:,}")
    print(f"  Compression ratio: {orig_size/text_size:.1f}x vs original")
    print("=" * 60)
    print()

    llm_tokens = stats["approx_llm_tokens"]
    if llm_tokens < 100000:
        print(f"  Fits in: Claude (200K), Gemini (1M), GPT-4o (128K)")
    else:
        print(f"  Fits in: Gemini (1M) only — consider reducing --seconds or --fps")
    print()

    if args.print_text:
        print("--- NEUTXT TEXT START ---")
        print(neutxt_text)
        print("--- NEUTXT TEXT END ---")
        print()

    if args.save_text:
        text_path = os.path.join(args.out_dir, f"{basename}.neutxt.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(neutxt_text)
        print(f"  Saved: {text_path}")

    # --- Step 4: Decode back ---
    print("Decoding NEUTXT text back to media ...")
    header, decoded_video_codes, decoded_audio_chunks = decode_text_av(neutxt_text)

    if decoded_video_codes and tok is not None:
        decoded_frames = [decode_packed_codes_to_frame(tok, c, 16, 16)
                          for c in decoded_video_codes]
        gif_path = os.path.join(args.out_dir, f"{basename}_neutxt.gif")
        _save_gif(decoded_frames, gif_path, args.fps)
        print(f"  Saved: {gif_path}")

    if decoded_audio_chunks and enc is not None:
        merged = np.concatenate(decoded_audio_chunks, axis=1)
        pcm_out = decode_codes_to_audio(enc, merged)
        wav_bytes = pcm_to_wav_bytes(pcm_out, args.audio_sr)
        wav_path = os.path.join(args.out_dir, f"{basename}_neutxt.wav")
        with open(wav_path, "wb") as f:
            f.write(wav_bytes)
        print(f"  Saved: {wav_path}  ({len(pcm_out)/args.audio_sr:.2f}s)")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
