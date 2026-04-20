"""NEUTXT + Claude API demo: send media as text, have Claude manipulate it.

Claude can't "see" what's in the base85 payloads, but it CAN manipulate the
structured text format — reversing frames, extracting keyframes, muting
audio sections, stripping streams, duplicating chunks, etc. The result
decodes back to valid media.

This proves the core value proposition: media-as-text that flows through any
text-based system, including LLM context windows.

Usage:
    # Video-only (default task: describe)
    python -m neutxt.llm_demo input.mp4 \\
        --vq_ckpt ckpt.pt --vq_config cfg.yaml --task reverse

    # Audiovisual
    python -m neutxt.llm_demo input.mp4 \\
        --vq_ckpt ckpt.pt --vq_config cfg.yaml --mode av --task strip_audio

    # Audio-only
    python -m neutxt.llm_demo input.mp4 --mode a --task reverse_audio

    # Available tasks: reverse, keyframes, slowmo, describe, freeform,
    #                  strip_audio, strip_video, reverse_audio, audio_loop

Requires: ANTHROPIC_API_KEY environment variable
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import Iterator

import numpy as np

from neutxt.vq import (
    pick_device, load_magvit2,
    encode_frame_to_packed_codes, decode_packed_codes_to_frame,
)
from neutxt.text_codec import (
    encode_av_to_text, decode_text_av, text_stats,
    HEADER_START, HEADER_END,
)
from neutxt.audio_codec import (
    load_encodec, load_audio_mono_pcm, pcm_to_wav_bytes,
    encode_audio_to_codes, decode_codes_to_audio,
    BANDWIDTH_TO_QUANTIZERS,
)


# ---------------------------------------------------------------------------
# System prompt that teaches Claude the NEUTXT format
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a media processing assistant. You work with NEUTXT v2, a text-based \
format for neural-encoded video and audio. Here is how the format works:

STRUCTURE:
- Header block between "--- NEUTXT v2 ---" and "---" lines, with key: value metadata
- After the header, one line per media chunk in temporal order
- K: prefix = video keyframe (self-contained)
- D: prefix = video delta frame (XOR difference from previous reconstructed frame)
- A:<timesteps>: prefix = audio chunk (independent, self-contained)

RULES YOU MUST FOLLOW:
1. Always preserve the header block exactly, updating only the counts \
("video_frames:", "audio_chunks:", "frames:") and "mode:" when the content changes
2. K:, D:, and A: payloads are base85-encoded compressed binary — do NOT modify \
payload content, only reorder/filter/duplicate whole lines
3. Video: the first video line MUST be a K:. If you remove a K:, all following \
D: lines until the next K: become invalid
4. Audio: A: chunks are INDEPENDENT — any order, any subset is valid. \
Safe to reorder, drop, or duplicate freely.
5. When reordering video, either work with complete groups (one K: + all \
following D: until the next K:) or use input where every video line is K: already

MODE VALUES:
- "v" = video only (K:, D: lines)
- "a" = audio only (A: lines)
- "av" = both (K:, D:, A: lines)
When you strip video, set mode to "a". When you strip audio, set mode to "v".

SAFE OPERATIONS:
- Reverse video frame order (if all are K:, simple reverse; else by groups)
- Extract only K: keyframes
- Duplicate video frames for slow-motion (preserve prefixes)
- Remove audio lines entirely → pure-video output (update mode to "v")
- Remove video lines entirely → pure-audio output (update mode to "a")
- Reorder audio chunks freely (they're independent)
- Duplicate audio chunks for looping
- Remove trailing frames (truncate)

UNSAFE OPERATIONS:
- Editing any base85 payload content
- Removing a K: frame while keeping its dependent D: frames
- Mixing chunks from different NEUTXT files with different models

When you return modified NEUTXT, output ONLY the NEUTXT text — no commentary \
before or after. Start with "--- NEUTXT v2 ---" and end with the last line.\
"""


# ---------------------------------------------------------------------------
# Task-specific prompts
# ---------------------------------------------------------------------------

TASK_PROMPTS = {
    "reverse": (
        "Reverse the temporal order of the VIDEO frames in this NEUTXT. "
        "All video frames are keyframes (K:), so simply reverse their order. "
        "If audio (A:) lines are present, also reverse their order so audio "
        "stays synchronized with video. Update the header counts. "
        "Return only the modified NEUTXT text."
    ),
    "keyframes": (
        "Extract only the video keyframes (K: lines) from this NEUTXT, "
        "discarding all delta (D:) and audio (A:) lines. Set mode to 'v'. "
        "Update video_frames in the header. Return only the modified NEUTXT text."
    ),
    "slowmo": (
        "Create a 2x slow-motion version by duplicating each video frame line. "
        "For each K:/D: line, output it twice in sequence. Keep prefixes as-is. "
        "Also duplicate each A: audio line to match. Halve the fps value in the "
        "header. Update video_frames and audio_chunks. Return only the modified NEUTXT."
    ),
    "strip_audio": (
        "Remove all audio (A:) lines from this NEUTXT. Keep all video (K:/D:) "
        "lines in order. Set mode to 'v'. Update audio_chunks to 0. "
        "Return only the modified NEUTXT text."
    ),
    "strip_video": (
        "Remove all video (K:/D:) lines from this NEUTXT. Keep all audio (A:) "
        "lines in order. Set mode to 'a'. Update video_frames to 0. "
        "Return only the modified NEUTXT text."
    ),
    "reverse_audio": (
        "Reverse the order of A: (audio) lines only. Leave K:/D: video lines "
        "untouched in their original order. Audio chunks are independent so "
        "reversing is safe. Return only the modified NEUTXT text."
    ),
    "audio_loop": (
        "Duplicate each A: audio line to create a 2x longer audio stream. "
        "For each A: line, output it twice in sequence. Leave video lines "
        "untouched. Update audio_chunks. Return only the modified NEUTXT text."
    ),
    "describe": (
        "I'm sending you media encoded in NEUTXT v2 text format. "
        "You cannot decode the perceptual content from the base85 payloads, "
        "but analyze what you CAN determine from the format:\n"
        "- What mode (v/a/av)? What video and audio models?\n"
        "- How many video frames / audio chunks? Duration? FPS?\n"
        "- Ratio of keyframes vs delta frames?\n"
        "- Approximate data size and compression?\n"
        "- How do payload sizes vary between K:, D:, and A: lines?\n"
        "Provide a concise technical analysis."
    ),
}


# ---------------------------------------------------------------------------
# Media helpers
# ---------------------------------------------------------------------------

def _ffmpeg_frames(path: str, fps: float, width: int, height: int,
                   max_seconds: float | None = None) -> Iterator[np.ndarray]:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path]
    if max_seconds:
        cmd.extend(["-t", str(max_seconds)])
    cmd.extend([
        "-vf", f"fps={fps},scale={width}:{height}:flags=lanczos",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ])
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert p.stdout is not None
    n = width * height * 3
    while True:
        buf = p.stdout.read(n)
        if not buf or len(buf) != n:
            break
        yield np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 3))
    p.stdout.close()
    p.wait()


def _save_gif(frames: list[np.ndarray], path: str, fps: float):
    from PIL import Image
    imgs = [Image.fromarray(f, "RGB") for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0, optimize=True)


def _chunk_audio_by_time(codes: np.ndarray, pcm_len: int, sample_rate: int,
                          chunk_seconds: float) -> list[np.ndarray]:
    n_q, total_T = codes.shape
    total_duration = pcm_len / sample_rate
    if total_duration <= chunk_seconds:
        return [codes]
    frames_per_sec = total_T / total_duration
    step = max(1, int(round(frames_per_sec * chunk_seconds)))
    return [codes[:, s:s + step].copy() for s in range(0, total_T, step)]


# ---------------------------------------------------------------------------
# Claude API call
# ---------------------------------------------------------------------------

def call_claude(neutxt_text: str, task_prompt: str) -> tuple[str, int, int]:
    """Send NEUTXT text to Claude and get the response."""
    import anthropic

    client = anthropic.Anthropic()
    user_message = f"{task_prompt}\n\nHere is the NEUTXT media:\n\n{neutxt_text}"

    print(f"  Sending to Claude API ({len(user_message):,} chars) ...")
    t0 = time.time()

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=16384,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    elapsed = time.time() - t0
    result = response.content[0].text

    print(f"  Response received in {elapsed:.1f}s")
    print(f"  Input tokens:  {response.usage.input_tokens:,}")
    print(f"  Output tokens: {response.usage.output_tokens:,}")

    return result, response.usage.input_tokens, response.usage.output_tokens


def _extract_neutxt(response_text: str) -> str | None:
    """Pull out the NEUTXT block from Claude's response, ignoring any commentary."""
    if HEADER_START not in response_text:
        return None
    start = response_text.index(HEADER_START)
    lines = response_text[start:].split("\n")
    clean: list[str] = []
    in_header = False
    header_done = False
    for line in lines:
        s = line.strip()
        if s == HEADER_START:
            in_header = True
            clean.append(line)
        elif in_header and s == HEADER_END:
            in_header = False
            header_done = True
            clean.append(line)
        elif in_header:
            clean.append(line)
        elif header_done and (s.startswith("K:") or s.startswith("D:") or s.startswith("A:")):
            clean.append(line)
        elif header_done and s and not s.startswith(("K:", "D:", "A:")):
            break  # trailing commentary
    return "\n".join(clean)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="NEUTXT + Claude API demo")
    ap.add_argument("input", help="input media file")
    ap.add_argument("--mode", default="v", choices=["v", "a", "av"],
                    help="v=video only, a=audio only, av=both")
    ap.add_argument("--vq_ckpt", help="MAGVIT2 checkpoint (required for v/av)")
    ap.add_argument("--vq_config", help="MAGVIT2 YAML config (required for v/av)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--keyint", type=int, default=8)
    ap.add_argument("--audio_sr", type=int, default=24000, choices=[24000, 48000])
    ap.add_argument("--audio_bandwidth", type=float, default=6.0,
                    choices=list(BANDWIDTH_TO_QUANTIZERS.keys()))
    ap.add_argument("--audio_chunk_seconds", type=float, default=1.0)
    ap.add_argument("--compression", default="zstd", choices=["zlib", "zstd"])
    ap.add_argument("--task", default="describe", choices=list(TASK_PROMPTS.keys()) + ["freeform"])
    ap.add_argument("--prompt", default=None, help="custom prompt (required for --task freeform)")
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()

    if args.task == "freeform" and not args.prompt:
        ap.error("--prompt required when --task is freeform")
    if args.mode in ("v", "av") and not (args.vq_ckpt and args.vq_config):
        ap.error("--vq_ckpt and --vq_config are required for mode 'v' or 'av'")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    # Reordering tasks need every video frame to be a keyframe
    REORDER_TASKS = {"reverse", "freeform"}
    if args.mode in ("v", "av") and args.task in REORDER_TASKS and args.keyint != 1:
        print(f"  NOTE: forcing --keyint 1 for '{args.task}' task "
              "(reordering requires all keyframes)")
        args.keyint = 1

    basename = os.path.splitext(os.path.basename(args.input))[0]
    device = pick_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    tok = None
    enc = None
    video_codes: list[np.ndarray] = []
    audio_codes_chunks: list[np.ndarray] = []

    # --- Step 1: Encode ---
    if args.mode in ("v", "av"):
        print("\n[1/5] Loading MAGVIT2 ...")
        tok = load_magvit2(args.vq_ckpt, args.vq_config, device=device, res=256)

        print(f"[2/5] Encoding video ({args.seconds}s @ {args.fps}fps) ...")
        for frame in _ffmpeg_frames(args.input, args.fps, 256, 256, args.seconds):
            codes, _, _ = encode_frame_to_packed_codes(tok, frame)
            video_codes.append(codes)
        print(f"  {len(video_codes)} video frames encoded")

    if args.mode in ("a", "av"):
        print(f"\n[1b/5] Loading EnCodec ({args.audio_sr}Hz, {args.audio_bandwidth}kbps) ...")
        enc = load_encodec(device, sample_rate=args.audio_sr, bandwidth=args.audio_bandwidth)

        print(f"[2b/5] Encoding audio ...")
        pcm = load_audio_mono_pcm(args.input, args.audio_sr, args.seconds)
        if len(pcm) == 0:
            if args.mode == "a":
                raise RuntimeError(f"No audio stream in {args.input}")
            print("  WARNING: no audio track found, falling back to video-only")
            args.mode = "v"
        else:
            codes = encode_audio_to_codes(enc, pcm)
            audio_codes_chunks = _chunk_audio_by_time(
                codes, len(pcm), args.audio_sr, args.audio_chunk_seconds)
            print(f"  {len(audio_codes_chunks)} audio chunks encoded")

    neutxt_text = encode_av_to_text(
        video_codes=video_codes if video_codes else None,
        audio_codes=audio_codes_chunks if audio_codes_chunks else None,
        fps=args.fps, keyint=args.keyint,
        audio_sr=args.audio_sr, audio_bandwidth=args.audio_bandwidth,
        audio_quantizers=BANDWIDTH_TO_QUANTIZERS.get(args.audio_bandwidth, 8),
        audio_chunk_seconds=args.audio_chunk_seconds,
        compression=args.compression, level=9,
    )

    stats = text_stats(neutxt_text)
    print(f"\n  NEUTXT: mode={stats['mode']}, "
          f"{stats['video_frames']} video + {stats['audio_chunks']} audio, "
          f"{len(neutxt_text):,} chars (~{stats['approx_llm_tokens']:,} LLM tokens)")

    # --- Step 2: Save original decode for comparison ---
    print("\n[3/5] Saving original decode ...")
    if video_codes and tok is not None:
        orig_frames = [decode_packed_codes_to_frame(tok, c, 16, 16) for c in video_codes]
        orig_gif = os.path.join(args.out_dir, f"{basename}_original.gif")
        _save_gif(orig_frames, orig_gif, args.fps)
        print(f"  Saved: {orig_gif}")
    if audio_codes_chunks and enc is not None:
        merged = np.concatenate(audio_codes_chunks, axis=1)
        pcm_out = decode_codes_to_audio(enc, merged)
        orig_wav = os.path.join(args.out_dir, f"{basename}_original.wav")
        with open(orig_wav, "wb") as f:
            f.write(pcm_to_wav_bytes(pcm_out, args.audio_sr))
        print(f"  Saved: {orig_wav}")

    # --- Step 3: Send to Claude ---
    task_prompt = args.prompt if args.task == "freeform" else TASK_PROMPTS[args.task]
    print(f"\n[4/5] Calling Claude API (task: {args.task}) ...")
    response_text, _, _ = call_claude(neutxt_text, task_prompt)

    # --- Step 4: Process response ---
    print(f"\n[5/5] Processing Claude's response ...")

    if args.task == "describe":
        print("\n" + "=" * 60)
        print("  CLAUDE'S ANALYSIS")
        print("=" * 60)
        print(response_text)
        print("=" * 60)
        return

    neutxt_output = _extract_neutxt(response_text)
    if neutxt_output is None:
        print("  WARNING: Claude's response doesn't contain NEUTXT format")
        print(response_text[:500])
        return

    try:
        out_stats = text_stats(neutxt_output)
        print(f"  Modified NEUTXT: {len(neutxt_output):,} chars, "
              f"mode={out_stats['mode']}, "
              f"{out_stats['video_frames']} video + {out_stats['audio_chunks']} audio")

        header, decoded_video, decoded_audio = decode_text_av(neutxt_output)

        if decoded_video and tok is not None:
            frames = [decode_packed_codes_to_frame(tok, c, 16, 16) for c in decoded_video]
            out_gif = os.path.join(args.out_dir, f"{basename}_{args.task}.gif")
            out_fps = header.fps if header.fps > 0 else args.fps
            _save_gif(frames, out_gif, out_fps)
            print(f"  Saved: {out_gif}")

        if decoded_audio and enc is not None:
            merged = np.concatenate(decoded_audio, axis=1)
            pcm_out = decode_codes_to_audio(enc, merged)
            out_wav = os.path.join(args.out_dir, f"{basename}_{args.task}.wav")
            with open(out_wav, "wb") as f:
                f.write(pcm_to_wav_bytes(pcm_out, args.audio_sr))
            print(f"  Saved: {out_wav} ({len(pcm_out)/args.audio_sr:.2f}s)")

    except Exception as e:
        print(f"  Error decoding Claude's output: {e}")
        err_path = os.path.join(args.out_dir, f"{basename}_{args.task}_raw.txt")
        with open(err_path, "w") as f:
            f.write(response_text)
        print(f"  Raw response saved to: {err_path}")
        return

    print()
    print("=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Task:        {args.task}")
    print(f"  Input mode:  {stats['mode']}  →  Output mode: {out_stats['mode']}")
    print(f"  Video:       {stats['video_frames']} → {out_stats['video_frames']} frames")
    print(f"  Audio:       {stats['audio_chunks']} → {out_stats['audio_chunks']} chunks")
    print("=" * 60)
    print()
    print("The media was transmitted as text through Claude's context window,")
    print("manipulated by the LLM, and decoded back to valid output.")
    print()


if __name__ == "__main__":
    main()
