"""NEUTXT + Claude API demo: send video as text, have Claude manipulate it.

Claude can't "see" what's in the base85 payloads, but it CAN manipulate the
structured text format — reversing frames, extracting keyframes, duplicating
for slow-motion, etc. The result decodes back to valid video.

This proves the core value proposition: video-as-text that flows through any
text-based system, including LLM context windows.

Usage:
    # Basic: encode video, send to Claude, decode result
    python -m neutxt.llm_demo input.mp4 \\
        --vq_ckpt path/to/ckpt.pt --vq_config path/to/cfg.yaml \\
        --task reverse

    # Available tasks: reverse, keyframes, slowmo, describe, freeform
    python -m neutxt.llm_demo input.mp4 \\
        --vq_ckpt ckpt.pt --vq_config cfg.yaml \\
        --task freeform --prompt "Keep only every 3rd frame"

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
    encode_frames_to_text, decode_text_to_frames, text_stats,
    HEADER_START, HEADER_END,
)


# ---------------------------------------------------------------------------
# System prompt that teaches Claude the NEUTXT format
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a video processing assistant. You work with NEUTXT v2, a text-based \
video encoding format. Here is how the format works:

STRUCTURE:
- Header block between "--- NEUTXT v2 ---" and "---" lines, with key: value metadata
- Frame lines after the header, one per video frame in temporal order
- K: prefix = keyframe (self-contained)
- D: prefix = delta frame (XOR difference from previous reconstructed frame)

RULES YOU MUST FOLLOW:
1. Always preserve the header block exactly, except update "frames:" count
2. K and D frames contain base85-encoded compressed binary data — do NOT modify payload content
3. The first frame MUST always be a K: (keyframe)
4. After any K: frame, subsequent D: frames depend on it — if you remove a K: frame, \
all D: frames until the next K: become invalid
5. When reordering, the safest operation is to work with complete "groups" \
(one K: frame + all following D: frames until the next K:)

SAFE OPERATIONS (you can do these):
- Reverse frame order (but first frame must still be K: — so re-label it)
- Extract only K: keyframes (always safe)
- Duplicate frames for slow-motion
- Remove frames from the end (truncate)
- Keep every Nth frame (but must start with a K:)
- Concatenate two NEUTXT sequences

UNSAFE OPERATIONS (will produce corrupt output):
- Editing the base85 payload content
- Removing a K: frame while keeping its dependent D: frames
- Mixing frames from different NEUTXT files with different models

When you return modified NEUTXT, output ONLY the NEUTXT text — no commentary \
before or after. Start with "--- NEUTXT v2 ---" and end with the last frame line.\
"""


# ---------------------------------------------------------------------------
# Task-specific prompts
# ---------------------------------------------------------------------------

TASK_PROMPTS = {
    "reverse": (
        "Reverse the temporal order of the frames in this NEUTXT video. "
        "All frames are keyframes (K:), so simply reverse their order. "
        "Update the header accordingly. Return only the modified NEUTXT text."
    ),
    "keyframes": (
        "Extract only the keyframes (K: lines) from this NEUTXT video, "
        "discarding all delta frames (D: lines). Update the frames count "
        "in the header. Return only the modified NEUTXT text."
    ),
    "slowmo": (
        "Create a 2x slow-motion version by duplicating each frame line. "
        "For each frame, output it twice in sequence. Keep prefixes as-is "
        "(K: stays K:, D: stays D:). Halve the fps value in the header. "
        "Update the frames count. Return only the modified NEUTXT text."
    ),
    "describe": (
        "I'm sending you a video encoded in NEUTXT v2 text format. "
        "You cannot decode the visual content from the base85 payloads, "
        "but analyze what you CAN determine from the format:\n"
        "- How many frames? Duration? FPS?\n"
        "- How many keyframes vs delta frames?\n"
        "- What's the approximate data size?\n"
        "- What compression and model are used?\n"
        "- How do the payload sizes vary between keyframes and delta frames?\n"
        "Provide a concise technical analysis."
    ),
}


# ---------------------------------------------------------------------------
# Video frame extraction
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


# ---------------------------------------------------------------------------
# Claude API call
# ---------------------------------------------------------------------------

def call_claude(neutxt_text: str, task_prompt: str) -> str:
    """Send NEUTXT text to Claude and get the response."""
    import anthropic

    client = anthropic.Anthropic()

    user_message = f"{task_prompt}\n\nHere is the NEUTXT video:\n\n{neutxt_text}"

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

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="NEUTXT + Claude API demo")
    ap.add_argument("input", help="input video file")
    ap.add_argument("--vq_ckpt", required=True, help="MAGVIT2 checkpoint")
    ap.add_argument("--vq_config", required=True, help="MAGVIT2 YAML config")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--keyint", type=int, default=8)
    ap.add_argument("--compression", default="zstd", choices=["zlib", "zstd"])
    ap.add_argument("--task", default="describe",
                    choices=["reverse", "keyframes", "slowmo", "describe", "freeform"])
    ap.add_argument("--prompt", default=None,
                    help="custom prompt (required for --task freeform)")
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()

    if args.task == "freeform" and not args.prompt:
        print("Error: --prompt required when --task is freeform")
        sys.exit(1)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: set ANTHROPIC_API_KEY environment variable")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    # Reordering tasks need every frame to be a keyframe (delta frames
    # are XOR differences that become invalid when moved out of order)
    REORDER_TASKS = {"reverse", "freeform"}
    if args.task in REORDER_TASKS and args.keyint != 1:
        print(f"  NOTE: forcing --keyint 1 for '{args.task}' task "
              "(reordering requires all keyframes)")
        args.keyint = 1

    basename = os.path.splitext(os.path.basename(args.input))[0]
    device = pick_device(args.device)

    # --- Step 1: Encode ---
    print("\n[1/5] Loading MAGVIT2 ...")
    tok = load_magvit2(args.vq_ckpt, args.vq_config, device=device, res=256)

    print(f"[2/5] Encoding video ({args.seconds}s @ {args.fps}fps) ...")
    all_codes: list[np.ndarray] = []
    for i, frame in enumerate(_ffmpeg_frames(args.input, args.fps, 256, 256, args.seconds)):
        codes, _, _ = encode_frame_to_packed_codes(tok, frame)
        all_codes.append(codes)
    print(f"  {len(all_codes)} frames encoded")

    neutxt_text = encode_frames_to_text(
        all_codes, fps=args.fps, keyint=args.keyint,
        compression=args.compression, level=9)

    stats = text_stats(neutxt_text)
    print(f"  NEUTXT: {len(neutxt_text):,} chars (~{stats['approx_llm_tokens']:,} LLM tokens)")

    # Save original GIF for comparison
    print("[3/5] Saving original decode ...")
    orig_frames = []
    for codes in all_codes:
        orig_frames.append(decode_packed_codes_to_frame(tok, codes, 16, 16))
    orig_gif = os.path.join(args.out_dir, f"{basename}_original.gif")
    _save_gif(orig_frames, orig_gif, args.fps)
    print(f"  Saved: {orig_gif}")

    # --- Step 2: Send to Claude ---
    task_prompt = args.prompt if args.task == "freeform" else TASK_PROMPTS[args.task]

    print(f"\n[4/5] Calling Claude API (task: {args.task}) ...")
    response_text = call_claude(neutxt_text, task_prompt)

    # --- Step 3: Process response ---
    print(f"\n[5/5] Processing Claude's response ...")

    if args.task == "describe":
        print("\n" + "=" * 60)
        print("  CLAUDE'S ANALYSIS OF THE NEUTXT VIDEO")
        print("=" * 60)
        print(response_text)
        print("=" * 60)
        print(f"\nOriginal GIF: {orig_gif}")
        print("Done!")
        return

    # For manipulation tasks, decode Claude's output back to video
    # Extract NEUTXT text from response (Claude might add commentary despite instructions)
    if HEADER_START in response_text:
        start = response_text.index(HEADER_START)
        neutxt_output = response_text[start:]
        # Trim any trailing non-NEUTXT text
        lines = neutxt_output.split("\n")
        clean_lines = []
        in_header = False
        header_done = False
        for line in lines:
            stripped = line.strip()
            if stripped == HEADER_START:
                in_header = True
                clean_lines.append(line)
            elif in_header and stripped == HEADER_END:
                in_header = False
                header_done = True
                clean_lines.append(line)
            elif in_header:
                clean_lines.append(line)
            elif header_done and (stripped.startswith("K:") or stripped.startswith("D:")):
                clean_lines.append(line)
            elif header_done and stripped and not stripped.startswith(("K:", "D:")):
                break  # trailing commentary
        neutxt_output = "\n".join(clean_lines)
    else:
        print("  WARNING: Claude's response doesn't contain NEUTXT format")
        print("  Response preview:")
        print(response_text[:500])
        return

    try:
        out_stats = text_stats(neutxt_output)
        print(f"  Modified NEUTXT: {len(neutxt_output):,} chars, {out_stats['frames']} frames")

        header, decoded_codes = decode_text_to_frames(neutxt_output)

        modified_frames = []
        for codes in decoded_codes:
            modified_frames.append(decode_packed_codes_to_frame(tok, codes, 16, 16))

        out_gif = os.path.join(args.out_dir, f"{basename}_{args.task}.gif")
        out_fps = header.fps if header.fps > 0 else args.fps
        _save_gif(modified_frames, out_gif, out_fps)
        print(f"  Saved: {out_gif}")

    except Exception as e:
        print(f"  Error decoding Claude's output: {e}")
        err_path = os.path.join(args.out_dir, f"{basename}_{args.task}_raw.txt")
        with open(err_path, "w") as f:
            f.write(response_text)
        print(f"  Raw response saved to: {err_path}")
        return

    # Summary
    print()
    print("=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Task:           {args.task}")
    print(f"  Original:       {len(all_codes)} frames → {orig_gif}")
    print(f"  After Claude:   {len(decoded_codes)} frames → {out_gif}")
    print(f"  Format:         Pure UTF-8 text, end to end")
    print("=" * 60)
    print()
    print("The video was transmitted as text through Claude's context window,")
    print("manipulated by the LLM, and decoded back to valid video.")
    print()


if __name__ == "__main__":
    main()
