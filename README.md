# NEUTXT

**Text-based Neural Media Encoding and Reconstruction System**

NEUTXT represents video **and audio** as self-describing UTF-8 text using neural codecs
(MAGVIT2 for video, EnCodec for audio). The resulting text can be pasted into LLM
context windows, stored in git repos, or transmitted through any text-based system —
then decoded back into valid media.

## Why

Modern neural codecs (MAGVIT2, EnCodec) compress media into compact latent tokens. NEUTXT
serializes those tokens as structured text, enabling:

- **Send video or audio to an LLM as part of a prompt** — have Claude/GPT manipulate it
- **Version-control media alongside code** — diffable, mergeable
- **Embed media in text-only systems** — chat, blockchain, databases
- **Synchronized A/V in one text stream** — `K:`/`D:` video lines interleaved with `A:` audio lines

## Quick Demo

```bash
# Video + audio (default): encode → text → decode to GIF + WAV
python -m neutxt demo input.mp4 --mode av \
  --vq_ckpt models/magvit2_256L.ckpt \
  --vq_config configs/imagenet_lfqgan_256_L.yaml

# Audio only (no MAGVIT2 needed)
python -m neutxt demo input.mp3 --mode a

# Send through Claude API, have it strip the audio track
python -m neutxt llm input.mp4 --mode av \
  --vq_ckpt models/magvit2_256L.ckpt \
  --vq_config configs/imagenet_lfqgan_256_L.yaml \
  --task strip_audio
```

## Size

- 2 seconds of 256×256 video + audio @ 8fps, 6 kbps → ~7,200 characters (~1,800 LLM tokens)
- 2 seconds of audio alone @ 6 kbps → ~2,000 characters (~500 LLM tokens)
- Fits comfortably in Claude 200K, GPT-4o 128K, Gemini 1M context windows

## Format

```
--- NEUTXT v2 ---
mode: av
video_model: MAGVIT2_256_L
fps: 8.0
resolution: 256x256
video_code_bits: 18
tokens_per_frame: 256
keyint: 8
video_frames: 16
audio_model: ENCODEC_24K
audio_sr: 24000
audio_bandwidth: 6.0
audio_quantizers: 8
audio_code_bits: 10
audio_chunk_seconds: 1.0
audio_chunks: 2
compression: zstd
checksum: a1b2c3d4
---
K:<base85 payload>      ← video keyframe
D:<base85 payload>      ← video delta (XOR vs previous)
A:<timesteps>:<base85>  ← audio chunk (independent)
...
```

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download MAGVIT2 checkpoint (~500MB)
python -c "from huggingface_hub import hf_hub_download; \
  hf_hub_download('TencentARC/Open-MAGVIT2', 'imagenet_256_L.ckpt', \
  local_dir='models')"

# For Claude API demos:
export ANTHROPIC_API_KEY=sk-ant-...
```

## Commands

| Command | Purpose |
|---|---|
| `python -m neutxt demo` | Encode video/audio → text → GIF + WAV (local only) |
| `python -m neutxt llm` | Encode → Claude API → decode back to media |
| `python -m neutxt mcp` | Run MCP server — gives any Claude Code / Claude desktop native `neutxt_*` tools |
| `python -m neutxt encode` | Encode to binary `.neutxt` container |
| `python -m neutxt play` | Play a `.neutxt` file (text or binary, auto-detected) |

## Use from Claude Code / Claude desktop (MCP)

Add this to your Claude MCP config (e.g. `~/.claude/mcp_servers.json`):

```json
{
  "neutxt": {
    "command": "/path/to/neutxt/.venv/bin/python",
    "args": ["-m", "neutxt", "mcp"],
    "env": {
      "NEUTXT_VQ_CKPT": "/path/to/magvit2_256L.ckpt",
      "NEUTXT_VQ_CONFIG": "/path/to/imagenet_lfqgan_256_L.yaml"
    }
  }
}
```

Then Claude can call these tools natively — no system prompt needed:

| Tool | Does |
|---|---|
| `neutxt_info` | Inspect metadata, duration, frame counts |
| `neutxt_trim` | Cut a time range |
| `neutxt_reverse` | Reverse playback (requires all keyframes) |
| `neutxt_strip_audio` / `neutxt_strip_video` | Remove one stream |
| `neutxt_concat` | Join multiple files |
| `neutxt_preview` | Decode frames to PNG so Claude can *see* the video |

## Status

- [x] Video encoding via MAGVIT2
- [x] UTF-8 text format with keyframes + XOR deltas
- [x] Claude API integration (reverse, keyframes, slowmo, describe, freeform)
- [x] Apple MPS / CUDA auto-detection
- [x] Audio encoding via EnCodec
- [x] Synchronized audiovisual mode (`--mode av`)
- [x] Audio-aware Claude tasks (strip_audio, strip_video, reverse_audio, audio_loop)

## License

TBD
