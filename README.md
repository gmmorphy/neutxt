# NEUTXT

**Text-based Neural Media Encoding and Reconstruction System**

NEUTXT represents video (and soon audio) as self-describing UTF-8 text using neural codecs.
The resulting text can be pasted into LLM context windows, stored in git repos, or transmitted
through any text-based system — then decoded back into valid media.

## Why

Modern neural codecs (MAGVIT2, EnCodec) compress media into compact latent tokens. NEUTXT
serializes those tokens as structured text, enabling:

- **Send video to an LLM as part of a prompt** — have Claude/GPT manipulate it
- **Version-control media alongside code** — diffable, mergeable
- **Embed media in text-only systems** — chat, blockchain, databases

## Quick Demo

```bash
# Encode video → NEUTXT text → decode to GIF
python -m neutxt demo input.mp4 \
  --vq_ckpt models/magvit2_256L.ckpt \
  --vq_config configs/imagenet_lfqgan_256_L.yaml

# Send through Claude API, have it reverse the video
python -m neutxt llm input.mp4 \
  --vq_ckpt models/magvit2_256L.ckpt \
  --vq_config configs/imagenet_lfqgan_256_L.yaml \
  --task reverse
```

## Size

- 3 seconds of 256×256 video @ 8fps → ~7,700 characters (~1,900 LLM tokens)
- Fits comfortably in Claude 200K, GPT-4o 128K, Gemini 1M context windows

## Format

```
--- NEUTXT v2 ---
model: MAGVIT2_256_L
fps: 8.0
resolution: 256x256
code_bits: 18
tokens_per_frame: 256
keyint: 8
compression: zstd
frames: 24
checksum: a1b2c3d4
---
K:<base85 payload>   ← keyframe
D:<base85 payload>   ← delta (XOR vs previous)
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
| `python -m neutxt demo` | Encode video → text → GIF (local only) |
| `python -m neutxt llm` | Encode → Claude API → decode back to GIF |
| `python -m neutxt encode` | Encode to binary `.neutxt` container |
| `python -m neutxt play` | Play a binary `.neutxt` file |

## Status

- [x] Video encoding via MAGVIT2
- [x] UTF-8 text format with keyframes + XOR deltas
- [x] Claude API integration (reverse, keyframes, slowmo, describe, freeform)
- [x] Apple MPS / CUDA auto-detection
- [ ] Audio encoding via EnCodec *(in progress)*
- [ ] Synchronized audiovisual mode

## License

TBD
