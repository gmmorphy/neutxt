"""EnCodec audio codec wrapper: encode/decode audio as neural latent codes."""
from __future__ import annotations

import io
import subprocess
import wave
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class EncodecWrapper:
    model: Any
    device: torch.device
    sample_rate: int       # 24000 or 48000
    bandwidth: float       # target kbps (1.5, 3.0, 6.0, 12.0, 24.0)
    n_quantizers: int      # derived from bandwidth
    code_bits: int = 10    # EnCodec codebook is 1024 entries per quantizer


BANDWIDTH_TO_QUANTIZERS = {
    1.5: 2,
    3.0: 4,
    6.0: 8,
    12.0: 16,
    24.0: 32,
}


def load_encodec(device: torch.device, sample_rate: int = 24000,
                 bandwidth: float = 6.0) -> EncodecWrapper:
    """Load Meta's EnCodec model for neural audio compression."""
    from encodec import EncodecModel

    if sample_rate == 24000:
        model = EncodecModel.encodec_model_24khz()
    elif sample_rate == 48000:
        model = EncodecModel.encodec_model_48khz()
    else:
        raise ValueError(f"EnCodec supports 24000 or 48000 Hz, got {sample_rate}")

    model.set_target_bandwidth(bandwidth)
    model.eval()
    # MPS doesn't support all ops EnCodec uses — CPU is safe and fast enough
    if device.type == "mps":
        device = torch.device("cpu")
    model.to(device)

    n_q = BANDWIDTH_TO_QUANTIZERS.get(bandwidth, 8)
    return EncodecWrapper(
        model=model,
        device=device,
        sample_rate=sample_rate,
        bandwidth=bandwidth,
        n_quantizers=n_q,
    )


# ---------------------------------------------------------------------------
# Audio I/O via ffmpeg
# ---------------------------------------------------------------------------

def load_audio_mono_pcm(path: str, sample_rate: int, max_seconds: float | None = None) -> np.ndarray:
    """Decode any media file to mono float32 PCM at the target sample rate."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path]
    if max_seconds:
        cmd.extend(["-t", str(max_seconds)])
    cmd.extend([
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-f", "f32le", "pipe:1",
    ])
    r = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype=np.float32)


def pcm_to_wav_bytes(pcm_float32: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 PCM to WAV bytes (16-bit signed)."""
    pcm16 = (np.clip(pcm_float32, -1.0, 1.0) * 32767).astype(np.int16)
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return bio.getvalue()


# ---------------------------------------------------------------------------
# Encode/Decode
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_audio_to_codes(enc: EncodecWrapper, pcm_mono: np.ndarray) -> np.ndarray:
    """
    Encode mono float32 PCM to EnCodec codes.

    Returns:
        codes: np.ndarray(uint16) shape (n_quantizers, n_timesteps)
               values in [0, 1024)
    """
    x = torch.from_numpy(pcm_mono).float().unsqueeze(0).unsqueeze(0).to(enc.device)
    encoded_frames = enc.model.encode(x)

    # encoded_frames is a list of (codes, scale) tuples, one per chunk
    all_codes = []
    for frame_codes, _scale in encoded_frames:
        all_codes.append(frame_codes[0].cpu().numpy())  # (n_q, T)
    codes = np.concatenate(all_codes, axis=1).astype(np.uint16)
    return codes


@torch.no_grad()
def decode_codes_to_audio(enc: EncodecWrapper, codes: np.ndarray) -> np.ndarray:
    """
    Decode EnCodec codes back to mono float32 PCM.

    Args:
        codes: np.ndarray shape (n_quantizers, n_timesteps)

    Returns:
        pcm: np.ndarray(float32) mono waveform in [-1, 1]
    """
    codes_t = torch.from_numpy(codes.astype(np.int64)).long().to(enc.device)
    codes_t = codes_t.unsqueeze(0)  # (1, n_q, T)

    # EnCodec decode expects list of (codes, scale) frames
    encoded_frames = [(codes_t, None)]
    y = enc.model.decode(encoded_frames)
    return y[0, 0].cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Bit packing for 10-bit codes
# ---------------------------------------------------------------------------

def bitpack10(codes: np.ndarray) -> bytes:
    """Pack uint16 codes (10-bit values) into a dense byte stream."""
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


def bitunpack10(packed: bytes, n_codes: int) -> np.ndarray:
    """Unpack dense byte stream back to uint16 codes."""
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
