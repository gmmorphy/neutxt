"""MAGVIT2 tokenizer wrapper: encode/decode video frames as latent codes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch


@dataclass
class Magvit2Tokenizer:
    model: Any
    device: torch.device
    res: int
    code_bits: int


def pick_device(device_str: str = "auto") -> torch.device:
    d = (device_str or "auto").lower()
    if d == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(d)


def load_magvit2(vq_ckpt: str, yaml_path: str, device: torch.device,
                 res: int = 256) -> Magvit2Tokenizer:
    from omegaconf import OmegaConf
    from open_magvit2.reconstruct import load_vqgan_new

    cfg = OmegaConf.load(yaml_path)
    vq = load_vqgan_new(cfg, vq_ckpt)
    vq.eval().to(device)
    return Magvit2Tokenizer(model=vq, device=device, res=res, code_bits=18)


# ---------------------------------------------------------------------------
# Encode: RGB frame -> packed codes
# ---------------------------------------------------------------------------

def _collect_tensors(obj: Any) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    if isinstance(obj, torch.Tensor):
        tensors.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            tensors.extend(_collect_tensors(v))
    elif isinstance(obj, (tuple, list)):
        for it in obj:
            tensors.extend(_collect_tensors(it))
    return tensors


def _bits_to_codes(bits: torch.Tensor, code_bits: int) -> torch.Tensor:
    if bits.dtype.is_floating_point:
        bits = (bits > 0.0).to(torch.int64)
    elif bits.dtype == torch.bool:
        bits = bits.to(torch.int64)
    else:
        bits = (bits != 0).to(torch.int64)
    weights = (1 << torch.arange(code_bits, dtype=torch.int64)).view(1, code_bits)
    return (bits * weights).sum(dim=1)


@torch.no_grad()
def encode_frame_to_packed_codes(tok: Magvit2Tokenizer,
                                  frame_rgb_uint8: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Encode an RGB frame (H,W,3 uint8) to uint32 codes. Returns (codes, gh, gw)."""
    from PIL import Image
    import torchvision.transforms as T

    if frame_rgb_uint8.dtype != np.uint8:
        frame_rgb_uint8 = np.clip(frame_rgb_uint8, 0, 255).astype(np.uint8)

    img = Image.fromarray(frame_rgb_uint8, "RGB").resize(
        (tok.res, tok.res), resample=Image.LANCZOS)
    x = T.ToTensor()(img).unsqueeze(0).to(tok.device)

    out = tok.model.encode(x)
    tensors = _collect_tensors(out)
    if not tensors:
        raise RuntimeError(f"encode() returned no tensors: {type(out)}")

    cb = tok.code_bits
    chosen: Optional[torch.Tensor] = None

    # Bitplanes (cb,16,16) or (1,cb,16,16)
    for t in tensors:
        s = t.detach().shape
        if len(s) == 3 and s == (cb, 16, 16):
            chosen = t.detach(); break
        if len(s) == 4 and s[0] == 1 and s[1:] == (cb, 16, 16):
            chosen = t.detach()[0]; break

    # Bits-per-token (256,cb) or (1,256,cb)
    if chosen is None:
        for t in tensors:
            s = t.detach().shape
            if len(s) == 2 and s == (256, cb):
                chosen = t.detach(); break
            if len(s) == 3 and s[0] == 1 and s[1:] == (256, cb):
                chosen = t.detach()[0]; break

    # Direct indices (256,)
    if chosen is None:
        for t in tensors:
            if t.detach().numel() == 256:
                chosen = t.detach(); break

    if chosen is None:
        chosen = tensors[-1].detach()

    t = chosen.to("cpu")
    gh, gw = 16, 16

    if t.ndim == 3 and t.shape == (cb, 16, 16):
        bits_hw = _bits_to_codes(
            t.permute(1, 2, 0).reshape(256, cb), cb)
        codes = bits_hw
    elif t.ndim == 2 and t.shape == (256, cb):
        codes = _bits_to_codes(t, cb)
    elif t.numel() == 256:
        tt = t.reshape(-1)
        codes = tt.round().to(torch.int64) if tt.dtype.is_floating_point else tt.to(torch.int64)
    else:
        raise RuntimeError(
            f"Cannot interpret encode() tensor shape={tuple(t.shape)} dtype={t.dtype}")

    return codes.numpy().astype(np.uint32), gh, gw


# ---------------------------------------------------------------------------
# Decode: packed codes -> RGB frame
# ---------------------------------------------------------------------------

@torch.no_grad()
def decode_packed_codes_to_frame(tok: Magvit2Tokenizer,
                                  packed_codes: np.ndarray,
                                  gh: int, gw: int) -> np.ndarray:
    """Decode integer codes back to an RGB uint8 frame (H,W,3)."""
    t = torch.as_tensor(packed_codes.astype(np.int64), dtype=torch.int64,
                        device=tok.device).view(1, gh * gw)

    vq = tok.model
    if hasattr(vq, "quantize"):
        quant = vq.quantize
    elif hasattr(vq, "quantizer"):
        quant = vq.quantizer
    elif hasattr(vq, "model") and hasattr(vq.model, "quantize"):
        quant = vq.model.quantize
    else:
        raise AttributeError("MAGVIT2 model missing quantizer")

    shape = (1, gh, gw, tok.code_bits)
    try:
        q = quant.get_codebook_entry(t, shape, order="")
    except TypeError:
        q = quant.get_codebook_entry(t, shape)

    y = vq.decode(q)[0].detach().to("cpu")

    mn, mx = float(y.min()), float(y.max())
    if mn < -0.1 and mx <= 1.1:
        y = (y + 1.0) / 2.0
    y = y.clamp(0.0, 1.0)
    return (y.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
