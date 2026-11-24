"""Enlarges the website painting to 4K on the GPU with Real-ESRGAN, without sharpening."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Project folder, model weights and tile sizes.
ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_URL = ("https://github.com/xinntao/Real-ESRGAN/releases/download/"
               "v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth")
WEIGHTS = ROOT / "models" / "upscaler" / "RealESRGAN_x4plus_anime_6B.pth"
TILE, PAD = 384, 16


def load_model(device):
    """Download (once) and load the Real-ESRGAN upscaler."""
    from spandrel import ModelLoader
    if not WEIGHTS.exists():
        WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading Real-ESRGAN weights (18 MB)...")
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS)
    return ModelLoader().load_from_file(str(WEIGHTS)).to(device).eval()


@torch.no_grad()
def upscale(model, img: np.ndarray, device) -> np.ndarray:
    """Upscale the image tile by tile so it fits in GPU memory."""
    h, w, _ = img.shape
    s = model.scale
    x = torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255).to(device)
    out = torch.zeros(1, 3, h * s, w * s, device=device)
    for y0 in range(0, h, TILE):
        for x0 in range(0, w, TILE):
            y1, x1 = min(y0 + TILE, h), min(x0 + TILE, w)
            ya, xa = max(y0 - PAD, 0), max(x0 - PAD, 0)
            yb, xb = min(y1 + PAD, h), min(x1 + PAD, w)
            tile = model(x[:, :, ya:yb, xa:xb])
            out[:, :, y0 * s:y1 * s, x0 * s:x1 * s] = \
                tile[:, :, (y0 - ya) * s:(y0 - ya + y1 - y0) * s, (x0 - xa) * s:(x0 - xa + x1 - x0) * s]
    return (out[0].clamp(0, 1).permute(1, 2, 0).mul(255).round().byte().cpu().numpy())


def main() -> None:
    """Upscale the painting and save 3840 and 1920 pixel JPEGs."""
    src = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "web" / "img" / "kolkata.png")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = np.array(Image.open(src).convert("RGB"))
    print(f"Input {img.shape[1]} x {img.shape[0]}, device {device}")
    big = Image.fromarray(upscale(load_model(device), img, device))
    out = src.parent
    for width in (3840, 1920):
        size = (width, round(width * big.height / big.width))
        big.resize(size, Image.LANCZOS).save(out / f"kolkata-{width}.jpg", quality=88,
                                             optimize=True, progressive=True)
        print(f"Saved {out / f'kolkata-{width}.jpg'}  {size[0]} x {size[1]}")


if __name__ == "__main__":
    main()
