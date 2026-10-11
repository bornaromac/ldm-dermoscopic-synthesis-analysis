"""
FID utilities for DDPM evaluation.

Provides:
  - SimpleImageDataset  : loads images from a directory tree
  - InceptionFeatureExtractor : extracts 2048-d pool3 features from InceptionV3
  - compute_fid_stats   : extract (mu, sigma) from a dataset
  - compute_fid         : FID from two (mu, sigma) pairs
  - main                : CLI entry point
"""
from __future__ import annotations

import os
import glob
import argparse
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from PIL import Image
from scipy import linalg


# ── Dataset ──────────────────────────────────────────────────────────────────

class SimpleImageDataset(Dataset):
    """Flat dataset that finds all .jpg/.png images under a root directory."""

    def __init__(self, image_dir: str, transform=None):
        exts = ("*.jpg", "*.jpeg", "*.png")
        self.paths: list = []
        for ext in exts:
            self.paths.extend(
                glob.glob(os.path.join(image_dir, "**", ext), recursive=True)
            )
        self.paths.sort()
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No images found under: {image_dir}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except Exception:
            # Return a blank image so the batch is not broken; affects stats minimally.
            img = Image.new("RGB", (299, 299))
        if self.transform is not None:
            img = self.transform(img)
        return img


# ── Feature extractor ─────────────────────────────────────────────────────────

class InceptionFeatureExtractor(nn.Module):
    """
    Wraps InceptionV3 and returns 2048-d pool3 features.
    Images must be [B, 3, 299, 299] in [0, 1].
    """

    def __init__(self):
        super().__init__()
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
        model.eval()
        # Drop the final FC; keep everything up to and including avgpool
        self.block1 = nn.Sequential(
            model.Conv2d_1a_3x3, model.Conv2d_2a_3x3, model.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.block2 = nn.Sequential(
            model.Conv2d_3b_1x1, model.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.block3 = nn.Sequential(
            model.Mixed_5b, model.Mixed_5c, model.Mixed_5d,
            model.Mixed_6a, model.Mixed_6b, model.Mixed_6c,
            model.Mixed_6d, model.Mixed_6e,
        )
        self.block4 = nn.Sequential(
            model.Mixed_7a, model.Mixed_7b, model.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.flatten(1)  # [B, 2048]


# ── Statistics extraction ─────────────────────────────────────────────────────

_INCEPTION_TRANSFORM = transforms.Compose([
    transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),          # [0, 1]
])


@torch.no_grad()
def compute_fid_stats(
    image_dir: str,
    model: InceptionFeatureExtractor,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mu, sigma) computed over all images in image_dir."""
    _nw = num_workers if torch.cuda.is_available() else 0
    ds = SimpleImageDataset(image_dir, transform=_INCEPTION_TRANSFORM)
    loader = DataLoader(
        ds, batch_size=batch_size,
        num_workers=_nw,
        pin_memory=torch.cuda.is_available(),
    )

    feats = []
    for batch in loader:
        batch = batch.to(device)
        feats.append(model(batch).cpu().numpy())

    feats = np.concatenate(feats, axis=0)          # [N, 2048]
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


# ── FID computation ───────────────────────────────────────────────────────────

def compute_fid(
    mu1: np.ndarray, sigma1: np.ndarray,
    mu2: np.ndarray, sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Frechet Inception Distance between two Gaussian distributions."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError("sqrtm produced large imaginary component")
        covmean = covmean.real

    # numerical stability
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    fid = float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))
    return fid


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Compute FID between two image directories.")
    p.add_argument("--real",       required=True,  help="Directory with real images")
    p.add_argument("--generated",  required=True,  help="Directory with generated images")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers",type=int, default=4)
    p.add_argument("--device",     default=None,   help="cuda / cpu (auto-detect if omitted)")
    args = p.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    model = InceptionFeatureExtractor().to(device).eval()

    print(f"Extracting stats for real images: {args.real}")
    mu_r, sig_r = compute_fid_stats(args.real, model, device, args.batch_size, args.num_workers)

    print(f"Extracting stats for generated images: {args.generated}")
    mu_g, sig_g = compute_fid_stats(args.generated, model, device, args.batch_size, args.num_workers)

    fid = compute_fid(mu_r, sig_r, mu_g, sig_g)
    print(f"\nFID = {fid:.4f}")


if __name__ == "__main__":
    main()
