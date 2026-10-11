"""
Precision and Recall for generative models.

Based on: Kynkäänniemi et al., "Improved Precision and Recall Metric for
Assessing Generative Models" (NeurIPS 2019).

Uses k-NN manifold estimation on InceptionV3 features.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

# _INCEPTION_TRANSFORM se importira iz src.compute_fid umjesto da se
# redeklarira ovdje. Prethodno je bila identična kopija konstante u oba
# modula — promjena u jednom nije automatski mijenjala drugi.
# Budući da compute_pr već ovisi o compute_fid, nema razloga za duplikaciju.
from src.compute_fid import (
    InceptionFeatureExtractor,
    SimpleImageDataset,
    _INCEPTION_TRANSFORM,
)


# ── k-NN helpers ─────────────────────────────────────────────────────────────

def _pairwise_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Vrati [N, M] Euklidsku matricu udaljenosti između redaka A i B."""
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    a2 = (A ** 2).sum(axis=1, keepdims=True)    # [N, 1]
    b2 = (B ** 2).sum(axis=1, keepdims=True).T  # [1, M]
    dist2 = a2 + b2 - 2.0 * (A @ B.T)
    return np.sqrt(np.maximum(dist2, 0.0))


def _knn_radii(feats: np.ndarray, k: int) -> np.ndarray:
    """Vrati udaljenost do k-tog najbližeg susjeda za svaki uzorak."""
    dists = _pairwise_distances(feats, feats)  # [N, N]
    dists_sorted = np.sort(dists, axis=1)      # uzlazno; indeks 0 je sam uzorak
    return dists_sorted[:, k]                  # k-ti susjed


def compute_precision_recall(
    real_feats: np.ndarray,
    gen_feats: np.ndarray,
    k: int = 3,
) -> tuple[float, float]:
    """
    Precision: udio generiranih uzoraka koji padaju unutar real manifolda.
    Recall:    udio realnih uzoraka pokrivenih generiranim manifoldom.
    """
    real_radii = _knn_radii(real_feats, k)
    gen_radii  = _knn_radii(gen_feats,  k)

    dist_gen_to_real = _pairwise_distances(gen_feats, real_feats)  # [N_gen, N_real]
    precision = float(np.mean(np.any(dist_gen_to_real <= real_radii[None, :], axis=1)))

    dist_real_to_gen = _pairwise_distances(real_feats, gen_feats)  # [N_real, N_gen]
    recall = float(np.mean(np.any(dist_real_to_gen <= gen_radii[None, :], axis=1)))

    return precision, recall


# ── Feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def _extract_features(
    image_dir: str,
    model: InceptionFeatureExtractor,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    _nw = num_workers if torch.cuda.is_available() else 0
    ds = SimpleImageDataset(image_dir, transform=_INCEPTION_TRANSFORM)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=_nw,
        pin_memory=torch.cuda.is_available(),
    )
    feats = []
    for batch in loader:
        feats.append(model(batch.to(device)).cpu().numpy())
    return np.concatenate(feats, axis=0)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Compute Precision & Recall between two image directories."
    )
    p.add_argument("--real",        required=True,  help="Directory with real images")
    p.add_argument("--generated",   required=True,  help="Directory with generated images")
    p.add_argument("--k",           type=int, default=3,
                   help="k for k-NN manifold estimation (default: 3)")
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device",      default=None)
    args = p.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    model = InceptionFeatureExtractor().to(device).eval()

    print(f"Extracting features for real images:      {args.real}")
    real_feats = _extract_features(
        args.real, model, device, args.batch_size, args.num_workers
    )

    print(f"Extracting features for generated images: {args.generated}")
    gen_feats = _extract_features(
        args.generated, model, device, args.batch_size, args.num_workers
    )

    precision, recall = compute_precision_recall(real_feats, gen_feats, k=args.k)
    print(f"\nPrecision = {precision:.4f}")
    print(f"Recall    = {recall:.4f}")


if __name__ == "__main__":
    main()