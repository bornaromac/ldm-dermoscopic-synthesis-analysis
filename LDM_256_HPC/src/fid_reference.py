"""
make_fid_reference.py — Build a fair FID reference set from the validation split.

The problem this solves:
    Previous FID compared generated images (128px PNGs, upscaled to 299 for
    InceptionV3) against real images at full resolution (600x450, downscaled to
    299). This resolution/sharpness mismatch systematically inflates FID,
    independent of model quality.

The fix:
    Real images go through the IDENTICAL preprocessing as generated images —
    Resize(res + 16) -> CenterCrop(res) -> saved as PNG in [0, 1]. FID then
    measures only the difference in content, not in resolution.

Methodology:
    The reference is built from the VAL split (images the model never saw during
    training), per class, so that per-class FID is fair and comparable to the
    literature.

The eval transform here mirrors data.build_transforms(train=False), minus the
[-1, 1] Normalize: saved PNGs are in [0, 1], matching the generated PNGs that
final_sample.py writes after mapping the VAE output to [0, 1].

Usage:
    python make_fid_reference.py \
        --val_csv data/splits/val.csv \
        --image_dirs data/HAM10000_images_part_1 data/HAM10000_images_part_2 \
        --res 256 \
        --dx_classes akiec bcc bkl df mel nv vasc \
        --out data/real_ref_256
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
from PIL import Image, ImageFile
from torchvision import transforms
from torchvision.utils import save_image

# Prevents crashes on truncated JPEG files (mirrors data.py).
# On Supek's Lustre filesystem, I/O errors are possible — without this,
# a single corrupted JPEG can abort the run.
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ── Path utilities ────────────────────────────────────────────────────────────

def build_image_index(image_dirs: List[str]) -> Dict[str, str]:
    """Map image_id (filename without extension) -> file path."""
    index: Dict[str, str] = {}
    for d in image_dirs:
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in glob.glob(os.path.join(d, ext)):
                stem = os.path.splitext(os.path.basename(p))[0]
                index.setdefault(stem, p)
    return index


def build_eval_transform(res: int):
    """Identical to data.build_transforms(train=False), without Normalize.

    Normalize([0.5], [0.5]) is omitted on purpose: the saved PNG must be in
    [0, 1] to match the generated PNGs, whereas Normalize is only needed when
    feeding pixels into the VAE during training.
    """
    return transforms.Compose([
        transforms.Resize(res + 16),
        transforms.CenterCrop(res),
        transforms.ToTensor(),  # -> [0, 1]
    ])


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Build a fair FID reference set from the validation split."
    )
    p.add_argument("--val_csv", required=True, help="Path to data/splits/val.csv")
    p.add_argument("--image_dirs", nargs="+", required=True,
                   help="Directories containing the HAM10000 images")
    p.add_argument("--res", type=int, default=256,
                   help="Target resolution (must match the generated images)")
    p.add_argument("--dx_classes", nargs="+",
                   default=["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
                   help="Ordered diagnosis codes -> class index. "
                        "MUST match data.dx_classes used in training.")
    p.add_argument("--out", required=True, help="Output directory for the reference set")
    args = p.parse_args()

    tfm = build_eval_transform(args.res)
    dx_to_idx = {dx.strip().lower(): i for i, dx in enumerate(args.dx_classes)}

    df = pd.read_csv(args.val_csv)
    required = {"image_id", "dx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"val_csv is missing columns: {missing}. Found: {list(df.columns)}"
        )

    image_index = build_image_index(args.image_dirs)
    if not image_index:
        raise RuntimeError(f"No images indexed under: {args.image_dirs}")

    out_root = Path(args.out)
    per_class_counts: Dict[str, int] = {dx: 0 for dx in dx_to_idx}
    skipped = 0

    for _, row in df.iterrows():
        dx = str(row["dx"]).strip().lower()
        image_id = str(row["image_id"])
        if dx not in dx_to_idx or image_id not in image_index:
            skipped += 1
            continue
        cls_dir = out_root / f"class_{dx}"
        cls_dir.mkdir(parents=True, exist_ok=True)
        try:
            img = Image.open(image_index[image_id]).convert("RGB")
        except Exception as exc:
            print(f"[warn] Skipping {image_id}: {exc}")
            skipped += 1
            continue
        save_image(tfm(img), str(cls_dir / f"{image_id}.png"))
        per_class_counts[dx] += 1

    print(f"\nReference set saved to: {out_root}  (resolution {args.res})")
    print("Per class:")
    for dx, n in per_class_counts.items():
        print(f"  class_{dx:6s}  {n}")
    print(f"Skipped (unknown class or image not found): {skipped}")
    print("\n[next] Per-class FID (example for mel):")
    print(f"  python src/compute_fid.py --real {out_root}/class_mel "
          f"--generated <generated_dir>/class_mel")


if __name__ == "__main__":
    main()