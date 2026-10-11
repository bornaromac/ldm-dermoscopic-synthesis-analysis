"""
resize_images.py — One-time pre-resize of HAM10000 images to speed up training.

The problem this solves:
    At 256, the DataLoader was decoding full 600x450 JPEGs and resizing them on
    every single fetch (millions of times), over the Lustre network filesystem.
    The GPU starved waiting for data — 16+ minutes without a single training step.

The fix:
    Resize every image ONCE to a short side of `short_side` (default 272 =
    image_size + 16, matching data.build_transforms) and save as PNG into a
    separate folder. The DataLoader then reads small images and does no heavy
    decode/resize per fetch. Training transforms (RandomResizedCrop(256),
    CenterCrop(256)) behave identically because the short side is still 272.

    PNG is used (lossless) to avoid adding compression artefacts, consistent
    with the generated images and the FID reference.

After running, point the config's image_dir_glob at --out, e.g.:
    image_dir_glob:
      - "data/HAM10000_images_256"

Usage (run on the login node — it only needs CPU):
    python resize_images.py \
        --image_dirs data/HAM10000_images_part_1 data/HAM10000_images_part_2 \
        --out data/HAM10000_images_256 \
        --short_side 272 \
        --workers 8
"""
from __future__ import annotations

import argparse
import glob
import os
from functools import partial
from multiprocessing import Pool

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def list_images(image_dirs):
    paths = []
    for d in image_dirs:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            paths.extend(glob.glob(os.path.join(d, ext)))
    # Deduplicate by stem (image_id), keeping the first occurrence.
    seen = {}
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        seen.setdefault(stem, p)
    return list(seen.items())  # [(image_id, path), ...]


def resize_one(item, out_dir, short_side):
    image_id, path = item
    out_path = os.path.join(out_dir, f"{image_id}.png")
    if os.path.exists(out_path):
        return "skip"
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        return f"error:{image_id}:{e}"
    w, h = img.size
    # Resize the SHORTER side to short_side, preserving aspect ratio.
    # This mirrors torchvision.transforms.Resize(short_side) with an int arg.
    if w <= h:
        new_w = short_side
        new_h = round(h * short_side / w)
    else:
        new_h = short_side
        new_w = round(w * short_side / h)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    img.save(out_path, format="PNG")
    return "ok"


def main():
    ap = argparse.ArgumentParser(description="Pre-resize images to short-side PNG.")
    ap.add_argument("--image_dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--short_side", type=int, default=272,
                    help="Target short side (default 272 = image_size + 16)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    items = list_images(args.image_dirs)
    print(f"Found {len(items)} unique images. Resizing to short side {args.short_side}...")

    worker = partial(resize_one, out_dir=args.out, short_side=args.short_side)
    ok = skipped = errors = 0
    err_examples = []
    with Pool(processes=args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(worker, items, chunksize=32), 1):
            if res == "ok":
                ok += 1
            elif res == "skip":
                skipped += 1
            else:
                errors += 1
                if len(err_examples) < 5:
                    err_examples.append(res)
            if i % 1000 == 0:
                print(f"  {i}/{len(items)} processed...")

    print(f"\nDone. Saved to: {args.out}")
    print(f"  ok={ok}  skipped(existing)={skipped}  errors={errors}")
    if err_examples:
        print("  first errors:")
        for e in err_examples:
            print("   ", e)
    print("\n[next] Point config image_dir_glob at this folder and resubmit training.")


if __name__ == "__main__":
    main()
