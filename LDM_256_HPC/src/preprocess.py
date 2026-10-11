from __future__ import annotations

import argparse
import yaml

from src.data import DataConfig, build_datasets


def main() -> None:
    p = argparse.ArgumentParser(
        description="Preprocess HAM10000 and generate train/val splits."
    )
    p.add_argument(
        "--config",
        default="configs/ddpm_ham_256_hpc.yaml",
        help="Path to YAML config file (default: configs/ddpm_ham_256_hpc.yaml)",
    )
    args = p.parse_args()

    try:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        raise SystemExit(
            f"ERROR: Config not found: {args.config}\n"
        )

    seed = int(cfg.get("seed", 42))
    data_cfg = DataConfig(**cfg["data"])
    train_ds, val_ds, stats = build_datasets(data_cfg, seed=seed)

    print("\nPreprocessing done.")
    print(f"- Config:                     {args.config}")
    print(f"- Image size:                 {data_cfg.image_size}x{data_cfg.image_size}")
    print(f"- Metadata rows total:        {stats['metadata_rows_total']}")
    print(f"- Rows after label map:       {stats['rows_after_label_map']}")
    print(f"- Images indexed:             {stats['images_indexed']}")
    print(f"- Rows with existing files:   {stats['rows_with_existing_files']}")
    print(f"- Train size:                 {stats['train_size']}")
    print(f"- Val size:                   {stats['val_size']}")
    print(f"- Splits:                     {data_cfg.root}/splits/train.csv, val.csv")

    print("\nImage dirs found:")
    for d in stats["image_dirs_found"]:
        print("  -", d)

    print("\nImage dirs used for indexing:")
    for d in stats["image_dirs_used_for_index"]:
        print("  -", d)


if __name__ == "__main__":
    main()