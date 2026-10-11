from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
from PIL import Image, ImageFile

# Prevents crashes on truncated JPEG files.
# On Supek's Lustre filesystem, I/O errors are possible — without this,
# a single corrupted JPEG will kill a DataLoader worker and stall training
# that may have been running for hours.
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import GroupShuffleSplit


@dataclass
class DataConfig:
    root: str
    image_dir_glob: List[str]
    metadata_glob: List[str]
    image_size: int
    num_workers: int
    val_ratio: float
    num_classes: int
    # malignant_dx / benign_dx are only used for the BINARY task (num_classes=2)
    # and for benign/malignant grouping in evaluation. Optional so the 7-class
    # config does not need to specify them.
    malignant_dx: Optional[List[str]] = None
    benign_dx: Optional[List[str]] = None
    # dx_classes: ordered list of diagnosis codes -> class index, e.g.
    #   ["akiec","bcc","bkl","df","mel","nv","vasc"]  (7-class per-diagnosis task)
    # If set, this overrides the binary mapping. The order DEFINES the integer
    # label of each class and MUST be identical across training, sampling, and
    # FID reference generation.
    dx_classes: Optional[List[str]] = None
    # Strength of class re-balancing for WeightedRandomSampler:
    #   0.0 = no balancing (original distribution)
    #   1.0 = full inverse-frequency (all classes equally likely)
    #   0.5 = compromise (recommended: helps rare classes without pushing the
    #         model into memorising the ~115 df images)
    sampler_temperature: float = 0.5
    # Subdirectory (under root) where train/val split CSVs are stored.
    # Keep this distinct per stage (e.g. "splits_256") so a new task does not
    # silently reuse split CSVs written with a different label mapping.
    splits_dirname: str = "splits"


# ── Path utilities ────────────────────────────────────────────────────────────

def find_first_existing(globs_list: List[str]) -> Optional[str]:
    for g in globs_list:
        matches = glob.glob(g, recursive=True)
        if matches:
            return sorted(matches, key=len)[0]
    return None


def find_all_dirs(globs_list: List[str]) -> List[str]:
    seen: set = set()
    dirs: List[str] = []
    for g in globs_list:
        for m in glob.glob(g, recursive=True):
            if os.path.isdir(m) and m not in seen:
                seen.add(m)
                dirs.append(m)
    return sorted(dirs, key=lambda x: (len(x), x))


def dir_has_images(d: str) -> bool:
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        if glob.glob(os.path.join(d, ext)):
            return True
    return False


def build_image_index_from_dirs(image_dirs: List[str]) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    for d in image_dirs:
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in glob.glob(os.path.join(d, ext)):
                base = os.path.splitext(os.path.basename(p))[0]
                if base not in idx:
                    idx[base] = p
    return idx


# ── Metadata ──────────────────────────────────────────────────────────────────

def load_metadata(metadata_path: str) -> pd.DataFrame:
    df = pd.read_csv(metadata_path)
    required = {"image_id", "lesion_id", "dx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Metadata CSV is missing columns: {missing}. "
            f"Found: {list(df.columns)}"
        )
    return df


def map_dx_to_binary_label(
    dx: str,
    malignant_dx: List[str],
    benign_dx: List[str],
) -> Optional[int]:
    dx = str(dx).strip().lower()
    if dx in malignant_dx:
        return 1
    if dx in benign_dx:
        return 0
    return None


def build_dx_index_map(dx_classes: List[str]) -> Dict[str, int]:
    """Ordered diagnosis list -> {dx_lower: index}. Order defines the labels."""
    return {dx.strip().lower(): i for i, dx in enumerate(dx_classes)}


def map_dx_to_multiclass_label(dx: str, dx_to_idx: Dict[str, int]) -> Optional[int]:
    return dx_to_idx.get(str(dx).strip().lower(), None)


# ── Weighted sampling (class imbalance mitigation) ────────────────────────────

def build_weighted_sampler(
    labels: List[int],
    num_classes: int,
    temperature: float = 0.5,
):
    """
    WeightedRandomSampler that oversamples rare classes.

    weight(class) ~ (1 / count(class)) ** temperature
      temperature=0.0 -> original distribution (no balancing)
      temperature=1.0 -> full inverse-frequency (all classes equal)
      temperature=0.5 -> compromise (recommended)

    Full inverse-frequency on HAM10000 7-class would oversample df (~115 imgs)
    ~60x, pushing a diffusion model toward memorising those few images. The
    temperature lets us help rare classes without that failure mode.
    """
    import numpy as np
    from torch.utils.data import WeightedRandomSampler

    labels_arr = np.asarray(labels)
    counts = np.bincount(labels_arr, minlength=num_classes).astype(np.float64)
    class_w = (1.0 / np.maximum(counts, 1)) ** float(temperature)
    sample_w = class_w[labels_arr]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(labels_arr),
        replacement=True,
    )


# ── Train / val split ─────────────────────────────────────────────────────────

def make_splits(
    df: pd.DataFrame,
    val_ratio: float,
    seed: int,
    splits_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(splits_dir, exist_ok=True)
    train_csv = os.path.join(splits_dir, "train.csv")
    val_csv = os.path.join(splits_dir, "val.csv")

    def _generate() -> Tuple[pd.DataFrame, pd.DataFrame]:
        if len(df) == 0:
            raise ValueError("Cannot create splits: dataframe has 0 samples.")
        gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        idx = np.arange(len(df))
        train_idx, val_idx = next(gss.split(idx, groups=df["lesion_id"].values))
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
        return train_df, val_df

    if os.path.isfile(train_csv) and os.path.isfile(val_csv):
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        if "image_id" in train_df.columns and "image_id" in val_df.columns:
            current_ids = set(df["image_id"].astype(str))
            if (
                len(train_df) > 0 and len(val_df) > 0
                and set(train_df["image_id"].astype(str)).issubset(current_ids)
                and set(val_df["image_id"].astype(str)).issubset(current_ids)
            ):
                return train_df, val_df
        return _generate()

    return _generate()


# ── Dataset ───────────────────────────────────────────────────────────────────

class HamBinaryDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_index: Dict[str, str],
        tfm,
        num_classes: int,
        image_size: int = 128,
    ) -> None:
        self.tfm = tfm
        self.num_classes = num_classes
        # image_size is stored so it can be used to produce a correctly
        # dimensioned blank fallback image when a file cannot be opened.
        self.image_size = image_size
        self.paths: List[str] = []
        self.labels: List[int] = []

        for _, row in df.iterrows():
            image_id = str(row["image_id"])
            if image_id in image_index:
                self.paths.append(image_index[image_id])
                self.labels.append(int(row["label"]))

        if not self.paths:
            raise RuntimeError(
                "Dataset is empty after filtering. "
                "This usually means no image files were indexed or the splits "
                "cache does not match the current dataset."
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        try:
            img = Image.open(self.paths[i]).convert("RGB")
        except Exception as e:
            # On Supek's Lustre filesystem, truncated or corrupted JPEGs are possible.
            # We return a blank image instead of crashing the DataLoader worker.
            print(f"[data] WARNING: Cannot open {self.paths[i]}: {e}. Using blank image.")
            img = Image.new("RGB", (self.image_size, self.image_size), color=0)
        return {
            "pixel_values": self.tfm(img),
            "class_labels": torch.tensor(self.labels[i], dtype=torch.long),
        }


# ── Transforms ────────────────────────────────────────────────────────────────

def build_transforms(image_size: int, train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize(image_size + 16),
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # RandomRotation intentionally removed.
            # Rotation augmentation creates black corners (zero-padding artefacts)
            # that the model learns to reproduce in generated images, making them
            # immediately identifiable as synthetic — a critical flaw for the
            # medical Turing test. Horizontal and vertical flips provide sufficient
            # orientation diversity without introducing this artefact.
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
    return transforms.Compose([
        transforms.Resize(image_size + 16),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])


# ── Entry point ───────────────────────────────────────────────────────────────

def build_datasets(cfg: DataConfig, seed: int):
    metadata_path = find_first_existing(cfg.metadata_glob)
    if metadata_path is None:
        raise FileNotFoundError(
            f"HAM10000_metadata.csv not found. Searched: {cfg.metadata_glob}"
        )

    df = load_metadata(metadata_path)
    stats: Dict[str, Any] = {
        "metadata_rows_total": int(len(df)),
        "metadata_path": metadata_path,
    }

    # Label mapping: per-diagnosis (7-class) if dx_classes is set, else binary.
    if cfg.dx_classes:
        dx_to_idx = build_dx_index_map(cfg.dx_classes)
        df["label"] = df["dx"].apply(lambda x: map_dx_to_multiclass_label(x, dx_to_idx))
        stats["label_mode"] = "multiclass"
        stats["dx_to_idx"] = dx_to_idx
    else:
        df["label"] = df["dx"].apply(
            lambda x: map_dx_to_binary_label(x, cfg.malignant_dx, cfg.benign_dx)
        )
        stats["label_mode"] = "binary"
    stats["rows_after_label_map"] = int(df["label"].notna().sum())
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)

    image_dirs = find_all_dirs(cfg.image_dir_glob)
    stats["image_dirs_found"] = image_dirs

    if not image_dirs:
        raise FileNotFoundError(
            f"No image directories found. Searched: {cfg.image_dir_glob}"
        )

    part_dirs = [
        d for d in image_dirs
        if os.path.basename(d).lower() in
        ("ham10000_images_part_1", "ham10000_images_part_2")
    ]
    merged_dirs = [
        d for d in image_dirs
        if os.path.basename(d).lower() == "ham10000_images" and dir_has_images(d)
    ]
    chosen_dirs = merged_dirs[:1] or part_dirs or image_dirs
    stats["image_dirs_used_for_index"] = chosen_dirs

    image_index = build_image_index_from_dirs(chosen_dirs)
    stats["images_indexed"] = int(len(image_index))

    if not image_index:
        raise RuntimeError(
            f"No images indexed (no .jpg/.png found in {chosen_dirs})."
        )

    exists_mask = df["image_id"].astype(str).isin(image_index)
    stats["rows_with_existing_files"] = int(exists_mask.sum())
    df = df[exists_mask].copy()

    if df.empty:
        raise RuntimeError(
            "Dataframe is empty after matching image_id to file. "
            "Check that metadata image_ids match the filenames on disk. "
            f"images_indexed={stats['images_indexed']}"
        )

    splits_dir = os.path.join(cfg.root, cfg.splits_dirname)
    train_df, val_df = make_splits(df, cfg.val_ratio, seed, splits_dir)

    train_ds = HamBinaryDataset(
        train_df, image_index,
        build_transforms(cfg.image_size, train=True),
        cfg.num_classes,
        image_size=cfg.image_size,
    )
    val_ds = HamBinaryDataset(
        val_df, image_index,
        build_transforms(cfg.image_size, train=False),
        cfg.num_classes,
        image_size=cfg.image_size,
    )

    stats["train_size"] = len(train_ds)
    stats["val_size"] = len(val_ds)

    # Per-class distribution of the train set — useful for verifying the
    # mapping and for setting sampler temperature.
    import numpy as _np
    _counts = _np.bincount(_np.asarray(train_ds.labels), minlength=cfg.num_classes)
    stats["train_class_counts"] = _counts.tolist()

    return train_ds, val_ds, stats