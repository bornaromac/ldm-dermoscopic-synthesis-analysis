"""
final_sample.py — Generate the full evaluation image set from a trained LDM with CFG.

Produces n_per_class images for each class and saves them as individual PNG
files for use with compute_fid.py and compute_pr.py. Folder names follow the
class layout described below (class_<dx> for the 7-class per-diagnosis task,
or class_0 / class_1 for the binary task).

This script is intended to be run once after training completes, on the
best checkpoint. For monitoring during training, use sample.py instead.

Usage:
    python -m src.final_sample \
        --config configs/ldm_ham_128.yaml \
        --ckpt runs/ldm_ham_128_cfg/checkpoints/best/ckpt.pt \
        --n_per_class 1000 \
        --batch_size 32 \
        --steps 50 \
        --guidance_scale 7.5 \
        --out_dir runs/ldm_ham_128_cfg/final_results

CFG note:
    guidance_scale=7.5 is the recommended default. You can experiment with
    different values without retraining:
      - 5.0: more diverse, slightly less class-specific
      - 7.5: good balance of quality and class specificity (recommended)
      - 10.0: strongly class-specific, less diverse
"""
from __future__ import annotations

import os
import argparse

import torch
import yaml
from diffusers import DDIMScheduler
from torchvision.utils import save_image

from src.model import ClassEmbedder, build_vae, build_unet
from src.data import DataConfig
from src.utils import ensure_dir, seed_everything, detect_mixed_precision


def _cfg_step(
    unet: torch.nn.Module,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """
    Apply one CFG denoising step.

    Runs the UNet twice — once with the class context (conditional) and once
    with a zero vector (unconditional) — then combines:
        noise_pred = uncond + guidance_scale * (cond - uncond)

    The model must have been trained with conditioning dropout for this to
    produce meaningful unconditional predictions.
    """
    uncond_context = torch.zeros_like(context)
    noise_pred_uncond = unet(latents, timestep, encoder_hidden_states=uncond_context).sample
    noise_pred_cond   = unet(latents, timestep, encoder_hidden_states=context).sample
    return noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)


@torch.no_grad()
def generate_final(
    config_path: str,
    ckpt_path: str,
    n_per_class: int = 1000,
    batch_size: int = 32,
    sampler_steps: int = 50,
    guidance_scale: float = 7.5,
    out_dir_override: str | None = None,
) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed_everything(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = DataConfig(**cfg["data"])
    scale_factor = float(cfg["vae"]["scale_factor"])
    u = cfg["unet"]

    # Load VAE (frozen pretrained encoder/decoder)
    vae = build_vae(cfg).to(device)

    # Build and load UNet + ClassEmbedder from checkpoint
    unet = build_unet(cfg).to(device)
    class_embedder = ClassEmbedder(
        num_classes=int(data_cfg.num_classes),
        embed_dim=int(u["cross_attention_dim"]),
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    if "unet" not in state or "class_embedder" not in state:
        raise KeyError(
            f"Checkpoint missing required keys. Found: {list(state.keys())}"
        )
    unet.load_state_dict(state["unet"])
    class_embedder.load_state_dict(state["class_embedder"])

    if "ema" in state:
        from diffusers.training_utils import EMAModel
        ema = EMAModel(unet.parameters())
        ema.load_state_dict(state["ema"])
        ema.copy_to(unet.parameters())
        print("[ema] EMA weights applied.")
    else:
        print("[ema] No EMA state found; using raw weights.")

    unet.eval()
    class_embedder.eval()

    sched = DDIMScheduler(
        num_train_timesteps=int(cfg["diffusion"]["num_train_timesteps"]),
        beta_schedule=str(cfg["diffusion"]["beta_schedule"]),
        prediction_type=str(cfg["diffusion"]["prediction_type"]),
        clip_sample=False,
    )
    sched.set_timesteps(sampler_steps, device=device)

    out_dir = out_dir_override or os.path.join(cfg["output"]["out_dir"], "final_results")

    print(f"[cfg] guidance_scale={guidance_scale}  steps={sampler_steps}")

    mp_mode = detect_mixed_precision()
    if device.type == "cuda" and mp_mode in ("bf16", "fp16"):
        _dtype = torch.bfloat16 if mp_mode == "bf16" else torch.float16
        autocast_ctx = torch.autocast(device_type="cuda", dtype=_dtype)
        print(f"[final_sample] Using {mp_mode} autocast for inference.")
    else:
        autocast_ctx = torch.autocast(device_type="cpu", enabled=False)

    # Folder naming.
    # If dx_classes is set (7-class per-diagnosis task), name folders
    # class_<dx> (e.g. class_mel) so they match make_fid_reference.py and
    # per-class FID can be computed folder-by-folder. Otherwise fall back to
    # the binary class_0 / class_1 layout.
    dx_classes = cfg["data"].get("dx_classes")
    if dx_classes:
        class_names = {i: f"class_{dx.lower()}" for i, dx in enumerate(dx_classes)}
        class_labels_desc = {i: dx.lower() for i, dx in enumerate(dx_classes)}
    else:
        class_names = {0: "class_0", 1: "class_1"}
        class_labels_desc = {0: "benign", 1: "malignant"}

    for cls, cls_name in class_names.items():
        cls_label = class_labels_desc.get(cls, cls_name)
        print(f"\n--- Generating {n_per_class} images: class {cls} ({cls_label}) ---")
        cls_dir = os.path.join(out_dir, cls_name)
        ensure_dir(cls_dir)

        generated = 0
        while generated < n_per_class:
            current_batch = min(batch_size, n_per_class - generated)

            labels = torch.full((current_batch,), cls, dtype=torch.long, device=device)
            context = class_embedder(labels)

            latents = torch.randn(
                current_batch,
                int(u["in_channels"]),
                int(u["sample_size"]),
                int(u["sample_size"]),
                device=device,
            )

            with autocast_ctx:
                for t in sched.timesteps:
                    # CFG dual forward pass at every denoising step
                    noise_pred = _cfg_step(unet, latents, t, context, guidance_scale)
                    latents = sched.step(noise_pred, t, latents).prev_sample

            imgs = (
                vae.decode((latents / scale_factor).to(vae.dtype)).sample.float()
                .clamp(-1, 1).add(1).div(2)
            )

            for i, img in enumerate(imgs):
                save_image(img, os.path.join(cls_dir, f"gen_{generated + i:04d}.png"))

            generated += current_batch
            print(f"  Progress: {generated}/{n_per_class}")

    print(f"\nFinal results saved to: {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate final evaluation image set from a trained LDM checkpoint with CFG."
    )
    p.add_argument("--config",          required=True,  help="Path to YAML config")
    p.add_argument("--ckpt",            required=True,  help="Path to ckpt.pt")
    p.add_argument("--n_per_class",     type=int,   default=1000)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--steps",           type=int,   default=50,  help="DDIM inference steps")
    p.add_argument("--guidance_scale",  type=float, default=7.5,
                   help="CFG guidance scale. 7.5 recommended for medical images.")
    p.add_argument("--out_dir",         default=None, help="Override output directory")
    args = p.parse_args()

    generate_final(
        config_path=args.config,
        ckpt_path=args.ckpt,
        n_per_class=args.n_per_class,
        batch_size=args.batch_size,
        sampler_steps=args.steps,
        guidance_scale=args.guidance_scale,
        out_dir_override=args.out_dir,
    )