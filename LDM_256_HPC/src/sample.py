"""
sample.py — Ad-hoc sampling from a trained LDM checkpoint with CFG.

Generates a visual grid of class-conditional dermoscopic images.
Useful for qualitative evaluation and thesis figures.

Usage:
    python -m src.sample \
        --config configs/ldm_ham_128.yaml \
        --ckpt runs/ldm_ham_128_cfg/checkpoints/best/ckpt.pt \
        --n_per_class 8 \
        --steps 50 \
        --guidance_scale 7.5 \
        --fp16

Sampling pipeline:
    Pure Gaussian noise in latent space [B, 4, 16, 16]
        → DDIM denoising with CFG (two UNet passes per step)
        → Denoised latents
        → VAE decoder
        → Pixel images [B, 3, 128, 128]

CFG note:
    At each denoising step, the UNet runs twice — once with the class
    context and once with a zero context vector. The predictions are
    combined as:  noise_pred = uncond + scale * (cond - uncond).
    This amplifies class-specific features and produces images that are
    more morphologically distinctive per class.
"""
from __future__ import annotations

import os
import re
import yaml

import torch
from PIL import Image
from torchvision.utils import make_grid, save_image

from diffusers import DDPMScheduler, DDIMScheduler

from src.data import DataConfig
from src.model import ClassEmbedder, build_vae, build_unet
from src.utils import ensure_dir, seed_everything, detect_mixed_precision


# ── Utilities ─────────────────────────────────────────────────────────────────

def ckpt_tag(ckpt_path: str) -> str:
    """Extract a short human-readable tag from a checkpoint path for file naming."""
    m = re.search(r"step_(\d+)", os.path.dirname(ckpt_path))
    if m:
        return m.group(1).zfill(7)
    if "best" in ckpt_path:
        return "best"
    if "final" in ckpt_path:
        return "final"
    return os.path.splitext(os.path.basename(ckpt_path))[0][:32]


def build_scheduler(cfg: dict, sampler: str):
    """Build DDIM or DDPM scheduler from config."""
    sampler = sampler.lower().strip()
    common = dict(
        num_train_timesteps=int(cfg["diffusion"]["num_train_timesteps"]),
        beta_schedule=str(cfg["diffusion"]["beta_schedule"]),
        prediction_type=str(cfg["diffusion"]["prediction_type"]),
        clip_sample=False,
    )
    if sampler == "ddim":
        return DDIMScheduler(**common)
    if sampler == "ddpm":
        return DDPMScheduler(**common)
    raise ValueError(f"Unknown sampler '{sampler}'. Choose 'ddim' or 'ddpm'.")


def _cfg_step(
    unet: torch.nn.Module,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """
    Apply one CFG denoising step.

    Runs the UNet twice per timestep — once with the class context and once
    with a zero (unconditional) context — then combines them:
        noise_pred = uncond + guidance_scale * (cond - uncond)

    This amplifies the direction from the unconditional distribution toward
    the class-conditional distribution by the guidance_scale factor.
    """
    uncond_context = torch.zeros_like(context)
    noise_pred_uncond = unet(latents, timestep, encoder_hidden_states=uncond_context).sample
    noise_pred_cond   = unet(latents, timestep, encoder_hidden_states=context).sample
    return noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)


# ── Sampling ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def main(
    config_path: str,
    ckpt_path: str,
    sampler: str = "ddim",
    steps: int = 50,
    n_per_class: int = 8,
    guidance_scale: float = 7.5,
    save_individual: bool = False,
    seed: int | None = None,
    device_str: str | None = None,
    fp16: bool = False,
    out_dir_override: str | None = None,
) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed_everything(int(seed if seed is not None else cfg.get("seed", 42)))

    data_cfg = DataConfig(**cfg["data"])
    scale_factor = float(cfg["vae"]["scale_factor"])
    u = cfg["unet"]

    device = torch.device(device_str) if device_str else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    tag = ckpt_tag(ckpt_path)
    out_root = out_dir_override or cfg.get("output", {}).get("save_images_dir", "runs/ldm_ham_128_cfg/samples")
    run_dir = os.path.join(out_root, f"ckpt_{tag}", f"{sampler}_steps{steps}_cfg{guidance_scale}_n{n_per_class}")
    ensure_dir(run_dir)

    # Load VAE (frozen pretrained)
    vae = build_vae(cfg).to(device)

    # Load UNet and ClassEmbedder from checkpoint
    unet = build_unet(cfg).to(device)
    class_embedder = ClassEmbedder(
        num_classes=int(data_cfg.num_classes),
        embed_dim=int(u["cross_attention_dim"]),
    ).to(device)

    state = torch.load(ckpt_path, map_location="cpu")
    if "unet" not in state or "class_embedder" not in state:
        raise KeyError(
            f"Checkpoint missing required keys. Found: {list(state.keys())}"
        )
    unet.load_state_dict(state["unet"], strict=True)
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

    sched = build_scheduler(cfg, sampler)
    sched.set_timesteps(steps, device=device)

    print(f"[cfg] guidance_scale={guidance_scale}  sampler={sampler}  steps={steps}")

    mp_mode = detect_mixed_precision()
    if fp16 and device.type == "cuda" and mp_mode in ("bf16", "fp16"):
        _dtype = torch.bfloat16 if mp_mode == "bf16" else torch.float16
        autocast_ctx = torch.autocast(device_type="cuda", dtype=_dtype)
        print(f"[sample] Using {mp_mode} autocast.")
    else:
        autocast_ctx = torch.autocast(device_type="cpu", enabled=False)

    all_imgs = []

    for cls in range(int(data_cfg.num_classes)):
        labels = torch.full((n_per_class,), cls, dtype=torch.long, device=device)
        context = class_embedder(labels)

        latents = torch.randn(
            n_per_class,
            int(u["in_channels"]),
            int(u["sample_size"]),
            int(u["sample_size"]),
            device=device,
        )

        with autocast_ctx:
            for t in sched.timesteps:
                noise_pred = _cfg_step(unet, latents, t, context, guidance_scale)
                latents = sched.step(noise_pred, t, latents).prev_sample

        # Decode latents → pixel images in [0, 1]
        imgs = (
            vae.decode((latents / scale_factor).to(vae.dtype)).sample.float()
            .clamp(-1, 1).add(1).div(2)
        )
        all_imgs.append(imgs)

        if save_individual:
            cls_dir = os.path.join(run_dir, f"cls{cls}")
            ensure_dir(cls_dir)
            for i, img in enumerate(imgs):
                save_image(img, os.path.join(cls_dir, f"{tag}_cls{cls}_{i:03d}.png"))

    grid = make_grid(torch.cat(all_imgs, dim=0), nrow=n_per_class).clamp(0, 1)
    grid_np = (grid * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    out_path = os.path.join(run_dir, f"grid_{tag}_{sampler}_cfg{guidance_scale}_steps{steps}.png")
    Image.fromarray(grid_np).save(out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Sample from a trained LDM checkpoint with CFG.")
    p.add_argument("--config",          required=True)
    p.add_argument("--ckpt",            required=True)
    p.add_argument("--sampler",         default="ddim", choices=["ddim", "ddpm"])
    p.add_argument("--steps",           type=int,   default=50)
    p.add_argument("--n_per_class",     type=int,   default=8)
    p.add_argument("--guidance_scale",  type=float, default=7.5,
                   help="CFG guidance scale. 1.0=no guidance, 7.5=recommended, >15=oversaturated.")
    p.add_argument("--save_individual", action="store_true")
    p.add_argument("--seed",            type=int,   default=None)
    p.add_argument("--device",          type=str,   default=None)
    p.add_argument("--fp16",            action="store_true",
                   help="Enable bf16/fp16 autocast during inference.")
    p.add_argument("--out_dir",         type=str,   default=None)
    args = p.parse_args()

    main(
        config_path=args.config,
        ckpt_path=args.ckpt,
        sampler=args.sampler,
        steps=args.steps,
        n_per_class=args.n_per_class,
        guidance_scale=args.guidance_scale,
        save_individual=args.save_individual,
        seed=args.seed,
        device_str=args.device,
        fp16=args.fp16,
        out_dir_override=args.out_dir,
    )