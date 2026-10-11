"""
train.py — LDM training script with Classifier-Free Guidance (CFG).

Training flow:
    1. Load pretrained VAE (frozen) and initialise UNet + ClassEmbedder
    2. For each batch:
       a. Encode pixel images to latent space via VAE encoder
       b. Sample random timesteps and add noise to latents
       c. With probability cfg_dropout_prob, zero out the class context
          so the model learns unconditional generation (required for CFG)
       d. Predict noise with UNet, conditioned on class embedding
       e. Compute MSE loss between predicted and actual noise
    3. Checkpoint, validate, and sample at configured intervals

CFG change vs. previous version:
    A single line added in the training loop zeros the class context vector
    with probability cfg_dropout_prob (default 0.1). This teaches the model
    to generate without any conditioning signal, which is the prerequisite
    for CFG amplification at inference time. No architectural changes needed.

The data pipeline (data.py) and all utility functions (utils.py) are
identical to the DDPM baseline, ensuring a fair architectural comparison.
"""
from __future__ import annotations

import os
import re
import math
import yaml
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from PIL import Image

from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_cosine_schedule_with_warmup

from src.data import DataConfig, build_datasets, build_weighted_sampler
from src.model import ClassEmbedder, build_vae, build_unet
from src.utils import (
    seed_everything, ensure_dir,
    detect_mixed_precision, get_num_workers,
    check_gpu, log_cluster_env,
    count_parameters, format_param_count, get_gpu_memory_summary,
)


# ── Config dataclass ──────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    batch_size: int
    grad_accum_steps: int
    lr: float
    weight_decay: float
    max_train_steps: int
    grad_clip_norm: float
    mixed_precision: str
    save_every_steps: int
    log_every_steps: int
    warmup_steps: int = 500
    cfg_dropout_prob: float = 0.1


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_eval_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    e = cfg.get("eval", {}) or {}
    return {
        "val_every_steps":    int(e.get("val_every_steps",    2000)),
        "max_val_batches":    int(e.get("max_val_batches",    20)),
        "sample_every_steps": int(e.get("sample_every_steps", 10000)),
        "sample_steps":       int(e.get("sample_steps",       50)),
        "sample_n_per_class": int(e.get("sample_n_per_class", 8)),
        "guidance_scale":     float(e.get("guidance_scale",   7.5)),
    }


def _find_latest_ckpt(out_dir: str) -> Optional[str]:
    """Return the path to the most recent periodic checkpoint for auto-resume."""
    ckpt_root = os.path.join(out_dir, "checkpoints")
    if not os.path.isdir(ckpt_root):
        return None
    step_dirs: List[tuple] = []
    for name in os.listdir(ckpt_root):
        m = re.match(r"step_(\d+)$", name)
        if m:
            ckpt_file = os.path.join(ckpt_root, name, "ckpt.pt")
            if os.path.isfile(ckpt_file):
                step_dirs.append((int(m.group(1)), ckpt_file))
    if step_dirs:
        step_dirs.sort(key=lambda x: x[0])
        return step_dirs[-1][1]
    return None


# ── VAE encode / decode ───────────────────────────────────────────────────────

@torch.no_grad()
def encode_images(
    vae: AutoencoderKL,
    images: torch.Tensor,
    scale_factor: float,
) -> torch.Tensor:
    """
    Encode a batch of pixel images into VAE latent space.

    We use .sample() from the posterior distribution rather than .mode()
    to introduce stochasticity during training, which improves diversity.
    The scale_factor normalises latents so their variance matches the unit
    Gaussian noise added during the diffusion forward process.

    Args:
        vae:          Frozen AutoencoderKL.
        images:       Pixel images [B, 3, H, W] normalised to [-1, 1].
        scale_factor: VAE-specific constant (0.18215 for SD VAE).

    Returns:
        Latents [B, 4, H/8, W/8].
    """
    latents = vae.encode(images.to(vae.dtype)).latent_dist.sample()
    return latents.float() * scale_factor


@torch.no_grad()
def decode_latents(
    vae: AutoencoderKL,
    latents: torch.Tensor,
    scale_factor: float,
) -> torch.Tensor:
    """
    Decode latents back to pixel space and normalise to [0, 1] for saving.

    Args:
        vae:          Frozen AutoencoderKL.
        latents:      Denoised latents [B, 4, H/8, W/8].
        scale_factor: Same constant used during encoding.

    Returns:
        Images [B, 3, H, W] in [0, 1].
    """
    latents = (latents / scale_factor).to(vae.dtype)
    images = vae.decode(latents).sample.float()
    return (images.clamp(-1, 1) + 1) / 2


# ── CFG sampling helper ───────────────────────────────────────────────────────

def _cfg_denoise(
    unet: torch.nn.Module,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """
    Single CFG denoising step: two UNet forward passes combined with guidance.

    The unconditional pass uses a zero context vector — the model has been
    trained to handle this via conditioning dropout. The conditional and
    unconditional predictions are then combined:

        noise_pred = uncond + guidance_scale * (cond - uncond)

    When guidance_scale=1.0, this reduces to pure conditional sampling.
    When guidance_scale=7.5, the class signal is amplified 7.5x relative
    to the unconditional baseline, producing images that are more strongly
    characteristic of the target class.

    Args:
        unet:           The denoising UNet (wrapped by Accelerate).
        latents:        Current noisy latents [B, 4, H, W].
        timestep:       Current diffusion timestep (scalar tensor).
        context:        Class conditioning context [B, 1, cross_attention_dim].
        guidance_scale: CFG amplification factor.

    Returns:
        Combined noise prediction [B, 4, H, W].
    """
    # Unconditional forward pass — zero context vector
    uncond_context = torch.zeros_like(context)
    noise_pred_uncond = unet(latents, timestep, encoder_hidden_states=uncond_context).sample

    # Conditional forward pass — class context vector
    noise_pred_cond = unet(latents, timestep, encoder_hidden_states=context).sample

    # CFG combination: amplify the direction from uncond toward cond
    return noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)


# ── Sampling (monitoring grids) ───────────────────────────────────────────────

@torch.no_grad()
def _sample_grid(
    accelerator: Accelerator,
    unet: torch.nn.Module,
    class_embedder: ClassEmbedder,
    vae: AutoencoderKL,
    ddim: DDIMScheduler,
    cfg: Dict[str, Any],
    out_path: str,
    steps: int,
    n_per_class: int,
    guidance_scale: float,
) -> None:
    """
    Generate a monitoring grid of class-conditional samples using DDIM + CFG.

    Sampling happens entirely in latent space. The VAE decoder is called
    once at the end to produce pixel images for visualisation.
    The DDIMScheduler is passed in rather than constructed here to avoid
    redundant object creation on every call.
    """
    device = accelerator.device
    scale_factor = float(cfg["vae"]["scale_factor"])
    u = cfg["unet"]
    ddim.set_timesteps(steps, device=device)

    all_imgs = []
    for cls in range(int(cfg["data"]["num_classes"])):
        labels = torch.full((n_per_class,), cls, dtype=torch.long, device=device)
        context = class_embedder(labels)  # [B, 1, cross_attention_dim]

        latents = torch.randn(
            n_per_class,
            int(u["in_channels"]),
            int(u["sample_size"]),
            int(u["sample_size"]),
            device=device,
        )

        for t in ddim.timesteps:
            noise_pred = _cfg_denoise(unet, latents, t, context, guidance_scale)
            latents = ddim.step(noise_pred, t, latents).prev_sample

        all_imgs.append(decode_latents(vae, latents, scale_factor))

    grid = make_grid(torch.cat(all_imgs, dim=0), nrow=n_per_class).clamp(0, 1)
    grid_np = (grid * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(grid_np).save(out_path)


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def _validate(
    accelerator: Accelerator,
    unet: torch.nn.Module,
    class_embedder: ClassEmbedder,
    vae: AutoencoderKL,
    noise_scheduler: DDPMScheduler,
    val_loader: DataLoader,
    scale_factor: float,
    max_batches: int,
) -> float:
    """
    Compute average validation denoising loss (MSE in latent space).

    Validation uses the full conditional context (no CFG dropout) to measure
    the model's best-case performance on the denoising objective.
    """
    unet.eval()
    class_embedder.eval()
    total, count = 0.0, 0

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        x = batch["pixel_values"]
        y = batch["class_labels"].long()

        latents = encode_images(vae, x, scale_factor)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=latents.device,
        ).long()
        noisy = noise_scheduler.add_noise(latents, noise, timesteps)
        context = class_embedder(y)
        pred = unet(noisy, timesteps, encoder_hidden_states=context).sample
        loss = F.mse_loss(pred, noise, reduction="mean")
        total += accelerator.gather_for_metrics(loss.detach()).mean().item()
        count += 1

    unet.train()
    class_embedder.train()
    return total / max(count, 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(config_path: str, resume_path: Optional[str] = None) -> None:
    log_cluster_env()
    check_gpu(min_memory_gb=8.0)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    eval_cfg = _get_eval_cfg(cfg)
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    data_cfg = DataConfig(**cfg["data"])
    train_cfg = TrainConfig(**cfg["train"])
    scale_factor = float(cfg["vae"]["scale_factor"])

    out_dir = cfg["output"]["out_dir"]
    samples_dir = cfg["output"]["save_images_dir"]
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    ensure_dir(out_dir)
    ensure_dir(samples_dir)
    ensure_dir(ckpt_dir)

    _mp = train_cfg.mixed_precision
    if _mp == "auto":
        _mp = detect_mixed_precision()
        print(f"[precision] auto-detected: {_mp}")

    accelerator = Accelerator(
        mixed_precision=None if _mp == "no" else _mp,
        gradient_accumulation_steps=int(train_cfg.grad_accum_steps),
        log_with="tensorboard",
        project_dir=out_dir,
    )

    # ── Data ─────────────────────────────────────────────────────────────────
    train_ds, val_ds, stats = build_datasets(data_cfg, seed=seed)
    _nw = get_num_workers(int(data_cfg.num_workers))
    train_sampler = build_weighted_sampler(
        train_ds.labels,
        num_classes=int(data_cfg.num_classes),
        temperature=float(data_cfg.sampler_temperature),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.batch_size),
        shuffle=False,            # sampler i shuffle se medusobno iskljucuju
        sampler=train_sampler,
        num_workers=_nw,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=(_nw > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg.batch_size),
        shuffle=False,
        num_workers=_nw,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(_nw > 0),
    )

    # ── Models ────────────────────────────────────────────────────────────────
    vae = build_vae(cfg).to(accelerator.device)
    unet = build_unet(cfg)
    class_embedder = ClassEmbedder(
        num_classes=int(data_cfg.num_classes),
        embed_dim=int(cfg["unet"]["cross_attention_dim"]),
    )

    params = count_parameters(unet)
    accelerator.print(
        f"[model] UNet: {format_param_count(params['trainable'])} trainable params "
        f"| {format_param_count(params['total'])} total"
    )
    accelerator.print(
        f"[model] ClassEmbedder: "
        f"{format_param_count(sum(p.numel() for p in class_embedder.parameters()))} params"
    )
    accelerator.print(
        f"[cfg] dropout_prob={train_cfg.cfg_dropout_prob}  "
        f"guidance_scale={eval_cfg['guidance_scale']}"
    )
    accelerator.print(f"[gpu] Memory after model init: {get_gpu_memory_summary()}")

    # ── Schedulers ────────────────────────────────────────────────────────────
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=int(cfg["diffusion"]["num_train_timesteps"]),
        beta_schedule=str(cfg["diffusion"]["beta_schedule"]),
        prediction_type=str(cfg["diffusion"]["prediction_type"]),
    )
    ddim_scheduler = DDIMScheduler(
        num_train_timesteps=int(cfg["diffusion"]["num_train_timesteps"]),
        beta_schedule=str(cfg["diffusion"]["beta_schedule"]),
        prediction_type=str(cfg["diffusion"]["prediction_type"]),
        clip_sample=False,
    )

    # ── Optimiser ─────────────────────────────────────────────────────────────
    trainable_params = list(unet.parameters()) + list(class_embedder.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(train_cfg.lr),
        weight_decay=float(train_cfg.weight_decay),
    )

    # ── Step counts ───────────────────────────────────────────────────────────
    steps_per_epoch = math.ceil(len(train_loader) / int(train_cfg.grad_accum_steps))
    total_steps = int(train_cfg.max_train_steps)
    epochs = math.ceil(total_steps / max(steps_per_epoch, 1))

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(train_cfg.warmup_steps),
        num_training_steps=total_steps,
    )

    # ── EMA ───────────────────────────────────────────────────────────────────
    use_ema = bool(cfg.get("ema", {}).get("use_ema", True))
    ema_decay = float(cfg.get("ema", {}).get("decay", 0.9999))
    ema: Optional[EMAModel] = None

    # ── Accelerate prepare ────────────────────────────────────────────────────
    unet, class_embedder, optimizer, train_loader, val_loader, lr_scheduler = (
        accelerator.prepare(
            unet, class_embedder, optimizer, train_loader, val_loader, lr_scheduler
        )
    )

    if use_ema and accelerator.is_main_process:
        ema = EMAModel(accelerator.unwrap_model(unet).parameters(), decay=ema_decay)

    accelerator.init_trackers(os.path.basename(out_dir))

    # ── Auto-resume ───────────────────────────────────────────────────────────
    if resume_path is None:
        resume_path = _find_latest_ckpt(out_dir)

    start_step = 0
    best_val = float("inf")

    if resume_path is not None and os.path.isfile(resume_path):
        try:
            ckpt = torch.load(resume_path, map_location="cpu")
            accelerator.unwrap_model(unet).load_state_dict(ckpt["unet"], strict=True)
            accelerator.unwrap_model(class_embedder).load_state_dict(ckpt["class_embedder"])
            optimizer.load_state_dict(ckpt["optimizer"])
            if "lr_scheduler" in ckpt:
                lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
            start_step = int(ckpt.get("step", 0))
            if ckpt.get("val_loss") is not None:
                best_val = float(ckpt["val_loss"])
            if ema is not None and "ema" in ckpt:
                ema.load_state_dict(ckpt["ema"])
            accelerator.print(
                f"Resumed from {resume_path} at step={start_step}, best_val={best_val:.6f}"
            )
        except Exception as exc:
            accelerator.print(f"[resume] Failed to load '{resume_path}': {exc}. Starting fresh.")
            start_step = 0
            best_val = float("inf")

    if accelerator.is_main_process:
        accelerator.print("=== DATA STATS ===")
        accelerator.print(stats)
        accelerator.print(
            f"steps/epoch={steps_per_epoch}  "
            f"total_steps={total_steps}  "
            f"epochs={epochs}"
        )

    # ── Checkpoint helper ─────────────────────────────────────────────────────
    def _save_ckpt(tag: str, step: int, val_loss: Optional[float] = None) -> None:
        if not accelerator.is_main_process:
            return
        path = os.path.join(ckpt_dir, tag, "ckpt.pt")
        ensure_dir(os.path.dirname(path))
        payload = {
            "step": step,
            "unet": accelerator.unwrap_model(unet).state_dict(),
            "class_embedder": accelerator.unwrap_model(class_embedder).state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "val_loss": val_loss,
            "config_path": config_path,
        }
        if ema is not None:
            payload["ema"] = ema.state_dict()
        tmp = path + ".tmp"
        try:
            torch.save(payload, tmp)
            os.replace(tmp, path)
            accelerator.print(f"saved: {path}")
        except Exception as exc:
            accelerator.print(f"[checkpoint] save failed: {exc}")
            if os.path.exists(tmp):
                os.remove(tmp)

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step = start_step
    running_loss, micro_steps = 0.0, 0
    unet.train()
    class_embedder.train()

    for _ in range(epochs):
        for batch in train_loader:
            with accelerator.accumulate(unet, class_embedder):
                x = batch["pixel_values"]
                y = batch["class_labels"].long()

                latents = encode_images(vae, x, scale_factor)
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=latents.device,
                ).long()

                noisy = noise_scheduler.add_noise(latents, noise, timesteps)
                context = class_embedder(y)

                # ── CFG conditioning dropout ──────────────────────────────────
                # With probability cfg_dropout_prob, replace the class context
                # with a zero vector. This teaches the model to generate without
                # any conditioning signal — a prerequisite for CFG at inference.
                # Without this, the model cannot produce a meaningful unconditional
                # prediction, and CFG guidance would amplify noise rather than
                # class-specific features.
                if train_cfg.cfg_dropout_prob > 0:
                    dropout_mask = torch.rand(context.shape[0], device=context.device)
                    dropout_mask = (dropout_mask < train_cfg.cfg_dropout_prob).float()
                    # Reshape to [B, 1, 1] to broadcast over seq_len and embed_dim
                    dropout_mask = dropout_mask.view(-1, 1, 1)
                    context = context * (1.0 - dropout_mask)

                noise_pred = unet(noisy, timesteps, encoder_hidden_states=context).sample
                loss = F.mse_loss(noise_pred, noise, reduction="mean")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    if float(train_cfg.grad_clip_norm) > 0:
                        accelerator.clip_grad_norm_(trainable_params, float(train_cfg.grad_clip_norm))
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            running_loss += loss.detach().item()
            micro_steps += 1

            if not accelerator.sync_gradients:
                continue

            if ema is not None:
                ema.step(accelerator.unwrap_model(unet).parameters())

            global_step += 1
            avg_loss = running_loss / micro_steps
            running_loss, micro_steps = 0.0, 0

            # Logging
            if global_step % int(train_cfg.log_every_steps) == 0:
                current_lr = lr_scheduler.get_last_lr()[0]
                if accelerator.is_main_process:
                    accelerator.print(
                        f"step {global_step}/{total_steps}  "
                        f"loss={avg_loss:.4f}  lr={current_lr:.2e}"
                    )
                accelerator.log(
                    {"train/loss": avg_loss, "train/lr": current_lr},
                    step=global_step,
                )

            # Validation
            if global_step % int(eval_cfg["val_every_steps"]) == 0:
                val_loss = _validate(
                    accelerator, unet, class_embedder, vae,
                    noise_scheduler, val_loader, scale_factor,
                    int(eval_cfg["max_val_batches"]),
                )
                if accelerator.is_main_process:
                    accelerator.print(f"[val] step {global_step}  val_loss={val_loss:.4f}")
                    accelerator.log({"val/loss": val_loss}, step=global_step)
                    if val_loss < best_val:
                        best_val = val_loss
                        _save_ckpt("best", step=global_step, val_loss=val_loss)

            # Monitoring samples with CFG
            if global_step % int(eval_cfg["sample_every_steps"]) == 0 and accelerator.is_main_process:
                unwrapped_unet = accelerator.unwrap_model(unet)
                unwrapped_emb = accelerator.unwrap_model(class_embedder)
                backup = None
                if ema is not None:
                    backup = {k: v.clone() for k, v in unwrapped_unet.state_dict().items()}
                    ema.copy_to(unwrapped_unet.parameters())

                unwrapped_unet.eval()
                unwrapped_emb.eval()
                out_path = os.path.join(samples_dir, f"sample_step_{global_step:07d}.png")
                _sample_grid(
                    accelerator, unwrapped_unet, unwrapped_emb, vae,
                    ddim_scheduler, cfg, out_path,
                    steps=int(eval_cfg["sample_steps"]),
                    n_per_class=int(eval_cfg["sample_n_per_class"]),
                    guidance_scale=float(eval_cfg["guidance_scale"]),
                )
                accelerator.print(f"sampled: {out_path}")
                unwrapped_unet.train()
                unwrapped_emb.train()

                if ema is not None and backup is not None:
                    unwrapped_unet.load_state_dict(backup)

            # Periodic checkpoint
            if global_step % int(train_cfg.save_every_steps) == 0:
                accelerator.wait_for_everyone()
                _save_ckpt(f"step_{global_step:07d}", step=global_step)

            if global_step >= total_steps:
                break

        if global_step >= total_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _save_ckpt("final", step=global_step, val_loss=best_val)

    accelerator.end_training()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument(
        "--resume", default=None,
        help="Path to ckpt.pt to resume from. "
             "If omitted, auto-detects the latest step checkpoint.",
    )
    args = p.parse_args()
    main(args.config, resume_path=args.resume)