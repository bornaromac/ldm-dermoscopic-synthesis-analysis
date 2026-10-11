"""
model.py — Model architecture for Latent Diffusion Model (LDM).

Provides three components:
  - ClassEmbedder  : maps integer class labels to cross-attention context vectors
  - build_vae      : loads the pretrained Stable Diffusion VAE (frozen)
  - build_unet     : builds the conditional UNet operating in latent space

Architecture overview:
    Images [B, 3, 128, 128]
        │
        ▼ VAE encoder (frozen, pretrained)
    Latents [B, 4, 16, 16]  ← diffusion process runs here
        │
        ▼ VAE decoder (frozen, pretrained)
    Images [B, 3, 256, 256]

The UNet never sees pixel values. It learns to denoise in the VAE's latent
space, which is 64x smaller than pixel space. This is why LDM is faster to
train and produces higher-quality outputs than pixel-space DDPM at the same
compute budget.
"""
from __future__ import annotations

from typing import Dict, Any

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet2DConditionModel


class ClassEmbedder(nn.Module):

    def __init__(self, num_classes: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            labels: Integer class labels of shape [B].

        Returns:
            Context tensor of shape [B, 1, embed_dim] suitable for passing
            as encoder_hidden_states to UNet2DConditionModel.
        """
        return self.embedding(labels).unsqueeze(1)  # [B] → [B, 1, embed_dim]


def build_vae(cfg: Dict[str, Any]) -> AutoencoderKL:
    vae = AutoencoderKL.from_pretrained(
        cfg["vae"]["pretrained"],
        torch_dtype=torch.bfloat16,
    )
    vae.requires_grad_(False)
    vae.eval()
    return vae


def build_unet(cfg: Dict[str, Any]) -> UNet2DConditionModel:

    u = cfg["unet"]
    unet = UNet2DConditionModel(
        sample_size=int(u["sample_size"]),
        in_channels=int(u["in_channels"]),
        out_channels=int(u["out_channels"]),
        layers_per_block=int(u["layers_per_block"]),
        block_out_channels=tuple(u["block_out_channels"]),
        down_block_types=tuple(u["down_block_types"]),
        up_block_types=tuple(u["up_block_types"]),
        cross_attention_dim=int(u["cross_attention_dim"]),
        dropout=float(u.get("dropout", 0.0)),
    )
    if bool(u.get("gradient_checkpointing", False)):
        unet.enable_gradient_checkpointing()
    return unet