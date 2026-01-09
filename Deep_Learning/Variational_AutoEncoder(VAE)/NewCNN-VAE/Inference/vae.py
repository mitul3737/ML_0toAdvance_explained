import json
from collections import namedtuple
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


LegacyParamSpec = namedtuple("LegacyParamSpec", ["key", "kind", "tf_shape"])
LEGACY_SCALE = 10000.0


def _device_from_flags(gpu_mode: bool, device: Optional[str]) -> torch.device:
    if device:
        return torch.device(device)
    if gpu_mode and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ConvVAE(nn.Module):
    """Convolutional VAE implemented with PyTorch."""

    def __init__(
        self,
        z_size: int = 32,
        batch_size: int = 1,
        learning_rate: float = 1e-4,
        kl_tolerance: float = 0.5,
        is_training: bool = False,
        reuse: bool = False,
        gpu_mode: bool = False,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.z_size = z_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.kl_tolerance = kl_tolerance
        self.is_training = is_training
        self.reuse = reuse
        self.device = _device_from_flags(gpu_mode, device)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(1024, z_size)
        self.fc_logvar = nn.Linear(1024, z_size)
        self.decoder_input = nn.Linear(z_size, 1024)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )
        self.to(self.device)
        self.eval()

    # ------------------------------------------------------------------
    # Core forward helpers
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tensor_x = self._prepare_input(x)
        mu, logvar = self._encode_latent(tensor_x)
        z = self._reparameterize(mu, logvar)
        recon = self._decode_latent(z)
        return recon, mu, logvar

    def encode(self, x: np.ndarray) -> np.ndarray:
        mu, logvar = self.encode_mu_logvar(x)
        noise = np.random.randn(*mu.shape).astype(np.float32)
        return mu + np.exp(0.5 * logvar) * noise

    def encode_mu_logvar(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            tensor_x = self._prepare_input(x)
            mu, logvar = self._encode_latent(tensor_x)
        return mu.cpu().numpy(), logvar.cpu().numpy()

    def decode(self, z: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor_z = self._prepare_latent(z)
            recon = self._decode_latent(tensor_z)
        recon = recon.permute(0, 2, 3, 1).cpu().numpy()
        return recon

    # ------------------------------------------------------------------
    # Loss utilities used during training
    def loss_components(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = F.mse_loss(recon_x, x, reduction="none")
        recon_loss = recon_loss.view(recon_loss.size(0), -1).sum(dim=1).mean()
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
        min_kl = self.kl_tolerance * mu.shape[1]

import numpy as np
import tensorflow as tf
import json
import os

# Building the CNN-VAE model within a class

