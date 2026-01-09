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
    kl_loss = torch.clamp(kl_loss, min=min_kl).mean()
    total_loss = recon_loss + kl_loss
    return total_loss, recon_loss, kl_loss

  # ------------------------------------------------------------------
  # Persistence helpers
  def load_json(self, filepath: Union[str, Path]) -> None:
    path = Path(filepath)
    if path.suffix in {".pt", ".pth"}:
      checkpoint = torch.load(path, map_location=self.device)
      self.load_state_dict(checkpoint)
      self.eval()
      return
    if not path.exists():
      raise FileNotFoundError(f"Missing VAE weights at {path}")
    self._load_legacy_json(path)
    self.eval()

  def save_json(self, filepath: Union[str, Path]) -> None:
    path = Path(filepath)
    if path.suffix in {".pt", ".pth"}:
      torch.save(self.state_dict(), path)
      return
    self._save_legacy_json(path)

  def set_random_params(self, stdev: float = 0.5) -> None:
    for param in self.parameters():
      nn.init.normal_(param, mean=0.0, std=stdev)

  def get_random_model_params(self, stdev: float = 0.5) -> List[np.ndarray]:
    random_params = []
    for tensor in self.parameters():
      shape = tensor.shape
      random_params.append(np.random.standard_cauchy(shape).astype(np.float32) * stdev)
    return random_params

  def set_model_params(self, params: Sequence[np.ndarray]) -> None:
    with torch.no_grad():
      named_params = list(self.named_parameters())
      if len(named_params) != len(params):
        raise ValueError("Parameter count mismatch when restoring ConvVAE weights")
      for (name, tensor), values in zip(named_params, params):
        array = np.asarray(values, dtype=np.float32)
        divisor = LEGACY_SCALE if array.dtype.kind in {"i", "u"} else 1.0
        array = array / divisor
        reshaped = array.reshape(tensor.shape)
        tensor.copy_(torch.from_numpy(reshaped).to(self.device))

  def get_model_params(self) -> tuple[List[List], List[tuple], List[str]]:
    params = []
    shapes = []
    names = []
    for name, tensor in self.named_parameters():
      array = tensor.detach().cpu().numpy()
      params.append(np.round(array * LEGACY_SCALE).astype(np.int32).tolist())
      shapes.append(tuple(array.shape))
      names.append(name)
    return params, shapes, names

  # ------------------------------------------------------------------
  # Internal helpers
  def _prepare_input(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(x, np.ndarray):
      tensor = torch.from_numpy(x).float()
    else:
      tensor = x.float()
    if tensor.dim() == 3:
      tensor = tensor.unsqueeze(0)
    if tensor.shape[-1] == 3:
      tensor = tensor.permute(0, 3, 1, 2)
    tensor = tensor.to(self.device)
    return tensor

  def _prepare_latent(self, z: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(z, np.ndarray):
      tensor = torch.from_numpy(z).float()
    else:
      tensor = z.float()
    if tensor.dim() == 1:
      tensor = tensor.unsqueeze(0)
    return tensor.to(self.device)

  def _encode_latent(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h = self.encoder(x)
    h = h.view(h.size(0), -1)
    mu = self.fc_mu(h)
    logvar = self.fc_logvar(h)
    return mu, logvar

  def _decode_latent(self, z: torch.Tensor) -> torch.Tensor:
    h = self.decoder_input(z)
    h = h.view(h.size(0), 1024, 1, 1)
    return self.decoder(h)

  @staticmethod
  def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

  def _load_legacy_json(self, path: Path) -> None:
    specs = self._legacy_param_specs()
    with path.open("r", encoding="utf-8") as handle:
      payload = json.load(handle)
    if len(payload) != len(specs):
      raise ValueError("Legacy VAE weight file has unexpected parameter count")
    params_map = {name: tensor for name, tensor in self.named_parameters()}
    with torch.no_grad():
      for spec, raw_values in zip(specs, payload):
        array = np.asarray(raw_values, dtype=np.float32) / LEGACY_SCALE
        reshaped = array.reshape(spec.tf_shape)
        if spec.kind in {"conv", "deconv"}:
          reshaped = reshaped.transpose(3, 2, 0, 1)
        elif spec.kind == "linear":
          reshaped = reshaped.T
        tensor = params_map[spec.key]
        tensor.copy_(torch.from_numpy(reshaped).to(self.device))

  def _save_legacy_json(self, path: Path) -> None:
    specs = self._legacy_param_specs()
    ordered_params = {name: tensor for name, tensor in self.named_parameters()}
    serialisable: List[List] = []
    for spec in specs:
      tensor = ordered_params[spec.key].detach().cpu().numpy()
      if spec.kind in {"conv", "deconv"}:
        tensor = tensor.transpose(2, 3, 1, 0)
      elif spec.kind == "linear":
        tensor = tensor.T
      tensor = np.round(tensor * LEGACY_SCALE).astype(np.int32)
      serialisable.append(tensor.tolist())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
      json.dump(serialisable, handle, separators=(",", ":"))

  @staticmethod
  def _legacy_param_specs() -> List[LegacyParamSpec]:
    return [
      LegacyParamSpec("encoder.0.weight", "conv", (4, 4, 3, 32)),
      LegacyParamSpec("encoder.0.bias", "bias", (32,)),
      LegacyParamSpec("encoder.2.weight", "conv", (4, 4, 32, 64)),
      LegacyParamSpec("encoder.2.bias", "bias", (64,)),
      LegacyParamSpec("encoder.4.weight", "conv", (4, 4, 64, 128)),
      LegacyParamSpec("encoder.4.bias", "bias", (128,)),
      LegacyParamSpec("encoder.6.weight", "conv", (4, 4, 128, 256)),
      LegacyParamSpec("encoder.6.bias", "bias", (256,)),
      LegacyParamSpec("fc_mu.weight", "linear", (1024, 32)),
      LegacyParamSpec("fc_mu.bias", "bias", (32,)),
      LegacyParamSpec("fc_logvar.weight", "linear", (1024, 32)),
      LegacyParamSpec("fc_logvar.bias", "bias", (32,)),
      LegacyParamSpec("decoder_input.weight", "linear", (32, 1024)),
      LegacyParamSpec("decoder_input.bias", "bias", (1024,)),
      LegacyParamSpec("decoder.0.weight", "deconv", (5, 5, 128, 1024)),
      LegacyParamSpec("decoder.0.bias", "bias", (128,)),
      LegacyParamSpec("decoder.2.weight", "deconv", (5, 5, 64, 128)),
      LegacyParamSpec("decoder.2.bias", "bias", (64,)),
      LegacyParamSpec("decoder.4.weight", "deconv", (6, 6, 32, 64)),
      LegacyParamSpec("decoder.4.bias", "bias", (32,)),
      LegacyParamSpec("decoder.6.weight", "deconv", (6, 6, 3, 32)),
      LegacyParamSpec("decoder.6.bias", "bias", (3,)),
    ]


def reset_graph() -> None:
  """Kept for backward compatibility with TensorFlow-era scripts."""
  torch.cuda.empty_cache()
