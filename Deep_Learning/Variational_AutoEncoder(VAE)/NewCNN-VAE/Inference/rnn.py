from __future__ import annotations

import json
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn


MODE_ZCH = 0
MODE_ZC = 1
MODE_Z = 2
MODE_Z_HIDDEN = 3
MODE_ZH = 4

LEGACY_SCALE = 10000.0
LOG_SQRT_2PI = float(np.log(np.sqrt(2.0 * np.pi)))

HyperParams = namedtuple(
    "HyperParams",
    [
        "num_steps",
        "max_seq_len",
        "input_seq_width",
        "output_seq_width",
        "rnn_size",
        "batch_size",
        "grad_clip",
        "num_mixture",
        "learning_rate",
        "decay_rate",
        "min_learning_rate",
        "use_layer_norm",
        "use_recurrent_dropout",
        "recurrent_dropout_prob",
        "use_input_dropout",
        "input_dropout_prob",
        "use_output_dropout",
        "output_dropout_prob",
        "is_training",
    ],
)


def default_hps() -> HyperParams:
    return HyperParams(
        num_steps=2000,
        max_seq_len=1000,
        input_seq_width=35,
        output_seq_width=32,
        rnn_size=256,
        batch_size=100,
        grad_clip=1.0,
        num_mixture=5,
        learning_rate=0.001,
        decay_rate=1.0,
        min_learning_rate=0.00001,
        use_layer_norm=0,
        use_recurrent_dropout=0,
        recurrent_dropout_prob=0.90,
        use_input_dropout=0,
        input_dropout_prob=0.90,
        use_output_dropout=0,
        output_dropout_prob=0.90,
        is_training=1,
    )


hps_model = default_hps()
hps_sample = hps_model._replace(batch_size=1, max_seq_len=1, use_recurrent_dropout=0, is_training=0)


@dataclass
class RNNState:
    h: torch.Tensor
    c: torch.Tensor

    def to(self, device: torch.device) -> "RNNState":
        return RNNState(self.h.to(device), self.c.to(device))

    def detach(self) -> "RNNState":
        return RNNState(self.h.detach(), self.c.detach())


def _device_from_flags(gpu_mode: bool, device: Optional[str]) -> torch.device:
    if device:
        return torch.device(device)
    if gpu_mode and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class MDNRNN(nn.Module):
    """MDN-RNN implemented with PyTorch."""

    def __init__(
        self,
        hps: HyperParams,
        reuse: bool = False,
        gpu_mode: bool = False,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.hps = hps
        self.device = _device_from_flags(gpu_mode, device)
        self.num_mixture = hps.num_mixture
        self.output_dim = hps.output_seq_width
        self.rnn = nn.LSTM(
            input_size=hps.input_seq_width,
            hidden_size=hps.rnn_size,
            batch_first=True,
        )
        self.proj = nn.Linear(hps.rnn_size, hps.output_seq_width * hps.num_mixture * 3)
        self.to(self.device)
        self.eval()

    # ------------------------------------------------------------------
    def forward(
        self,
        inputs: Union[np.ndarray, torch.Tensor],
        state: Optional[RNNState] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, RNNState]:
        x = self._prepare_inputs(inputs)
        batch_size = x.size(0)
        initial_state = self._resolve_state(state, batch_size)
        outputs, (h_n, c_n) = self.rnn(x, (initial_state.h, initial_state.c))
        logits = self.proj(outputs)
        logmix, mean, logstd = self._split_mixture_params(logits)
        logmix = logmix - torch.logsumexp(logmix, dim=-1, keepdim=True)
        return logmix, mean, logstd, RNNState(h_n, c_n)

    def loss(
        self,
        targets: torch.Tensor,
        logmix: torch.Tensor,
        mean: torch.Tensor,
        logstd: torch.Tensor,
    ) -> torch.Tensor:
        y = targets.unsqueeze(-1)
        inv_std = torch.exp(-logstd)
        centered = (y - mean) * inv_std
        log_probs = logmix - logstd - LOG_SQRT_2PI - 0.5 * centered.pow(2)
        log_probs = torch.logsumexp(log_probs, dim=-1)
        return -log_probs.mean()

    def sample(
        self,
        logmix: torch.Tensor,
        mean: torch.Tensor,
        logstd: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        logmix = logmix / max(temperature, 1e-5)
        logmix = logmix - torch.logsumexp(logmix, dim=-1, keepdim=True)
        categorical = torch.distributions.Categorical(logits=logmix)
        mixture_idx = categorical.sample()
        gather_idx = mixture_idx.unsqueeze(-1)
        chosen_mean = torch.gather(mean, -1, gather_idx)
        chosen_logstd = torch.gather(logstd, -1, gather_idx)
        rand = torch.randn_like(chosen_mean) * temperature
        sample = chosen_mean + torch.exp(chosen_logstd) * rand
        return sample.squeeze(-1)

    # ------------------------------------------------------------------
    def init_state(self, batch_size: int) -> RNNState:
        h = torch.zeros(1, batch_size, self.hps.rnn_size, device=self.device)
        c = torch.zeros_like(h)
        return RNNState(h, c)

    def reset_parameters(self, stdev: float = 0.1) -> None:
        for param in self.parameters():
            nn.init.normal_(param, mean=0.0, std=stdev)

    def get_model_params(self) -> tuple[List[List], List[tuple], List[str]]:
        params: List[List] = []
        shapes: List[tuple] = []
        names: List[str] = []
        for name, tensor in self.named_parameters():
            array = tensor.detach().cpu().numpy()
            params.append(np.round(array * LEGACY_SCALE).astype(np.int32).tolist())
            shapes.append(tuple(array.shape))
            names.append(name)
        return params, shapes, names

    def get_random_model_params(self, stdev: float = 0.5) -> List[np.ndarray]:
        random_params = []
        for tensor in self.parameters():
            shape = tensor.shape
            random_params.append(np.random.standard_cauchy(shape).astype(np.float32) * stdev)
        return random_params

    def set_random_params(self, stdev: float = 0.5) -> None:
        with torch.no_grad():
            for tensor in self.parameters():
                tensor.copy_(torch.from_numpy(np.random.standard_cauchy(tensor.shape)).float().to(self.device) * stdev)

    def set_model_params(self, params: Sequence[np.ndarray]) -> None:
        with torch.no_grad():
            named_params = list(self.named_parameters())
            if len(named_params) != len(params):
                raise ValueError("Parameter count mismatch when restoring MDNRNN weights")
            for (name, tensor), values in zip(named_params, params):
                array = np.asarray(values, dtype=np.float32)
                divisor = LEGACY_SCALE if array.dtype.kind in {"i", "u"} else 1.0
                array = array / divisor
                reshaped = array.reshape(tensor.shape)
                tensor.copy_(torch.from_numpy(reshaped).to(self.device))

    def load_json(self, filepath: Union[str, Path]) -> None:
        path = Path(filepath)
        if path.suffix in {".pt", ".pth"}:
            checkpoint = torch.load(path, map_location=self.device)
            self.load_state_dict(checkpoint)
            self.eval()
            return
        if path.suffix == ".json":
            if not path.exists():
                raise FileNotFoundError(f"Missing RNN weights at {path}")
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.set_model_params(payload)
            self.eval()
            return
        raise ValueError(f"Unsupported weight file extension for MDNRNN: {path.suffix}")

    def save_json(self, filepath: Union[str, Path]) -> None:
        path = Path(filepath)
        if path.suffix in {".pt", ".pth"}:
            torch.save(self.state_dict(), path)
            return
        if path.suffix == ".json":
            params, _, _ = self.get_model_params()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                json.dump(params, handle, separators=(",", ":"))
            return
        raise ValueError(f"Unsupported weight file extension for MDNRNN: {path.suffix}")

    def close_sess(self) -> None:
        """Compatibility shim for legacy TensorFlow callers."""

    # ------------------------------------------------------------------
    def _prepare_inputs(self, inputs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(inputs, np.ndarray):
            tensor = torch.from_numpy(inputs).float()
        else:
            tensor = inputs.float()
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)

    def _resolve_state(self, state: Optional[RNNState], batch_size: int) -> RNNState:
        if state is None:
            return self.init_state(batch_size)
        h = state.h
        c = state.c
        if h.dim() == 2:
            h = h.unsqueeze(0)
        if c.dim() == 2:
            c = c.unsqueeze(0)
        return RNNState(h.to(self.device), c.to(self.device))

    def _split_mixture_params(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = logits.shape
        reshaped = logits.view(batch, seq_len, self.output_dim, self.num_mixture * 3)
        logmix, mean, logstd = torch.split(reshaped, self.num_mixture, dim=-1)
        logmix = logmix.view(batch, seq_len, self.output_dim, self.num_mixture)
        mean = mean.view(batch, seq_len, self.output_dim, self.num_mixture)
        logstd = logstd.view(batch, seq_len, self.output_dim, self.num_mixture)
        return logmix, mean, logstd


def get_pi_idx(x: float, pdf: np.ndarray) -> int:
    accumulate = 0.0
    for idx, prob in enumerate(pdf):
        accumulate += prob
        if accumulate >= x:
            return idx
    return len(pdf) - 1


def sample_sequence(
    model: MDNRNN,
    init_z: np.ndarray,
    actions: np.ndarray,
    temperature: float = 1.0,
    seq_len: int = 1000,
) -> np.ndarray:
    model.eval()
    state = model.init_state(batch_size=1)
    z_prev = torch.from_numpy(init_z).float().view(1, 1, -1).to(model.device)
    outputs = []
    with torch.no_grad():
        for idx in range(seq_len):
            action = torch.from_numpy(actions[idx]).float().view(1, 1, -1).to(model.device)
            inp = torch.cat([z_prev, action], dim=-1)
            logmix, mean, logstd, state = model(inp, state)
            sample = model.sample(logmix, mean, logstd, temperature=temperature)
            outputs.append(sample.squeeze(0).cpu().numpy())
            z_prev = sample.unsqueeze(0)
    return np.asarray(outputs, dtype=np.float32)


def rnn_init_state(rnn: MDNRNN) -> RNNState:
    return rnn.init_state(batch_size=1)


def rnn_next_state(rnn: MDNRNN, z: np.ndarray, a: np.ndarray, prev_state: RNNState) -> RNNState:
    rnn.eval()
    with torch.no_grad():
        z_tensor = torch.from_numpy(z.astype(np.float32)).view(1, 1, -1)
        a_tensor = torch.from_numpy(a.astype(np.float32)).view(1, 1, -1)
        inputs = torch.cat([z_tensor, a_tensor], dim=-1).to(rnn.device)
        _, _, _, next_state = rnn(inputs, prev_state.to(rnn.device))
    return next_state.detach()


def rnn_output_size(mode: int) -> int:
    if mode == MODE_ZCH:
        return 32 + 256 + 256
    if mode in (MODE_ZC, MODE_ZH):
        return 32 + 256
    return 32


def rnn_output(state: RNNState, z: np.ndarray, mode: int) -> np.ndarray:
    z_np = np.asarray(z, dtype=np.float32)
    h_np = state.h.detach().cpu().numpy()
    c_np = state.c.detach().cpu().numpy()
    if mode == MODE_ZCH:
        hc = np.concatenate((c_np, h_np), axis=2)[0, 0]
        return np.concatenate([z_np, hc])
    if mode == MODE_ZC:
        return np.concatenate([z_np, c_np[0, 0]])
    if mode == MODE_ZH:
        return np.concatenate([z_np, h_np[0, 0]])
    return z_np
