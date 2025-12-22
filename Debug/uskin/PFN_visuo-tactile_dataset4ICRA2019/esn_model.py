# esn_model.py  (drop-in replacement)
# - No global `device` dependency
# - Follows the input tensor device (x.device) and model.to(device)
# - Fixes t==0 negative-index bug
# - Ridge regression uses X.device
# - Accepts x as [B, ch, T, D] or [B, T, D] (auto-unsqueeze ch)
# - Keeps your public class names: EchoStateNetwork, ReadOut, ESN

import itertools
from typing import Optional, Tuple

import torch
import torch.nn as nn


class EchoStateNetwork(nn.Module):
    def __init__(self, model_params, dataset_params):
        super().__init__()

        self.reservoir_size = int(model_params["reservoir_size"])
        self.reservoir_weights_scale = float(model_params["reservoir_weights_scale"])

        self.input_size = int(model_params["input_size"])
        self.channel_size = int(model_params["channel_size"])
        self.input_weights_scale = float(model_params["input_weights_scale"])
        self.spectral_radius = float(model_params["spectral_radius"])
        self.density = float(model_params["reservoir_density"])
        self.leak_rate = float(model_params["leak_rate"])

        self.sequence_length = int(int(dataset_params["sequence_length"]) / int(dataset_params["slicing_size"]))
        # self.sequence_length = int(dataset_params["seq_end"]) - int(dataset_params["seq_start"])

        # ---- Parameters (NO .to(device) here; model.to(device) will move them) ----
        rw = torch.empty((self.reservoir_size, self.reservoir_size)).uniform_(
            -self.reservoir_weights_scale, self.reservoir_weights_scale
        )
        iw = torch.empty((self.reservoir_size, self.input_size * self.channel_size)).uniform_(
            -self.input_weights_scale, self.input_weights_scale
        )

        self.register_parameter("reservoir_weights", nn.Parameter(rw, requires_grad=False))
        self.register_parameter("input_weights", nn.Parameter(iw, requires_grad=False))

        # ---- Sparse mask (buffer so it moves with model.to(device)) ----
        # boolean mask -> float mask to multiply
        mask_bool = (torch.rand((self.reservoir_size, self.reservoir_size)) < self.density)
        self.register_buffer("reservoir_weights_mask", mask_bool.to(dtype=self.reservoir_weights.dtype))
        self.reservoir_weights.data.mul_(self.reservoir_weights_mask)

        # ---- Spectral radius scaling ----
        self._apply_spectral_radius_()

        # ---- State holders (buffers so they move; but overwritten per forward anyway) ----
        self.register_buffer("last_reservoir_state_matrix", torch.zeros(1, self.channel_size, self.reservoir_size))

    def _apply_spectral_radius_(self) -> None:
        """Scale reservoir_weights to target spectral radius (approx via largest singular value)."""
        with torch.no_grad():
            W = self.reservoir_weights.data
            # try torch.linalg.svdvals (new), fallback to torch.svd (old)
            try:
                s = torch.linalg.svdvals(W)
                rho = float(s.max().item())
            except Exception:
                # torch.svd returns U,S,V
                _, s, _ = torch.svd(W)
                rho = float(s.max().item())

            if rho > 0:
                W.mul_(self.spectral_radius / rho)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:
          - [B, ch, T, D]  (recommended)
          - [B, T, D]      (auto unsqueeze ch=1)
        returns:
          - reservoir_state_matrix: [B, ch, T, H]
        """
        if x.dim() == 3:
            # [B,T,D] -> [B,1,T,D]
            x = x.unsqueeze(1)

        if x.dim() != 4:
            raise ValueError(f"ESN input must be [B,ch,T,D] or [B,T,D], got shape={tuple(x.shape)}")

        B, ch, T, D = x.shape
        if ch != self.channel_size:
            # 厳密に合わせたいならここで raise にしてもOK
            # raise ValueError(f"channel_size mismatch: x has ch={ch}, model expects {self.channel_size}")
            pass
        if D != self.input_size:
            raise ValueError(f"input_size mismatch: x has D={D}, model expects {self.input_size}")

        dev = x.device
        dtype = x.dtype

        # 状態行列 [B,ch,T,H]
        reservoir_state_matrix = torch.zeros((B, ch, T, self.reservoir_size), device=dev, dtype=dtype)

        # last state [B,ch,H]
        last_state = torch.zeros((B, ch, self.reservoir_size), device=dev, dtype=dtype)

        # 入力重みは Parameter なので model.to(device) で同じ dev にいる想定だが、
        # 念のため dev を合わせる（異なると matmul が落ちる）
        W_in = self.input_weights.to(device=dev, dtype=dtype)
        W_res = self.reservoir_weights.to(device=dev, dtype=dtype)

        for t in range(T):
            # x[:, :, t, :] -> [B,ch,D] -> flatten ch*D
            u_t = x[:, :, t, :].reshape(B, -1)                 # [B, ch*D]
            input_at_t = torch.matmul(u_t, W_in.t())           # [B, H]
            input_at_t = input_at_t.unsqueeze(1)               # [B, 1, H] (broadcast OK for ch)

            if t == 0:
                prev_state = torch.zeros((B, ch, self.reservoir_size), device=dev, dtype=dtype)
                state_update = torch.matmul(last_state, W_res)  # last_state is zeros initially
            else:
                prev_state = reservoir_state_matrix[:, :, t - 1, :]
                state_update = torch.matmul(prev_state, W_res)

            new_state = self.leak_rate * torch.tanh(input_at_t + state_update) + (1.0 - self.leak_rate) * prev_state
            reservoir_state_matrix[:, :, t, :] = new_state

        # store last state as buffer shape [B,ch,H]
        self.last_reservoir_state_matrix = reservoir_state_matrix[:, :, -1, :].detach()
        return reservoir_state_matrix

    def reset_hidden_state(self) -> None:
        # shape is arbitrary here; will be overwritten on next forward
        self.last_reservoir_state_matrix = torch.zeros_like(self.last_reservoir_state_matrix)


class ReadOut(nn.Module):
    def __init__(self, model_params, dataset_params):
        super().__init__()

        self.reservoir_state_matrix_size = int(model_params["reservoir_size"])
        self.output_size = int(model_params["ReadOut_output_size"])
        self.batch_training = bool(model_params["Batch_Training"])
        self.channel_size = int(model_params["channel_size"])

        self.sequence_length = int(int(dataset_params["sequence_length"]) / int(dataset_params["slicing_size"]))
        # self.sequence_length = int(dataset_params["seq_end"]) - int(dataset_params["seq_start"])

        self.readout_dense = nn.Linear(self.reservoir_state_matrix_size, self.output_size, bias=False)

        if self.batch_training:
            self.readout_dense.weight.requires_grad = False

        nn.init.xavier_uniform_(self.readout_dense.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:
          - [B, ch, T, H]
          - [B, T, H] (auto unsqueeze ch=1)
        returns:
          - [B, ch, T, C] or [B, T, C] depending on input
        """
        squeezed = False
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [B,1,T,H]
            squeezed = True
        if x.dim() != 4:
            raise ValueError(f"ReadOut input must be [B,ch,T,H] or [B,T,H], got shape={tuple(x.shape)}")

        out = self.readout_dense(x)  # linear over last dim: H->C, keeps [B,ch,T,C]
        if squeezed:
            out = out.squeeze(1)     # [B,T,C]
        return out

    # ---- Ridge regression (kept compatible with your original math, but device-safe) ----
    @staticmethod
    def ridge_regression(X: torch.Tensor, Y: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        X: [n, p] (your usage: [H, N])
        Y: [m, p] (your usage: [C, N])
        returns coefficients: [m, n] (your usage: [C, H])
        """
        n, p = X.shape
        dev = X.device

        X = X.float()
        Y = Y.float()

        ridge = (float(alpha) * torch.eye(n, device=dev, dtype=X.dtype))
        # coefficients = (X X^T + aI)^-1 (X Y^T)  then transpose -> [m,n]
        coeff = torch.linalg.solve(X @ X.t() + ridge, X @ Y.t()).t()
        return coeff

    @staticmethod
    def ridge_regression_update(outputs: torch.Tensor, targets: torch.Tensor, model: nn.Module, alpha: float = 0.0) -> None:
        """
        outputs: [H, N]  (states.T)
        targets: [C, N]  (targets.T)
        """
        with torch.no_grad():
            new_weights = ReadOut.ridge_regression(outputs.squeeze(), targets.squeeze(), alpha)
            model.ReadOut.readout_dense.weight.copy_(new_weights)

    @staticmethod
    def model_params_candinate(model_params):
        model_params_combinations = list(itertools.product(*model_params.values()))
        param_dicts = [dict(zip(model_params.keys(), combination)) for combination in model_params_combinations]
        return param_dicts

    @staticmethod
    def model_sturcture_dict(model):
        layers_dict = {}
        for name, module in model.named_modules():
            layers_dict[name] = {
                "type": type(module).__name__,
                "parameters": {p: getattr(module, p) for p in module.__dict__ if not p.startswith("_")},
            }
        if "" in layers_dict:
            del layers_dict[""]
        return layers_dict


class ESN(nn.Module):
    def __init__(self, model_params, training_params, dataset_params):
        super().__init__()
        self.ESN = EchoStateNetwork(model_params, dataset_params)
        self.ReadOut = ReadOut(model_params, dataset_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        returns:
          - [B,T,C]  (because we squeeze ch in the end, matching your original behavior)
        """
        reservoir_states = self.ESN(x)              # [B,ch,T,H]
        logits = self.ReadOut(reservoir_states)     # [B,ch,T,C] (or [B,T,C] if input was [B,T,D])
        if logits.dim() == 4:
            logits = logits.squeeze(1)             # remove channel dim -> [B,T,C]
        return logits

    def reset_hidden_state(self) -> None:
        self.ESN.reset_hidden_state()
