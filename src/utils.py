"""Training and tensor helper utilities."""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mask = mask.to(dtype=x.dtype)
    num = (x * mask.unsqueeze(-1)).sum(dim=dim)
    den = mask.sum(dim=dim, keepdim=True).clamp_min(1.0)
    return num / den


def sequence_recovery(pred_tokens: torch.Tensor, true_tokens: torch.Tensor, mask: torch.Tensor) -> float:
    correct = ((pred_tokens == true_tokens) & mask).sum().item()
    total = mask.sum().item()
    if total == 0:
        return 0.0
    return float(correct) / float(total)


class MomentumQueue:
    def __init__(self, dim: int, size: int = 4096, device: torch.device | None = None) -> None:
        self.size = int(size)
        self.device = device if device is not None else torch.device("cpu")
        self.buffer = torch.empty((0, dim), device=self.device)

    def enqueue(self, z: torch.Tensor) -> None:
        z = z.detach()
        if z.ndim != 2:
            raise ValueError("MomentumQueue expects 2D embeddings")
        if self.buffer.numel() == 0:
            self.buffer = z[-self.size :]
            return
        self.buffer = torch.cat([self.buffer, z], dim=0)[-self.size :]

    def get_queue(self) -> torch.Tensor:
        return self.buffer


def compute_pearson_r(pred: torch.Tensor, target: torch.Tensor) -> float:
    if pred.numel() < 2:
        return float("nan")
    x = pred - pred.mean()
    y = target - target.mean()
    den = torch.sqrt((x * x).sum() * (y * y).sum()).clamp_min(1e-12)
    return float((x * y).sum() / den)


def compute_spearman_r(pred: torch.Tensor, target: torch.Tensor) -> float:
    if pred.numel() < 2:
        return float("nan")
    rp = torch.argsort(torch.argsort(pred)).float()
    rt = torch.argsort(torch.argsort(target)).float()
    return compute_pearson_r(rp, rt)


def compute_r2(pred: torch.Tensor, target: torch.Tensor) -> float:
    if pred.numel() < 2:
        return float("nan")
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum().clamp_min(1e-12)
    return float(1.0 - ss_res / ss_tot)
