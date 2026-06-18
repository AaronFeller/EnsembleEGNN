"""Loss functions for pretraining and downstream regression."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .config import VOCAB_SIZE


def pretraining_losses(
    outputs: Dict[str, torch.Tensor],
    residue_target: torch.Tensor,
    coords_target: torch.Tensor,
    node_mask: torch.Tensor,
    conformer_mask: torch.Tensor,
    conformer_weights: torch.Tensor,
    residue_mask: torch.Tensor,
    w_residue: float = 0.3,
    w_coord: float = 0.5,
    w_dist: float = 0.2,
) -> Dict[str, torch.Tensor]:
    residue_logits = outputs["residue_logits"]
    coord_pred = outputs["coord_pred"]
    pred_dist = outputs["pred_dist"]

    valid_residue = residue_mask & node_mask & (residue_target >= 0) & (residue_target < VOCAB_SIZE)
    if valid_residue.any():
        residue_loss = F.cross_entropy(
            residue_logits[valid_residue],
            residue_target[valid_residue],
        )
    else:
        residue_loss = residue_logits.sum() * 0.0

    coord_mask = conformer_mask[:, :, None] & node_mask[:, None, :]
    coord_mask_f = coord_mask.float().unsqueeze(-1)
    sqerr = (coord_pred - coords_target) ** 2
    coord_loss = (sqerr * coord_mask_f).sum() / coord_mask_f.sum().clamp_min(1.0)

    mean_coords = (coords_target * conformer_weights[:, :, None, None]).sum(dim=1)
    true_dist = torch.cdist(mean_coords, mean_coords)
    pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
    dist_loss = F.l1_loss(pred_dist[pair_mask], true_dist[pair_mask])

    total = w_residue * residue_loss + w_coord * coord_loss + w_dist * dist_loss
    return {
        "total": total,
        "residue": residue_loss,
        "coord": coord_loss,
        "dist": dist_loss,
    }


def regression_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)
