"""Task heads for pretraining and regression."""

from __future__ import annotations

import torch.nn as nn

from ..config import VOCAB_SIZE


class PretrainHead(nn.Module):
    """Heads for masked residue, geometry denoising, and distance reconstruction."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.residue_classifier = nn.Linear(hidden_dim, VOCAB_SIZE)
        self.coord_decoder = nn.Linear(hidden_dim, 3)
        self.dist_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, residue_embedding: torch.Tensor, per_conformer_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        residue_logits = self.residue_classifier(residue_embedding)
        coord_pred = self.coord_decoder(per_conformer_embedding)

        z = self.dist_proj(residue_embedding)
        pred_dist = torch.cdist(z, z)

        return {
            "residue_logits": residue_logits,
            "coord_pred": coord_pred,
            "pred_dist": pred_dist,
        }


class RegressionHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, ensemble_embedding: torch.Tensor) -> torch.Tensor:
        return self.net(ensemble_embedding).squeeze(-1)
