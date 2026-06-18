"""Unified ensemble model with pretraining and regression heads."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig
from .backbone import EnsembleBackbone
from .heads import PretrainHead, RegressionHead


class EnsembleMPNN(nn.Module):
    """Shared architecture for pretraining and downstream regression."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.backbone = EnsembleBackbone(
            hidden_dim=cfg.hidden_dim,
            egnn_layers=cfg.egnn_layers,
            k_neighbors=cfg.k_neighbors,
            set_inducing_points=cfg.set_inducing_points,
            attn_heads=cfg.attn_heads,
            dropout=cfg.dropout,
            fusion_mode=cfg.fusion_mode,
            use_conformer_prior=cfg.use_conformer_prior,
        )
        self.pretrain_head = PretrainHead(cfg.hidden_dim)
        self.regression_head = RegressionHead(cfg.hidden_dim)

    def encode(
        self,
        residue_type: torch.Tensor,
        coords: torch.Tensor,
        node_mask: torch.Tensor,
        conformer_mask: torch.Tensor,
        conformer_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.backbone(
            residue_type=residue_type,
            coords=coords,
            node_mask=node_mask,
            conformer_mask=conformer_mask,
            conformer_weights=conformer_weights,
        )

    def forward_pretrain(
        self,
        residue_type: torch.Tensor,
        coords: torch.Tensor,
        node_mask: torch.Tensor,
        conformer_mask: torch.Tensor,
        conformer_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        enc = self.encode(residue_type, coords, node_mask, conformer_mask, conformer_weights)
        heads = self.pretrain_head(enc["residue_embedding"], enc["per_conformer_embedding"])
        # Use the EGNN-updated coordinates as the denoising target prediction.
        # This keeps the coordinate objective aligned with geometric message passing.
        heads["coord_pred"] = enc["encoded_coords"]
        return {**enc, **heads}

    def forward_regression(
        self,
        residue_type: torch.Tensor,
        coords: torch.Tensor,
        node_mask: torch.Tensor,
        conformer_mask: torch.Tensor,
        conformer_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        enc = self.encode(residue_type, coords, node_mask, conformer_mask, conformer_weights)
        pred = self.regression_head(enc["ensemble_embedding"])
        return {**enc, "prediction": pred}
