"""Backbone model for conformer-wise encoding and ensemble fusion."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import VOCAB_SIZE
from .layers import EGNNLayer, SetAttentionBlock


class ConformerSetFusion(nn.Module):
    """Permutation-invariant conformer fusion using inducing points."""

    def __init__(
        self,
        hidden_dim: int,
        inducing_points: int,
        heads: int,
        dropout: float = 0.1,
        mode: str = "setattn",
        use_conformer_prior: bool = True,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.use_conformer_prior = use_conformer_prior
        self.inducing = nn.Parameter(torch.randn(inducing_points, hidden_dim) * 0.02)
        self.seed = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)

        self.block1 = SetAttentionBlock(hidden_dim, heads, dropout)
        self.block2 = SetAttentionBlock(hidden_dim, heads, dropout)

        self.weight_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        conformer_tokens: torch.Tensor,
        conformer_mask: torch.Tensor,
        conformer_prior: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = conformer_tokens.shape[0]

        if self.mode == "single":
            if conformer_prior is not None:
                idx_src = conformer_prior.masked_fill(~conformer_mask, float("-inf"))
                idx = idx_src.argmax(dim=-1)
            else:
                idx = conformer_mask.float().argmax(dim=-1)
            conformer_attn = torch.zeros_like(conformer_mask, dtype=conformer_tokens.dtype)
            conformer_attn.scatter_(1, idx.unsqueeze(-1), 1.0)
            conformer_attn = conformer_attn * conformer_mask.float()
            conformer_attn = conformer_attn / conformer_attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            ensemble_embedding = (conformer_tokens * conformer_attn.unsqueeze(-1)).sum(dim=1)
            return ensemble_embedding, conformer_attn

        if self.mode == "mean":
            conformer_attn = conformer_mask.float()
            conformer_attn = conformer_attn / conformer_attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            ensemble_embedding = (conformer_tokens * conformer_attn.unsqueeze(-1)).sum(dim=1)
            return ensemble_embedding, conformer_attn

        if self.mode != "setattn":
            raise ValueError(f"Unsupported fusion mode: {self.mode}")

        inducing = self.inducing.unsqueeze(0).expand(bsz, -1, -1)
        h = self.block1(inducing, conformer_tokens, key_padding_mask=~conformer_mask)

        seed = self.seed.unsqueeze(0).expand(bsz, -1, -1)
        pooled = self.block2(seed, h)
        ensemble_embedding = pooled[:, 0]

        logits = self.weight_mlp(conformer_tokens).squeeze(-1)
        if self.use_conformer_prior and conformer_prior is not None:
            logits = logits + torch.log(conformer_prior.clamp_min(1e-8))

        logits = logits.masked_fill(~conformer_mask, float("-inf"))
        conformer_attn = torch.softmax(logits, dim=-1)
        conformer_attn = torch.nan_to_num(conformer_attn)

        return ensemble_embedding, conformer_attn


class EnsembleBackbone(nn.Module):
    """Shared backbone for pretraining, regression, and inverse folding."""

    def __init__(
        self,
        hidden_dim: int,
        egnn_layers: int,
        k_neighbors: int,
        set_inducing_points: int,
        attn_heads: int,
        dropout: float,
        fusion_mode: str = "setattn",
        use_conformer_prior: bool = True,
    ) -> None:
        super().__init__()
        self.k_neighbors = k_neighbors

        self.residue_embed = nn.Embedding(VOCAB_SIZE, hidden_dim)
        self.egnn_layers = nn.ModuleList([EGNNLayer(hidden_dim, dropout=dropout) for _ in range(egnn_layers)])
        self.fusion = ConformerSetFusion(
            hidden_dim,
            set_inducing_points,
            attn_heads,
            dropout=dropout,
            mode=fusion_mode,
            use_conformer_prior=use_conformer_prior,
        )
        self.residue_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        residue_type: torch.Tensor,
        coords: torch.Tensor,
        node_mask: torch.Tensor,
        conformer_mask: torch.Tensor,
        conformer_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        bsz, n_conf, n_nodes, _ = coords.shape
        hidden_dim = self.residue_embed.embedding_dim

        h0 = self.residue_embed(residue_type)  # [B, N, D]
        h = h0.unsqueeze(1).expand(bsz, n_conf, n_nodes, hidden_dim).reshape(bsz * n_conf, n_nodes, hidden_dim)
        x = coords.reshape(bsz * n_conf, n_nodes, 3)

        node_mask_bc = node_mask.unsqueeze(1).expand(bsz, n_conf, n_nodes).reshape(bsz * n_conf, n_nodes)

        for layer in self.egnn_layers:
            h, x = layer(h, x, node_mask_bc, self.k_neighbors)

        h = h.view(bsz, n_conf, n_nodes, hidden_dim)
        x = x.view(bsz, n_conf, n_nodes, 3)

        node_mask_f = node_mask.float()
        den = node_mask_f.sum(dim=1, keepdim=True).clamp_min(1.0).unsqueeze(-1)  # [B,1,1]
        conf_tokens = (h * node_mask_f[:, None, :, None]).sum(dim=2) / den

        ensemble_embedding, conf_attn = self.fusion(conf_tokens, conformer_mask, conformer_weights)
        residue_embedding = (h * conf_attn[:, :, None, None]).sum(dim=1)
        residue_embedding = self.residue_norm(residue_embedding)

        return {
            "ensemble_embedding": ensemble_embedding,
            "residue_embedding": residue_embedding,
            "per_conformer_embedding": h,
            "encoded_coords": x,
            "conformer_attention": conf_attn,
        }
