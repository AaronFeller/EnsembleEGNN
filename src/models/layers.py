"""Neural layers for geometric message passing and set fusion."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn_indices(x: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    """Compute kNN indices for each node in each graph of a batch.

    Args:
        x: [B, N, 3]
        mask: [B, N] bool
        k: number of neighbors
    Returns:
        idx: [B, N, K]
    """
    bsz, n_nodes, _ = x.shape
    k = min(k, n_nodes)

    dist = torch.cdist(x, x)  # [B, N, N]
    valid_pair = mask.unsqueeze(2) & mask.unsqueeze(1)
    dist = dist.masked_fill(~valid_pair, float("inf"))

    eye = torch.eye(n_nodes, device=x.device, dtype=torch.bool).unsqueeze(0)
    dist = dist.masked_fill(eye, float("inf"))

    idx = torch.topk(dist, k=k, dim=-1, largest=False).indices

    invalid_rows = ~mask
    if invalid_rows.any():
        idx[invalid_rows] = 0
    return idx


def gather_neighbors(values: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather neighbor features.

    Args:
        values: [B, N, D]
        idx: [B, N, K]
    Returns:
        out: [B, N, K, D]
    """
    bsz = values.shape[0]
    batch_idx = torch.arange(bsz, device=values.device)[:, None, None]
    return values[batch_idx, idx]


class EGNNLayer(nn.Module):
    """A lightweight E(n)-style message passing block."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, x: torch.Tensor, node_mask: torch.Tensor, k_neighbors: int) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, n_nodes, _ = h.shape
        idx = knn_indices(x, node_mask, k_neighbors)  # [B, N, K]
        k = idx.shape[-1]

        h_i = h.unsqueeze(2).expand(bsz, n_nodes, k, h.shape[-1])
        h_j = gather_neighbors(h, idx)

        x_i = x.unsqueeze(2).expand(bsz, n_nodes, k, 3)
        x_j = gather_neighbors(x, idx)
        rel = x_j - x_i
        dist2 = (rel * rel).sum(dim=-1, keepdim=True)

        msg_in = torch.cat([h_i, h_j, dist2], dim=-1)
        m_ij = self.edge_mlp(msg_in)

        src_mask = node_mask.unsqueeze(-1).unsqueeze(-1).float()
        dst_mask = gather_neighbors(node_mask.float().unsqueeze(-1), idx)
        edge_mask = src_mask * dst_mask
        m_ij = m_ij * edge_mask

        m_i = m_ij.sum(dim=2)
        dh = self.node_mlp(torch.cat([h, m_i], dim=-1))
        h_new = self.norm(h + dh)
        h_new = torch.where(node_mask.unsqueeze(-1), h_new, h)

        coord_gate = self.coord_mlp(m_ij)
        dx = (rel * coord_gate * edge_mask).sum(dim=2) / float(max(k, 1))
        x_new = x + dx
        x_new = torch.where(node_mask.unsqueeze(-1), x_new, x)

        return h_new, x_new


class EGNNLayerWithAngles(EGNNLayer):
    """EGNN layer variant with a lightweight angular feature channel."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__(hidden_dim=hidden_dim, dropout=dropout)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, h: torch.Tensor, x: torch.Tensor, node_mask: torch.Tensor, k_neighbors: int) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, n_nodes, _ = h.shape
        idx = knn_indices(x, node_mask, k_neighbors)
        k = idx.shape[-1]

        h_i = h.unsqueeze(2).expand(bsz, n_nodes, k, h.shape[-1])
        h_j = gather_neighbors(h, idx)

        x_i = x.unsqueeze(2).expand(bsz, n_nodes, k, 3)
        x_j = gather_neighbors(x, idx)
        rel = x_j - x_i
        dist = rel.norm(dim=-1, keepdim=True)

        rel_unit = rel / dist.clamp_min(1e-8)
        center_dir = x_i / x_i.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        cos_theta = (rel_unit * center_dir).sum(dim=-1, keepdim=True)

        msg_in = torch.cat([h_i, h_j, dist, dist * dist, cos_theta], dim=-1)
        m_ij = self.edge_mlp(msg_in)

        src_mask = node_mask.unsqueeze(-1).unsqueeze(-1).float()
        dst_mask = gather_neighbors(node_mask.float().unsqueeze(-1), idx)
        edge_mask = src_mask * dst_mask
        m_ij = m_ij * edge_mask

        m_i = m_ij.sum(dim=2)
        dh = self.node_mlp(torch.cat([h, m_i], dim=-1))
        h_new = self.norm(h + dh)
        h_new = torch.where(node_mask.unsqueeze(-1), h_new, h)

        coord_gate = self.coord_mlp(m_ij)
        dx = (rel * coord_gate * edge_mask).sum(dim=2) / float(max(k, 1))
        x_new = x + dx
        x_new = torch.where(node_mask.unsqueeze(-1), x_new, x)

        return h_new, x_new


class Transformer1DLayer(nn.Module):
    """Standard pre-norm transformer block for sequence context."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        xn = self.ln1(x)
        out, _ = self.attn(xn, xn, xn, key_padding_mask=key_padding_mask)
        x = x + out
        x = x + self.ff(self.ln2(x))
        return x


class ConformerCrossAttention(nn.Module):
    """Cross-attention from sequence tokens to per-conformer geometric tokens."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.ln_out = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self.ln_q(query)
        kv = self.ln_kv(key_value)
        out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask)
        x = query + out
        x = x + self.ff(self.ln_out(x))
        return x


class HypergraphConvLayer(nn.Module):
    """Hypergraph-style convolution where conformers act as hyperedges."""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        node_features: torch.Tensor,
        per_conformer_features: torch.Tensor,
        node_mask: torch.Tensor,
        conformer_mask: torch.Tensor,
        conformer_weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # node_features: [B, N, D], per_conformer_features: [B, C, N, D]
        bsz, n_conf, _, dim = per_conformer_features.shape
        conf_tokens = per_conformer_features.mean(dim=2)  # [B, C, D]
        conf_tokens = self.edge_mlp(conf_tokens)

        logits = conf_tokens.mean(dim=-1)
        if conformer_weights is not None:
            logits = logits + torch.log(conformer_weights.clamp_min(1e-8))
        logits = logits.masked_fill(~conformer_mask, float("-inf"))
        conf_attn = torch.softmax(logits, dim=-1)
        conf_attn = torch.nan_to_num(conf_attn)

        msg = (per_conformer_features * conf_attn[:, :, None, None]).sum(dim=1)
        out = self.node_mlp(torch.cat([node_features, msg], dim=-1))
        out = self.norm(node_features + out)
        out = torch.where(node_mask.unsqueeze(-1), out, node_features)
        return out, conf_attn


class SetAttentionBlock(nn.Module):
    """Transformer-like attention block for set inputs."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out, _ = self.attn(query, key_value, key_value, key_padding_mask=key_padding_mask)
        x = self.ln1(query + attn_out)
        ff = self.ff(x)
        return self.ln2(x + ff)
