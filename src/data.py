"""Dataset and collation for peptide conformer ensembles."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .config import AA_TO_ID, ATOM_TO_ID, ATOM_UNK_TOKEN_ID, PAD_TOKEN_ID, UNK_TOKEN_ID


class EnsembleDataset(Dataset):
    """Loads ensemble examples from .pt or .jsonl files.

    Expected sample format keys:
    - sequence: str (required)
    - coords: list or tensor shaped [C, N, 3] (required)
    - conformer_weights: list shaped [C] (optional)
    - target: float (optional, required for regression training)
    """

    def __init__(self, path: str, require_target: bool = False) -> None:
        self.path = path
        self.require_target = require_target
        self.samples = self._load(path)

    def _load(self, path: str) -> List[Dict[str, Any]]:
        if path.endswith(".pt"):
            data = torch.load(path, map_location="cpu")
            if isinstance(data, dict) and "samples" in data:
                return data["samples"]
            if isinstance(data, list):
                return data
            raise ValueError("Unsupported .pt format. Use list[dict] or {'samples': list[dict]}.")
        if path.endswith(".jsonl"):
            out = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.append(json.loads(line))
            return out
        raise ValueError("Dataset path must be .pt or .jsonl.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        seq = item["sequence"]
        coords = torch.as_tensor(item["coords"], dtype=torch.float32)
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ValueError("coords must have shape [C, N, 3]")

        node_tokens = item.get("node_tokens")
        if node_tokens is not None:
            if len(node_tokens) != coords.shape[1]:
                raise ValueError("node_tokens length and coordinate length mismatch")
            if len(node_tokens) == 0:
                raise ValueError("node_tokens cannot be empty")
            if isinstance(node_tokens[0], str):
                token_ids = [ATOM_TO_ID.get(sym, ATOM_UNK_TOKEN_ID) for sym in node_tokens]
            else:
                token_ids = [int(x) for x in node_tokens]
            residue_type = torch.tensor(token_ids, dtype=torch.long)
        else:
            if coords.shape[1] != len(seq):
                raise ValueError("sequence length and coordinate length mismatch")
            residue_type = torch.tensor([AA_TO_ID.get(aa, UNK_TOKEN_ID) for aa in seq], dtype=torch.long)

        out: Dict[str, Any] = {
            "sequence": seq,
            "residue_type": residue_type,
            "coords": coords,
        }

        conf_w = item.get("conformer_weights")
        if conf_w is None:
            conf_w = torch.ones(coords.shape[0], dtype=torch.float32) / float(coords.shape[0])
        else:
            conf_w = torch.as_tensor(conf_w, dtype=torch.float32)
        out["conformer_weights"] = conf_w

        if "target" in item:
            out["target"] = torch.tensor(float(item["target"]), dtype=torch.float32)
        elif self.require_target:
            raise ValueError("Missing target for sample while require_target=True")

        return out


def collate_ensemble(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    bsz = len(batch)
    c_max = max(item["coords"].shape[0] for item in batch)
    n_max = max(item["coords"].shape[1] for item in batch)

    residue = torch.full((bsz, n_max), PAD_TOKEN_ID, dtype=torch.long)
    coords = torch.zeros((bsz, c_max, n_max, 3), dtype=torch.float32)
    node_mask = torch.zeros((bsz, n_max), dtype=torch.bool)
    conformer_mask = torch.zeros((bsz, c_max), dtype=torch.bool)
    conf_w = torch.zeros((bsz, c_max), dtype=torch.float32)
    target = torch.full((bsz,), float("nan"), dtype=torch.float32)

    for i, item in enumerate(batch):
        n = item["residue_type"].shape[0]
        c = item["coords"].shape[0]
        residue[i, :n] = item["residue_type"]
        coords[i, :c, :n] = item["coords"]
        node_mask[i, :n] = True
        conformer_mask[i, :c] = True

        cw = item["conformer_weights"]
        if cw.shape[0] != c:
            raise ValueError("conformer_weights length does not match conformer count")
        conf_w[i, :c] = cw

        if "target" in item:
            target[i] = item["target"]

    conf_w = conf_w * conformer_mask.float()
    conf_w = conf_w / conf_w.sum(dim=1, keepdim=True).clamp_min(1e-8)

    return {
        "residue_type": residue,
        "coords": coords,
        "node_mask": node_mask,
        "conformer_mask": conformer_mask,
        "conformer_weights": conf_w,
        "target": target,
    }
