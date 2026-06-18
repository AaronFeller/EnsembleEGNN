#!/usr/bin/env python3
"""Run regression inference from a finetuned checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from peptide_ensemble_mpnn.config import ModelConfig
from peptide_ensemble_mpnn.data import EnsembleDataset, collate_ensemble
from peptide_ensemble_mpnn.models import EnsembleMPNN


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    state = ckpt.get("model_state", ckpt)
    target_scaling = ckpt.get("target_scaling")

    # Some finetune checkpoints keep launcher args (e.g., hidden_dim=192)
    # even when model architecture is loaded from init checkpoint args (e.g., 448).
    # Infer hidden_dim from actual state tensor shape to avoid load mismatches.
    inferred_hidden_dim = None
    embed_w = state.get("backbone.residue_embed.weight") if isinstance(state, dict) else None
    if embed_w is not None and hasattr(embed_w, "shape") and len(embed_w.shape) == 2:
        inferred_hidden_dim = int(embed_w.shape[1])

    cfg = ModelConfig(
        hidden_dim=int(inferred_hidden_dim if inferred_hidden_dim is not None else ckpt_args.get("hidden_dim", 192)),
        egnn_layers=int(ckpt_args.get("egnn_layers", 6)),
        k_neighbors=int(ckpt_args.get("k_neighbors", 12)),
        fusion_mode=ckpt_args.get("fusion_mode", "setattn"),
        use_conformer_prior=not bool(ckpt_args.get("disable_conformer_prior", False)),
    )
    model = EnsembleMPNN(cfg)
    model.load_state_dict(state, strict=False)

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    ds = EnsembleDataset(args.data, require_target=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_ensemble)

    rows: List[Dict] = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            bsz = batch["residue_type"].shape[0]
            batch_dev = move_batch(batch, device)
            out = model.forward_regression(
                residue_type=batch_dev["residue_type"],
                coords=batch_dev["coords"],
                node_mask=batch_dev["node_mask"],
                conformer_mask=batch_dev["conformer_mask"],
                conformer_weights=batch_dev["conformer_weights"],
            )
            pred_t = out["prediction"].detach().cpu()
            if target_scaling and "target_min" in target_scaling and "target_max" in target_scaling:
                tmin = float(target_scaling["target_min"])
                tmax = float(target_scaling["target_max"])
                pred_t = pred_t * (tmax - tmin) + tmin
            pred = pred_t.tolist()
            tgt = batch["target"].detach().cpu().tolist()

            for i in range(bsz):
                sample = ds.samples[offset + i]
                rows.append(
                    {
                        "index": offset + i,
                        "prediction": float(pred[i]),
                        "target": float(tgt[i]) if tgt[i] == tgt[i] else None,
                        "summary_smiles": sample.get("summary_smiles", ""),
                        "cluster": sample.get("cluster"),
                    }
                )
            offset += bsz

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print({"predictions": len(rows), "out": str(out_path)})


if __name__ == "__main__":
    main()
