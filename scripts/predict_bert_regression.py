#!/usr/bin/env python3
"""Run BERT-only regression inference from a finetuned checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from finetune_bert_regression import BertRegressor, canonicalize_smiles
from peptide_ensemble_mpnn.data import EnsembleDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-nonfinite-rate", type=float, default=0.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


class BertDataset(Dataset):
    def __init__(self, path: str) -> None:
        self.base = EnsembleDataset(path, require_target=False)
        self.smiles = [canonicalize_smiles(str(r.get("summary_smiles", ""))) for r in self.base.samples]

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict:
        item = self.base[idx]
        target = item.get("target")
        if torch.is_tensor(target):
            target_val = float(target.item())
        else:
            target_val = float(target)
        return {
            "idx": int(idx),
            "smiles": self.smiles[idx],
            "target": target_val,
        }


def collate_bert(batch: List[Dict]) -> Dict:
    return {
        "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
        "smiles": [b["smiles"] for b in batch],
        "target": torch.tensor([b["target"] for b in batch], dtype=torch.float32),
    }


def move_batch(batch: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def get_target_scaling(ckpt: Dict) -> Optional[Tuple[float, float]]:
    scaling = ckpt.get("target_scaling")
    if isinstance(scaling, dict) and scaling.get("target_min") is not None and scaling.get("target_max") is not None:
        return (float(scaling["target_min"]), float(scaling["target_max"]))
    return None


def inverse_scale(pred: torch.Tensor, scaling: Optional[Tuple[float, float]]) -> torch.Tensor:
    if scaling is None:
        return pred
    tmin, tmax = scaling
    return pred * (tmax - tmin) + tmin


def finite_stats(t: torch.Tensor) -> Dict[str, float]:
    flat = t.reshape(-1)
    finite = flat[torch.isfinite(flat)]
    if flat.numel() == 0 or finite.numel() == 0:
        return {
            "finite_ratio": 0.0,
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
        }
    return {
        "finite_ratio": float(torch.isfinite(flat).float().mean().item()),
        "min": float(finite.min().item()),
        "max": float(finite.max().item()),
        "mean": float(finite.mean().item()),
        "std": float(finite.std(unbiased=False).item()),
    }


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt

    model = BertRegressor(
        clm_model_name=str(ckpt_args.get("hf_model", "aaronfeller/peptideclm-2-hybrid-base")),
        dropout=float(ckpt_args.get("head_dropout", 0.1)),
        head_hidden_dim=int(ckpt_args.get("head_hidden_dim", 0)),
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    print({"missing_keys": len(missing), "unexpected_keys": len(unexpected), "loaded_state_keys": len(state)})

    scaling = get_target_scaling(ckpt if isinstance(ckpt, dict) else {})
    max_smiles_len = int(ckpt_args.get("max_smiles_len", 256))

    device = torch.device(args.device)
    model.to(device)
    model.eval()

    ds = BertDataset(args.data)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_bert)

    rows: List[Dict] = []
    offset = 0
    batch_diagnostics: List[Dict[str, float]] = []
    total_pred = 0
    nonfinite_pred = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            bsz = batch["target"].shape[0]
            batch_dev = move_batch(batch, device)
            pred_scaled = model(batch_dev, device=device, max_smiles_len=max_smiles_len)
            stats_scaled = finite_stats(pred_scaled.detach())
            pred_raw = inverse_scale(pred_scaled, scaling)
            stats_raw = finite_stats(pred_raw.detach())

            total_pred += int(pred_raw.numel())
            nonfinite_pred += int((~torch.isfinite(pred_raw)).sum().item())
            batch_diagnostics.append(
                {
                    "batch": float(batch_idx),
                    "scaled_finite_ratio": stats_scaled["finite_ratio"],
                    "scaled_min": stats_scaled["min"],
                    "scaled_max": stats_scaled["max"],
                    "scaled_mean": stats_scaled["mean"],
                    "scaled_std": stats_scaled["std"],
                    "raw_finite_ratio": stats_raw["finite_ratio"],
                    "raw_min": stats_raw["min"],
                    "raw_max": stats_raw["max"],
                    "raw_mean": stats_raw["mean"],
                    "raw_std": stats_raw["std"],
                }
            )
            print(
                {
                    "phase": "predict",
                    "batch": batch_idx,
                    "scaled_finite_ratio": stats_scaled["finite_ratio"],
                    "raw_finite_ratio": stats_raw["finite_ratio"],
                },
                flush=True,
            )

            pred = pred_raw.detach().cpu().tolist()
            tgt = batch["target"].detach().cpu().tolist()
            for i in range(bsz):
                sample = ds.base.samples[offset + i]
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

    if total_pred == 0:
        raise RuntimeError("Prediction produced zero outputs")

    nonfinite_rate = float(nonfinite_pred) / float(total_pred)
    if nonfinite_rate > float(args.max_nonfinite_rate):
        raise RuntimeError(
            f"Non-finite prediction rate {nonfinite_rate:.6f} exceeded threshold {args.max_nonfinite_rate:.6f}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    diagnostics_path = out_path.with_suffix(out_path.suffix + ".diagnostics.json")
    diagnostics_payload = {
        "predictions": len(rows),
        "total_pred_values": total_pred,
        "nonfinite_pred_values": nonfinite_pred,
        "nonfinite_rate": nonfinite_rate,
        "threshold": float(args.max_nonfinite_rate),
        "batches": batch_diagnostics,
    }
    diagnostics_path.write_text(json.dumps(diagnostics_payload, indent=2), encoding="utf-8")

    print({"predictions": len(rows), "out": str(out_path), "diagnostics": str(diagnostics_path), "nonfinite_rate": nonfinite_rate})


if __name__ == "__main__":
    main()
