#!/usr/bin/env python3
"""Run hybrid regression inference from a finetuned checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from finetune_hybrid_regression import HybridRegressor, canonicalize_smiles
from peptide_ensemble_mpnn.config import ModelConfig
from peptide_ensemble_mpnn.data import EnsembleDataset, collate_ensemble
from peptide_ensemble_mpnn.models import EnsembleMPNN


def _filter_compatible_state_dict(model: nn.Module, state: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], int]:
    model_state = model.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    dropped = 0
    for key, value in state.items():
        if key not in model_state:
            dropped += 1
            continue
        if model_state[key].shape != value.shape:
            dropped += 1
            continue
        filtered[key] = value
    return filtered, dropped


def _infer_hidden_dim_from_state(state: Dict[str, torch.Tensor], fallback: int) -> int:
    for key in ("ensemble.backbone.residue_embed.weight", "backbone.residue_embed.weight"):
        tensor = state.get(key)
        if torch.is_tensor(tensor) and tensor.ndim == 2 and tensor.shape[1] > 0:
            return int(tensor.shape[1])
    return fallback


def _infer_egnn_layers_from_state(state: Dict[str, torch.Tensor], fallback: int) -> int:
    prefixes = ("ensemble.backbone.egnn_layers.", "backbone.egnn_layers.")
    layer_ids = set()
    for key in state.keys():
        for prefix in prefixes:
            if key.startswith(prefix):
                rem = key[len(prefix) :]
                idx = rem.split(".", 1)[0]
                if idx.isdigit():
                    layer_ids.add(int(idx))
                break
    if layer_ids:
        return max(layer_ids) + 1
    return fallback


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-nonfinite-rate", type=float, default=0.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


class HybridDataset(Dataset):
    def __init__(self, path: str) -> None:
        self.base = EnsembleDataset(path, require_target=False)
        self.smiles = [canonicalize_smiles(str(r.get("summary_smiles", ""))) for r in self.base.samples]

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict:
        item = self.base[idx]
        item["idx"] = idx
        item["smiles"] = self.smiles[idx]
        return item


def collate_hybrid(batch: List[Dict]) -> Dict:
    smiles = [b["smiles"] for b in batch]
    idxs = torch.tensor([int(b["idx"]) for b in batch], dtype=torch.long)
    core_batch = []
    for b in batch:
        core_batch.append({k: v for k, v in b.items() if k not in {"smiles", "idx"}})
    out = collate_ensemble(core_batch)
    out["smiles"] = smiles
    out["idx"] = idxs
    return out


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

    hidden_dim = _infer_hidden_dim_from_state(state, int(ckpt_args.get("hidden_dim", 192)))
    egnn_layers = _infer_egnn_layers_from_state(state, int(ckpt_args.get("egnn_layers", 6)))

    cfg = ModelConfig(
        hidden_dim=hidden_dim,
        egnn_layers=egnn_layers,
        k_neighbors=int(ckpt_args.get("k_neighbors", 12)),
        fusion_mode=str(ckpt_args.get("fusion_mode", "setattn")),
        use_conformer_prior=not bool(ckpt_args.get("disable_conformer_prior", False)),
    )
    ensemble = EnsembleMPNN(cfg)
    model = HybridRegressor(
        ensemble=ensemble,
        clm_model_name=str(ckpt_args.get("hf_model", "aaronfeller/peptideclm-2-hybrid-base")),
        dropout=float(ckpt_args.get("head_dropout", 0.1)),
        head_hidden_dim=int(ckpt_args.get("head_hidden_dim", 0)),
        head_num_layers=int(ckpt_args.get("head_num_layers", 2)),
    )
    filtered_state, dropped = _filter_compatible_state_dict(model, state)
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    print(
        {
            "loaded_keys": len(filtered_state),
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
            "dropped_mismatch_or_unknown": dropped,
            "hidden_dim": hidden_dim,
            "egnn_layers": egnn_layers,
        }
    )

    scaling = get_target_scaling(ckpt if isinstance(ckpt, dict) else {})
    max_smiles_len = int(ckpt_args.get("max_smiles_len", 256))

    device = torch.device(args.device)
    model.to(device)
    model.eval()

    ds = HybridDataset(args.data)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_hybrid)

    rows: List[Dict] = []
    offset = 0
    batch_diagnostics: List[Dict[str, float]] = []
    total_pred = 0
    nonfinite_pred = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            bsz = batch["residue_type"].shape[0]
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
