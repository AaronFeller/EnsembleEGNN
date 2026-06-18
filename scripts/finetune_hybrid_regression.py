#!/usr/bin/env python3
"""Finetune hybrid model: EnsembleMPNN + peptide CLM for PAMPA regression."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from peptide_ensemble_mpnn.config import ModelConfig
from peptide_ensemble_mpnn.data import EnsembleDataset, collate_ensemble
from peptide_ensemble_mpnn.models import EnsembleMPNN
from peptide_ensemble_mpnn.utils import ensure_dir, save_json, set_seed


def canonicalize_smiles(smiles: str) -> str:
    s = (smiles or "").strip()
    # Keep raw SMILES to avoid an RDKit dependency in hybrid training.
    return s


def finite_ratio(t: torch.Tensor) -> float:
    if t.numel() == 0:
        return 0.0
    return float(torch.isfinite(t).float().mean().item())


def finite_stats(t: torch.Tensor) -> Dict[str, float]:
    flat = t.reshape(-1)
    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        return {
            "finite_ratio": finite_ratio(t),
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
        }
    return {
        "finite_ratio": finite_ratio(t),
        "min": float(finite.min().item()),
        "max": float(finite.max().item()),
        "mean": float(finite.mean().item()),
        "std": float(finite.std(unbiased=False).item()),
    }


def model_has_nonfinite_parameters(model: nn.Module) -> bool:
    for p in model.parameters():
        if not torch.isfinite(p).all():
            return True
    return False


def model_has_nonfinite_gradients(model: nn.Module) -> bool:
    for p in model.parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            return True
    return False


def _load_checkpoint_state(ckpt_path: str) -> Tuple[Dict[str, torch.Tensor], Dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state", ckpt)
        args = ckpt.get("args", {})
    else:
        state = ckpt
        args = {}
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")
    return state, args


def _extract_ensemble_state_from_checkpoint(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    ensemble_state: Dict[str, torch.Tensor] = {}
    prefixes = ("backbone.", "regression_head.", "pretrain_head.")
    for k, v in state.items():
        if k.startswith("ensemble."):
            ensemble_state[k[len("ensemble.") :]] = v
        elif k.startswith(prefixes):
            ensemble_state[k] = v
    return ensemble_state


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


def valid_target_mask(target: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(target)


def pearson_r(pred: torch.Tensor, target: torch.Tensor) -> float:
    if pred.numel() < 2:
        return float("nan")
    x = pred - pred.mean()
    y = target - target.mean()
    den = torch.sqrt((x * x).sum() * (y * y).sum()).clamp_min(1e-12)
    return float((x * y).sum() / den)


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    if pred.numel() < 2:
        return float("nan")
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum().clamp_min(1e-12)
    return float(1.0 - ss_res / ss_tot)


def load_target_scaling(path: str) -> Optional[Tuple[float, float]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Target scaling file not found: {path}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    tmin = float(payload.get("target_min"))
    tmax = float(payload.get("target_max"))
    if not math.isfinite(tmin) or not math.isfinite(tmax):
        raise ValueError("Invalid target scaling values")
    if tmax <= tmin:
        raise ValueError("Invalid target scaling range; expected target_max > target_min")
    return (tmin, tmax)


def scale_target(target: torch.Tensor, scaling: Optional[Tuple[float, float]]) -> torch.Tensor:
    if scaling is None:
        return target
    tmin, tmax = scaling
    return (target - tmin) / max(tmax - tmin, 1e-8)


def inverse_scale_target(target: torch.Tensor, scaling: Optional[Tuple[float, float]]) -> torch.Tensor:
    if scaling is None:
        return target
    tmin, tmax = scaling
    return target * (tmax - tmin) + tmin


def infer_clm_hidden_dim(clm: nn.Module, tokenizer: Any) -> int:
    cfg = getattr(clm, "config", None)
    for attr in ("hidden_size", "d_model", "n_embd", "dim", "embedding_size"):
        val = getattr(cfg, attr, None)
        if isinstance(val, int) and val > 0:
            return int(val)

    with torch.no_grad():
        sample = tokenizer(["C"], return_tensors="pt", padding=True, truncation=True, max_length=8)
        out = clm(**sample)
        hid = getattr(out, "last_hidden_state", None)
        if torch.is_tensor(hid) and hid.ndim == 3 and hid.shape[-1] > 0:
            return int(hid.shape[-1])

    raise ValueError("Unable to infer CLM hidden dimension from config or model output")



class HybridDataset(Dataset):
    def __init__(self, path: str, require_target: bool) -> None:
        self.base = EnsembleDataset(path, require_target=require_target)
        self.smiles: List[str] = []
        for row in self.base.samples:
            smi = str(row.get("summary_smiles", ""))
            self.smiles.append(canonicalize_smiles(smi) if smi else "")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict:
        item = self.base[idx]
        item["idx"] = idx
        item["smiles"] = self.smiles[idx]
        return item


def collate_hybrid(batch: List[Dict]) -> Dict[str, torch.Tensor | List[str] | torch.Tensor]:
    smiles = [b["smiles"] for b in batch]
    idxs = torch.tensor([int(b["idx"]) for b in batch], dtype=torch.long)
    core_batch = []
    for b in batch:
        core_batch.append({k: v for k, v in b.items() if k not in {"smiles", "idx"}})
    out = collate_ensemble(core_batch)
    out["smiles"] = smiles
    out["idx"] = idxs
    return out


class HybridRegressor(nn.Module):
    def __init__(
        self,
        ensemble: EnsembleMPNN,
        clm_model_name: str,
        dropout: float = 0.1,
        head_hidden_dim: int = 0,
        head_num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.ensemble = ensemble
        self.tokenizer = AutoTokenizer.from_pretrained(clm_model_name, trust_remote_code=True)
        self.clm = AutoModel.from_pretrained(clm_model_name, trust_remote_code=True)

        d_ens = ensemble.backbone.residue_embed.embedding_dim
        d_clm = infer_clm_hidden_dim(self.clm, self.tokenizer)
        self.ensemble_norm = nn.LayerNorm(d_ens)
        self.clm_norm = nn.LayerNorm(d_clm)
        head_hidden = int(head_hidden_dim) if int(head_hidden_dim) > 0 else int(d_ens)
        if int(head_num_layers) < 1:
            raise ValueError("head_num_layers must be >= 1")
        head_layers: List[nn.Module] = []
        in_dim = int(d_ens + d_clm)
        if int(head_num_layers) == 1:
            head_layers.append(nn.Linear(in_dim, 1))
        else:
            head_layers.extend([nn.Linear(in_dim, head_hidden), nn.GELU(), nn.Dropout(dropout)])
            for _ in range(int(head_num_layers) - 2):
                head_layers.extend([nn.Linear(head_hidden, head_hidden), nn.GELU(), nn.Dropout(dropout)])
            head_layers.append(nn.Linear(head_hidden, 1))
        self.head = nn.Sequential(*head_layers)

    def forward(
        self,
        batch: Dict[str, torch.Tensor | List[str]],
        device: torch.device,
        max_smiles_len: int,
        return_debug: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, float]]:
        coords = torch.nan_to_num(batch["coords"], nan=0.0, posinf=0.0, neginf=0.0)
        conformer_weights = torch.nan_to_num(batch["conformer_weights"], nan=0.0, posinf=0.0, neginf=0.0)
        enc = self.ensemble.encode(
            residue_type=batch["residue_type"],
            coords=coords,
            node_mask=batch["node_mask"],
            conformer_mask=batch["conformer_mask"],
            conformer_weights=conformer_weights,
        )
        ens_emb = torch.nan_to_num(enc["ensemble_embedding"], nan=0.0, posinf=0.0, neginf=0.0)
        ens_emb = self.ensemble_norm(ens_emb)

        smiles = batch["smiles"]
        tok = self.tokenizer(
            smiles,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_smiles_len,
        )
        tok = {k: v.to(device) for k, v in tok.items()}
        if "attention_mask" in tok:
            tok["attention_mask"] = tok["attention_mask"].bool()
        out = self.clm(**tok)
        maybe_mean_pool = getattr(out, "mean_pool", None)
        if torch.is_tensor(maybe_mean_pool) and maybe_mean_pool.ndim == 2:
            clm_emb = maybe_mean_pool
        else:
            raise RuntimeError("CLM forward did not return mean_pool; expected mean_pool from hosted model")
        clm_emb = torch.nan_to_num(clm_emb, nan=0.0, posinf=0.0, neginf=0.0)
        clm_emb = self.clm_norm(clm_emb)

        fused = torch.cat([ens_emb, clm_emb], dim=-1)
        pred = self.head(fused).squeeze(-1)

        if not return_debug:
            return pred

        pred_stats = finite_stats(pred)
        debug = {
            "ensemble_finite_ratio": finite_ratio(ens_emb),
            "clm_finite_ratio": finite_ratio(clm_emb),
            "fused_finite_ratio": finite_ratio(fused),
            "head_finite_ratio": finite_ratio(pred),
            "pred_min": pred_stats["min"],
            "pred_max": pred_stats["max"],
            "pred_mean": pred_stats["mean"],
            "pred_std": pred_stats["std"],
        }
        return pred, debug


def train_epoch(
    model: HybridRegressor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    target_scaling: Optional[Tuple[float, float]],
    max_smiles_len: int,
    epoch: int,
    batch_log_interval: int,
    max_nonfinite_batch_fraction: float,
    checkpoint_interval_fraction: float = 0.5,
    interval_checkpoint_callback: Optional[Callable[[int, float], None]] = None,
) -> Dict[str, float]:
    model.train()
    sum_loss = 0.0
    sum_mae = 0.0
    batches = 0
    pred_total = 0
    pred_finite = 0
    diagnostics_batches: List[Dict[str, float]] = []
    half_checkpoint_saved = False
    total_batches = max(len(loader), 1)
    for batch_idx, batch in enumerate(loader):
        batch_t = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        pred_out = model(batch_t, device, max_smiles_len=max_smiles_len, return_debug=True)
        pred, debug = pred_out
        head_nonfinite_fraction = 1.0 - float(debug["head_finite_ratio"])
        if head_nonfinite_fraction > max_nonfinite_batch_fraction:
            raise RuntimeError(
                f"Train batch {batch_idx} exceeded non-finite prediction threshold: {head_nonfinite_fraction:.6f} > {max_nonfinite_batch_fraction:.6f}"
            )
        pred_total += int(pred.numel())
        pred_finite += int(torch.isfinite(pred).sum().item())
        diagnostics_batches.append(
            {
                "epoch": float(epoch),
                "batch": float(batch_idx),
                **debug,
            }
        )

        if batch_idx % max(batch_log_interval, 1) == 0:
            print(
                {
                    "phase": "train",
                    "epoch": epoch,
                    "batch": batch_idx,
                    "ensemble_finite_ratio": debug["ensemble_finite_ratio"],
                    "clm_finite_ratio": debug["clm_finite_ratio"],
                    "fused_finite_ratio": debug["fused_finite_ratio"],
                    "head_finite_ratio": debug["head_finite_ratio"],
                },
                flush=True,
            )

        valid = valid_target_mask(batch_t["target"])
        pred = pred[valid]
        target_raw = batch_t["target"][valid]
        if pred.numel() == 0:
            continue
        target = scale_target(target_raw, target_scaling)

        finite = torch.isfinite(pred) & torch.isfinite(target)
        pred = pred[finite]
        target = target[finite]
        target_raw = target_raw[finite]
        if pred.numel() == 0:
            continue

        loss = nn.functional.mse_loss(pred, target)
        if not torch.isfinite(loss):
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if model_has_nonfinite_gradients(model):
            raise RuntimeError(f"Non-finite gradients detected at train batch {batch_idx}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if model_has_nonfinite_parameters(model):
            raise RuntimeError(f"Non-finite parameters detected after optimizer step at train batch {batch_idx}")

        # Save an interval checkpoint once per epoch when configured (e.g., 0.5 for half-epoch).
        if (
            interval_checkpoint_callback is not None
            and checkpoint_interval_fraction > 0.0
            and checkpoint_interval_fraction < 1.0
            and (not half_checkpoint_saved)
        ):
            progress = float(batch_idx + 1) / float(total_batches)
            if progress >= checkpoint_interval_fraction:
                try:
                    interval_checkpoint_callback(epoch, checkpoint_interval_fraction)
                except Exception as exc:
                    print(
                        {
                            "warning": "interval_checkpoint_failed",
                            "epoch": epoch,
                            "progress_fraction": float(checkpoint_interval_fraction),
                            "error": str(exc),
                        },
                        flush=True,
                    )
                half_checkpoint_saved = True

        pred_raw = inverse_scale_target(pred, target_scaling)
        mae = (pred_raw - target_raw).abs().mean()
        sum_loss += loss.item()
        sum_mae += mae.item()
        batches += 1

    if pred_total == 0:
        raise RuntimeError("Train epoch produced zero predictions; dataset/collate path is invalid")
    if pred_finite == 0:
        raise RuntimeError("Train epoch produced only non-finite predictions")
    if batches == 0:
        raise RuntimeError("Train epoch had no finite supervised batches")

    return {
        "loss": sum_loss / max(batches, 1),
        "mae": sum_mae / max(batches, 1),
        "pred_finite_ratio": float(pred_finite) / float(max(pred_total, 1)),
        "diagnostics_batches": diagnostics_batches,
    }


def eval_epoch(
    model: HybridRegressor,
    loader: DataLoader,
    device: torch.device,
    target_scaling: Optional[Tuple[float, float]],
    max_smiles_len: int,
    epoch: int,
    batch_log_interval: int,
    max_nonfinite_batch_fraction: float,
) -> Dict[str, float]:
    model.eval()
    sum_loss = 0.0
    sum_mae = 0.0
    batches = 0
    pred_all = []
    target_all = []
    pred_total = 0
    pred_finite = 0
    diagnostics_batches: List[Dict[str, float]] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch_t = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            pred_out = model(batch_t, device, max_smiles_len=max_smiles_len, return_debug=True)
            pred, debug = pred_out
            head_nonfinite_fraction = 1.0 - float(debug["head_finite_ratio"])
            if head_nonfinite_fraction > max_nonfinite_batch_fraction:
                raise RuntimeError(
                    f"Validation batch {batch_idx} exceeded non-finite prediction threshold: {head_nonfinite_fraction:.6f} > {max_nonfinite_batch_fraction:.6f}"
                )
            pred_total += int(pred.numel())
            pred_finite += int(torch.isfinite(pred).sum().item())
            diagnostics_batches.append(
                {
                    "epoch": float(epoch),
                    "batch": float(batch_idx),
                    **debug,
                }
            )

            if batch_idx % max(batch_log_interval, 1) == 0:
                print(
                    {
                        "phase": "val",
                        "epoch": epoch,
                        "batch": batch_idx,
                        "ensemble_finite_ratio": debug["ensemble_finite_ratio"],
                        "clm_finite_ratio": debug["clm_finite_ratio"],
                        "fused_finite_ratio": debug["fused_finite_ratio"],
                        "head_finite_ratio": debug["head_finite_ratio"],
                    },
                    flush=True,
                )

            valid = valid_target_mask(batch_t["target"])
            pred = pred[valid]
            target_raw = batch_t["target"][valid]
            if pred.numel() == 0:
                continue
            target = scale_target(target_raw, target_scaling)

            finite = torch.isfinite(pred) & torch.isfinite(target)
            pred = pred[finite]
            target = target[finite]
            target_raw = target_raw[finite]
            if pred.numel() == 0:
                continue

            loss = nn.functional.mse_loss(pred, target)
            if not torch.isfinite(loss):
                continue
            pred_raw = inverse_scale_target(pred, target_scaling)
            mae = (pred_raw - target_raw).abs().mean()
            sum_loss += loss.item()
            sum_mae += mae.item()
            batches += 1
            pred_all.append(pred_raw.detach().cpu())
            target_all.append(target_raw.detach().cpu())

    if pred_all:
        p = torch.cat(pred_all)
        t = torch.cat(target_all)
        r = pearson_r(p, t)
        r2 = r2_score(p, t)
    else:
        r = float("nan")
        r2 = float("nan")

    if pred_total == 0:
        raise RuntimeError("Validation epoch produced zero predictions; dataset/collate path is invalid")
    if pred_finite == 0:
        raise RuntimeError("Validation epoch produced only non-finite predictions")
    if batches == 0:
        raise RuntimeError("Validation epoch had no finite supervised batches")

    return {
        "loss": sum_loss / max(batches, 1),
        "mae": sum_mae / max(batches, 1),
        "pearson_r": r,
        "r2": r2,
        "pred_finite_ratio": float(pred_finite) / float(max(pred_total, 1)),
        "diagnostics_batches": diagnostics_batches,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-data", required=True)
    p.add_argument("--val-data", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--init-checkpoint", required=True)
    p.add_argument("--hf-model", default="aaronfeller/peptideclm-2-hybrid-base")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--early-stopping-patience", type=int, default=15)
    p.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    p.add_argument("--checkpoint-interval-fraction", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--lr-head", type=float, default=2e-4)
    p.add_argument("--lr-clm", type=float, default=2e-5)
    p.add_argument("--lr-ensemble", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--head-hidden-dim", type=int, default=0)
    p.add_argument("--head-num-layers", type=int, default=2)
    p.add_argument("--head-dropout", type=float, default=0.1)
    p.add_argument("--hidden-dim", type=int, default=192)
    p.add_argument("--egnn-layers", type=int, default=6)
    p.add_argument("--k-neighbors", type=int, default=12)
    p.add_argument("--fusion-mode", choices=["setattn", "mean", "single"], default="setattn")
    p.add_argument("--disable-conformer-prior", action="store_true")
    p.add_argument("--max-smiles-len", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-scale-json", default="")
    p.add_argument("--batch-log-interval", type=int, default=50)
    p.add_argument("--max-nonfinite-batch-fraction", type=float, default=0.0)
    p.add_argument("--from-checkpoint-args", action="store_true", default=True)
    p.add_argument("--no-from-checkpoint-args", action="store_false", dest="from_checkpoint_args")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_hybrid_optimizer(model: HybridRegressor, args: argparse.Namespace) -> torch.optim.Optimizer:
    head_params = list(model.head.parameters()) + list(model.ensemble_norm.parameters()) + list(model.clm_norm.parameters())
    ensemble_params = list(model.ensemble.parameters())
    clm_params = list(model.clm.parameters())

    for p in head_params:
        p.requires_grad = True
    for p in ensemble_params:
        p.requires_grad = True
    for p in clm_params:
        p.requires_grad = True

    param_groups = [
        {"params": [p for p in head_params if p.requires_grad], "lr": float(args.lr_head), "weight_decay": float(args.weight_decay), "group": "head"},
        {"params": [p for p in ensemble_params if p.requires_grad], "lr": float(args.lr_ensemble), "weight_decay": float(args.weight_decay), "group": "ensemble"},
        {"params": [p for p in clm_params if p.requires_grad], "lr": float(args.lr_clm), "weight_decay": float(args.weight_decay), "group": "clm"},
    ]

    # Remove empty groups while preserving intended order.
    param_groups = [g for g in param_groups if len(g["params"]) > 0]

    if not param_groups:
        raise RuntimeError("No trainable parameters found for optimizer")

    optimizer_groups = [{"params": g["params"], "lr": g["lr"], "weight_decay": g["weight_decay"]} for g in param_groups]
    optimizer = torch.optim.AdamW(optimizer_groups)
    print(
        {
            "optimizer_param_groups": [
                {
                    "group": g["group"],
                    "lr": g["lr"],
                    "params": sum(int(p.numel()) for p in g["params"]),
                }
                for g in param_groups
            ]
        },
        flush=True,
    )
    return optimizer


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.outdir)

    device = torch.device(args.device)
    target_scaling = load_target_scaling(args.target_scale_json)

    train_ds = HybridDataset(args.train_data, require_target=True)
    val_ds = HybridDataset(args.val_data, require_target=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_hybrid)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_hybrid)

    state, ckpt_args = _load_checkpoint_state(args.init_checkpoint)

    hidden_dim = int(ckpt_args.get("hidden_dim", args.hidden_dim)) if args.from_checkpoint_args else args.hidden_dim
    egnn_layers = int(ckpt_args.get("egnn_layers", args.egnn_layers)) if args.from_checkpoint_args else args.egnn_layers
    k_neighbors = int(ckpt_args.get("k_neighbors", args.k_neighbors)) if args.from_checkpoint_args else args.k_neighbors
    fusion_mode = str(ckpt_args.get("fusion_mode", args.fusion_mode)) if args.from_checkpoint_args else args.fusion_mode
    disable_conformer_prior = bool(ckpt_args.get("disable_conformer_prior", args.disable_conformer_prior)) if args.from_checkpoint_args else args.disable_conformer_prior

    cfg = ModelConfig(
        hidden_dim=hidden_dim,
        egnn_layers=egnn_layers,
        k_neighbors=k_neighbors,
        fusion_mode=fusion_mode,
        use_conformer_prior=not disable_conformer_prior,
    )
    ensemble = EnsembleMPNN(cfg)
    model = HybridRegressor(
        ensemble=ensemble,
        clm_model_name=args.hf_model,
        dropout=float(args.head_dropout),
        head_hidden_dim=int(args.head_hidden_dim),
        head_num_layers=int(args.head_num_layers),
    ).to(device)

    ensemble_state = _extract_ensemble_state_from_checkpoint(state)
    filtered_state, dropped = _filter_compatible_state_dict(model.ensemble, ensemble_state)
    missing, unexpected = model.ensemble.load_state_dict(filtered_state, strict=False)
    print(
        {
            "ensemble_loaded_keys": len(filtered_state),
            "ensemble_missing_keys": len(missing),
            "ensemble_unexpected_keys": len(unexpected),
            "dropped_mismatch_or_unknown": dropped,
        }
    )

    optimizer = build_hybrid_optimizer(model, args)

    def save_interval_checkpoint(epoch_idx: int, progress_fraction: float) -> None:
        tag = f"p{int(round(progress_fraction * 100.0)):02d}"
        ensure_dir(args.outdir)
        ckpt = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
            "target_scaling": {
                "target_min": target_scaling[0],
                "target_max": target_scaling[1],
            }
            if target_scaling is not None
            else None,
            "epoch": epoch_idx,
            "epoch_progress": float(progress_fraction),
        }
        torch.save(ckpt, Path(args.outdir) / f"checkpoint_epoch_{epoch_idx:03d}_{tag}.pt")

    best_val = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            target_scaling,
            max_smiles_len=args.max_smiles_len,
            epoch=epoch,
            batch_log_interval=args.batch_log_interval,
            max_nonfinite_batch_fraction=args.max_nonfinite_batch_fraction,
            checkpoint_interval_fraction=float(args.checkpoint_interval_fraction),
            interval_checkpoint_callback=save_interval_checkpoint,
        )
        val_metrics = eval_epoch(
            model,
            val_loader,
            device,
            target_scaling,
            max_smiles_len=args.max_smiles_len,
            epoch=epoch,
            batch_log_interval=args.batch_log_interval,
            max_nonfinite_batch_fraction=args.max_nonfinite_batch_fraction,
        )

        epoch_diag = {
            "epoch": epoch,
            "train": {
                "pred_finite_ratio": train_metrics["pred_finite_ratio"],
                "batches": train_metrics["diagnostics_batches"],
            },
            "val": {
                "pred_finite_ratio": val_metrics["pred_finite_ratio"],
                "batches": val_metrics["diagnostics_batches"],
            },
        }
        ensure_dir(args.outdir)
        save_json(str(Path(args.outdir) / f"diagnostics_epoch_{epoch:03d}.json"), epoch_diag)

        row = {
            "epoch": epoch,
            "train_loss": float(train_metrics["loss"]),
            "train_mae": float(train_metrics["mae"]),
            "train_pred_finite_ratio": float(train_metrics["pred_finite_ratio"]),
            "val_loss": float(val_metrics["loss"]),
            "val_mae": float(val_metrics["mae"]),
            "val_pearson_r": float(val_metrics["pearson_r"]),
            "val_r2": float(val_metrics["r2"]),
            "val_pred_finite_ratio": float(val_metrics["pred_finite_ratio"]),
        }
        history.append(row)
        print(row, flush=True)

        if not math.isfinite(float(val_metrics["loss"])):
            raise RuntimeError(f"Validation loss is non-finite at epoch {epoch}; aborting to avoid saving unusable checkpoint")

        improved = float(val_metrics["loss"]) < (best_val - float(args.early_stopping_min_delta))
        if improved:
            best_val = float(val_metrics["loss"])
            best_epoch = epoch
            epochs_without_improvement = 0
            ckpt = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "args": vars(args),
                "target_scaling": {
                    "target_min": target_scaling[0],
                    "target_max": target_scaling[1],
                }
                if target_scaling is not None
                else None,
                "epoch": epoch,
                "best_val": best_val,
            }
            ensure_dir(args.outdir)
            torch.save(ckpt, Path(args.outdir) / "best_finetune.pt")
        else:
            epochs_without_improvement += 1

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(
                {
                    "early_stopping": True,
                    "stopped_epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val,
                    "patience": int(args.early_stopping_patience),
                    "min_delta": float(args.early_stopping_min_delta),
                },
                flush=True,
            )
            break

    ensure_dir(args.outdir)
    save_json(str(Path(args.outdir) / "history.json"), {"history": history})


if __name__ == "__main__":
    main()
