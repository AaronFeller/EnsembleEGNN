#!/usr/bin/env python3
"""Finetune BERT/CLM-only regression model for PAMPA."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from peptide_ensemble_mpnn.data import EnsembleDataset
from peptide_ensemble_mpnn.utils import ensure_dir, save_json, set_seed


def canonicalize_smiles(smiles: str) -> str:
    return (smiles or "").strip()


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


class BertDataset(Dataset):
    def __init__(self, path: str, require_target: bool) -> None:
        self.base = EnsembleDataset(path, require_target=require_target)
        self.smiles: List[str] = []
        for row in self.base.samples:
            smi = str(row.get("summary_smiles", ""))
            self.smiles.append(canonicalize_smiles(smi))

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


def collate_bert(batch: List[Dict]) -> Dict[str, torch.Tensor | List[str]]:
    return {
        "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
        "smiles": [b["smiles"] for b in batch],
        "target": torch.tensor([b["target"] for b in batch], dtype=torch.float32),
    }


class BertRegressor(nn.Module):
    def __init__(self, clm_model_name: str, dropout: float = 0.1, head_hidden_dim: int = 0) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(clm_model_name, trust_remote_code=True)
        self.clm = AutoModel.from_pretrained(clm_model_name, trust_remote_code=True)

        d_clm = infer_clm_hidden_dim(self.clm, self.tokenizer)
        hidden = int(head_hidden_dim) if int(head_hidden_dim) > 0 else int(d_clm)
        self.clm_norm = nn.LayerNorm(d_clm)
        self.head = nn.Sequential(
            nn.Linear(d_clm, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        batch: Dict[str, torch.Tensor | List[str]],
        device: torch.device,
        max_smiles_len: int,
        return_debug: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, float]]:
        tok = self.tokenizer(
            batch["smiles"],
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
        pred = self.head(clm_emb).squeeze(-1)

        if not return_debug:
            return pred

        pred_stats = finite_stats(pred)
        debug = {
            "clm_finite_ratio": finite_ratio(clm_emb),
            "head_finite_ratio": finite_ratio(pred),
            "pred_min": pred_stats["min"],
            "pred_max": pred_stats["max"],
            "pred_mean": pred_stats["mean"],
            "pred_std": pred_stats["std"],
        }
        return pred, debug


def train_epoch(
    model: BertRegressor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    target_scaling: Optional[Tuple[float, float]],
    max_smiles_len: int,
    epoch: int,
    batch_log_interval: int,
    max_nonfinite_batch_fraction: float,
) -> Dict[str, float]:
    model.train()
    sum_loss = 0.0
    sum_mae = 0.0
    batches = 0
    pred_total = 0
    pred_finite = 0
    diagnostics_batches: List[Dict[str, float]] = []

    for batch_idx, batch in enumerate(loader):
        batch_t = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        pred, debug = model(batch_t, device, max_smiles_len=max_smiles_len, return_debug=True)

        head_nonfinite_fraction = 1.0 - float(debug["head_finite_ratio"])
        if head_nonfinite_fraction > max_nonfinite_batch_fraction:
            raise RuntimeError(
                f"Train batch {batch_idx} exceeded non-finite prediction threshold: "
                f"{head_nonfinite_fraction:.6f} > {max_nonfinite_batch_fraction:.6f}"
            )

        pred_total += int(pred.numel())
        pred_finite += int(torch.isfinite(pred).sum().item())
        diagnostics_batches.append({"epoch": float(epoch), "batch": float(batch_idx), **debug})

        if batch_idx % max(batch_log_interval, 1) == 0:
            print(
                {
                    "phase": "train",
                    "epoch": epoch,
                    "batch": batch_idx,
                    "clm_finite_ratio": debug["clm_finite_ratio"],
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

        pred_raw = inverse_scale_target(pred, target_scaling)
        mae = (pred_raw - target_raw).abs().mean()
        sum_loss += loss.item()
        sum_mae += mae.item()
        batches += 1

    if pred_total == 0:
        raise RuntimeError("Train epoch produced zero predictions")
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
    model: BertRegressor,
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
    pred_all: List[torch.Tensor] = []
    target_all: List[torch.Tensor] = []
    pred_total = 0
    pred_finite = 0
    diagnostics_batches: List[Dict[str, float]] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch_t = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            pred, debug = model(batch_t, device, max_smiles_len=max_smiles_len, return_debug=True)

            head_nonfinite_fraction = 1.0 - float(debug["head_finite_ratio"])
            if head_nonfinite_fraction > max_nonfinite_batch_fraction:
                raise RuntimeError(
                    f"Validation batch {batch_idx} exceeded non-finite prediction threshold: "
                    f"{head_nonfinite_fraction:.6f} > {max_nonfinite_batch_fraction:.6f}"
                )

            pred_total += int(pred.numel())
            pred_finite += int(torch.isfinite(pred).sum().item())
            diagnostics_batches.append({"epoch": float(epoch), "batch": float(batch_idx), **debug})

            if batch_idx % max(batch_log_interval, 1) == 0:
                print(
                    {
                        "phase": "val",
                        "epoch": epoch,
                        "batch": batch_idx,
                        "clm_finite_ratio": debug["clm_finite_ratio"],
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
        raise RuntimeError("Validation epoch produced zero predictions")
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
    p.add_argument("--hf-model", default="aaronfeller/peptideclm-2-hybrid-base")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--early-stopping-patience", type=int, default=3)
    p.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    p.add_argument("--checkpoint-interval-fraction", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--head-hidden-dim", type=int, default=0)
    p.add_argument("--head-dropout", type=float, default=0.1)
    p.add_argument("--max-smiles-len", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-scale-json", default="")
    p.add_argument("--batch-log-interval", type=int, default=50)
    p.add_argument("--max-nonfinite-batch-fraction", type=float, default=0.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_optimizer(model: BertRegressor, args: argparse.Namespace) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found")
    opt = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay))
    print(
        {
            "optimizer": "AdamW",
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "params": sum(int(p.numel()) for p in params),
        },
        flush=True,
    )
    return opt


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.outdir)

    device = torch.device(args.device)
    target_scaling = load_target_scaling(args.target_scale_json)

    train_ds = BertDataset(args.train_data, require_target=True)
    val_ds = BertDataset(args.val_data, require_target=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_bert)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_bert)

    model = BertRegressor(
        clm_model_name=args.hf_model,
        dropout=float(args.head_dropout),
        head_hidden_dim=int(args.head_hidden_dim),
    ).to(device)

    optimizer = build_optimizer(model, args)

    def save_interval_checkpoint(epoch_idx: int, progress_fraction: float) -> None:
        ensure_dir(args.outdir)
        tag = f"p{int(round(progress_fraction * 100.0)):02d}"
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
    history: List[Dict[str, float]] = []

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
        )

        # Save a progress checkpoint every X fraction of epoch (one snapshot per epoch).
        if 0.0 < float(args.checkpoint_interval_fraction) < 1.0:
            try:
                save_interval_checkpoint(epoch, float(args.checkpoint_interval_fraction))
            except Exception as exc:
                print({"warning": "interval_checkpoint_failed", "epoch": epoch, "error": str(exc)}, flush=True)

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
            raise RuntimeError(f"Validation loss is non-finite at epoch {epoch}")

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
