#!/usr/bin/env python3
"""Finetune ensemble model for regression targets."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from peptide_ensemble_mpnn.config import ModelConfig
from peptide_ensemble_mpnn.data import EnsembleDataset, collate_ensemble
from peptide_ensemble_mpnn.losses import regression_loss
from peptide_ensemble_mpnn.models import EnsembleMPNN
from peptide_ensemble_mpnn.utils import ensure_dir, save_json, set_seed


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


def _filter_compatible_state_dict(model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], int]:
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


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def valid_target_mask(target: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(target)


def pearson_r(pred: torch.Tensor, target: torch.Tensor) -> float:
    if pred.numel() < 2:
        return float("nan")
    x = pred - pred.mean()
    y = target - target.mean()
    den = torch.sqrt((x * x).sum() * (y * y).sum()).clamp_min(1e-12)
    return float((x * y).sum() / den)


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


def train_epoch(
    model: EnsembleMPNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    target_scaling: Optional[Tuple[float, float]],
) -> Dict[str, float]:
    model.train()
    sum_loss = 0.0
    sum_mae = 0.0
    batches = 0

    for batch in loader:
        batch = move_batch(batch, device)
        out = model.forward_regression(
            residue_type=batch["residue_type"],
            coords=batch["coords"],
            node_mask=batch["node_mask"],
            conformer_mask=batch["conformer_mask"],
            conformer_weights=batch["conformer_weights"],
        )

        valid = valid_target_mask(batch["target"])
        pred = out["prediction"][valid]
        target_raw = batch["target"][valid]
        if pred.numel() == 0:
            continue
        target = scale_target(target_raw, target_scaling)

        loss = regression_loss(pred, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        pred_raw = inverse_scale_target(pred, target_scaling)
        mae = (pred_raw - target_raw).abs().mean()
        sum_loss += loss.item()
        sum_mae += mae.item()
        batches += 1

    return {
        "loss": sum_loss / max(batches, 1),
        "mae": sum_mae / max(batches, 1),
    }


def eval_epoch(model: EnsembleMPNN, loader: DataLoader, device: torch.device, target_scaling: Optional[Tuple[float, float]]) -> Dict[str, float]:
    model.eval()
    sum_loss = 0.0
    sum_mae = 0.0
    batches = 0
    pred_all = []
    target_all = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            out = model.forward_regression(
                residue_type=batch["residue_type"],
                coords=batch["coords"],
                node_mask=batch["node_mask"],
                conformer_mask=batch["conformer_mask"],
                conformer_weights=batch["conformer_weights"],
            )

            valid = valid_target_mask(batch["target"])
            pred = out["prediction"][valid]
            target_raw = batch["target"][valid]
            if pred.numel() == 0:
                continue
            target = scale_target(target_raw, target_scaling)

            loss = regression_loss(pred, target)
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
    else:
        r = float("nan")

    return {
        "loss": sum_loss / max(batches, 1),
        "mae": sum_mae / max(batches, 1),
        "pearson_r": r,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-data", required=True)
    p.add_argument("--val-data", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--init-checkpoint", default="")
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=192)
    p.add_argument("--egnn-layers", type=int, default=6)
    p.add_argument("--k-neighbors", type=int, default=12)
    p.add_argument("--fusion-mode", choices=["setattn", "mean", "single"], default="setattn")
    p.add_argument("--disable-conformer-prior", action="store_true")
    p.add_argument(
        "--representation",
        choices=["auto", "residue", "all_atom", "a_carbon"],
        default="auto",
        help="Expected node representation in input JSONL.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-scale-json", default="", help="JSON file with global target_min/target_max for min-max scaling to [0,1].")
    p.add_argument("--from-checkpoint-args", action="store_true", default=True)
    p.add_argument("--no-from-checkpoint-args", action="store_false", dest="from_checkpoint_args")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.outdir)

    device = torch.device(args.device)
    target_scaling = load_target_scaling(args.target_scale_json)

    train_ds = EnsembleDataset(args.train_data, require_target=True)
    val_ds = EnsembleDataset(args.val_data, require_target=True)

    if len(train_ds.samples) == 0:
        raise ValueError("Training dataset is empty")
    train_has_node_tokens = "node_tokens" in train_ds.samples[0]
    if args.representation == "all_atom" and not train_has_node_tokens:
        raise ValueError("--representation all_atom requires node_tokens in dataset rows")
    if args.representation in {"residue", "a_carbon"} and train_has_node_tokens:
        raise ValueError("Dataset has node_tokens; choose --representation all_atom or use residue-style data")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_ensemble)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_ensemble)

    ckpt_state: Dict[str, torch.Tensor] = {}
    ckpt_args: Dict = {}
    if args.init_checkpoint:
        ckpt_state, ckpt_args = _load_checkpoint_state(args.init_checkpoint)

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
    model = EnsembleMPNN(cfg).to(device)

    if args.init_checkpoint:
        load_state, dropped = _filter_compatible_state_dict(model, ckpt_state)
        missing, unexpected = model.load_state_dict(load_state, strict=False)
        print({"missing_keys": len(missing), "unexpected_keys": len(unexpected), "dropped_mismatch_or_unknown": dropped})

    if args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    best_val = math.inf
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device, target_scaling)
        val_metrics = eval_epoch(model, val_loader, device, target_scaling)

        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(row)

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
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
            torch.save(ckpt, Path(args.outdir) / "best_finetune.pt")

    save_json(str(Path(args.outdir) / "history.json"), {"history": history})


if __name__ == "__main__":
    main()
