#!/usr/bin/env python3
"""Self-supervised pretraining for ensemble MPNN on PDB-style data."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from peptide_ensemble_mpnn.config import MASK_TOKEN_ID, ModelConfig, PAD_TOKEN_ID, UNK_TOKEN_ID, VOCAB_SIZE
from peptide_ensemble_mpnn.data import EnsembleDataset, collate_ensemble
from peptide_ensemble_mpnn.losses import pretraining_losses
from peptide_ensemble_mpnn.models import EnsembleMPNN
from peptide_ensemble_mpnn.utils import ensure_dir, save_json, set_seed


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def make_residue_mask(node_mask: torch.Tensor, prob: float) -> torch.Tensor:
    rand = torch.rand_like(node_mask.float())
    mask = (rand < prob) & node_mask

    # Ensure at least one supervised token for each non-empty example.
    needs_one = (mask.sum(dim=1) == 0) & (node_mask.sum(dim=1) > 0)
    if needs_one.any():
        for row_idx in torch.nonzero(needs_one, as_tuple=False).squeeze(-1):
            valid_idx = torch.nonzero(node_mask[row_idx], as_tuple=False).squeeze(-1)
            pick = valid_idx[torch.randint(valid_idx.numel(), (1,), device=node_mask.device)]
            mask[row_idx, pick] = True
    return mask


def apply_masked_token_corruption(
    token_target: torch.Tensor,
    residue_mask: torch.Tensor,
    replace_prob: float,
    random_prob: float,
    keep_prob: float,
) -> torch.Tensor:
    if abs((replace_prob + random_prob + keep_prob) - 1.0) > 1e-6:
        raise ValueError("mask corruption probabilities must sum to 1.0")

    token_input = token_target.clone()
    if not residue_mask.any():
        return token_input

    roll = torch.rand_like(token_target.float())
    replace_mask = residue_mask & (roll < replace_prob)
    random_mask = residue_mask & (roll >= replace_prob) & (roll < replace_prob + random_prob)

    token_input[replace_mask] = MASK_TOKEN_ID

    if random_mask.any():
        num_random = int(random_mask.sum().item())
        random_tokens = torch.randint(low=0, high=VOCAB_SIZE, size=(num_random,), device=token_target.device)
        invalid = (random_tokens == PAD_TOKEN_ID) | (random_tokens == MASK_TOKEN_ID) | (random_tokens == UNK_TOKEN_ID)
        while invalid.any():
            random_tokens[invalid] = torch.randint(low=0, high=VOCAB_SIZE, size=(int(invalid.sum().item()),), device=token_target.device)
            invalid = (random_tokens == PAD_TOKEN_ID) | (random_tokens == MASK_TOKEN_ID) | (random_tokens == UNK_TOKEN_ID)
        token_input[random_mask] = random_tokens

    return token_input


def train_epoch(
    model: EnsembleMPNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mask_prob: float,
    noise_std: float,
    mask_replace_prob: float,
    mask_random_prob: float,
    mask_keep_prob: float,
    log_interval: int,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    sums = {"total": 0.0, "residue": 0.0, "coord": 0.0, "dist": 0.0, "mask_frac": 0.0}
    count = 0

    total_steps = len(loader)
    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        residue_target = batch["residue_type"]
        coords_target = batch["coords"]

        residue_mask = make_residue_mask(batch["node_mask"], mask_prob)
        residue_input = apply_masked_token_corruption(
            token_target=residue_target,
            residue_mask=residue_mask,
            replace_prob=mask_replace_prob,
            random_prob=mask_random_prob,
            keep_prob=mask_keep_prob,
        )

        coord_mask = (batch["conformer_mask"][:, :, None] & batch["node_mask"][:, None, :]).float().unsqueeze(-1)
        noise = torch.randn_like(coords_target) * noise_std
        coords_noisy = coords_target + noise * coord_mask

        out = model.forward_pretrain(
            residue_type=residue_input,
            coords=coords_noisy,
            node_mask=batch["node_mask"],
            conformer_mask=batch["conformer_mask"],
            conformer_weights=batch["conformer_weights"],
        )
        loss_dict = pretraining_losses(
            outputs=out,
            residue_target=residue_target,
            coords_target=coords_target,
            node_mask=batch["node_mask"],
            conformer_mask=batch["conformer_mask"],
            conformer_weights=batch["conformer_weights"],
            residue_mask=residue_mask,
        )

        optimizer.zero_grad(set_to_none=True)
        loss_dict["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for k in sums:
            if k == "mask_frac":
                masked_count = residue_mask.float().sum()
                valid_count = batch["node_mask"].float().sum().clamp_min(1.0)
                sums[k] += (masked_count / valid_count).item()
            else:
                sums[k] += loss_dict[k].item()
        count += 1

        if step % log_interval == 0 or step == total_steps:
            print(
                {
                    "phase": "train",
                    "epoch": epoch,
                    "step": step,
                    "steps_total": total_steps,
                    "train_total": sums["total"] / max(count, 1),
                    "train_residue": sums["residue"] / max(count, 1),
                    "train_coord": sums["coord"] / max(count, 1),
                    "train_dist": sums["dist"] / max(count, 1),
                    "train_mask_frac": sums["mask_frac"] / max(count, 1),
                },
                flush=True,
            )

    return {k: v / max(count, 1) for k, v in sums.items()}


def eval_epoch(
    model: EnsembleMPNN,
    loader: DataLoader,
    device: torch.device,
    mask_prob: float,
    noise_std: float,
    mask_replace_prob: float,
    mask_random_prob: float,
    mask_keep_prob: float,
    log_interval: int,
    epoch: int,
) -> Dict[str, float]:
    model.eval()
    sums = {"total": 0.0, "residue": 0.0, "coord": 0.0, "dist": 0.0, "mask_frac": 0.0}
    count = 0

    with torch.no_grad():
        total_steps = len(loader)
        for step, batch in enumerate(loader, start=1):
            batch = move_batch(batch, device)
            residue_target = batch["residue_type"]
            coords_target = batch["coords"]

            residue_mask = make_residue_mask(batch["node_mask"], mask_prob)
            residue_input = apply_masked_token_corruption(
                token_target=residue_target,
                residue_mask=residue_mask,
                replace_prob=mask_replace_prob,
                random_prob=mask_random_prob,
                keep_prob=mask_keep_prob,
            )

            coord_mask = (batch["conformer_mask"][:, :, None] & batch["node_mask"][:, None, :]).float().unsqueeze(-1)
            noise = torch.randn_like(coords_target) * noise_std
            coords_noisy = coords_target + noise * coord_mask

            out = model.forward_pretrain(
                residue_type=residue_input,
                coords=coords_noisy,
                node_mask=batch["node_mask"],
                conformer_mask=batch["conformer_mask"],
                conformer_weights=batch["conformer_weights"],
            )
            loss_dict = pretraining_losses(
                outputs=out,
                residue_target=residue_target,
                coords_target=coords_target,
                node_mask=batch["node_mask"],
                conformer_mask=batch["conformer_mask"],
                conformer_weights=batch["conformer_weights"],
                residue_mask=residue_mask,
            )
            for k in sums:
                if k == "mask_frac":
                    masked_count = residue_mask.float().sum()
                    valid_count = batch["node_mask"].float().sum().clamp_min(1.0)
                    sums[k] += (masked_count / valid_count).item()
                else:
                    sums[k] += loss_dict[k].item()
            count += 1

            if step % log_interval == 0 or step == total_steps:
                print(
                    {
                        "phase": "val",
                        "epoch": epoch,
                        "step": step,
                        "steps_total": total_steps,
                        "val_total": sums["total"] / max(count, 1),
                        "val_residue": sums["residue"] / max(count, 1),
                        "val_coord": sums["coord"] / max(count, 1),
                        "val_dist": sums["dist"] / max(count, 1),
                        "val_mask_frac": sums["mask_frac"] / max(count, 1),
                    },
                    flush=True,
                )

    return {k: v / max(count, 1) for k, v in sums.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-data", required=True)
    p.add_argument("--val-data", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--mask-prob", type=float, default=0.15)
    p.add_argument("--noise-std", type=float, default=0.15)
    p.add_argument("--mask-replace-prob", type=float, default=0.8)
    p.add_argument("--mask-random-prob", type=float, default=0.1)
    p.add_argument("--mask-keep-prob", type=float, default=0.1)
    p.add_argument("--hidden-dim", type=int, default=192)
    p.add_argument("--egnn-layers", type=int, default=6)
    p.add_argument("--k-neighbors", type=int, default=12)
    p.add_argument("--fusion-mode", choices=["setattn", "mean", "single"], default="setattn")
    p.add_argument("--disable-conformer-prior", action="store_true")
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if abs((args.mask_replace_prob + args.mask_random_prob + args.mask_keep_prob) - 1.0) > 1e-6:
        raise ValueError("--mask-replace-prob + --mask-random-prob + --mask-keep-prob must sum to 1.0")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    set_seed(args.seed)
    ensure_dir(args.outdir)

    device = torch.device(args.device)

    train_ds = EnsembleDataset(args.train_data, require_target=False)
    val_ds = EnsembleDataset(args.val_data, require_target=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_ensemble)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_ensemble)

    cfg = ModelConfig(
        hidden_dim=args.hidden_dim,
        egnn_layers=args.egnn_layers,
        k_neighbors=args.k_neighbors,
        fusion_mode=args.fusion_mode,
        use_conformer_prior=not args.disable_conformer_prior,
    )
    model = EnsembleMPNN(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = math.inf
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.mask_prob,
            args.noise_std,
            args.mask_replace_prob,
            args.mask_random_prob,
            args.mask_keep_prob,
            args.log_interval,
            epoch,
        )
        val_metrics = eval_epoch(
            model,
            val_loader,
            device,
            args.mask_prob,
            args.noise_std,
            args.mask_replace_prob,
            args.mask_random_prob,
            args.mask_keep_prob,
            args.log_interval,
            epoch,
        )

        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(row, flush=True)

        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            ckpt = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "best_val": best_val,
            }
            torch.save(ckpt, Path(args.outdir) / "best_pretrain.pt")

    save_json(str(Path(args.outdir) / "history.json"), {"history": history})


if __name__ == "__main__":
    main()
