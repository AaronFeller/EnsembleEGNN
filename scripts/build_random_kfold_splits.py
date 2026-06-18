#!/usr/bin/env python3
"""Build random k-fold CV splits with one model per holdout fold.

For each holdout fold H:
- test: fold H
- val: next fold ((H % k) + 1)
- train: all remaining folds

This yields exactly k model fits (no ensemble averaging per holdout).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-jsonl", nargs="+", required=True, help="Prepared JSONL files to pool.")
    p.add_argument("--outdir", required=True)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--report-md",
        default="",
        help="Optional markdown report path. Defaults to <outdir>/split_report.md.",
    )
    return p.parse_args()


def read_rows(paths: List[str]) -> List[Dict]:
    rows: List[Dict] = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_report(path: Path, n_rows: int, n_folds: int, seed: int, fold_sizes: Dict[int, int]) -> None:
    lines = [
        "# Random K-Fold Split Report",
        "",
        f"- Total pooled rows: {n_rows}",
        f"- Number of folds: {n_folds}",
        f"- Seed: {seed}",
        "- Scheme: random k-fold, one model per holdout (no ensemble)",
        "",
        "## Fold Sizes",
        "",
    ]
    for fold in sorted(fold_sizes):
        lines.append(f"- Fold {fold}: {fold_sizes[fold]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.n_folds < 2:
        raise ValueError("--n-folds must be >= 2")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pooled = read_rows(args.input_jsonl)
    if len(pooled) < args.n_folds:
        raise ValueError(f"Need at least {args.n_folds} rows, found {len(pooled)}")

    rng = random.Random(args.seed)
    idxs = list(range(len(pooled)))
    rng.shuffle(idxs)

    by_fold: Dict[int, List[Dict]] = {f: [] for f in range(1, args.n_folds + 1)}
    for i, idx in enumerate(idxs):
        fold = (i % args.n_folds) + 1
        row = dict(pooled[idx])
        row["fold"] = fold
        by_fold[fold].append(row)

    manifest_runs = []
    for holdout in range(1, args.n_folds + 1):
        val = (holdout % args.n_folds) + 1

        test_rows = by_fold[holdout]
        val_rows = by_fold[val]
        train_rows: List[Dict] = []
        for f in range(1, args.n_folds + 1):
            if f not in (holdout, val):
                train_rows.extend(by_fold[f])

        model_dir = outdir / f"holdout_{holdout}" / f"val_{val}"
        test_path = outdir / f"holdout_{holdout}" / "test.jsonl"
        train_path = model_dir / "train.jsonl"
        val_path = model_dir / "val.jsonl"

        write_jsonl(test_path, test_rows)
        write_jsonl(train_path, train_rows)
        write_jsonl(val_path, val_rows)

        manifest_runs.append(
            {
                "holdout_cluster": holdout,
                "val_cluster": val,
                "train_data": str(train_path),
                "val_data": str(val_path),
                "test_data": str(test_path),
                "train_size": len(train_rows),
                "val_size": len(val_rows),
                "test_size": len(test_rows),
            }
        )

    targets = [float(r["target"]) for r in pooled]
    target_min = min(targets)
    target_max = max(targets)
    target_range = target_max - target_min

    scaling_path = outdir / "target_scaling.json"
    scaling_payload = {
        "mode": "minmax_0_1",
        "target_min": target_min,
        "target_max": target_max,
        "target_range": target_range,
        "n_assigned_rows": len(pooled),
    }
    scaling_path.write_text(json.dumps(scaling_payload, indent=2), encoding="utf-8")

    fold_sizes = {f: len(rows) for f, rows in by_fold.items()}
    manifest = {
        "scheme": "random_kfold_no_ensemble",
        "folds": list(range(1, args.n_folds + 1)),
        "n_folds": args.n_folds,
        "seed": args.seed,
        "n_models": len(manifest_runs),
        "n_assigned_rows": len(pooled),
        "n_unmatched_rows": 0,
        "fold_sizes": fold_sizes,
        "target_scaling_json": str(scaling_path),
        "target_scaling": {
            "mode": "minmax_0_1",
            "target_min": target_min,
            "target_max": target_max,
            "target_range": target_range,
        },
        "runs": manifest_runs,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report_md = Path(args.report_md) if args.report_md else (outdir / "split_report.md")
    write_report(report_md, len(pooled), args.n_folds, args.seed, fold_sizes)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
