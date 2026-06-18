#!/usr/bin/env python3
"""Aggregate 5-model ensemble predictions per holdout cluster and report CV metrics."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--pred-root", required=True)
    p.add_argument("--out-json", required=True)
    return p.parse_args()


def pearson_r(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    den = max(denx * deny, 1e-12)
    return num / den


def r2_score(pred: List[float], tgt: List[float]) -> float:
    n = len(pred)
    if n < 2:
        return float("nan")
    mean_t = sum(tgt) / n
    ss_res = sum((t - p) ** 2 for p, t in zip(pred, tgt))
    ss_tot = sum((t - mean_t) ** 2 for t in tgt)
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


def metrics(pred: List[float], tgt: List[float]) -> Dict[str, float]:
    n = len(pred)
    if n == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "pearson_r": float("nan"), "r2": float("nan")}
    err = [p - t for p, t in zip(pred, tgt)]
    mae = sum(abs(e) for e in err) / n
    rmse = math.sqrt(sum(e * e for e in err) / n)
    return {"n": n, "mae": mae, "rmse": rmse, "pearson_r": pearson_r(pred, tgt), "r2": r2_score(pred, tgt)}


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))

    by_holdout: Dict[int, List[Dict]] = defaultdict(list)
    for run in manifest["runs"]:
        by_holdout[int(run["holdout_cluster"])].append(run)

    holdout_results = {}
    all_pred: List[float] = []
    all_tgt: List[float] = []

    pred_root = Path(args.pred_root)
    for holdout, runs in sorted(by_holdout.items()):
        bag: Dict[str, Dict[str, List[float]]] = {}
        for run in runs:
            val = int(run["val_cluster"])
            legacy_pred_file = pred_root / f"holdout_{holdout}" / f"val_{val}" / "test_predictions.jsonl"
            retry_pred_file = pred_root / f"holdout_{holdout}" / f"val_{val}" / "predictions.jsonl"
            hybrid_pred_file = pred_root / f"holdout_{holdout}_val_{val}" / "predictions.jsonl"
            if legacy_pred_file.exists():
                pred_file = legacy_pred_file
            elif retry_pred_file.exists():
                pred_file = retry_pred_file
            elif hybrid_pred_file.exists():
                pred_file = hybrid_pred_file
            else:
                continue
            with pred_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    key = row.get("summary_smiles") or str(row.get("index"))
                    if key not in bag:
                        bag[key] = {"pred": [], "target": row.get("target")}
                    bag[key]["pred"].append(float(row["prediction"]))

        pred_h: List[float] = []
        tgt_h: List[float] = []
        for key, item in bag.items():
            t = item.get("target")
            preds = item.get("pred", [])
            if t is None or not preds:
                continue
            p = sum(preds) / len(preds)
            pred_h.append(float(p))
            tgt_h.append(float(t))

        m = metrics(pred_h, tgt_h)
        holdout_results[str(holdout)] = m
        all_pred.extend(pred_h)
        all_tgt.extend(tgt_h)

    report = {
        "holdout_metrics": holdout_results,
        "overall": metrics(all_pred, all_tgt),
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
