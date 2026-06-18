#!/usr/bin/env python3
"""Build a clean PAMPA-only random k-fold dataset for finetuning.

Pipeline:
1) Rebuild PAMPA-matched rows from CycPept clusters CSV via existing matcher.
2) Pool unique assigned rows (from holdout test shards).
3) Drop rows with exact target in drop_targets (default: -10.0).
4) Build random k-fold splits for finetuning.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Hashable, List, Set, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-jsonl", nargs="+", required=True)
    p.add_argument("--clusters-csv", required=True)
    p.add_argument("--summary-csv", default="")
    p.add_argument("--tmp-cv-outdir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--drop-target",
        dest="drop_targets",
        action="append",
        type=float,
        default=None,
        help="Exact target value to exclude. Repeat to exclude multiple values.",
    )
    p.add_argument("--dedup-sequence", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def run_cmd(args: list[str]) -> None:
    proc = subprocess.run(args, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(args)}")


def row_key(row: Dict) -> Hashable:
    sid = str(row.get("summary_id", "")).strip()
    if sid:
        return ("summary_id", sid)
    return (
        "fallback",
        str(row.get("summary_smiles", "")).strip(),
        str(row.get("source_stem", "")).strip(),
        str(row.get("sequence", "")).strip(),
    )


def _stable_row_id(row: Dict) -> Tuple[int, str, str]:
    sid = str(row.get("summary_id", "")).strip()
    sid_num = int(sid) if sid.isdigit() else 10**12
    smi = str(row.get("summary_smiles", "")).strip()
    stem = str(row.get("source_stem", "")).strip()
    return (sid_num, smi, stem)


def dedup_rows_by_sequence(rows: List[Dict]) -> Tuple[List[Dict], int]:
    by_seq: Dict[str, List[Dict]] = {}
    for row in rows:
        seq = str(row.get("sequence", "")).strip()
        by_seq.setdefault(seq, []).append(row)

    selected: List[Dict] = []
    dropped = 0
    for seq, members in by_seq.items():
        if len(members) == 1:
            selected.append(members[0])
            continue

        targets = [float(m["target"]) for m in members]
        sorted_t = sorted(targets)
        n = len(sorted_t)
        if n % 2 == 1:
            median = sorted_t[n // 2]
        else:
            median = 0.5 * (sorted_t[n // 2 - 1] + sorted_t[n // 2])

        representative = min(
            members,
            key=lambda m: (abs(float(m["target"]) - median),) + _stable_row_id(m),
        )
        selected.append(representative)
        dropped += len(members) - 1

    selected.sort(key=_stable_row_id)
    return selected, dropped


def main() -> None:
    args = parse_args()
    drop_targets = args.drop_targets if args.drop_targets is not None else [-10.0]
    drop_target_set = {float(x) for x in drop_targets}

    root = Path(__file__).resolve().parents[1]
    py = sys.executable

    tmp_cv = Path(args.tmp_cv_outdir)
    outdir = Path(args.outdir)
    pooled_path = outdir / "pooled_pampa_no10.jsonl"

    if args.overwrite:
        if tmp_cv.exists():
            shutil.rmtree(tmp_cv)
        if outdir.exists():
            shutil.rmtree(outdir)

    tmp_cv.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    # Step 1: rebuild PAMPA-matched assignment from clusters CSV.
    build_cluster_script = root / "scripts" / "build_cycpept_pampa_splits.py"
    cmd = [
        py,
        str(build_cluster_script),
        "--input-jsonl",
        *args.input_jsonl,
        "--clusters-csv",
        args.clusters_csv,
        "--outdir",
        str(tmp_cv),
        "--matching-report-md",
        str(tmp_cv / "matching_report.md"),
    ]
    if args.summary_csv:
        cmd.extend(["--summary-csv", args.summary_csv])
    run_cmd(cmd)

    manifest_path = tmp_cv / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Step 2/3: pool unique rows and drop exact target values.
    seen: Set[Hashable] = set()
    dropped = 0
    unique_before_drop = 0
    filtered_rows: List[Dict] = []

    for c in manifest["clusters"]:
        test_fp = tmp_cv / f"holdout_{c}" / "test.jsonl"
        with test_fp.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = row_key(row)
                if key in seen:
                    continue
                seen.add(key)
                unique_before_drop += 1

                tgt = float(row["target"])
                if tgt in drop_target_set:
                    dropped += 1
                    continue
                filtered_rows.append(row)

    dedup_seq_dropped = 0
    if args.dedup_sequence:
        filtered_rows, dedup_seq_dropped = dedup_rows_by_sequence(filtered_rows)

    with pooled_path.open("w", encoding="utf-8") as out:
        for row in filtered_rows:
            out.write(json.dumps(row) + "\n")

    kept = len(filtered_rows)

    summary = {
        "tmp_cv_outdir": str(tmp_cv),
        "random_split_outdir": str(outdir),
        "unique_assigned_rows_before_drop": unique_before_drop,
        "dropped_target_eq": sorted(drop_target_set),
        "dropped_rows": dropped,
        "dedup_sequence": bool(args.dedup_sequence),
        "dedup_sequence_dropped_rows": dedup_seq_dropped,
        "kept_rows": kept,
        "pooled_jsonl": str(pooled_path),
        "source_manifest": str(manifest_path),
    }
    (outdir / "filter_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Step 4: rebuild random split from filtered pooled dataset.
    build_random_script = root / "scripts" / "build_random_kfold_splits.py"
    run_cmd(
        [
            py,
            str(build_random_script),
            "--input-jsonl",
            str(pooled_path),
            "--outdir",
            str(outdir),
            "--n-folds",
            str(int(args.n_folds)),
            "--seed",
            str(int(args.seed)),
            "--report-md",
            str(outdir / "split_report.md"),
        ]
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
