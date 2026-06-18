#!/usr/bin/env python3
"""Build 6-fold holdout splits for CycPeptMPDB PAMPA using cluster assignments.

For each holdout cluster H:
- test: cluster H
- for each validation cluster V != H:
  - train: all clusters except {H, V}
  - val: cluster V

This yields 5 models per holdout and 30 model fits total for 6 clusters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-jsonl", nargs="+", required=True, help="Prepared JSONL files to pool (e.g., finetune_train/val).")
    p.add_argument("--clusters-csv", required=True, help="CSV with columns SMILES,PAMPA,cluster.")
    p.add_argument(
        "--summary-csv",
        default="",
        help="Optional CycPept summary CSV with columns including CycPeptMPDB_ID,sequence,helm,smiles.",
    )
    p.add_argument(
        "--matching-report-md",
        default="",
        help="Optional output path for markdown report of matched/unmatched rows. Defaults to <outdir>/matching_report.md.",
    )
    p.add_argument("--outdir", required=True)
    return p.parse_args()


def canonicalize_smiles(smiles: str, isomeric: bool = True) -> str:
    smiles = smiles.strip()
    if not smiles:
        return ""
    try:
        from rdkit import Chem  # type: ignore

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)
    except Exception:
        return ""


def canonicalize_helm(helm: str, isomeric: bool = True) -> str:
    helm = helm.strip()
    if not helm:
        return ""
    try:
        from rdkit import Chem  # type: ignore

        mol = Chem.MolFromHELM(helm)
        if mol is None:
            return ""
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)
    except Exception:
        return ""


def source_stem_to_helm(stem: str) -> str:
    aa = set("ACDEFGHIKLMNPQRSTVWY")
    toks = [t for t in stem.strip().split(".") if t]
    if not toks:
        return ""
    parts = []
    for t in toks:
        if len(t) == 1 and t in aa:
            parts.append(t)
        else:
            parts.append(f"[{t}]")
    n = len(parts)
    return f"PEPTIDE1{{{'.'.join(parts)}}}$PEPTIDE1,PEPTIDE1,1:R1-{n}:R2$$$"


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


def read_cluster_maps(path: str) -> Tuple[Dict[str, Tuple[float, int]], Dict[str, Tuple[float, int]], int]:
    by_iso: Dict[str, Tuple[float, int]] = {}
    noiso_multi: Dict[str, set[Tuple[float, int]]] = defaultdict(set)
    with Path(path).open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            smi_iso = canonicalize_smiles(row.get("SMILES", ""), isomeric=True)
            smi_noiso = canonicalize_smiles(row.get("SMILES", ""), isomeric=False)
            if not smi_iso or not smi_noiso:
                continue
            pampa_raw = row.get("PAMPA", "")
            cluster_raw = row.get("cluster", "")
            try:
                pampa = float(pampa_raw)
                cluster = int(cluster_raw)
            except ValueError:
                continue
            if math.isfinite(pampa):
                val = (pampa, cluster)
                by_iso[smi_iso] = val
                noiso_multi[smi_noiso].add(val)

    by_noiso: Dict[str, Tuple[float, int]] = {}
    ambiguous = 0
    for key, vals in noiso_multi.items():
        if len(vals) == 1:
            by_noiso[key] = next(iter(vals))
        else:
            ambiguous += 1
    return by_iso, by_noiso, ambiguous


def read_summary_lookup(path: str) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    if not path:
        return {}, {}
    by_id: Dict[str, Dict[str, str]] = {}
    by_seq: Dict[str, Dict[str, str]] = {}
    p = Path(path)
    if not p.exists():
        return {}, {}
    with p.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            sid = str(row.get("CycPeptMPDB_ID", "")).strip()
            seq = str(row.get("sequence", "")).strip()
            if sid:
                by_id[sid] = row
            if seq:
                by_seq[seq] = row
    return by_id, by_seq


def write_matching_report(
    out_path: Path,
    n_total: int,
    n_assigned: int,
    method_counts: Dict[str, int],
    unmatched_reasons: Dict[str, int],
    unmatched_rows: List[Dict[str, str]],
    target_min: Optional[float],
    target_max: Optional[float],
) -> None:
    lines = [
        "# CycPept Cluster Matching Report",
        "",
        f"- Total pooled rows: {n_total}",
        f"- Assigned rows: {n_assigned}",
        f"- Unmatched rows: {n_total - n_assigned}",
    ]
    if target_min is not None and target_max is not None:
        lines.extend(
            [
                f"- Global target min: {target_min:.6f}",
                f"- Global target max: {target_max:.6f}",
                "- Global target scaling: min-max to [0, 1]",
            ]
        )

    lines.extend(["", "## Match Methods", ""])
    for k in sorted(method_counts):
        lines.append(f"- {k}: {method_counts[k]}")

    lines.extend(["", "## Unmatched Reasons", ""])
    for k in sorted(unmatched_reasons):
        lines.append(f"- {k}: {unmatched_reasons[k]}")

    lines.extend(["", "## Sample Unmatched Rows (first 50)", ""])
    lines.append("| summary_id | source_stem | reason | summary_smiles |")
    lines.append("|---|---|---|---|")
    for row in unmatched_rows[:50]:
        smi = row.get("summary_smiles", "").replace("|", "\\|")
        if len(smi) > 120:
            smi = smi[:117] + "..."
        lines.append(
            f"| {row.get('summary_id', '')} | {row.get('source_stem', '')} | {row.get('reason', '')} | {smi} |"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pooled = read_rows(args.input_jsonl)
    cluster_map_iso, cluster_map_noiso, n_ambiguous_noiso = read_cluster_maps(args.clusters_csv)
    summary_by_id, summary_by_seq = read_summary_lookup(args.summary_csv)

    assigned_rows: List[Dict] = []
    unmatched_rows: List[Dict[str, str]] = []
    match_methods: Counter[str] = Counter()
    unmatched_reasons: Counter[str] = Counter()

    def try_assign(row: Dict) -> Tuple[Optional[Tuple[float, int]], str]:
        smi_raw = str(row.get("summary_smiles", "") or "")
        smi_iso = canonicalize_smiles(smi_raw, isomeric=True)
        smi_noiso = canonicalize_smiles(smi_raw, isomeric=False)

        if smi_noiso and smi_noiso in cluster_map_noiso:
            return cluster_map_noiso[smi_noiso], "summary_smiles_nonisomeric"
        if smi_iso and smi_iso in cluster_map_iso:
            return cluster_map_iso[smi_iso], "summary_smiles_isomeric"

        sid = str(row.get("summary_id", "") or "").strip()
        if sid and sid in summary_by_id:
            ref = summary_by_id[sid]
            ref_smi = str(ref.get("smiles", "") or "")
            ref_helm = str(ref.get("helm", "") or "")

            ref_smi_iso = canonicalize_smiles(ref_smi, isomeric=True)
            ref_smi_noiso = canonicalize_smiles(ref_smi, isomeric=False)
            ref_helm_iso = canonicalize_helm(ref_helm, isomeric=True)
            ref_helm_noiso = canonicalize_helm(ref_helm, isomeric=False)

            if ref_smi_noiso and ref_smi_noiso in cluster_map_noiso:
                return cluster_map_noiso[ref_smi_noiso], "summary_id_smiles_nonisomeric"
            if ref_helm_noiso and ref_helm_noiso in cluster_map_noiso:
                return cluster_map_noiso[ref_helm_noiso], "summary_id_helm_nonisomeric"
            if ref_smi_iso and ref_smi_iso in cluster_map_iso:
                return cluster_map_iso[ref_smi_iso], "summary_id_smiles_isomeric"
            if ref_helm_iso and ref_helm_iso in cluster_map_iso:
                return cluster_map_iso[ref_helm_iso], "summary_id_helm_isomeric"

        stem = str(row.get("source_stem", "") or "").strip()
        if stem and stem in summary_by_seq:
            ref = summary_by_seq[stem]
            ref_smi = str(ref.get("smiles", "") or "")
            ref_helm = str(ref.get("helm", "") or "")

            ref_smi_iso = canonicalize_smiles(ref_smi, isomeric=True)
            ref_smi_noiso = canonicalize_smiles(ref_smi, isomeric=False)
            ref_helm_iso = canonicalize_helm(ref_helm, isomeric=True)
            ref_helm_noiso = canonicalize_helm(ref_helm, isomeric=False)

            if ref_smi_noiso and ref_smi_noiso in cluster_map_noiso:
                return cluster_map_noiso[ref_smi_noiso], "source_stem_summary_smiles_nonisomeric"
            if ref_helm_noiso and ref_helm_noiso in cluster_map_noiso:
                return cluster_map_noiso[ref_helm_noiso], "source_stem_summary_helm_nonisomeric"
            if ref_smi_iso and ref_smi_iso in cluster_map_iso:
                return cluster_map_iso[ref_smi_iso], "source_stem_summary_smiles_isomeric"
            if ref_helm_iso and ref_helm_iso in cluster_map_iso:
                return cluster_map_iso[ref_helm_iso], "source_stem_summary_helm_isomeric"

        if stem:
            stem_helm = source_stem_to_helm(stem)
            stem_helm_iso = canonicalize_helm(stem_helm, isomeric=True)
            stem_helm_noiso = canonicalize_helm(stem_helm, isomeric=False)
            if stem_helm_noiso and stem_helm_noiso in cluster_map_noiso:
                return cluster_map_noiso[stem_helm_noiso], "source_stem_helm_nonisomeric"
            if stem_helm_iso and stem_helm_iso in cluster_map_iso:
                return cluster_map_iso[stem_helm_iso], "source_stem_helm_isomeric"

        if not smi_raw.strip():
            return None, "missing_summary_smiles"
        if not smi_iso:
            return None, "invalid_summary_smiles"
        return None, "no_cluster_match"

    for row in pooled:
        assigned, method = try_assign(row)
        if assigned is None:
            unmatched_reasons[method] += 1
            unmatched_rows.append(
                {
                    "summary_id": str(row.get("summary_id", "") or ""),
                    "source_stem": str(row.get("source_stem", "") or ""),
                    "summary_smiles": str(row.get("summary_smiles", "") or ""),
                    "reason": method,
                }
            )
            continue

        pampa, cluster = assigned
        new_row = dict(row)
        new_row["target"] = pampa
        new_row["cluster"] = cluster
        assigned_rows.append(new_row)
        match_methods[method] += 1

    clusters = sorted({int(r["cluster"]) for r in assigned_rows})
    by_cluster: Dict[int, List[Dict]] = {c: [] for c in clusters}
    for row in assigned_rows:
        by_cluster[int(row["cluster"])].append(row)

    manifest_runs = []
    for holdout in clusters:
        test_rows = by_cluster[holdout]
        test_path = outdir / f"holdout_{holdout}" / "test.jsonl"
        write_jsonl(test_path, test_rows)

        for val in clusters:
            if val == holdout:
                continue
            train_rows: List[Dict] = []
            for c in clusters:
                if c not in (holdout, val):
                    train_rows.extend(by_cluster[c])
            val_rows = by_cluster[val]

            model_dir = outdir / f"holdout_{holdout}" / f"val_{val}"
            train_path = model_dir / "train.jsonl"
            val_path = model_dir / "val.jsonl"
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

    target_min = None
    target_max = None
    target_range = None
    if assigned_rows:
        targets = [float(r["target"]) for r in assigned_rows]
        target_min = min(targets)
        target_max = max(targets)
        target_range = target_max - target_min

    scaling_path = outdir / "target_scaling.json"
    if target_min is not None and target_max is not None and target_range is not None:
        scaling_payload = {
            "mode": "minmax_0_1",
            "target_min": target_min,
            "target_max": target_max,
            "target_range": target_range,
            "n_assigned_rows": len(assigned_rows),
        }
        scaling_path.write_text(json.dumps(scaling_payload, indent=2), encoding="utf-8")

    manifest = {
        "clusters": clusters,
        "n_clusters": len(clusters),
        "n_models": len(manifest_runs),
        "n_assigned_rows": len(assigned_rows),
        "n_unmatched_rows": len(pooled) - len(assigned_rows),
        "matching_method_counts": dict(match_methods),
        "unmatched_reason_counts": dict(unmatched_reasons),
        "n_ambiguous_nonisomeric_cluster_keys": n_ambiguous_noiso,
        "target_scaling_json": str(scaling_path),
        "target_scaling": {
            "mode": "minmax_0_1",
            "target_min": target_min,
            "target_max": target_max,
            "target_range": target_range,
        },
        "runs": manifest_runs,
    }
    with (outdir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    report_md = Path(args.matching_report_md) if args.matching_report_md else (outdir / "matching_report.md")
    write_matching_report(
        out_path=report_md,
        n_total=len(pooled),
        n_assigned=len(assigned_rows),
        method_counts=dict(match_methods),
        unmatched_reasons=dict(unmatched_reasons),
        unmatched_rows=unmatched_rows,
        target_min=target_min,
        target_max=target_max,
    )

    print(manifest)


if __name__ == "__main__":
    main()
