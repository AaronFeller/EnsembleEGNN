#!/usr/bin/env python3
"""Prepare CREMP/CycPeptMPDB from pickle.tar.gz into ensemble JSONL.

Requires RDKit (the pickle files contain RDKit Mol objects).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

AA_SET = set("ACDEFGHIKLMNPQRSTVWY")
AA3 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pickle-archive", required=True, help="Path to pickle.tar.gz")
    p.add_argument("--summary-csv", default="", help="Optional summary csv for numeric targets")
    p.add_argument("--target-column", default="ensemblefreeenergy")
    p.add_argument("--outdir", required=True)
    p.add_argument("--max-conformers", type=int, default=5)
    p.add_argument(
        "--representation",
        choices=["chunked_residue", "all_atom", "a_carbon"],
        default="chunked_residue",
        help="Coordinate representation to write.",
    )
    p.add_argument("--heavy-only", action="store_true")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-by-cycle", action="store_true", default=True)
    p.add_argument("--no-split-by-cycle", action="store_false", dest="split_by_cycle")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=500)
    return p.parse_args()


def read_summary(path: str) -> Dict[str, Dict[str, str]]:
    if not path:
        return {}
    p = Path(path)
    out: Dict[str, Dict[str, str]] = {}
    with p.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            out[row["sequence"]] = row
    return out


def token_to_aa(token: str) -> str:
    clean = "".join(ch for ch in token if ch.isalpha())
    if not clean:
        return "X"
    up = clean.upper()
    if len(up) == 1 and up in AA_SET:
        return up
    if up in AA3:
        return AA3[up]
    if up.startswith("ME") and len(up) > 2:
        rem = up[2:]
        if rem in AA3:
            return AA3[rem]
        if rem[-1] in AA_SET:
            return rem[-1]
    if up[-1] in AA_SET:
        return up[-1]
    return "X"


def canonicalize_sequence(stem: str) -> str:
    toks = [t for t in stem.split(".") if t]
    return "".join(token_to_aa(t) for t in toks)


def cyclic_key(seq: str) -> str:
    if not seq:
        return ""
    rots = [seq[i:] + seq[:i] for i in range(len(seq))]
    rev = seq[::-1]
    rev_rots = [rev[i:] + rev[:i] for i in range(len(rev))]
    return min(rots + rev_rots)


def normalize_weights(weights: List[float]) -> List[float]:
    w = [max(float(x), 0.0) for x in weights]
    s = sum(w)
    if s <= 0:
        return [1.0 / len(w)] * len(w)
    return [x / s for x in w]


def get_target(summary_row: Optional[Dict[str, str]], col: str) -> Optional[float]:
    if summary_row is None or not col:
        return None
    raw = summary_row.get(col)
    if raw is None or raw == "":
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    if not math.isfinite(v):
        return None
    return v


def extract_coords_from_mol(mol, heavy_only: bool) -> tuple[List[List[List[float]]], List[int], List[str]]:
    # Import lazily so the script can print a clear message if RDKit is absent.
    from rdkit import Chem  # type: ignore

    atom_ids = []
    atom_symbols = []
    for i, a in enumerate(mol.GetAtoms()):
        if heavy_only and a.GetAtomicNum() == 1:
            continue
        atom_ids.append(i)
        atom_symbols.append(a.GetSymbol())

    out: List[List[List[float]]] = []
    for conf in mol.GetConformers():
        pts = []
        for i in atom_ids:
            p = conf.GetAtomPosition(i)
            pts.append([float(p.x), float(p.y), float(p.z)])
        if pts:
            out.append(pts)
    return out, atom_ids, atom_symbols


def _is_carbonyl_carbon(atom) -> bool:
    if atom.GetSymbol() != "C":
        return False
    for b in atom.GetBonds():
        if b.GetBondTypeAsDouble() == 2.0:
            other = b.GetOtherAtom(atom)
            if other.GetSymbol() == "O":
                return True
    return False


def alpha_carbon_atom_ids(mol, atom_ids: List[int]) -> List[int]:
    atom_id_set = set(atom_ids)
    out = []
    for idx in atom_ids:
        a = mol.GetAtomWithIdx(idx)
        if a.GetSymbol() != "C":
            continue
        has_n = False
        has_carbonyl_c = False
        for n in a.GetNeighbors():
            if n.GetIdx() not in atom_id_set:
                continue
            if n.GetSymbol() == "N":
                has_n = True
            if _is_carbonyl_carbon(n):
                has_carbonyl_c = True
        if has_n and has_carbonyl_c:
            out.append(idx)
    out.sort()
    return out


def chunk_project(confs: List[List[List[float]]], n_tokens: int) -> List[List[List[float]]]:
    if not confs or n_tokens <= 0:
        return []
    n_atoms = len(confs[0])
    n_tokens = min(n_tokens, n_atoms)
    bounds = [int(round(i * n_atoms / n_tokens)) for i in range(n_tokens + 1)]
    for i in range(1, len(bounds)):
        if bounds[i] <= bounds[i - 1]:
            bounds[i] = bounds[i - 1] + 1
    bounds[-1] = n_atoms

    proj: List[List[List[float]]] = []
    for conf in confs:
        row = []
        for i in range(n_tokens):
            s, e = bounds[i], bounds[i + 1]
            pts = conf[s:e] or [conf[min(s, n_atoms - 1)]]
            x = sum(p[0] for p in pts) / len(pts)
            y = sum(p[1] for p in pts) / len(pts)
            z = sum(p[2] for p in pts) / len(pts)
            row.append([x, y, z])
        proj.append(row)
    return proj


def main() -> None:
    args = parse_args()

    try:
        import rdkit  # noqa: F401
    except Exception as e:
        raise RuntimeError("RDKit is required to read CREMP pickle archives. Install RDKit and retry.") from e

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = read_summary(args.summary_csv)
    summary_norm = {".".join(k.split(".")): v for k, v in summary.items()}

    rng = random.Random(args.seed)
    split_map: Dict[str, bool] = {}

    fp = {
        "pretrain_train": (outdir / "pretrain_train.jsonl").open("w", encoding="utf-8"),
        "pretrain_val": (outdir / "pretrain_val.jsonl").open("w", encoding="utf-8"),
        "inverse_train": (outdir / "inverse_train.jsonl").open("w", encoding="utf-8"),
        "inverse_val": (outdir / "inverse_val.jsonl").open("w", encoding="utf-8"),
        "finetune_train": (outdir / "finetune_train.jsonl").open("w", encoding="utf-8"),
        "finetune_val": (outdir / "finetune_val.jsonl").open("w", encoding="utf-8"),
    }

    seen = 0
    skipped = 0
    a_carbon_fallback = 0

    with tarfile.open(args.pickle_archive, "r:gz") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".pickle"):
                continue
            stem = Path(m.name).stem
            seq = canonicalize_sequence(stem)
            if not seq:
                skipped += 1
                continue

            data = pickle.load(tf.extractfile(m))
            mol = data.get("rd_mol")
            if mol is None:
                skipped += 1
                continue

            confs, atom_ids, atom_symbols = extract_coords_from_mol(mol, args.heavy_only)
            if not confs:
                skipped += 1
                continue

            atom_pos = {atom_id: i for i, atom_id in enumerate(atom_ids)}
            ca_ids = alpha_carbon_atom_ids(mol, atom_ids)
            ca_pos = [atom_pos[i] for i in ca_ids if i in atom_pos]

            weights = data.get("conformerweights")
            if not isinstance(weights, list) or len(weights) < len(confs):
                meta = data.get("conformers", [])
                if isinstance(meta, list) and len(meta) >= len(confs):
                    weights = [float(meta[i].get("boltzmannweight", 0.0)) for i in range(len(confs))]
                else:
                    weights = [1.0] * len(confs)

            order = sorted(range(len(confs)), key=lambda i: float(weights[i]), reverse=True)
            keep = order[: min(args.max_conformers, len(order))]
            confs = [confs[i] for i in keep]
            cw = normalize_weights([weights[i] for i in keep])

            node_tokens: Optional[List[str]] = None
            if args.representation == "all_atom":
                coords = confs
                node_tokens = atom_symbols
            elif args.representation == "a_carbon":
                if len(ca_pos) == len(seq) and len(ca_pos) > 0:
                    coords = [[[conf[j][0], conf[j][1], conf[j][2]] for j in ca_pos] for conf in confs]
                    node_tokens = ["C"] * len(ca_pos)
                else:
                    # Fall back to stable residue proxy if alpha-carbon detection is ambiguous.
                    coords = chunk_project(confs, len(seq))
                    a_carbon_fallback += 1
            else:
                coords = chunk_project(confs, len(seq))
            if not coords:
                skipped += 1
                continue

            sample = {
                "sequence": seq,
                "coords": coords,
                "conformer_weights": cw,
                "source_stem": stem,
            }
            if node_tokens is not None:
                sample["node_tokens"] = node_tokens
            if args.representation == "all_atom":
                sample["ca_indices"] = ca_pos

            row = summary.get(stem)
            if row is None:
                row = summary_norm.get(".".join(stem.split(".")))
            if row is not None:
                smi = row.get("smiles")
                if smi:
                    sample["summary_smiles"] = smi
                sid = row.get("CycPeptMPDB_ID")
                if sid:
                    sample["summary_id"] = sid
                perm = row.get("permeability")
                if perm not in (None, ""):
                    try:
                        sample["summary_permeability"] = float(perm)
                    except ValueError:
                        pass
            target = get_target(row, args.target_column)
            if target is not None:
                sample["target"] = target

            key = cyclic_key(seq) if args.split_by_cycle else stem
            if key not in split_map:
                split_map[key] = rng.random() < args.val_frac
            is_val = split_map[key]

            line = json.dumps(sample) + "\n"
            if is_val:
                fp["pretrain_val"].write(line)
                fp["inverse_val"].write(line)
            else:
                fp["pretrain_train"].write(line)
                fp["inverse_train"].write(line)

            if "target" in sample:
                if is_val:
                    fp["finetune_val"].write(line)
                else:
                    fp["finetune_train"].write(line)

            seen += 1
            if args.progress_every > 0 and seen % args.progress_every == 0:
                print({"seen": seen, "skipped": skipped}, flush=True)
            if args.max_samples > 0 and seen >= args.max_samples:
                break

    for h in fp.values():
        h.close()

    stats = {
        "rows_total": seen,
        "skipped": skipped,
        "representation": args.representation,
        "a_carbon_fallback": a_carbon_fallback,
        "max_conformers": args.max_conformers,
        "target_column": args.target_column,
        "pickle_archive": args.pickle_archive,
    }
    with (outdir / "prep_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(stats)


if __name__ == "__main__":
    main()
