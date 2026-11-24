# EnsembleEGNN

### THIS MODEL IS CURRENTLY IN DEVELOPMENT, UPDATE COMING SHORTLY; I WOULD ADVISE NOT USING THIS VERSION.
-Aaron Feller

---

EnsembleEGNN is the public release for the code and manuscript assets behind the cyclic-peptide conformer-ensemble study accepted to ICML 2026 GFM Workshop. The release surface is intentionally narrow: it contains the all-atom data-preparation path used for the reported experiments, the training and evaluation scripts needed to reproduce the manuscript workflow, and the final manuscript source with figures.

Submisson and reviews can be found here: https://openreview.net/forum?id=PKFtafpZFI

## What This Repo Contains

1. Instructions for CREMP and CycPeptMPDB-CREMP datasets
2. All-atom CREMP/CycPeptMPDB JSONL preparation from the unified pickle archive
3. Self-supervised pretraining for the geometric ensemble model
4. Boundary-filtered random 5-fold PAMPA evaluation splits
5. Geometric, BERT-only, and hybrid finetuning plus held-out prediction
6. Aggregation of fold-level metrics and the final manuscript source

## Repository Layout

1. `src/`: core PepHGNN model code, data loaders, config, and utilities
2. `scripts/`: release scripts for data preparation, splitting, training, prediction, and aggregation
3. `manuscript/`: paper source, bibliography, and final figure assets

## Environment

This repo no longer ships packaging metadata, so set up a clean environment and
install the runtime dependencies directly. At minimum, the retained scripts use
PyTorch, NumPy, RDKit, and Hugging Face Transformers.

Example:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy torch rdkit transformers
```

Run commands from the repository root. The retained scripts resolve the local
`src/` tree directly.

## Release Workflow

### 1) Prepare all-atom JSONL datasets

```bash
python3 scripts/prepare_cremp_pickle_dataset.py \
  --pickle-archive data/CREMP/pickle.tar.gz \
  --summary-csv data/CREMP/summary.csv \
  --representation all_atom \
  --outdir data/cremp_prepared_all_atom
```

This produces the pretraining and finetuning JSONL files consumed by the rest
of the workflow.

### 2) Build the manuscript PAMPA split

The reported protocol removes exact boundary targets at `-10.0`, `-8.0`, and
`-4.0`, then rebuilds the random 5-fold evaluation split.

```bash
python3 scripts/rebuild_pampa_no10_random_split.py \
  --input-jsonl data/cremp_prepared_all_atom/finetune_train.jsonl \
  --input-jsonl data/cremp_prepared_all_atom/finetune_val.jsonl \
  --clusters-csv data/CycPeptMPDB/pampa_clusters.csv \
  --summary-csv data/CycPeptMPDB/summary.csv \
  --tmp-cv-outdir runs/tmp_pampa_cluster_match \
  --outdir data/cycpept_pampa_random5_no10_no8_no4 \
  --drop-target -10.0 \
  --drop-target -8.0 \
  --drop-target -4.0
```

The helper scripts `build_cycpept_pampa_splits.py` and
`build_random_kfold_splits.py` are retained because this step uses them
internally.

### 3) Pretrain the geometric backbone

```bash
python3 scripts/pretrain.py \
  --train-data data/cremp_prepared_all_atom/pretrain_train.jsonl \
  --val-data data/cremp_prepared_all_atom/pretrain_val.jsonl \
  --outdir runs/pretrain_cremp_all_atom
```

### 4) Finetune downstream regressors

Geometric model:

```bash
python3 scripts/finetune_regression.py \
  --train-data <train.jsonl> \
  --val-data <val.jsonl> \
  --outdir runs/finetune_geom \
  --init-checkpoint runs/pretrain_cremp_all_atom/best_pretrain.pt \
  --target-scale-json <target_scaling.json>
```

BERT-only model:

```bash
python3 scripts/finetune_bert_regression.py \
  --train-data <train.jsonl> \
  --val-data <val.jsonl> \
  --outdir runs/finetune_bert \
  --target-scale-json <target_scaling.json>
```

Hybrid model:

```bash
python3 scripts/finetune_hybrid_regression.py \
  --train-data <train.jsonl> \
  --val-data <val.jsonl> \
  --outdir runs/finetune_hybrid \
  --init-checkpoint runs/pretrain_cremp_all_atom/best_pretrain.pt \
  --target-scale-json <target_scaling.json>
```

### 5) Predict held-out folds and aggregate metrics

Use the prediction entry point matching the model family:

1. `scripts/predict_regression.py`
2. `scripts/predict_bert_regression.py`
3. `scripts/predict_hybrid_regression.py`

Then aggregate held-out predictions with:

1. `scripts/aggregate_cycpept_pampa_cv.py`

## Manuscript Assets

The final paper source is in `manuscript/main.tex`. The checked-in figures under
`manuscript/figures/` are the final release artifacts referenced by the paper.


## Reproducibility Notes

1. The reported experiments use all-atom coordinates and atom-level node tokens.
2. The PAMPA evaluation set removes exact targets at `-10.0`, `-8.0`, and `-4.0`.
3. Each random-5 run uses 3 folds for training, 1 for validation, and 1 held-out test fold.

## Citation

If you use this repository, please cite the accompanying manuscript.

```
@inproceedings{
  feller2026graph,
  title={Graph Learning on Ensembles of Cyclic Peptides: An Investigation of Molecular Ensemble Modeling},
  author={Feller, Aaron L and Deibler, Kristine and Secor, Maxim},
  booktitle={Workshop on Graph Foundation Models: A New Era for Graph Machine Learning},
  year={2026},
  url={https://openreview.net/forum?id=PKFtafpZFI}
}
```
