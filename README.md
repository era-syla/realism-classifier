# realism-classifier

A binary classifier for evaluating image realism from precomputed DINO embedding vectors.

**Class 1** = realistic images  
**Class 0** = synthetic / less-realistic images

The model takes fixed-length DINO embedding vectors as input — no raw images are used during training or inference. After training, the model produces a *realism score* (probability in [0, 1]) for each sample, which can be used to compare how realistic different datasets appear.

A pre-trained checkpoint (`run_03`, val AUC = 0.9767) is included at `outputs/runs/run_03/best_model.pt`.

---

## Installation

```bash
pip install -e ".[dev]"
# or just
pip install -r requirements.txt
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.1.

---

## Quickstart — score a pkl file

The fastest way to run the classifier on a new dataset:

```bash
python scripts/run_inference.py \
    --pkl       /path/to/embeddings.pkl \
    --checkpoint outputs/runs/run_03/best_model.pt \
    --output    outputs/my_dataset_scores.csv
```

**Input pkl format** — must contain a dict with:
```python
{
    "embeddings": np.ndarray,  # shape (N, 768), float32, raw DINOv2-base CLS tokens
    "paths":      list[str],   # N file paths (optional)
}
```

**Output CSV** columns: `filename`, `path`, `realism_score`

**Important:** embeddings must be **raw / unscaled** DINOv2-base (ViT-B/14 or ViT-B/16) CLS token outputs — do not normalise or scale before passing in. The model was trained on `facebook/dinov2-base` / `facebook/dinov3-vitb16-pretrain-lvd1689m` embeddings (768-dim).

---

## All scripts

### `scripts/run_inference.py` — score a pkl file (recommended for inference)

```bash
python scripts/run_inference.py \
    --pkl        /path/to/embeddings.pkl \
    --checkpoint outputs/runs/run_03/best_model.pt \
    --output     outputs/scores.csv \
    --batch-size 512 \
    --device     auto   # auto | cpu | cuda | mps
```

---

### `scripts/train.py` — train a new model

```bash
python scripts/train.py \
    --config   configs/default.yaml \
    --run-name run_01 \
    --override training.learning_rate=3e-4 \
    --override model.dropout=0.2
```

Best checkpoint → `outputs/runs/run_01/best_model.pt`  
Per-epoch metrics → `outputs/runs/run_01/metrics.csv`

---

### `scripts/evaluate.py` — evaluate on labelled test split

```bash
python scripts/evaluate.py \
    --config     configs/default.yaml \
    --checkpoint outputs/runs/run_01/best_model.pt \
    --split      test \
    --output-dir outputs/evaluation/run_01
```

Outputs: `evaluation_report.json`, `per_dataset_metrics.csv`, ROC curve, score distribution plots.

---

### `scripts/infer.py` — score new datasets (npz/npy + metadata CSV)

```bash
python scripts/infer.py \
    --config     configs/default.yaml \
    --checkpoint outputs/runs/run_01/best_model.pt \
    --embeddings data/dataset_A_embeddings.npy \
    --metadata   data/dataset_A_meta.csv \
    --output     outputs/scores/dataset_A.csv
```

---

### `scripts/compare_datasets.py` — compare realism across datasets

```bash
python scripts/compare_datasets.py \
    --config     configs/default.yaml \
    --scores-dir outputs/scores \
    --output-dir outputs/comparison/run_01 \
    --top-k 20
```

Outputs: `dataset_ranking.csv`, violin plot, bar chart, per-dataset histograms.

---

### `scripts/test_analysis.py` — full test-set analysis (9 plots)

```bash
python scripts/test_analysis.py \
    --config     configs/default.yaml \
    --checkpoint outputs/runs/run_03/best_model.pt \
    --output-dir outputs/test_analysis/run_03
```

Generates: ROC/PR curves, score distributions, per-dataset violin plot, confusion matrix, accuracy bars, composition pies, CDF, summary metrics table, dataset breakdown table.

---

## Pre-trained checkpoint (run_03)

| Property | Value |
|---|---|
| Val AUC | 0.9767 |
| Architecture | MLP 768→512→256→128→1 |
| BatchNorm | yes |
| Dropout | 0.3 |
| Activation | ReLU |
| Epochs | 100 |
| Embedding input | DINOv2-base, 768-dim, raw (unscaled) |

Training data: ABC sketch+extrude, ABC all ops, DeepCAD, Fusion360 (real); CADRecode 100k, SynCAD, SynCAD-17 (synthetic). Balanced 1:1 (~154k each class).

---

## Data format (training)

### Embeddings

Place your embedding files under `data/`. Two formats are supported:

| Format | Description |
|--------|-------------|
| `.npz` (preferred) | Keys `train`, `val`, `test` — each `(N, D)` float32 array |
| `.npy` | Single `(N, D)` float32 array; split determined by CSV column |

### Metadata CSV

| Column | Required | Notes |
|--------|----------|-------|
| `sample_id` | yes | globally unique identifier |
| `label` | yes | 1 = realistic, 0 = synthetic |
| `split` | yes | `train` / `val` / `test` |
| `dataset` | yes | source dataset name |
| `object_id` | no | optional |
| `view` | no | optional |

---

## Configuration

All hyperparameters and paths are in `configs/default.yaml`. Override at CLI with `--override key=value` (dot-notation).

---

## Project structure

```
realism-classifier/
├── configs/default.yaml          # hyperparameters and paths
├── data/                         # place your embeddings + CSVs here
├── outputs/
│   └── runs/run_03/
│       ├── best_model.pt         # pre-trained checkpoint
│       ├── config.yaml           # run config
│       └── metrics.csv           # per-epoch training metrics
├── src/realism_classifier/
│   ├── config.py                 # typed config dataclasses + YAML loading
│   ├── dataset.py                # EmbeddingDataset + DataLoader factories
│   ├── model.py                  # RealismMLP + checkpoint save/load
│   ├── train.py                  # training loop, early stopping
│   ├── evaluate.py               # metrics, ROC, per-dataset reports
│   ├── inference.py              # score new embedding files
│   └── compare.py                # cross-dataset comparison + plots
├── scripts/
│   ├── run_inference.py          # quickstart: score a pkl file
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   ├── compare_datasets.py
│   └── test_analysis.py
└── tests/
```

---

## Key design choices

- **Input is always DINO embeddings** — no raw images used anywhere in the pipeline.
- **Logits in `forward()`; sigmoid at inference** — `BCEWithLogitsLoss` during training for numerical stability; `predict_proba()` applies sigmoid to return [0, 1] scores.
- **Self-describing checkpoints** — every `.pt` file stores the model architecture, so it can be reconstructed with no config file needed.
- **Early stopping on `val_auc`** — optimises for ranking quality rather than calibrated loss, which matters most for score-based dataset comparison.
- **`evaluate.py` and `compare.py` are separate** — `evaluate.py` requires ground-truth labels; `compare.py` works on any scored CSV.

---

## Running tests

```bash
pytest tests/ -v
```
