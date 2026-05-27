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

## Scripts

### `scripts/infer.py` — score a dataset

The main inference script. Accepts a `.pkl` file or `.npy`/`.npz` + metadata CSV.

```bash
# from a pkl file (quickstart)
python scripts/infer.py \
    --input      /path/to/embeddings.pkl \
    --checkpoint outputs/runs/run_03/best_model.pt \
    --output     outputs/scores.csv

# from npy/npz + metadata CSV
python scripts/infer.py \
    --input      data/embeddings.npz \
    --metadata   data/metadata.csv \
    --checkpoint outputs/runs/run_03/best_model.pt \
    --output     outputs/scores.csv
```

**pkl format** — must be a dict with:
```python
{
    "embeddings": np.ndarray,  # shape (N, 768), float32
    "paths":      list[str],   # N file paths (optional)
}
```

**Important:** embeddings must be **raw / unscaled** DINOv2-base CLS token outputs (768-dim). Do not normalise or scale before passing in. Trained on `facebook/dinov2-base` / `facebook/dinov3-vitb16-pretrain-lvd1689m`.

Output CSV columns: `filename`, `path`, `realism_score`

---

### `scripts/train.py` — train a new model

```bash
python scripts/train.py \
    --config   configs/default.yaml \
    --run-name run_01 \
    --override training.learning_rate=3e-4 \
    --override model.dropout=0.2
```

Saves the best checkpoint to `outputs/runs/run_01/best_model.pt` and per-epoch metrics to `outputs/runs/run_01/metrics.csv`. Early stopping on `val_auc`.

---

### `scripts/evaluate.py` — evaluate on labelled test set

Runs the full test-set analysis and generates 9 diagnostic plots + summary CSVs.

```bash
python scripts/evaluate.py \
    --config     configs/default.yaml \
    --checkpoint outputs/runs/run_03/best_model.pt \
    --output-dir outputs/test_analysis/run_03
```

Outputs:

| File | Description |
|---|---|
| `01_roc_pr.png` | ROC and Precision-Recall curves |
| `02_score_dist_global.png` | Score distribution (real vs synthetic) |
| `03_per_dataset_violin.png` | Per-dataset score violin plot |
| `04_confusion_matrix.png` | Confusion matrix (row-normalised %) |
| `05_per_dataset_accuracy.png` | Per-dataset accuracy bar chart |
| `06_dataset_composition.png` | Test set composition pie charts |
| `07_score_cdf.png` | Cumulative score distribution |
| `08_summary_metrics.png` | Global metrics table |
| `09_dataset_breakdown.png` | Per-dataset breakdown table |
| `dataset_breakdown.csv` | Per-dataset metrics CSV |
| `global_metrics.csv` | Global metrics CSV |

---

### `scripts/compare_datasets.py` — compare scores across datasets

Compares realism score distributions across multiple scored CSVs.

```bash
python scripts/compare_datasets.py \
    --config     configs/default.yaml \
    --scores-dir outputs/scores \
    --output-dir outputs/comparison/run_03 \
    --top-k      20
```

Outputs: `dataset_ranking.csv`, violin plot, bar chart, per-dataset histograms.

---

## Architecture

The classifier is a **RealismMLP** — a feedforward neural network that takes a 768-dim DINO embedding and outputs a realism probability in [0, 1].

```
Input (768-dim DINO embedding)
        │
        ▼
  Linear(768 → 512)
  BatchNorm1d(512)
  ReLU
  Dropout(0.3)
        │
        ▼
  Linear(512 → 256)
  BatchNorm1d(256)
  ReLU
  Dropout(0.3)
        │
        ▼
  Linear(256 → 128)
  BatchNorm1d(128)
  ReLU
  Dropout(0.3)
        │
        ▼
  Linear(128 → 1)
        │
        ▼
  raw logit  ──[training]──►  BCEWithLogitsLoss
                              (sigmoid applied internally)
        │
   [inference]
        ▼
  sigmoid → realism score ∈ [0, 1]
```

**Key points:**
- `forward()` returns a raw logit — `BCEWithLogitsLoss` is used during training, which applies sigmoid internally for numerical stability
- `predict_proba()` applies sigmoid explicitly to produce the final [0, 1] realism score
- BatchNorm and Dropout(0.3) are applied after every hidden layer to regularise training
- The checkpoint is self-describing — it stores the model architecture, so no config file is needed to load it at inference time

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
│   ├── model.py                  # RealismMLP + checkpoint utilities
│   ├── train.py                  # training loop + early stopping
│   ├── evaluate.py               # metrics, ROC, per-dataset reports
│   ├── inference.py              # score new embedding files
│   └── compare.py                # cross-dataset comparison + plots
├── scripts/
│   ├── infer.py                  # score a dataset (pkl or npz/npy)
│   ├── train.py                  # train a new model
│   ├── evaluate.py               # full test-set evaluation + 9 plots
│   ├── compare_datasets.py       # compare scores across datasets
│   └── test_analysis.py          # underlying analysis implementation
└── tests/
```

---

## Key design choices

- **Input is always DINO embeddings** — no raw images used anywhere in the pipeline.
- **Logits in `forward()`; sigmoid at inference** — `BCEWithLogitsLoss` during training for numerical stability; `predict_proba()` applies sigmoid to return [0, 1] scores.
- **Self-describing checkpoints** — every `.pt` file stores the model architecture, so it can be reconstructed with no config file needed.
- **Early stopping on `val_auc`** — optimises for ranking quality rather than calibrated loss, which matters most for score-based dataset comparison.
- **`evaluate.py` and `compare_datasets.py` are separate** — `evaluate.py` requires ground-truth labels; `compare_datasets.py` works on any scored CSV.

---

## Running tests

```bash
pytest tests/ -v
```
