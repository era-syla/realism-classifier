# realism-classifier

A binary classifier for evaluating image realism from precomputed DINO embedding vectors.

**Class 1** = realistic images  
**Class 0** = synthetic / less-realistic images

The model takes fixed-length DINO embedding vectors as input — no raw images are used during training or inference. After training, the model produces a *realism score* (probability in [0, 1]) for each sample, which can be used to compare how realistic different datasets appear.

---

## Installation

```bash
pip install -e ".[dev]"
# or just
pip install -r requirements.txt
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.1.

---

## Data format

### Embeddings

Place your embedding files under `data/`. Two formats are supported:

| Format | Description |
|--------|-------------|
| `.npz` (preferred) | Keys `train`, `val`, `test` — each `(N, D)` float32 array |
| `.npy` | Single `(N, D)` float32 array; split determined by CSV column |

### Metadata CSV (training / evaluation)

| Column | Required | Notes |
|--------|----------|-------|
| `sample_id` | yes | globally unique identifier |
| `label` | yes | 1 = realistic, 0 = synthetic |
| `split` | yes | `train` / `val` / `test` |
| `dataset` | yes | source dataset name |
| `object_id` | no | optional |
| `view` | no | optional |

### Metadata CSV (inference — no labels needed)

Same structure but `label` and `split` columns are omitted.

### Inference output CSV

`sample_id`, `realism_score`, plus any extra columns from the input metadata.

---

## Configuration

All hyperparameters and paths are set in `configs/default.yaml`. You can override any value at the CLI with `--override key=value` (dot-notation).

```yaml
data:
  embeddings_path: "data/embeddings.npz"
  metadata_csv:    "data/metadata.csv"
  embedding_dim:   768        # must match your DINO model

model:
  hidden_dims:    [512, 256, 128]
  dropout:        0.3
  activation:     "relu"
  use_batch_norm: true

training:
  batch_size:     256
  max_epochs:     100
  learning_rate:  1.0e-3
  ...
```

---

## Typical workflow

### 1. Train

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --run-name run_01 \
    --override training.learning_rate=3e-4 \
    --override model.dropout=0.2
```

The best checkpoint is saved to `outputs/runs/run_01/best_model.pt`.  
Per-epoch metrics are logged to `outputs/runs/run_01/metrics.csv`.

### 2. Evaluate on the test split

```bash
python scripts/evaluate.py \
    --config configs/default.yaml \
    --checkpoint outputs/runs/run_01/best_model.pt \
    --split test \
    --output-dir outputs/evaluation/run_01
```

Outputs: `evaluation_report.json`, `per_dataset_metrics.csv`, ROC curve plot, score distribution plots.

### 3. Score new datasets

```bash
python scripts/infer.py \
    --config configs/default.yaml \
    --checkpoint outputs/runs/run_01/best_model.pt \
    --embeddings data/dataset_A_embeddings.npy \
    --metadata   data/dataset_A_meta.csv \
    --output     outputs/scores/dataset_A.csv
```

Repeat for each dataset you want to score.

### 4. Compare datasets

```bash
python scripts/compare_datasets.py \
    --config     configs/default.yaml \
    --scores-dir outputs/scores \
    --output-dir outputs/comparison/run_01 \
    --top-k 20
```

Outputs: `dataset_ranking.csv`, violin plot, bar chart, per-dataset histograms.

---

## Project structure

```
realism-classifier/
├── configs/default.yaml          # hyperparameters and paths
├── data/                         # place your embeddings + CSVs here
├── outputs/                      # checkpoints, logs, results
├── src/realism_classifier/
│   ├── config.py                 # typed config dataclasses + YAML loading
│   ├── dataset.py                # EmbeddingDataset + DataLoader factories
│   ├── model.py                  # RealismMLP + checkpoint save/load
│   ├── train.py                  # training loop, early stopping
│   ├── evaluate.py               # metrics, ROC, per-dataset reports
│   ├── inference.py              # score new embedding files
│   └── compare.py                # cross-dataset comparison + plots
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   └── compare_datasets.py
└── tests/
```

---

## Key design choices

- **Input is always DINO embeddings** — no raw images used anywhere in the pipeline.
- **Logits only in `forward()`** — `BCEWithLogitsLoss` during training; `predict_proba()` applies sigmoid at inference time.
- **Self-describing checkpoints** — every `.pt` file stores the model architecture so it can be reconstructed without a config file.
- **Early stopping on `val_auc`** — optimises for ranking quality rather than calibrated loss, which matters most for score-based dataset comparison.
- **`evaluate.py` and `compare.py` are separate** — `evaluate.py` requires ground-truth labels; `compare.py` works on any scored CSV.

---

## Running tests

```bash
pytest tests/ -v
```
