"""Configuration dataclasses, YAML loading, and CLI override helpers."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


# ---------------------------------------------------------------------------
# Nested config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    embeddings_path: str = "data/embeddings.npz"
    metadata_csv: str = "data/metadata.csv"
    embedding_dim: int = 768
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ModelConfig:
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    dropout: float = 0.3
    activation: str = "relu"
    use_batch_norm: bool = True


@dataclass
class SchedulerConfig:
    type: str = "cosine"
    warmup_epochs: int = 3
    T_max: int = 50
    step_size: int = 20
    gamma: float = 0.5
    patience: int = 5


@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 10
    metric: str = "val_auc"
    min_delta: float = 1e-4
    mode: str = "max"


@dataclass
class TrainingConfig:
    batch_size: int = 256
    max_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adam"
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    pos_weight: Optional[float] = None


@dataclass
class CheckpointingConfig:
    output_dir: str = "outputs/runs"
    save_best_only: bool = True
    checkpoint_metric: str = "val_auc"


@dataclass
class EvaluationConfig:
    threshold: float = 0.5
    per_dataset_report: bool = True


@dataclass
class InferenceConfig:
    checkpoint_path: str = ""
    embeddings_path: Optional[str] = None
    metadata_csv: Optional[str] = None
    output_csv: str = "outputs/inference_scores.csv"
    batch_size: int = 512


@dataclass
class ComparisonConfig:
    scores_dir: str = "outputs/scores"
    output_dir: str = "outputs/comparison"
    plot_format: str = "png"
    top_k: Optional[int] = None


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    comparison: ComparisonConfig = field(default_factory=ComparisonConfig)
    seed: int = 42
    device: str = "auto"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dict_to_config(d: dict) -> Config:
    """Recursively populate Config dataclasses from a plain dict."""
    def _build(cls, data):
        if data is None:
            return cls()
        kwargs = {}
        for f in cls.__dataclass_fields__:
            if f not in data:
                continue
            ftype = cls.__dataclass_fields__[f].type
            val = data[f]
            # Resolve string annotation to actual class if needed
            if isinstance(ftype, str):
                ftype = _NESTED_TYPES.get(ftype)
            if ftype is not None and hasattr(ftype, "__dataclass_fields__") and isinstance(val, dict):
                kwargs[f] = _build(ftype, val)
            else:
                kwargs[f] = val
        return cls(**kwargs)

    return _build(Config, d)


# Map of string type names to their dataclass classes (needed when annotations
# are stored as strings due to from __future__ import annotations).
_NESTED_TYPES = {
    "DataConfig": DataConfig,
    "ModelConfig": ModelConfig,
    "TrainingConfig": TrainingConfig,
    "SchedulerConfig": SchedulerConfig,
    "EarlyStoppingConfig": EarlyStoppingConfig,
    "CheckpointingConfig": CheckpointingConfig,
    "EvaluationConfig": EvaluationConfig,
    "InferenceConfig": InferenceConfig,
    "ComparisonConfig": ComparisonConfig,
}


def load_config(yaml_path: str) -> Config:
    """Load a Config from a YAML file."""
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return _dict_to_config(raw)


def merge_cli_overrides(config: Config, overrides: dict) -> Config:
    """
    Apply flat dot-notation overrides onto an existing Config.

    Example:
        overrides = {"training.learning_rate": 5e-4, "model.dropout": 0.1}
    """
    config = copy.deepcopy(config)
    for key, value in overrides.items():
        parts = key.split(".")
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        leaf = parts[-1]
        # Cast to the existing field type if possible
        existing = getattr(obj, leaf, None)
        if existing is not None and not isinstance(existing, type(value)):
            try:
                value = type(existing)(value)
            except (TypeError, ValueError):
                pass
        setattr(obj, leaf, value)
    return config


def resolve_device(device_str: str) -> str:
    """Resolve 'auto' to the best available device string for torch."""
    if device_str != "auto":
        return device_str
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def save_config(config: Config, path: str) -> None:
    """Serialize config to YAML for run reproducibility."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(asdict(config), f, default_flow_style=False, sort_keys=False)


def parse_overrides(override_list: list[str]) -> dict:
    """
    Convert a list of 'key=value' strings (from argparse) to a dict.

    Example:
        ["training.learning_rate=5e-4", "model.dropout=0.1"]
        -> {"training.learning_rate": 5e-4, "model.dropout": 0.1}

    Values are parsed with yaml.safe_load so floats, ints, bools, and null
    are all handled correctly.
    """
    result = {}
    for item in override_list or []:
        if "=" not in item:
            raise ValueError(f"Override must be in 'key=value' format, got: {item!r}")
        key, _, raw_val = item.partition("=")
        result[key.strip()] = yaml.safe_load(raw_val.strip())
    return result
