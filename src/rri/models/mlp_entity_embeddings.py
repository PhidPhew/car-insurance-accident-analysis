"""MLP + entity embeddings (CLAUDE.md §10 L2, Phase 6 Step 5) -- the severity classifier's
neural counterpart to the LightGBM classifier (`rri.models.lightgbm_baseline`).

Same record-level severity-classification frame, same 4-class target, same time-based
split -- the only thing that differs is the model. Justified by CLAUDE.md §10's own framing:
high cardinality (120 causes, 152 crash types, thousands of road codes) is exactly the
regime where a learned embedding can beat one-hot encoding, since one-hot would blow up the
input dimension and give every rare category its own untied weight.

**Embedding columns: province, cause, crash type, road geometry, road code** -- CLAUDE.md
§10's own list. `province_th` is used instead of the LightGBM classifier's coarser `region`
column (10 categories) -- CLAUDE.md explicitly names "province" as an embedding target, and
province subsumes region's information at finer granularity, so keeping both would just be
redundant. Every other feature (numeric, boolean, and the remaining low-cardinality
categoricals) matches `rri.models.lightgbm_baseline.CLASSIFIER_NUMERIC_COLS`/
`CLASSIFIER_BOOLEAN_COLS` exactly, so the two classifiers see the same non-embedded feature
space -- CLAUDE.md Step 5's "fair comparison, not to force a win."

**Regularization-sensitive by design** (CLAUDE.md Step 5): 2 hidden layers, dropout, and
early stopping on 2024 validation loss, restoring the best-validation-loss weights rather
than whatever the last epoch happened to land on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")  # headless -- this module only ever writes PNGs, never displays
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from sklearn.manifold import TSNE
from torch import nn

from config.simulation_params import RANDOM_SEED
from rri.models.lightgbm_baseline import CLASSIFIER_BOOLEAN_COLS, CLASSIFIER_NUMERIC_COLS

#: sklearn's TSNE default perplexity (30) needs more samples than the 77 provinces this
#: module projects -- the rule of thumb is perplexity < n_samples / 3.
_TSNE_PERPLEXITY = 15.0

EMBEDDING_COLS = ["province_th", "cause_raw", "crash_type_raw", "road_geometry_raw", "road_code"]
ONEHOT_COLS = ["season", "vehicle_group", "weather_band", "agency", "covid_disruption_level"]
NUMERIC_COLS = CLASSIFIER_NUMERIC_COLS
BOOLEAN_COLS = CLASSIFIER_BOOLEAN_COLS

HIDDEN_SIZES = (64, 32)
DROPOUT = 0.3
MAX_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 15
BATCH_SIZE = 512
LEARNING_RATE = 1e-3

#: Index 0 is reserved for "unknown/unseen category" in every embedding table.
_UNK_INDEX = 0


def embedding_dim_for_cardinality(cardinality: int) -> int:
    """CLAUDE.md §10's "embedding dim 4-8", scaled gently with cardinality within that band."""
    return int(np.clip(round(cardinality**0.25 * 1.6), 4, 8))


@dataclass(frozen=True)
class CategoryEncoder:
    mapping: dict[str, int]
    vocab_size: int  # includes the reserved unknown index


def fit_category_encoder(train_col: pl.Series) -> CategoryEncoder:
    """Category -> index (1..N) from `train_col`'s distinct values; index 0 = unknown."""
    levels = sorted(train_col.cast(pl.Utf8).unique().drop_nulls().to_list())
    mapping = {level: i + 1 for i, level in enumerate(levels)}
    return CategoryEncoder(mapping=mapping, vocab_size=len(levels) + 1)


def encode_column(series: pl.Series, encoder: CategoryEncoder) -> np.ndarray:
    return np.array(
        [encoder.mapping.get(v, _UNK_INDEX) for v in series.cast(pl.Utf8).to_list()], dtype=np.int64
    )


@dataclass(frozen=True)
class OnehotEncoder:
    categories: list[str]


def fit_onehot_encoder(train_col: pl.Series) -> OnehotEncoder:
    return OnehotEncoder(categories=sorted(train_col.cast(pl.Utf8).unique().drop_nulls().to_list()))


def encode_onehot(series: pl.Series, encoder: OnehotEncoder) -> np.ndarray:
    """One-hot matrix, `(n, len(categories))`; an unseen/null value is the all-zero row."""
    values = series.cast(pl.Utf8).to_list()
    out = np.zeros((len(values), len(encoder.categories)), dtype=np.float32)
    index_by_category = {c: i for i, c in enumerate(encoder.categories)}
    for row, v in enumerate(values):
        idx = index_by_category.get(v)
        if idx is not None:
            out[row, idx] = 1.0
    return out


@dataclass(frozen=True)
class NumericScaler:
    means: dict[str, float]
    stds: dict[str, float]


def fit_numeric_scaler(train: pl.DataFrame, numeric_cols: list[str]) -> NumericScaler:
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for col in numeric_cols:
        mean = train[col].mean()
        mean_f = float(mean) if mean is not None else 0.0  # type: ignore[arg-type]
        std = train[col].std()
        std_f = float(std) if std is not None else 0.0  # type: ignore[arg-type]
        means[col] = mean_f
        stds[col] = std_f if std_f > 1e-8 else 1.0
    return NumericScaler(means=means, stds=stds)


def transform_numeric(
    df: pl.DataFrame, numeric_cols: list[str], scaler: NumericScaler
) -> np.ndarray:
    """Mean-fill (train mean) then standardize (train mean/std) -- fit on train only."""
    columns = []
    for col in numeric_cols:
        values = df[col].fill_null(scaler.means[col]).to_numpy().astype(np.float32)
        columns.append((values - scaler.means[col]) / scaler.stds[col])
    return np.stack(columns, axis=1) if columns else np.zeros((df.height, 0), dtype=np.float32)


def transform_boolean(df: pl.DataFrame, boolean_cols: list[str]) -> np.ndarray:
    columns = [df[col].fill_null(False).cast(pl.Float32).to_numpy() for col in boolean_cols]
    return np.stack(columns, axis=1) if columns else np.zeros((df.height, 0), dtype=np.float32)


class EntityEmbeddingMlp(nn.Module):
    def __init__(
        self,
        embedding_vocab_sizes: dict[str, int],
        embedding_dims: dict[str, int],
        dense_input_dim: int,
        n_classes: int,
        hidden_sizes: tuple[int, ...] = HIDDEN_SIZES,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.embedding_cols = list(embedding_vocab_sizes.keys())
        self.embeddings = nn.ModuleDict(
            {
                col: nn.Embedding(embedding_vocab_sizes[col], embedding_dims[col])
                for col in self.embedding_cols
            }
        )
        total_input = sum(embedding_dims[c] for c in self.embedding_cols) + dense_input_dim

        layers: list[nn.Module] = []
        prev = total_input
        for hidden in hidden_sizes:
            layers += [nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(dropout)]
            prev = hidden
        layers.append(nn.Linear(prev, n_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self, embedding_inputs: dict[str, torch.Tensor], dense: torch.Tensor
    ) -> torch.Tensor:
        embedded = [self.embeddings[col](embedding_inputs[col]) for col in self.embedding_cols]
        x = torch.cat([*embedded, dense], dim=1)
        output: torch.Tensor = self.mlp(x)
        return output

    def embedding_weights(self, col: str) -> np.ndarray:
        embedding_layer = cast(nn.Embedding, self.embeddings[col])
        return np.asarray(embedding_layer.weight.detach().cpu().numpy())


@dataclass(frozen=True)
class FittedMlp:
    model: EntityEmbeddingMlp
    category_encoders: dict[str, CategoryEncoder]
    onehot_encoders: dict[str, OnehotEncoder]
    numeric_scaler: NumericScaler
    class_names: list[str]
    train_losses: list[float] = field(default_factory=list)
    valid_losses: list[float] = field(default_factory=list)
    best_epoch: int = 0


def _build_tensors(
    df: pl.DataFrame,
    category_encoders: dict[str, CategoryEncoder],
    onehot_encoders: dict[str, OnehotEncoder],
    numeric_scaler: NumericScaler,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    embedding_inputs = {
        col: torch.from_numpy(encode_column(df[col], category_encoders[col]))
        for col in EMBEDDING_COLS
    }
    onehot_parts = [encode_onehot(df[col], onehot_encoders[col]) for col in ONEHOT_COLS]
    numeric_part = transform_numeric(df, NUMERIC_COLS, numeric_scaler)
    boolean_part = transform_boolean(df, BOOLEAN_COLS)
    dense = np.concatenate([*onehot_parts, numeric_part, boolean_part], axis=1)
    return embedding_inputs, torch.from_numpy(dense.astype(np.float32))


def _dense_input_dim(onehot_encoders: dict[str, OnehotEncoder]) -> int:
    return (
        sum(len(onehot_encoders[c].categories) for c in ONEHOT_COLS)
        + len(NUMERIC_COLS)
        + len(BOOLEAN_COLS)
    )


def fit_mlp_classifier(train: pl.DataFrame, valid: pl.DataFrame) -> FittedMlp:
    """Fit the entity-embedding MLP with early stopping on validation loss."""
    torch.manual_seed(RANDOM_SEED)
    generator = torch.Generator().manual_seed(RANDOM_SEED)

    class_names = sorted(train["severity_class"].unique().to_list())
    class_index = {name: i for i, name in enumerate(class_names)}

    category_encoders = {col: fit_category_encoder(train[col]) for col in EMBEDDING_COLS}
    onehot_encoders = {col: fit_onehot_encoder(train[col]) for col in ONEHOT_COLS}
    numeric_scaler = fit_numeric_scaler(train, NUMERIC_COLS)

    embedding_dims = {
        col: embedding_dim_for_cardinality(category_encoders[col].vocab_size)
        for col in EMBEDDING_COLS
    }
    vocab_sizes = {col: category_encoders[col].vocab_size for col in EMBEDDING_COLS}

    model = EntityEmbeddingMlp(
        embedding_vocab_sizes=vocab_sizes,
        embedding_dims=embedding_dims,
        dense_input_dim=_dense_input_dim(onehot_encoders),
        n_classes=len(class_names),
    )

    train_embed, train_dense = _build_tensors(
        train, category_encoders, onehot_encoders, numeric_scaler
    )
    valid_embed, valid_dense = _build_tensors(
        valid, category_encoders, onehot_encoders, numeric_scaler
    )
    train_y = torch.tensor(
        [class_index[c] for c in train["severity_class"].to_list()], dtype=torch.long
    )
    valid_y = torch.tensor(
        [class_index[c] for c in valid["severity_class"].to_list()], dtype=torch.long
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    n = train_dense.shape[0]
    best_valid_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    patience_left = EARLY_STOPPING_PATIENCE
    train_losses: list[float] = []
    valid_losses: list[float] = []

    for epoch in range(MAX_EPOCHS):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        epoch_loss = 0.0
        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start : start + BATCH_SIZE]
            batch_embed = {col: train_embed[col][idx] for col in EMBEDDING_COLS}
            batch_dense = train_dense[idx]
            batch_y = train_y[idx]

            optimizer.zero_grad()
            logits = model(batch_embed, batch_dense)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        train_losses.append(epoch_loss / n)

        model.eval()
        with torch.no_grad():
            valid_logits = model(valid_embed, valid_dense)
            valid_loss = loss_fn(valid_logits, valid_y).item()
        valid_losses.append(valid_loss)

        if valid_loss < best_valid_loss - 1e-4:
            best_valid_loss = valid_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_left = EARLY_STOPPING_PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return FittedMlp(
        model=model,
        category_encoders=category_encoders,
        onehot_encoders=onehot_encoders,
        numeric_scaler=numeric_scaler,
        class_names=class_names,
        train_losses=train_losses,
        valid_losses=valid_losses,
        best_epoch=best_epoch,
    )


def predict_mlp_classifier(fitted: FittedMlp, df: pl.DataFrame) -> np.ndarray:
    embed, dense = _build_tensors(
        df, fitted.category_encoders, fitted.onehot_encoders, fitted.numeric_scaler
    )
    fitted.model.eval()
    with torch.no_grad():
        logits = fitted.model(embed, dense)
        predicted_index = torch.argmax(logits, dim=1).cpu().numpy()
    return np.array([fitted.class_names[i] for i in predicted_index])


def province_embedding_table(fitted: FittedMlp) -> pl.DataFrame:
    """The learned `province_th` embedding vectors -- the 77 real provinces only.

    Excludes both the reserved OOV index 0 and the literal `"Unknown"` geography
    category (fact_accident rows with an unresolved province, CLAUDE.md N7) -- CLAUDE.md
    Step 5 asks for "the 77 province embeddings" specifically, and neither of those two
    rows is a real province to plot on a province risk-profile map.
    """
    encoder = fitted.category_encoders["province_th"]
    weights = fitted.model.embedding_weights("province_th")
    real_provinces = {p: i for p, i in encoder.mapping.items() if p != "Unknown"}
    provinces = sorted(real_provinces.keys())
    indices = [real_provinces[p] for p in provinces]
    dim_cols = {f"dim_{i}": weights[indices, i] for i in range(weights.shape[1])}
    return pl.DataFrame({"province_th": provinces, **dim_cols})


def project_province_embeddings_2d(embedding_table: pl.DataFrame) -> pl.DataFrame:
    """t-SNE projection of `province_embedding_table`'s output to 2-D, seeded (N9)."""
    dim_cols = [c for c in embedding_table.columns if c.startswith("dim_")]
    x = embedding_table.select(dim_cols).to_numpy()
    n_samples = x.shape[0]
    perplexity = min(_TSNE_PERPLEXITY, max(2.0, (n_samples - 1) / 3))
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=perplexity, init="pca")
    coords = tsne.fit_transform(x)
    return embedding_table.select("province_th").with_columns(
        pl.Series("tsne_x", coords[:, 0]), pl.Series("tsne_y", coords[:, 1])
    )


def save_province_embedding_plot(
    projected: pl.DataFrame, dim_geography: pl.DataFrame, output_path: Path
) -> Path:
    """Scatter plot of the 2-D projection, colored by region and labeled by province.

    Labels use `province_en` (dim_geography), not `province_th` -- this environment has
    no verified Thai-font fallback for matplotlib, and an unrendered/tofu label would be
    worse than an English one on what is an internal analysis artifact, not the Thai-UI
    dashboard (CLAUDE.md N10 scopes the Thai-UI requirement to the dashboard/report/notebook
    markdown, not every intermediate chart).
    """
    labeled = projected.join(
        dim_geography.select("province_th", "province_en", "region"), on="province_th", how="left"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 9))
    regions = sorted(labeled["region"].unique().to_list())
    cmap = plt.get_cmap("tab10")
    for i, region in enumerate(regions):
        subset = labeled.filter(pl.col("region") == region)
        ax.scatter(
            subset["tsne_x"], subset["tsne_y"], label=region, color=cmap(i % 10), s=60, alpha=0.85
        )
    for row in labeled.iter_rows(named=True):
        ax.annotate(
            row["province_en"],
            (row["tsne_x"], row["tsne_y"]),
            fontsize=6,
            alpha=0.7,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.set_title("Province risk-profile embeddings (t-SNE of MLP entity embedding layer)")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(title="Region", loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
