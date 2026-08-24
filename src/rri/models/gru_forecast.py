"""Global GRU forecast (CLAUDE.md §10 L3, Phase 6 Step 6) -- proof of capability, not the
centerpiece. Scoped deliberately light: 48 months x 77 provinces (the frequency/severity
frame's own window, CLAUDE.md §12) is thin for a sequence model, so a small hidden size and
an honest comparison against a seasonal-naive baseline are the point, not squeezing out
accuracy. If the GRU doesn't beat seasonal-naive, that is reported as a legitimate result,
not hidden (CLAUDE.md Step 6).

**Grain**: one province x month accident count, summed across vehicle groups (`vehicle_group`
does not appear in CLAUDE.md's description of this step -- "forecasting monthly accident
counts," not per-vehicle-group counts, unlike the GLM/LightGBM frequency models).

**Windowing**: 12-month lookback -> 6-month-ahead forecast (CLAUDE.md's "6-month-ahead"
paired with "rainfall forecast lead 0-5" -- 6 lead columns for 6 forecast steps, so this
reads the pairing as horizons 0..5 months ahead of the window's own first target month, not
literally month+6). Each of the 6 target months' `rain_lead{h}` is read from *that target
month's own row* -- the exact same "known-future forecast" convention CLAUDE.md's Step 1
guard rail already established for the GLM/LightGBM frequency frame (`rri.features.
frequency_severity`), reused here rather than re-litigated.

**Split**: a window belongs to train/valid/test by its *target span's year* -- CLAUDE.md N6
applies to windows, not just rows. Only windows whose full 6-month target span sits inside a
single calendar year are kept (target start month January-July) -- a window straddling two
years would need rows from both, and assigning it to one split while it draws information
from another would blur the boundary. This does cost some windows (5 of 12 possible start
months per province-year), but a leaky window is worse than a missing one.

**Known limitation this module is honest about**: 12-month lookback leaves only 3 usable
year-windows total per province (2023 train / 2024 valid / 2025 test) x 7 start months each
-- 539 windows per split across 77 provinces. Small. That is the whole point of scoping this
"light" (CLAUDE.md Step 6) rather than pretending a thin sequence-model dataset is a large
one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
import torch
from torch import nn

from config.simulation_params import RANDOM_SEED

LOOKBACK_MONTHS = 12
FORECAST_HORIZON = 6

HIDDEN_SIZE = 16
DROPOUT = 0.2
#: Full-batch training (539 windows -- one gradient step per epoch, see fit_gru_forecast),
#: so convergence is slow in epoch-count even though each epoch is cheap (~2ms). An earlier
#: MAX_EPOCHS=300/patience=20 stopped while validation loss was still falling steadily
#: (13.1 at epoch 300 vs. 8.0 once allowed to run to convergence) -- an honest "GRU vs.
#: seasonal-naive" comparison needs the GRU actually converged first, not cut short.
MAX_EPOCHS = 2000
EARLY_STOPPING_PATIENCE = 100
LEARNING_RATE = 5e-3

RAIN_LEAD_COLS = [f"rain_lead{h}" for h in range(FORECAST_HORIZON)]


def build_province_month_series(freq_sev_frame: pl.DataFrame) -> pl.DataFrame:
    """Collapse the frequency/severity frame's vehicle_group split into one series per
    province x month: total accident_count, plus the (vehicle_group-invariant) rainfall
    and calendar columns carried through via `.first()`.
    """
    return (
        freq_sev_frame.group_by(["province_th", "year", "month"])
        .agg(
            pl.col("accident_count").sum().alias("accident_count"),
            *[pl.col(c).first() for c in RAIN_LEAD_COLS],
        )
        .sort(["province_th", "year", "month"])
    )


def _month_index(year: int, month: int, base_year: int) -> int:
    """0-based month index from `base_year`-01, e.g. 2023-03 with base 2022 -> 14."""
    return (year - base_year) * 12 + (month - 1)


def _cyclical_month(month: int) -> tuple[float, float]:
    angle = 2.0 * np.pi * (month - 1) / 12.0
    return float(np.sin(angle)), float(np.cos(angle))


@dataclass(frozen=True)
class GruWindows:
    history: np.ndarray  # (n, LOOKBACK_MONTHS, 3) -- scaled count, month_sin, month_cos
    target_covariates: np.ndarray  # (n, FORECAST_HORIZON, 3) -- rain_lead_h, month_sin, month_cos
    y: np.ndarray  # (n, FORECAST_HORIZON) -- raw accident counts
    province: np.ndarray  # (n,)
    target_start_year: np.ndarray  # (n,)
    scale: np.ndarray  # (n,) -- per-window province scale used to normalize history


def _province_scale(counts: np.ndarray) -> float:
    """A stable per-province normalizer for the history channel -- its own mean count,
    floored so a near-zero-accident province doesn't blow up to a huge scaled value."""
    return float(max(counts.mean(), 1.0))


def build_gru_windows(series: pl.DataFrame) -> GruWindows:
    """Slide 12-month lookback / 6-month-target windows over every province's series."""
    base_year = int(series["year"].min())  # type: ignore[arg-type]
    histories: list[np.ndarray] = []
    target_covs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    provinces: list[str] = []
    target_years: list[int] = []
    scales: list[float] = []

    for province, group in series.group_by("province_th"):
        province_name = province[0] if isinstance(province, tuple) else province
        g = group.sort(["year", "month"])
        n_months = g.height
        month_index = np.array(
            [_month_index(y, m, base_year) for y, m in zip(g["year"], g["month"], strict=True)]
        )
        # series should be contiguous (48 consecutive months); assert rather than
        # silently produce misaligned windows if a gap ever appears.
        assert np.array_equal(month_index, np.arange(n_months)), (
            f"non-contiguous month series for province {province_name}"
        )

        counts = g["accident_count"].to_numpy().astype(np.float64)
        years = g["year"].to_numpy()
        months = g["month"].to_numpy()
        rain = g.select(RAIN_LEAD_COLS).to_numpy().astype(np.float64)
        scale = _province_scale(counts)

        for lookback_start in range(0, n_months - LOOKBACK_MONTHS - FORECAST_HORIZON + 1):
            lookback_end = lookback_start + LOOKBACK_MONTHS  # exclusive
            target_start = lookback_end
            target_end = target_start + FORECAST_HORIZON  # exclusive

            target_start_year = int(years[target_start])
            if int(years[target_end - 1]) != target_start_year:
                continue  # keep only windows whose target span fits one calendar year

            hist_counts = counts[lookback_start:lookback_end] / scale
            hist_months = months[lookback_start:lookback_end]
            hist_cyc = np.array([_cyclical_month(int(m)) for m in hist_months])
            history = np.concatenate([hist_counts[:, None], hist_cyc], axis=1)

            target_months = months[target_start:target_end]
            target_cyc = np.array([_cyclical_month(int(m)) for m in target_months])
            # target_covariates[h] uses rain_lead{h} from the target month at position h
            # of this window's target span (h = 0..FORECAST_HORIZON-1) -- see module
            # docstring on why this reuses the frequency/severity frame's own convention.
            lead_values = np.array([rain[target_start + h, h] for h in range(FORECAST_HORIZON)])
            target_covariates = np.concatenate([lead_values[:, None], target_cyc], axis=1)

            histories.append(history)
            target_covs.append(target_covariates)
            ys.append(counts[target_start:target_end])
            provinces.append(str(province_name))
            target_years.append(target_start_year)
            scales.append(scale)

    return GruWindows(
        history=np.stack(histories).astype(np.float32),
        target_covariates=np.stack(target_covs).astype(np.float32),
        y=np.stack(ys).astype(np.float32),
        province=np.array(provinces),
        target_start_year=np.array(target_years),
        scale=np.array(scales, dtype=np.float32),
    )


def split_windows(windows: GruWindows) -> dict[str, GruWindows]:
    """Time-based split by each window's target-span year (CLAUDE.md N6)."""
    out = {}
    for name, years in [("train", (None, 2023)), ("valid", (2024, 2024)), ("test", (2025, 2025))]:
        lo, hi = years
        mask = windows.target_start_year <= hi if lo is None else windows.target_start_year == lo
        out[name] = GruWindows(
            history=windows.history[mask],
            target_covariates=windows.target_covariates[mask],
            y=windows.y[mask],
            province=windows.province[mask],
            target_start_year=windows.target_start_year[mask],
            scale=windows.scale[mask],
        )
    return out


def seasonal_naive_forecast(series: pl.DataFrame, windows: GruWindows) -> np.ndarray:
    """Seasonal-naive baseline: forecast month m with the actual count from m-12."""
    lookup = {
        (row["province_th"], row["year"], row["month"]): row["accident_count"]
        for row in series.iter_rows(named=True)
    }
    predictions = np.zeros_like(windows.y)
    for i in range(len(windows.province)):
        year = int(windows.target_start_year[i])
        for h in range(FORECAST_HORIZON):
            month = h + 1
            predictions[i, h] = lookup.get((windows.province[i], year - 1, month), np.nan)
    return predictions


class GlobalGru(nn.Module):
    """One GRU encoder shared across every province's sequence, plus a direct multi-step
    decoder head (all 6 horizons predicted at once from the encoder's final state, not
    autoregressively -- covariates for every horizon are already known, so there is
    nothing an autoregressive loop would gain here)."""

    def __init__(
        self,
        history_input_size: int = 3,
        covariate_size: int = 3,
        hidden_size: int = HIDDEN_SIZE,
        forecast_horizon: int = FORECAST_HORIZON,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.gru = nn.GRU(history_input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + covariate_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Softplus(),  # count forecast, must stay non-negative
        )

    def forward(self, history: torch.Tensor, target_covariates: torch.Tensor) -> torch.Tensor:
        _, final_hidden = self.gru(history)
        hidden = self.dropout(final_hidden.squeeze(0))  # (batch, hidden_size)
        outputs = []
        for h in range(self.forecast_horizon):
            step_input = torch.cat([hidden, target_covariates[:, h, :]], dim=1)
            outputs.append(self.head(step_input))
        result: torch.Tensor = torch.cat(outputs, dim=1)
        return result


@dataclass(frozen=True)
class FittedGru:
    model: GlobalGru
    train_losses: list[float] = field(default_factory=list)
    valid_losses: list[float] = field(default_factory=list)
    best_epoch: int = 0


def fit_gru_forecast(train: GruWindows, valid: GruWindows) -> FittedGru:
    """Fit the global GRU with Poisson NLL loss (count target) and early stopping."""
    torch.manual_seed(RANDOM_SEED)
    model = GlobalGru()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.PoissonNLLLoss(log_input=False, full=True, eps=1e-6)

    train_history = torch.from_numpy(train.history)
    train_cov = torch.from_numpy(train.target_covariates)
    train_y = torch.from_numpy(train.y)
    valid_history = torch.from_numpy(valid.history)
    valid_cov = torch.from_numpy(valid.target_covariates)
    valid_y = torch.from_numpy(valid.y)

    best_valid_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    patience_left = EARLY_STOPPING_PATIENCE
    train_losses: list[float] = []
    valid_losses: list[float] = []

    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        pred = model(train_history, train_cov)
        loss = loss_fn(pred, train_y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            valid_pred = model(valid_history, valid_cov)
            valid_loss = loss_fn(valid_pred, valid_y).item()
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

    return FittedGru(
        model=model, train_losses=train_losses, valid_losses=valid_losses, best_epoch=best_epoch
    )


def predict_gru_forecast(fitted: FittedGru, windows: GruWindows) -> np.ndarray:
    fitted.model.eval()
    with torch.no_grad():
        pred = fitted.model(
            torch.from_numpy(windows.history), torch.from_numpy(windows.target_covariates)
        )
    return np.asarray(pred.cpu().numpy())
