"""Number formatting shared across pages — every KPI card and chart tooltip
should render through one of these so formatting stays consistent."""

from __future__ import annotations


def fmt_int(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:,.0f}"


def fmt_money_thb(value: float | None, decimals: int = 0) -> str:
    if value is None:
        return "N/A"
    return f"฿{value:,.{decimals}f}"


def fmt_pct(value: float | None, decimals: int = 1) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def fmt_ratio_pct(value: float | None, decimals: int = 1) -> str:
    """For values already expressed as a ratio meant to be shown as a percentage
    (e.g. loss_ratio = 0.653 -> "65.3%")."""
    return fmt_pct(value, decimals)


def fmt_index(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def fmt_delta_pct(value: float | None, decimals: int = 1) -> str:
    if value is None:
        return "N/A"
    sign = "+" if value >= 0 else "−"
    return f"{sign}{abs(value):.{decimals}f}%"
