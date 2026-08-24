"""SiamShield dashboard theme — CSS on top of `.streamlit/config.toml`, plus
the KPI-card component used across pages. Palette follows the dataviz skill's
validated default (see the project's dataviz skill, `references/palette.md`):
sequential blue for magnitude, the fixed status palette for delta good/bad,
never color-alone (every delta pairs an arrow glyph + colored text).
"""

from __future__ import annotations

import streamlit as st

# -- palette tokens (dataviz skill reference palette, light mode) --
SURFACE = "#fcfcfb"
PAGE_PLANE = "#f9f9f7"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
BORDER = "rgba(11, 11, 11, 0.10)"

BLUE = "#2a78d6"
BLUE_DARK = "#184f95"
STATUS_GOOD = "#006300"
STATUS_WARNING = "#fab219"
STATUS_SERIOUS = "#ec835a"
STATUS_CRITICAL = "#d03b3b"

CATEGORICAL = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]


def inject_theme() -> None:
    st.markdown(
        f"""
        <style>
        .block-container {{
            padding-top: 1.5rem;
            max-width: 1200px;
        }}
        .siamshield-header {{
            background: linear-gradient(90deg, {BLUE_DARK} 0%, {BLUE} 100%);
            color: white;
            padding: 1.1rem 1.5rem;
            border-radius: 10px;
            margin-bottom: 1.25rem;
        }}
        .siamshield-header h1 {{
            font-size: 1.4rem;
            font-weight: 700;
            margin: 0;
            color: white;
        }}
        .siamshield-header p {{
            font-size: 0.9rem;
            margin: 0.2rem 0 0 0;
            color: rgba(255,255,255,0.85);
        }}
        .kpi-row {{
            display: flex;
            gap: 0.9rem;
            flex-wrap: wrap;
            margin-bottom: 1rem;
        }}
        .kpi-card {{
            background: {SURFACE};
            border: 1px solid {BORDER};
            border-radius: 10px;
            padding: 0.9rem 1.1rem;
            flex: 1 1 150px;
            min-width: 150px;
        }}
        .kpi-label {{
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: {SECONDARY_INK};
            font-weight: 600;
            margin-bottom: 0.35rem;
        }}
        .kpi-value {{
            font-size: 1.55rem;
            font-weight: 700;
            color: {PRIMARY_INK};
            line-height: 1.1;
        }}
        .kpi-delta {{
            font-size: 0.78rem;
            font-weight: 600;
            margin-top: 0.3rem;
        }}
        .kpi-delta.good {{ color: {STATUS_GOOD}; }}
        .kpi-delta.bad {{ color: {STATUS_CRITICAL}; }}
        .kpi-delta.neutral {{ color: {MUTED_INK}; }}
        .kpi-caveat {{
            font-size: 0.72rem;
            color: {MUTED_INK};
            margin-top: 0.35rem;
        }}
        .bq-tag {{
            display: inline-block;
            background: {PAGE_PLANE};
            border: 1px solid {BORDER};
            color: {SECONDARY_INK};
            font-size: 0.72rem;
            font-weight: 600;
            padding: 0.15rem 0.55rem;
            border-radius: 999px;
            margin-bottom: 0.6rem;
        }}
        section[data-testid="stSidebar"] {{
            background: {PAGE_PLANE};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str) -> None:
    # Single line, no leading whitespace, no embedded blank line -- st.markdown
    # runs content through a CommonMark parser, and an indented / blank-line-
    # interrupted multi-line HTML block gets misread as an indented code block
    # (see kpi_card_html's docstring note below for the mechanism).
    st.markdown(
        f'<div class="siamshield-header"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def bq_tag(text: str) -> None:
    st.markdown(f'<span class="bq-tag">{text}</span>', unsafe_allow_html=True)


def kpi_card_html(
    label: str,
    value: str,
    delta_pct: float | None,
    *,
    higher_is_worse: bool = True,
    caveat: str | None = None,
) -> str:
    """Returns ONE line, no leading whitespace, no embedded blank line.

    `st.markdown` runs its content through a CommonMark parser. A raw `<div>`
    block only stays "raw HTML" across multiple lines up to the next blank
    (or whitespace-only) line -- past that, indented lines are read as an
    *indented code block* and printed literally instead of rendered. Joining
    several indented, multi-line card templates (the original bug here)
    reliably breaks after the first card for exactly this reason. Building
    everything as one line sidesteps the ambiguity entirely.
    """
    if delta_pct is None:
        delta_html = '<div class="kpi-delta neutral">YoY: N/A</div>'
    else:
        is_bad = (delta_pct > 0) if higher_is_worse else (delta_pct < 0)
        arrow = "▲" if delta_pct >= 0 else "▼"
        cls = "bad" if is_bad else "good"
        delta_html = f'<div class="kpi-delta {cls}">{arrow} {abs(delta_pct):.1f}% YoY</div>'
    caveat_html = f'<div class="kpi-caveat">{caveat}</div>' if caveat else ""
    return (
        '<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f"{delta_html}"
        f"{caveat_html}"
        "</div>"
    )


def render_kpi_row(cards: list[str]) -> None:
    st.markdown(f'<div class="kpi-row">{"".join(cards)}</div>', unsafe_allow_html=True)
