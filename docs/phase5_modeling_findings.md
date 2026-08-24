# Phase 5 Modeling Findings — Business-Sense Review Addendum

> Checkpoint reviewed and decided 2026-08-24. Full metric tables and SHAP outputs live in
> `notebooks/05_modeling_baseline.ipynb` and `reports/`; this doc captures the reasoning
> behind four decisions made at the Step 4 business-sense checkpoint, before Phase 6 started.

---

## 1. `ev_share` is a confound, not a causal effect — keep the feature, restrict its use

**Finding:** `ev_share` is the strongest or second-strongest SHAP driver in every Phase 5
model (frequency, severity), always with a large negative effect (GLM coefficient −40.8 —
0%→5% EV share would imply an ~86% frequency reduction if read causally). That is
implausibly large for what should be a minor factor.

**Diagnostic run to check this (per the Step 4 review decision):**

| Check | Result |
|---|---|
| Corr(`ev_share`, total registered vehicles, raw scale) | **r = 0.829** |
| Corr(`ev_share`, log(total registered vehicles)) | **r = 0.718** |
| Top `ev_share` province | Bangkok, 6.2% — more than **double** the #2 province (Phuket, 3.0%) |
| Mean `ev_share` by region | Central 1.18% > South 1.10% > West 1.06% > East 1.00% > North 0.97% > Northeast 0.72% |

`ev_share` is heavily concentrated in a handful of major urban/tourist-hub provinces
(Bangkok, Phuket, Chiang Mai, Songkhla) and tracks overall vehicle-fleet scale closely. This
is exactly the signature of a **province-identity/urbanization proxy**, not a dose-response
EV effect: the model isn't learning "EVs make roads safer," it's learning "this numeric
column happens to separate Bangkok-and-similar from everywhere else."

**Decision:** keep `ev_share` in the predictive models — it is a legitimate, non-leaking,
knowable-at-prediction-time feature, and dropping a working proxy would only hurt predictive
performance for no benefit. **But it must never be used to answer BQ8** ("do high-EV-share
provinces show higher severity, is a separate EV rating factor warranted"). BQ8 is answered
only through the stated limitation: `fact_accident` has no per-vehicle fuel-type field, so
any EV-severity comparison is necessarily at the **province (ecological) level**, not the
per-vehicle level the question actually asks. Reporting "we verified this is a confound, not
a causal effect, with the correlation evidence above" is the honest answer to BQ8 — stronger
than silently dropping the feature or (worse) reporting the raw SHAP effect as if it were
causal.

---

## 2. `avg_days_to_next_holiday` dropped from the LightGBM frequency/severity feature set

**Finding:** ranked #2 by SHAP importance in the frequency model (mean |SHAP| = 0.264,
second only to `vehicle_group`), with a positive value/SHAP correlation (more days until the
next holiday → higher predicted frequency) that looked odd next to `n_holiday_days`/
`has_songkran`/`has_newyear` all *also* independently raising predicted frequency.

**Partial-dependence check** (frequency LightGBM model, all other features held at each
row's own value, `avg_days_to_next_holiday` swept across its observed range):

```
  5.9 days -> 20.78 predicted accidents
  7.9      -> 21.07
  9.9      -> 21.07
 11.8      -> 21.07
 13.8      -> 20.19
 15.8      -> 20.19
 17.8      -> 20.19
 19.8      -> 20.19
 21.8      -> 46.92   <- sharp, non-monotonic jump
 23.7      -> 46.17
 25.7      -> 46.27
 27.7      -> 37.67
 29.7      -> 31.16
 31.7      -> 23.03
 33.6      -> 23.03
```

Not a coherent trend — a flat plateau, then a sharp spike, then decay. This is the signature
of a tree model exploiting a specific calendar-cluster split rather than a genuine "distance
from holiday" effect. It also correlates **−0.71** with `n_holiday_days` in this frame
(mechanically: many holidays nearby ⇒ few days until the next one), so it is very plausibly
just fighting that cleaner, already-included signal collinearly.

**Decision:** dropped from `rri.models.lightgbm_baseline.FREQ_SEV_NUMERIC_COLS` (it was never
in the GLM's formula to begin with). `n_holiday_days`/`has_songkran`/`has_newyear` already
cover BQ3's holiday-risk question without this ambiguity. Models refit and all Phase 5
artifacts (metrics, SHAP tables/plots, `reports/phase5_model_comparison.csv`) regenerated
after the drop.

---

## 3. `r35mm_avg_days_2014_2023` flips sign between frequency and severity — a coherent story (BQ2)

**Finding:** provinces with more historical heavy-rain days (`r35mm_avg_days_2014_2023`)
show **lower** predicted frequency but **higher** predicted severity — opposite signs across
the two models, on the same feature.

**Read:** this is not a contradiction, it's a mechanistically coherent pattern worth
promoting as a genuine insight for BQ2 ("how much do rain and flood-prone areas raise risk"):
heavy-rain-prone provinces tend to be less densely trafficked (lower baseline accident
*frequency*, consistent with lower urbanization/traffic volume — the same axis `ev_share`
proxies), but the accidents that do happen there skew more *severe* — plausible drivers
include higher-speed rural/highway roads, weather-hazard conditions (standing water,
reduced visibility, hydroplaning), and worse post-crash emergency response times in
less-developed areas. **Fewer but more severe crashes in heavy-rain provinces.**

This complements (not duplicates) the flood-risk finding: `flood_pct_high_risk_tambons` is
also a positive severity driver, so both of this project's two rain-related geography
features point the same direction on severity, from two different data sources
(`thailand_heavy_rain_R35mm.csv` and `thailand_flood_risk_areas.csv`) — a form of internal
corroboration worth citing in the BQ2 write-up.

**Decision:** promote this to the report as a headline BQ2 finding, framed exactly as above
("fewer but more severe crashes"), not as two conflicting numbers to explain away.

---

## 4. `rain_lead0..5` collinearity — carried into Phase 6 as a GRU sanity check

**Finding:** in both the frequency and severity SHAP tables, `rain_lead4`/`rain_lead5`
(forecasts 4-5 months ahead) ranked *above* `rain_lead0` (the nearest-term forecast). If the
forecast leads were adding genuine month-specific predictive signal, the nearest-term lead
would be expected to matter at least as much as the furthest one.

**Read:** the 6 lead columns are issued together for the same (province, month) and are
highly autocorrelated with each other and with the province's general wet/dry-season
identity. It's plausible LightGBM is picking whichever lead happens to reduce split variance
best among near-duplicates, without that lead being meaningfully more informative than its
neighbors — i.e., the leads may collectively function as "how rainy is this province
generally" rather than each one carrying distinct month-specific forecast information.

**This project's stated differentiator is "the rainfall forecast predicts the future"** —
so this matters. **Carried forward to Phase 6**: when building the GRU forecast (which uses
the leads as known-future covariates), check whether the model's learned sensitivity
actually differs meaningfully across `rain_lead0..5` for different forecast horizons, or
whether it collapses to a static per-province level. Report the result honestly either way —
a GRU that can't tell the leads apart is itself a legitimate, reportable finding about how
much value the forecast-lead differentiator actually adds beyond a province-level baseline.
