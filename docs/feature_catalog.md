# Feature catalog

Contract C5: `data/processed/features/{case}.parquet`. Every feature is computed **per channel**
from one 4 s window `x` (1024 samples at fs = 256 Hz, in µV, already notch- and band-pass-filtered
per Contract C3), giving 18 channels × 32 features = **576** columns named `f_{feature}_{channel_slug}`
(30 features / 540 columns if `features.entropy.enabled: false`; see Section 12.3's fallback ladder).

> **Status: real findings from the first data slice** (`chb01`, `chb02`, `chb21`→`chb01`;
> `python check_feature_sanity.py`, 26,481 windows, **387 ictal / 26,094 interictal**). The "Expected
> during seizure" column below states what was actually observed, channel-averaged, on this slice —
> not a literature guess anymore. **Caveats that still apply:** only 2 distinct patients, only 387
> ictal windows total, no per-patient or per-channel breakdown, no significance testing — treat
> directions as a first-pass signal to recheck once the full cohort lands, especially for any
> feature whose % change is small. Two features contradicted the original literature-derived
> hypothesis outright (flagged below with **‼**) rather than being silently corrected — that
> disagreement is itself worth a line in `docs/baseline_report.md`, not just an edit here.

## Time domain (`features/time_domain.py`, 8 features)

| Feature | Formula | Unit | Observed during seizure (this slice) |
|---|---|---|---|
| `mean` | `mean(x)` | µV | Near zero both ways (-0.0004 → 0.0082); no physiologically meaningful direction, as expected — the % change (+2393%) is an artifact of dividing by a near-zero baseline, not a real effect. |
| `std` | `sqrt(var(x))`, ddof=0 | µV | **Up** (36.5 → 88.9, +144%). Matches expectation: ictal discharges are higher-amplitude than background. |
| `skew` | `m3 / std**3` (population, biased) | unitless | Down, but both values near zero (0.11 → -0.004); no strong effect either direction on this slice. |
| `kurt` | `m4 / m2**2 - 3` (excess) | unitless | **‼ Down** (1.41 → 0.66), contradicting the "up, from spikes" hypothesis. Plausible explanation: averaged over a full 4 s window, a *sustained rhythmic* ictal discharge is closer to a smooth periodic signal (which is platykurtic, negative excess kurtosis) than the interictal background, which may include isolated large-amplitude artifacts contributing heavier tails. Kurtosis measured on a shorter, spike-centered window might show the opposite — worth checking once per-seizure-phase windows are available. |
| `line_length` | `sum(abs(diff(x)))` | µV | **Up** (6526 → 11259, +73%). Matches expectation and the standard seizure-detection literature. |
| `hjorth_activity` | `var(x)` | µV² | **Up** (2260 → 9688, +329%), same driver as `std`. |
| `hjorth_mobility` | `sqrt(var(dx)/var(x))`, `dx=diff(x)` | unitless (~rad/sample) | **Down** (0.238 → 0.173, -27%) on this slice — i.e. ictal activity here is dominated by *relatively slower* content than background, consistent with the low-frequency shift also seen in `rel_delta`/`rel_theta` below. |
| `hjorth_complexity` | `mobility(dx)/mobility(x)` | unitless | **‼ Up** (3.16 → 3.89, +23%), contradicting the "down toward 1, near-periodic" hypothesis. The interictal baseline complexity is already well above 1 (3.16, not ~1), so "toward 1" was the wrong reference point for this population — real pediatric interictal EEG isn't close to periodic either. The *increase* suggests these seizures are less simply periodic than the synthetic single-tone assumption in `test_extract.py`; likely mixed-frequency or evolving morphology rather than a clean single rhythm. |

Divisions guarded to `0.0` (not NaN) wherever the relevant variance is 0 (e.g. a flat/clipped channel);
in practice, zero non-finite replacements were needed anywhere on this real slice.

## Frequency domain (`features/frequency_domain.py`, 10 features)

Welch PSD: 1 s segments, 50 % overlap, Hann window (`features.welch`). Band power = trapezoidal
integral of the PSD over each band (`features.bands`); `rel_{band}` normalises by total power in
0.5–45 Hz.

| Feature | Formula | Unit | Observed during seizure (this slice) |
|---|---|---|---|
| `logpow_delta` | `log10(power_[0.5,4)Hz + 1e-12)` | log₁₀(µV²/Hz · Hz) | **Up** (+36%). All five `logpow_*` bands rise together (32-40%) — consistent with the overall amplitude increase (`std`/`hjorth_activity`) dominating, as expected; `logpow_*` alone doesn't distinguish *which* band matters most, `rel_*` does. |
| `logpow_theta` | same, [4,8) Hz | " | **Up** (+39%). |
| `logpow_alpha` | same, [8,13) Hz | " | **Up** (+33%). |
| `logpow_beta` | same, [13,30) Hz | " | **Up** (+31%). |
| `logpow_gamma` | same, [30,45) Hz | " | **Up** (+40%). |
| `rel_delta` | `power_delta / power_[0.5,45)Hz` | fraction, [0,1] | **Up** (0.541 → 0.651, +20%). |
| `rel_theta` | same | " | Flat (0.213 → 0.216, +2%) — essentially no shift. |
| `rel_alpha` | same | " | **Down** (0.089 → 0.055, -38%). |
| `rel_beta` | same | " | **Down** (0.091 → 0.051, -45%). |
| `rel_gamma` | same | " | **Down** (0.066 → 0.027, -59%). |

**Clear, coherent finding on this slice:** relative power shifts from higher frequencies toward
delta (and holds steady in theta) during seizure — `rel_delta` up, `rel_alpha`/`rel_beta`/`rel_gamma`
all down by 38-59%. This is consistent with classic rhythmic delta/theta ictal discharges, common in
pediatric CHB-MIT-type recordings, and it independently cross-validates the wavelet finding below
(same low-frequency-dominance story, different feature family).

## Nonlinear / entropy (`features/nonlinear.py`, 2 features; omitted if `entropy.enabled: false`)

Computed on the window **decimated by `entropy.decimate`** (plain stride-slicing; see the module
docstring for why not `scipy.signal.decimate`), with `m = entropy.m`, `r = entropy.r_factor * std(decimated window)`.

| Feature | Formula | Unit | Observed during seizure (this slice) |
|---|---|---|---|
| `sampen` | `-ln(A/B)`, template matches of length `m`/`m+1` within Chebyshev distance `r` | unitless (nats) | **Down** (1.05 → 0.84, -20%). Matches the classic literature finding directly: ictal activity is more regular/synchronised than background. |
| `apen` | `Phi^m(r) - Phi^(m+1)(r)` (Pincus) | unitless (nats) | **Down** (0.95 → 0.83, -13%), same direction as `sampen`, smaller magnitude — consistent with `apen`'s known self-match bias making it a more conservative (damped) estimator on short windows. |

Zero non-finite replacements occurred anywhere on this real slice — the `r<=0`/no-template-match
`nan` path is genuinely rare on real EEG, not something silently papering over bad data in practice.

**Real-world cost, measured (not the earlier synthetic-noise estimate):** 179-226 ms/window (18 ch,
all four feature groups combined) on the engineer's machine, ~4.5x higher than the ~40 ms/window
synthetic-noise benchmark from Step 8. Cause: `sample_entropy`/`approximate_entropy`'s early-exit
comparison loop breaks less often on autocorrelated real EEG than on i.i.d. Gaussian noise, so more
comparisons run to completion. Plan full-cohort runtime from this number, not the synthetic one.

## Wavelet (`features/wavelet.py`, 12 features)

5-level `db4` DWT (`features.wavelet`), `pywt.wavedec(..., level=5)` → `[a5, d5, d4, d3, d2, d1]`.
At fs = 256 Hz, empirically verified sub-band peak frequencies for this config (see
`test_nonlinear_wavelet_features.py`): 6 Hz → d5, 12 Hz → d4, 24 Hz → d3 (approximate ranges:
d1 64–128 Hz, mostly empty above the 45 Hz band-pass but kept for a fixed catalog; d2 32–64 Hz; d3
16–32 Hz; d4 8–16 Hz; d5 4–8 Hz; a5 0–4 Hz).

| Feature | Formula | Unit | Observed during seizure (this slice) |
|---|---|---|---|
| `dwt_logE_d1` | `log10(sum(d1**2) + 1e-12)` | log₁₀(µV²) | **Up** (+19%). All six `dwt_logE_*` sub-bands rise together (11-19%), fairly uniformly — expected once aggregating across many different seizures/rhythms rather than one clean tone (see `dwt_std_*` below for the differential story `dwt_logE_*` alone doesn't show). |
| `dwt_logE_d2` | same, d2 (32–64 Hz) | " | **Up** (+13%). |
| `dwt_logE_d3` | same, d3 (16–32 Hz) | " | **Up** (+11%). |
| `dwt_logE_d4` | same, d4 (8–16 Hz) | " | **Up** (+12%). |
| `dwt_logE_d5` | same, d5 (4–8 Hz) | " | **Up** (+16%). |
| `dwt_logE_a5` | same, a5 (0–4 Hz) | " | **Up** (+15%). |
| `dwt_std_d1` | `std(coeffs)` | µV | **Up** (+48%), smallest jump of the six — matches `dwt_logE_d1`'s smaller share too (least ictal-relevant sub-band). |
| `dwt_std_d2` | " | " | **Up** (+59%). |
| `dwt_std_d3` | " | " | **Up** (+73%). |
| `dwt_std_d4` | " | " | **Up** (+103%). |
| `dwt_std_d5` | " | " | **Up** (+165%), largest jump. |
| `dwt_std_a5` | " | " | **Up** (+157%), second largest. |

**Cross-validates the frequency-domain finding above:** `dwt_std_*` growth is *not* uniform the way
`dwt_logE_*` looked — d4/d5/a5 (roughly the 0-16 Hz range, i.e. delta/theta/low-alpha) grow far more
(103-165%) than d1/d2/d3 (roughly 16-128 Hz, i.e. beta/gamma and above) at 48-73%. Two independent
feature families (Welch `rel_*` and wavelet `dwt_std_*`) both point to the same story on this slice:
seizure activity concentrates in lower frequencies at the relative expense of higher ones. That
convergence is a stronger basis for the claim than either family alone.
