# jump_test: do Kou or Bates fix Merton's too-fast skew decay?

The market's implied-vol skew decays roughly like maturity^-0.4, Merton's like maturity^-0.6 to -0.7. Does (a) a Kou
double-exponential jump model or (b) Bates (Heston stochastic variance + Merton jumps) get the decay right, and does the
better model also price future quotes better?

Data: BOL and VOLVB daily closing bid/ask quotes, 2 May to 30 Dec 2025 (167 trading days), r = 2 %, q = 0, the same filters
and the same error measure z = (model - mid) / half-spread as `../seb_merton_analys.py`, which is imported, not copied.

## Short answer

| Skew-decay exponent (all 167 days) | BOL | VOLVB |
|---|---|---|
| Market | -0.364 [-0.391, -0.335] | -0.410 [-0.423, -0.393] |
| Merton | -0.603 [-0.631, -0.574] | -0.733 [-0.763, -0.705] |
| Kou | -0.598 [-0.622, -0.576] | -0.705 [-0.729, -0.681] |
| Bates | -0.460 [-0.488, -0.432] | -0.432 [-0.448, -0.417] |
| Heston, no jumps (overnight run, no day-to-day penalty) | -0.466 | -0.419 |

95 % bootstrap CIs over dates. Merton, Kou and Bates from `output/skew_exponents.csv`; Heston from
`output_overnight/horizon_exponents.csv` (h = 0).

1. **Kou decays as fast as Merton.** Kou - Merton = +0.004 [-0.005, +0.015] (BOL) and +0.029 [+0.020, +0.039] (VOLVB). Kou also
   fits no better than Merton, and its up-jump rate eta1 sits on its upper bound on 105/167 (BOL) and 142/167 (VOLVB) days, so it
   collapses into "diffusion + exponential down-jumps", which behaves like Merton.
2. **Bates moves the exponent to the market's.** Bates - Merton = +0.14 [+0.12, +0.16] (BOL) and +0.30 [+0.28, +0.33] (VOLVB).
   On VOLVB Bates sits on the market (-0.43 vs -0.41); on BOL it is 0.10 steeper than the market.
3. **Bates is worth its extra parameters.** Gaussian AIC prefers Bates to Merton on 78 % (BOL) and 98 % (VOLVB) of days, BIC on
   69 % and 96 %.
4. **It is mainly the stochastic variance, not the jumps.** Heston without jumps already has the market's decay, and gives 80-90 %
   of Bates' out-of-sample gain over Merton; the jumps add a small, significant gain up to about 5 trading days.
5. **Out of sample over time** (overnight run): with today's parameters, Bates prices the quotes of the next 1-10 trading days
   significantly better than Merton on both stocks (1 day ahead: RMSE_z 0.60 vs 0.87 on BOL, 0.90 vs 1.41 on VOLVB). The edge is
   gone at 20 days, and at 40 days Merton is better on BOL. Bates' skew exponent stays at the market's at every horizon.

## How to run

Needs numpy, scipy, pandas, matplotlib (`pip install -r ../requirements.txt`). Run `python seb_merton_analys.py` in the folder
above first (the tests compare against its results), then from inside `jump_test`:

```
python run_all.py --jobs 2           # tests -> calibration (all days) -> analysis -> figures, about 20 min on 2 cores
python run_all.py --overnight        # also the overnight run (about 1 h on 5 worker processes)
```

or one script at a time:

| Script | What it does | Runtime (2 cores) |
|---|---|---|
| `python tests.py [--quick]` | pricing tests (below), `output/test_report.csv` | 1 min |
| `python calibrate.py [--jobs 2] [--day-step 5] [--stocks BOL] [--max-days 3]` | daily calibration of Merton, Kou, Bates | about 12 min |
| `python analyze.py [--in-dir output_step5]` | tables, bootstrap CIs, charts | 10 s |
| `python overnight.py [--jobs 6] [--hours 10] [--smoke] [--analysis-only]` | out-of-sample windows in time, maturity and parameters | about 1 h on 5 workers |
| `python plot_skew_powerlaw.py` | slide figure in `output_figures/` | 5 s |
| `python make_vol_surface_3d.py` | interactive 3D page in `output_3d/` | 1 min |

The first run reads the Excel files once and caches them in `cache/` (delete it to force a re-read).

## Method

**Models** (`jump_models.py`). All are forward-based and European: X = ln(S_T/F_T) with E[e^X] = 1, priced from the characteristic
function.

| Model | Free parameters | Compensator |
|---|---|---|
| Merton | 4: sigma, lambda, mu_J, sigma_J | omega = -sigma^2/2 - lambda*kappa_J |
| Kou | 5: sigma, lambda, p, eta1, eta2 | zeta = p*eta1/(eta1-1) + (1-p)*eta2/(eta2+1) - 1 |
| Bates | 8: v0, kappa, theta, xi, rho, lambda, mu_J, sigma_J | jump term T*lambda*(phi_J - 1 - iu*kappa_J) on a Heston martingale exponent |
| Heston (control) | 5: v0, kappa, theta, xi, rho | none needed |
| BS_slice (reference) | one sigma per expiry, about 4.6 on average | |

**Pricing.** COS method (Fang-Oosterlee) with a truncation range from numerically differenced cumulants (plus a tail-aware
extension for Kou). 384 cosine terms inside the optimiser, 2048 for every reported price; if the two differ by more than
0.001 SEK at the optimum, it re-optimises on a 4x finer grid. Heston uses the "little trap" form of the characteristic function.

**Calibration** (`jump_calib.py`, `calibrate.py`). One parameter set per model and day, fitted to all expiries of that day at once,
least squares on z with the half-spread floored at 0.05 SEK. Bounds: Merton as in the main script; Kou eta in [1.5, 80], p in
[0.01, 0.99]; Bates v0, theta in [1e-4, 1], kappa in [0.1, 20], xi in [0.05, 3], rho in [-0.99, 0.99]. Starts: yesterday's
parameters, a Merton-based start for Kou/Bates, and fixed starts. A mild penalty `3 * sum(((coord - coord_prev)/scale)^2)` against
day-to-day jumps; the weight 3 is the main script's, the per-parameter scales for Kou/Bates are judgement.

**Metrics** (`jump_metrics.py`, `analyze.py`).

1. Fit: RMSE of z and the share with |z| <= 1, per stock, model and maturity bucket (0-30, 30-60, 60-90, 90-150, 150-230,
   230-400 days), plus next-trading-day out-of-sample values (today's parameters on tomorrow's quotes).
2. Skew term structure: per (date, expiry) the weighted slope of implied vol on log-moneyness (|k| < 0.2, at least 5 quotes
   spanning k = 0, weights 1/half-spread) for the market (IV_Mid) and each model. Decay exponent = OLS slope of ln|skew| on ln T
   over pairs with a negative market skew; 95 % CI by bootstrapping dates. Differences between models are bootstrapped the same way.
3. IV-error heatmap: mean (model IV - market IV) in vol points on a maturity x moneyness grid.

Per-parameter comparison: Gaussian AIC = n ln(RSS/n) + 2k and BIC = n ln(RSS/n) + k ln n per day (RSS = sum of z^2).

## Tests before the real fit (`tests.py`, all 68 pass)

| Block | Check | Result |
|---|---|---|
| A | E[exp(X)] = 1 for 300 random parameter sets, Merton, Kou, Bates | error <= 3e-15 |
| B | Merton COS vs the Poisson-weighted Black-Scholes series | 2e-8 SEK |
| C | Merton, Kou, Bates COS vs a Lewis (2001) integral | <= 8e-8 SEK |
| D | Merton, Kou, Bates (and pure Heston) vs Monte Carlo (exact for Merton/Kou, Andersen QE for Heston/Bates) | max 0.46 x 4 SE |
| E | Bates(lambda=0) vs Heston-1993 two-probability pricer | 2e-11 SEK |
| F | Bates -> Merton as xi -> 0 (rho = 0: error ~ xi^2; rho != 0: ~ xi) | log-log slopes 2 and 1 to within 4e-4 |
| G | Kou(lambda=0) = Black-Scholes | 6e-13 |
| H | Put-call parity and forward, q = 0 and q = 1.5 % | <= 4e-8 SEK |
| I | COS grid convergence (384 vs 2048 terms), including heavy-jump corners | <= 5e-2 SEK on the optimiser grid in the worst heavy-Kou corner, <= 1.2e-6 SEK on the reporting grid |
| J | Truncation range vs closed-form cumulants and Monte Carlo paths | no path outside the range |
| K | Reproduce the main script's Merton results: price diff 3e-7 SEK, RMSE_z 0.805 (BOL) / 1.283 (VOLVB), 85.2 % / 68.4 % inside, refit of the first days matches to 1e-5, skew exponents -0.603 / -0.733 | all match |

## Results (all 167 trading days per stock, `output/`)

| Stock | Model | Params | RMSE_z | Inside \|z\|<=1 | Next day: RMSE_z / inside | Decay exponent |
|---|---|---|---|---|---|---|
| BOL | BS_slice | ~4.6 | 1.091 | 55.3 % | | |
| BOL | Merton | 4 | 0.805 | 85.2 % | 0.884 / 82.2 % | -0.603 |
| BOL | Kou | 5 | 0.821 | 85.5 % | 0.891 / 82.5 % | -0.598 |
| BOL | Bates | 8 | 0.536 | 94.0 % | 0.664 / 88.6 % | -0.460 |
| BOL | Market | | | | | -0.364 |
| VOLVB | BS_slice | ~4.7 | 2.050 | 23.5 % | | |
| VOLVB | Merton | 4 | 1.283 | 68.4 % | 1.411 / 64.3 % | -0.733 |
| VOLVB | Kou | 5 | 1.282 | 70.3 % | 1.400 / 66.2 % | -0.705 |
| VOLVB | Bates | 8 | 0.918 | 79.9 % | 1.081 / 74.9 % | -0.432 |
| VOLVB | Market | | | | | -0.410 |

The next-day exponents are within 0.003 of the in-sample ones. Restricting to expiries of 20+ days or 20-200 days makes Merton and
Kou steeper still (-0.68 to -0.81) while Bates stays closer to the market (BOL -0.51 to -0.56 vs -0.40 to -0.43; VOLVB -0.44 to
-0.47 vs -0.42).

**Per extra parameter, versus Merton** (Gaussian AIC/BIC, heuristic)

| Stock | Model | RSS reduction | per extra param | mean dAIC | AIC better on | mean dBIC | BIC better on |
|---|---|---|---|---|---|---|---|
| BOL | Kou (+1) | -4.0 % | -2.0 | +3.0 | 28 % of days | +5.4 | 5 % |
| BOL | Bates (+4) | 55.6 % | 6.9 | -37.0 | 78 % | -27.7 | 69 % |
| VOLVB | Kou (+1) | 0.1 % | 0.2 | -0.3 | 45 % | +1.9 | 28 % |
| VOLVB | Bates (+4) | 48.7 % | 14.2 | -44.6 | 98 % | -35.6 | 96 % |

**By maturity** (`summary_by_maturity.csv`, RMSE_z Merton / Kou / Bates). Bates helps most at the two ends. BOL: 0-30 days
1.35 / 1.41 / 0.68, 230-400 days 0.69 / 0.69 / 0.42; VOLVB: 0-30 days 2.03 / 2.09 / 1.53, 230-400 days 1.16 / 1.09 / 0.44. In the
90-150 day bucket on BOL the three are about equal. Merton's skew at 150-400 days is only 54-65 % of the market's, Bates' 87-103 %.

**Implied-vol error heatmaps** (`heatmaps.png`). Merton under-prices short-dated (0-30 d) vol around and above the money by 1.5-3.5
vol points and over-prices long-dated (150-400 d) calls (k > 0.07) by 1.4-2.0 points; on VOLVB it also under-prices the long-dated
deep put wing (k < -0.25) by 1.6 points. Bates removes almost all of the long-dated error (|error| at most 0.8) and cuts the
short-dated error to 0.2-1.4 points on BOL, but only modestly on VOLVB (1.5-2.5 points). Kou looks like Merton.

## Overnight run: out-of-sample windows in time, maturity and parameters (`overnight.py`)

Every trading day of both stocks and four models (Merton, Kou, Heston, Bates), with no day-to-day penalty, so every day is an
independent calibration and the work spreads over all CPU cores. Each fit gets the normal start points plus 4 extra restarts
(perturbed copies of the best solution and one random start), so it cannot end up worse than the normal fit.

| # | Information used for the fit | What is predicted | What it tests |
|---|---|---|---|
| 1 Time | all options of day t | all options of trading day t+h, h = 1, 2, 5, 10, 20, 40 (h = 0: in sample) | how fast parameters go stale; skew-decay exponent of the later quotes |
| 2 Maturity | free: only the options of day t with Days <= s, s = 45, 60, 75, 90, 120. prior: the same plus a penalty (weight 3) towards the full fit of day t-1 | the options of day t with Days > s; "far" = Days >= max(90, s + 60) | whether the model carries short-maturity information out to long maturities |
| 3 Structure | structural parameters from the previous W days, W = 1, 3, 5, 10, 20, 40, 60, plus the state (sigma for Merton/Kou, v0 for Heston/Bates) fitted to day t | options of day t+h | whether the slow parameters are constant (a long window wins) or drift (the daily re-fit wins); W = 0 is the daily re-fit |

The run is resumable: every finished (stock, day) is saved in `output_overnight/checkpoints_<hash>/` and skipped on a restart.
Days are processed every 5th day first, so an interrupted run still covers the whole sample. Keep the computer awake. Measured on
the author's PC (5 worker processes): 0.85 h in total. Outputs in `output_overnight/`: `OVERNIGHT_SUMMARY.txt` (read this first),
the horizon / maturity-window / parameter-window tables and DM tests, `fits_full_<STOCK>.csv`, the per-option files
`*_options_<STOCK>.csv.gz` and the charts `horizons.png`, `extrapolation.png`, `structural_window.png`.

**1. Time: parameters of day t, quotes of day t+h** (`horizon_summary.csv`, `horizon_dm.csv`; RMSE_z, h = 0 is in sample)

| Stock | Model | h=0 | 1 | 2 | 5 | 10 | 20 | 40 |
|---|---|---|---|---|---|---|---|---|
| BOL | Merton | 0.779 | 0.874 | 0.923 | 1.080 | 1.309 | 1.534 | 1.894 |
| BOL | Kou | 0.783 | 0.876 | 0.925 | 1.079 | 1.307 | 1.525 | 1.877 |
| BOL | Heston | 0.518 | 0.653 | 0.742 | 0.984 | 1.275 | 1.604 | 2.041 |
| BOL | Bates | 0.446 | 0.602 | 0.699 | 0.953 | 1.251 | 1.603 | 2.052 |
| VOLVB | Merton | 1.269 | 1.410 | 1.502 | 1.748 | 2.089 | 2.299 | 2.720 |
| VOLVB | Kou | 1.262 | 1.391 | 1.482 | 1.735 | 2.072 | 2.275 | 2.696 |
| VOLVB | Heston | 0.758 | 0.946 | 1.125 | 1.523 | 1.942 | 2.351 | 2.990 |
| VOLVB | Bates | 0.698 | 0.903 | 1.091 | 1.513 | 1.942 | 2.371 | 3.024 |

- Diebold-Mariano (per-day loss, p < 0.05): Bates beats Merton at every h <= 10 on both stocks (1 day ahead p < 1e-9); no difference
  at h = 20; Merton better at h = 40 on BOL (p < 0.001), not significant on VOLVB (p = 0.09).
- Bates vs Heston (the jump part alone): Bates better for h <= 5 on both stocks (p < 0.01) and at h = 10 on BOL; no difference at 20;
  Bates worse at h = 40 on VOLVB (p = 0.02). Merton -> Heston -> Bates at h = 1: 0.874 -> 0.653 -> 0.602 (BOL), 1.410 -> 0.946 ->
  0.903 (VOLVB), so stochastic variance gives 80-90 % of the improvement.
- Skew-decay exponent of the h-day-ahead quotes (`horizon_exponents.csv`): Bates -0.39 to -0.42 (BOL) and -0.37 to -0.44 (VOLVB) at
  every h; market -0.35 to -0.36 and -0.40 to -0.41; Merton -0.61 to -0.63 and -0.70 to -0.72. The exponent result is not an
  in-sample artefact.
- After about 20 trading days every model misses by more than a half-spread on average, so parameters must be re-fitted daily.

**2. Maturity: fit on expiries up to s days, price the far expiries** (`extrap_summary.csv`, `extrap_dm.csv`, far-expiry RMSE_z)

| Stock | Variant | s | Merton | Kou | Heston | Bates | far skew / market: Merton | Bates |
|---|---|---|---|---|---|---|---|---|
| BOL | free | 120 | 0.876 | 0.970 | 1.921 | 1.721 | 0.53 | 0.84 |
| BOL | prior | 45 | 1.569 | 1.533 | 0.760 | 0.855 | 0.57 | 0.93 |
| BOL | prior | 120 | 0.766 | 0.791 | 0.692 | 0.537 | 0.62 | 0.93 |
| VOLVB | free | 120 | 1.301 | 1.264 | 2.331 | 3.267 | 0.53 | 0.85 |
| VOLVB | prior | 45 | 2.064 | 2.053 | 1.548 | 1.460 | 0.52 | 0.95 |
| VOLVB | prior | 120 | 1.258 | 1.208 | 1.140 | 0.805 | 0.55 | 0.91 |

- With the prior (yesterday's full fit as a penalty target, as a desk would update), Bates beats Merton in 10 of 10 cells (both
  stocks x 5 splits). Merton's far-expiry skew is only 0.51-0.62 of the market's; Bates' is 0.88-0.95.
- Free (no prior), Merton wins 9 of 10 cells: from 2-3 expiries Heston and Bates cannot pin down mean reversion and long-run variance.

**3. Structure: slow parameters frozen over W days, only the state re-fitted** (`window_summary.csv`, `window_dm.csv`)

- The daily full re-fit (W = 0) is best for every model and stock. BOL, h = 1: Bates 0.603 (W = 0), 0.618 (W = 1), 0.668 (W = 10),
  0.876 (W = 60); Merton 0.875 -> 0.942. Bates still beats Merton at W = 60. On VOLVB Heston overtakes Bates once the structure is
  frozen for 20 days or more: the jump parameters are the least stable.
- The Kou W = 1 cell on BOL (RMSE_z 1.001 at h = 1) is one failed state re-fit on 2025-12-09; without that day it is 0.873.

**Reading.** Bates is the better model for the maturity structure and for pricing up to about two weeks ahead, mainly through
stochastic variance; the jumps add a small short-horizon gain. Merton is adequate for the short end and the jump-risk analyses.

## Caveats

1. **Parameters are weakly identified.** Merton's lambda and mu_J are not separately identified (only the jump share of variance
   is), and Kou's p and eta1 and Bates' kappa and rho are poorly pinned down by one day's quotes. Quote prices, skews and exponents,
   not individual parameter values.
2. **BOL's Merton exponent depends on the calibration chain.** Because of the flat lambda/mu_J valley, where the warm-started chain
   ends up matters: -0.60 here (all days, penalty 3), -0.62 without the penalty (overnight run), -0.44 to -0.60 with other start days.
   For VOLVB "Merton is too steep" is robust (-0.67 to -0.74).
3. **Bound hits** (all days): Kou eta1 at its upper bound on 105/167 (BOL) and 142/167 (VOLVB) days; Bates kappa on 38/167 (BOL) and
   54/167 (VOLVB), sigma_J on 61/167 (VOLVB). The fits are good, but those parameters say nothing on their own.
4. **AIC/BIC are heuristic.** They treat the residuals as independent; they are correlated across strikes within an expiry.
5. **Bootstrap intervals cover the sampling of dates only,** not the calibration-path variation in caveat 2. The DM tests in the
   overnight run reuse the same days across cells, so read patterns that repeat across stocks and horizons, not single p-values.
6. **Scope.** Daily closing quotes, two stocks, May-Dec 2025, wide spreads. Event days are treated in the main pipeline
   (`seb_handelsrisk.py`), not here.
7. **|z| <= 1** uses the half-spread floored at 0.05 SEK, so for very tight quotes it is slightly looser than "inside bid/ask"; both
   are in the CSVs (`Inside_pct`, `InsideBidAsk_pct`) and agree to within 0.1 percentage points.

## Earlier side tests (not in this repository)

These were run during the project and are summarised here; their code was removed to keep the repository small.

- **Recovery test** (simulate quotes from known parameters, add noise, refit): prices and the skew decay exponent are recovered well
  (median exponent error 0.01-0.09 at noise of one half-spread), individual parameters are not (median errors: lambda 39-76 %,
  Kou p and eta1 80-100 %, Bates kappa 75 %).
- **Other chain set-ups** (every 5th day with offsets 0/2/4, no penalty): Merton -0.44 to -0.60 (BOL) and -0.67 to -0.74 (VOLVB),
  Bates -0.36 to -0.46 and -0.42 to -0.44; the conclusions above hold in every set-up.
- **Parametric bootstrap** (60 simulations per day, 20 days): quote noise cannot explain Bates' in-sample gain over Merton on 20 of
  20 days, and over Heston on 12 of 20.
- **Surface dynamics:** constant-maturity ATM vol moves against the stock (close to sticky strike), which a Levy model with constant
  parameters such as Merton cannot produce. Return i.i.d. tests had too little power to say anything.
- **Reachable exponents:** Merton can produce an exponent near -0.4, but only in a corner of its parameter space (rare, large,
  widely spread negative jumps with small diffusion).
- **First skew-decay analysis** (`rough_vol_test`): found market -0.36/-0.41 vs Merton -0.60/-0.73, now reproduced by `analyze.py`
  and test K; an exploratory rough Bergomi fit was also tried.

## Visuals for the presentation

- `plot_skew_powerlaw.py` -> `output_figures/skew_powerlaw_light.png` and `_dark.png` (16:9): skew against days to expiry on log-log
  axes, market vs Merton vs Bates, with the exponent range -0.3 to -0.5 documented for S&P 500 ATM skew (Gatheral, Rough Volatility,
  2018).
- `make_vol_surface_3d.py` + `vol_surface_3d_template.html` -> `output_3d/vol_surface_3d.html`: interactive 3D implied-vol surfaces,
  market quotes vs Merton and Bates, every trading day (Swedish text, technical terms in English; needs internet for Plotly).
  `output_3d/vol_surface_3d.mp4` (22 s, 1080p) and `vol_surface_3d_spin.gif` are recordings of the page.
- The 3D surface shrinks towards December because the data's last expiry is 20 March 2026: the longest option gets shorter every day.

## Files

| File | Content |
|---|---|
| `jump_models.py` | characteristic functions, model classes (bounds, starts, penalty coordinates), COS pricer |
| `jump_calib.py` | data loading (reuses `seb_merton_analys.py`), one-day calibrator, daily loop |
| `jump_metrics.py` | fit table, skew table, decay exponents with bootstrap, IV-error grid, Newey-West t |
| `tests.py`, `calibrate.py`, `analyze.py`, `overnight.py`, `run_all.py` | the pipeline (see How to run) |
| `output/` | all-days calibration: `params_daily_<STOCK>.csv`, `options_fit_<STOCK>.csv`, `options_oos_<STOCK>.csv`, `summary_table.csv`, `summary_by_maturity.csv`, `skew_exponents.csv`, `skew_exponent_differences.csv`, `skew_pairs_*.csv`, `heatmap_iv_error.csv`, `skew_loglog.png`, `heatmaps.png`, `test_report.csv` |
| `output_overnight/` | the overnight run (see above) |
| `plot_skew_powerlaw.py`, `output_figures/` | the skew power-law slide figure |
| `make_vol_surface_3d.py`, `vol_surface_3d_template.html`, `output_3d/` | the 3D surface page, its video and GIF |
