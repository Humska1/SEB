# SEB Project 5 – Black-Scholes vs Merton jump-diffusion: full summary

Status as of 29 September 2026. All numbers come from the frozen final run of 29 September 2026, which reproduced identically on the author's computer and in the cloud.

---

## 1. The assignment

SEB's equity-derivatives desk makes markets in options. Their central question:

> Can jump risk improve the understanding and pricing of the parts of the option market where Black-Scholes deviates most from market prices?

Sub-questions from SEB:
1. Where are the largest differences between BS valuation and market prices?
2. Can they be explained by jump risk?
3. For which strikes and maturities is the effect largest?
4. How are results affected ahead of reports and other events?
5. Are there situations where the model difference matters for market making and trading (bid/ask, risk)?

SEB also asked specifically about **long vs short gamma**.

SEB's data note (docs/seb_data_guide.txt): the options are American, but early exercise only matters around dividends and the period has no dividend, so BS-type valuation is fine with dividend yield 0.

---

## 2. Data

- Two Swedish stocks: **Boliden (BOL)** and **Volvo B (VOLVB)**.
- Daily closing bid/ask for the stock and all its options, 1 May – 30 Dec 2025. 167 trading days with option quotes; 7 holidays have no option data.
- Expiries from May 2025 to March 2026, all before the spring 2026 dividends, so no dividend inside any option's life.
- Some Volvo series are "adjusted" (names ending in X, e.g. strike 462.11). Their implied vols match normal series, so they are kept.
- Risk-free rate 2%. Put-call parity in the data implies roughly 1.6–2.0%, so 2% is consistent.

**Cleaning funnel (BOL / VOLVB):**

| Step | BOL | VOLVB |
|---|---|---|
| Option rows in file | 56,832 | 52,254 |
| Bid > 0.01 and ask ≥ bid | 48,704 | 42,876 |
| At least 5 days to expiry | 45,204 | 41,078 |
| Out-of-the-money only (puts K<F, calls K≥F) | 16,924 | 13,919 |
| Spread no wider than the mid price | **12,768** | **11,857** |

Out-of-the-money options only: they are the liquid side, and early exercise is negligible for them. A binomial-tree check shows the American premium on a 10% OTM put is 0.2–0.75% of the price, below the spread.

---

## 3. Models and method

**Black-Scholes benchmark (`BS_slice`)**
- One volatility per expiry per day, chosen to fit that expiry's prices as well as possible.
- This gives BS about 5 numbers per day, so it is a fair, not a straw-man, benchmark.
- SEB itself uses a full volatility surface. The comparison therefore shows how much of the market's price pattern a jump model explains structurally.

**Merton jump-diffusion (`Merton_dag`)**
- Stock = normal random movement (σ) plus sudden jumps. Jumps arrive λ times per year on average; jump size (in log terms) is normal with mean μ_J and spread σ_J.
- One set of 4 numbers (σ, λ, μ_J, σ_J) per day, fitted to all expiries at once.
- Priced with Merton's exact series formula (a weighted sum of BS prices), not Monte Carlo: fast and noise-free.
- **Bug found and fixed** in the original team code: the per-jump drift term used `n·μ_J/T` instead of `n·ln(1+κ)/T`. The old version broke put-call parity by up to 1.04 SEK and was 0.2–0.4 SEK off Monte Carlo. The fixed version matches Monte Carlo within the confidence interval and satisfies parity to 1e-14.

**How fit is measured**
- Every error is in **half-spreads**: z = (model − mid) / (half the bid-ask spread). |z| ≤ 1 means the model price is inside bid/ask.
- Calibration minimises the sum of z². This way cheap, short-dated OTM options count as much as expensive ATM ones. A plain price error would be dominated by expensive options: a 1-vol-point miss on a 6-month ATM option weighs about 20× more than on a 1-month 10%-OTM put.

**Stabilisation**
- Without help, λ jumped around from day to day and often hit its limit of 10.
- Fix: a penalty for moving far from yesterday's parameters (REG_WEIGHT = 3).
- The strength was chosen from a trade-off curve: the largest value costing under 1 percentage point of inside-spread fit. Fit dropped only 86.2 → 85.2% (BOL) and 68.9 → 68.4% (VOLVB). Pricing tomorrow with today's parameters was essentially unchanged at every penalty strength, so the day-to-day jumps were noise.

**Identification check (forward/backward test)**
- The calibration was run forward and backward in time.
- λ and μ_J individually follow different paths depending on direction (correlations −0.08 and 0.12 for BOL), so they are **not separately identified** by the data.
- What the data do pin down: prices, σ_J (correlation 0.80–0.90), σ, and the **share of variance coming from jumps: about 41% (BOL) and 47% (VOLVB)**.
- Conclusion: the data tell how much risk comes from jumps, not whether it is few big jumps or many small ones. Never quote a daily λ.

**Why not Kou (double-exponential jumps)?**
- Test: fit Merton separately per short expiry (≤45 days). Switch to Kou only if (A) many short OTM puts fall outside bid/ask, (B) there is a wing pattern (puts too cheap and calls too expensive at the same time), and (C) there are enough OTM calls to estimate Kou's upside parameters.
- Result: A = 0.0% (BOL) and 0.4% (VOLVB) outside the spread; B = 15% and 37% of expiries show the pattern (threshold 50%). So Merton is sufficient; the spreads are too wide to tell Merton and Kou apart.
- Merton's remaining failures are across maturities (skew fading too fast) and around reports, which Kou would not fix either.

---

## 4. Code and how to run

All code is in `<repo>`:

| File | What it does |
|---|---|
| `seb_merton_analys.py` | Data cleaning, implied vols, BS vs Merton calibration, Kou test, heatmaps, smile examples, parameter plots. With `--full` also the trade-off curve (`lcurve`) and forward/backward test (`fram_bak_test`). |
| `seb_gamma_analys.py` | Long/short gamma: daily delta-hedged P&L (BS vs Merton delta), hold-to-expiry market-maker simulation, bootstrap confidence intervals. |
| `seb_handelsrisk.py` | Report days (priced vs actual move), gap test (overnight shocks), spreads vs jump risk. |
| `kor_allt.ps1` | Runs all three scripts and writes a log. |

Run in PowerShell:
- `cd <repo>`
- `powershell -ExecutionPolicy Bypass -File .\kor_allt.ps1 -Quick` for the main + gamma + trading-risk steps, about 5 minutes.
- Without `-Quick` it also runs the trade-off curve and forward/backward test, about 15 minutes.

Data are read from `data\clean\*_rensad.xlsx`. Results go to `RESULTAT\`.

---

## 5. Results

### 5.1 Pricing: Merton is clearly better

| | BS (one σ per expiry) | Merton (4 numbers per day) |
|---|---|---|
| BOL: share of prices inside bid/ask | 55.3% | **85.2%** |
| VOLVB: share inside bid/ask | 23.5% | **68.4%** |
| BOL: median miss in implied vol | 1.87 vol points | **0.91** |
| VOLVB: median miss in implied vol | 2.39 vol points | **0.87** |
| BOL: RMSE in half-spreads | 1.09 | **0.81** |
| VOLVB: RMSE in half-spreads | 2.05 | **1.28** |

- Merton wins with fewer numbers per day than BS (4 vs about 5).
- Out-of-sample check (yesterday's Merton parameters on today's prices): 82% inside bid/ask for BOL and 65% for VOLVB, only about 3 points below same-day. So it is not overfitting.
- **Where BS fails:** it prices crash protection (OTM puts) too cheap and upside options (OTM calls) too expensive. That is the market's "skew".
  - BOL 1–3 month OTM puts: BS average error −1.28 half-spreads; Merton −0.08.
  - BOL 1–3 month OTM calls: BS +1.12; Merton +0.06.
  - VOLVB 1–3 month OTM puts: BS inside bid/ask about 1%; Merton about 76%.
  - VOLVB 1–3 month OTM calls: BS about 2%; Merton about 62%.
- **Where Merton still fails:**
  - Options with under 1 month left are slightly too cheap. This is worst on expiries spanning a report: Volvo's 17 Oct and both stocks' 18 July expiries were off by 4–6 half-spreads. A Poisson jump cannot know the report date.
  - Long-dated (>6 month) OTM calls are too expensive (VOLVB z about +1.1 to +1.6). Merton's skew fades too fast with maturity (Das & Sundaram 1999).

### 5.2 Long vs short gamma

**Idea:** short gamma is like selling insurance. You collect a small premium (theta) on quiet days and lose big on jump days. Market makers are often short gamma because clients buy options from them.

**Setup:** each option is held long or short and delta-hedged daily with the stock. Delta comes from BS (the option's own implied vol, as a desk would do) or from Merton.

Daily results (short gamma, per option per day, mid to mid):
- Delta hedging cuts daily P&L swings about 4× (std 1.79 → 0.48 SEK for BOL; 1.12 → 0.28 SEK for VOLVB).
- Quiet days: tiny premium. VOLVB +0.027 SEK per option per day, which is significant only when the 4 biggest-move days are excluded (2.5σ cutoff). BOL: not significant.
- Report days: short gamma earned, BOL +1.38 and VOLVB +0.28 SEK per option. This came mainly from implied vol falling after the report.
- Unexpected jump: on 23 July 2025, Volvo rose +6.9% with no report, and short gamma lost on every option, −1.77 SEK average (about 4 half-spreads).
- Stock trading costs for the hedge are negligible: about 0.003 SEK per option per day.
- **Merton delta does not hedge better.** It was 29% worse on VOLVB quiet days (significant: interval −49% to −9%) and no different for BOL. Merton helps pricing, not delta hedging.

Hold to expiry (market maker sells at the ask and hedges daily until expiry):
- BOL: +0.93 SEK per option after stock costs (significant: interval +0.19 to +1.46). Most of it is the bid-ask spread (half-spread about 0.75 SEK).
- VOLVB: +0.25 SEK, not significant.

**Caveat:** only 8 months, 2 stocks, 2 reports each and a handful of jump days. Positions overlap, so the effective sample is small. Read the gamma results as case studies.

### 5.3 Report days: priced move vs actual move

The priced report move comes from option prices the day before, using two expiries that both span the report (term-structure method, Dubinsky, Johannes, Kaeck & Seeger 2019). A second method, the implied-vol drop across the report, gave similar numbers (e.g. BOL July 5.7% vs 5.7%).

| Report | Priced expected size of move | Actual move | Short gamma that day | Of which: move / vol drop |
|---|---|---|---|---|
| BOL 18 Jul 2025 | 4.5% | −2.4% | +1.08 SEK | −0.08 / +1.22 |
| BOL 22 Oct 2025 | 4.1% | +0.6% | +1.61 SEK | +0.12 / +1.56 |
| VOLVB 17 Jul 2025 | 3.5% | −2.1% | +0.51 SEK | −0.08 / +0.60 |
| VOLVB 17 Oct 2025 | 4.3% | −7.0% | +0.00 SEK | −1.43 / +1.39 |

- In 3 of 4 reports the move was smaller than priced: the insurance was too expensive and short gamma won.
- Implied vol falls 5–6 vol points after a report (the "vol crush"). This cushions the insurance seller, who only loses when the move is far bigger than priced. Even Volvo's −7.0% only broke even.
- For the desk: before each report, compare the priced move with the stock's history of report-day moves.

### 5.4 Gap test: what an overnight jump costs

A delta-hedged short position is shocked by an instant move. Average loss across all OTM options:

| Jump | BOL loss (SEK/option) | BOL with vol spike (+5 vol points) | VOLVB loss | VOLVB with vol spike |
|---|---|---|---|---|
| −10% | −4.47 (about 39 days of theta) | −6.54 | −3.71 (about 58 days of theta) | −5.28 |
| −5% | −1.16 | −3.51 | −0.95 | −2.73 |
| +5% | −1.11 | – | −0.91 | – |
| +10% | −4.14 | – | −3.35 | – |

- A 10% overnight gap wipes out roughly 1–2 months of time-value income. Losses grow roughly with the square of the jump.
- **Validated against real big days (no report):** predicted vs actual loss within 0–0.3 SEK. For example: VOLVB 23 Jul +6.9%, predicted −1.59 vs actual −1.77; BOL 3 Sep +4.9%, −1.02 vs −1.01; VOLVB 4 Dec +3.7%, −0.58 vs −0.57.
- On report days the actual loss was smaller than predicted, because implied vol fell after the report.
- **Merton's view:** 32% (BOL) and 44% (VOLVB) of the daily time value is payment for jump risk; for OTM puts it is about 40–50%.

### 5.5 Spreads vs jump risk

Spreads are measured in vol points (IV at ask − IV at bid).
- Spreads are mostly mechanical: cheap options and options with little sensitivity to volatility (low vega) have wide spreads in vol terms. Vega and option price explain 74–81% of the variation.
- Around reports, spreads are about 7–9% wider (significant for both stocks).
- Jump risk: BOL spreads are about 4% wider per standard deviation of Merton jump risk (significant). VOLVB shows no effect.
- OTM puts are barely wider than OTM calls, even though Merton says they carry 2–3× more jump risk.
- **Conclusion:** the market charges for jump risk mainly in the price level (the skew), not in the bid-ask width.
- Same options before vs after a report: spread in SEK roughly unchanged for BOL (1.80 → 1.73; 1.74 → 1.78) and narrower after Volvo's July report (1.11 → 0.77). BOL's apparent vol-point widening after the October report is mechanical, because vol fell and vega changed.

---

## 6. The plots (in RESULTAT)

**`BOL_felkarta.png` / `VOLVB_felkarta.png`: error heatmap, BS (left) vs Merton (right)**
- Rows: time to expiry (<1M, 1–3M, 3–6M, >6M).
- Columns: strike relative to forward (K/F), from deep OTM puts (crash protection) on the left to deep OTM calls on the right.
- Each box: average error in half-spreads (0 = at mid, ±1 = at bid/ask). Blue = model too cheap, red = too expensive. n = number of quotes.
- BS: blue on the left, red on the right, which is the skew BS cannot produce.
- Merton: mostly pale, so the skew error is gone. Left over: a slightly blue top row (short options around reports) and slightly red bottom right (long-dated upside options).
- Ignore boxes with very small n (e.g. n = 1, n = 24).

**`BOL_smile_exempel.png` / `VOLVB_smile_exempel.png`: market vs models on three days (shortest expiry ≥14 days)**
- x: K/F. y: implied vol. Gray bars: market bid-ask range per option; tall = wide spread.
- Orange: BS, flat by construction. Blue: Merton, sloped.
- BOL: 26 May, Merton follows the market and BS doesn't. 1 Sept, spreads so wide both fit. 2 Dec, Merton too steep and below the market on the upside, BS better; this is Merton's one-set-for-all-expiries compromise.
- Planned improvement: choose example days at the 25th/50th/75th percentile of Merton's fit, so they are representative, not cherry-picked.

**`BOL_parametrar.png` / `VOLVB_parametrar.png` (backup): Merton's daily settings**
- Diffusion vol σ, jump share of variance (about 40–47%), jump-size spread σ_J, total vol, share inside bid/ask per day, stock price.
- λ and μ_J are deliberately not plotted, because they are not identified.
- Inside bid/ask dips in July around the reports.

**`BOL_gamma_kumulativ.png` / `VOLVB_gamma_kumulativ.png`: short gamma book over time**
- Cumulative P&L (SEK) of a book short one of every OTM option, delta-hedged daily, excluding spread income.
- Orange vertical lines: report days. Dotted: big-move days.
- Slow climb (premium) and sharp drops (jumps). VOLVB: about +200 by early July, a big drop on 23 July, ending about +125.

**`Rapport_prisad_vs_faktisk.png`: report days**
- Blue: move priced by options the day before. Orange: actual move (sign labelled).
- 3 of 4 smaller than priced; Volvo 17 Oct (−7.0% vs 4.3%) larger.

**`BOL_gap_test.png` / `VOLVB_gap_test.png`: overnight jump test**
- Left: loss (in days of theta) vs instant jump size for 4 option groups (<1M and 1–3M, OTM puts and OTM calls). U-shaped; a 10% jump costs about 35–100 days of premium. Puts hurt most on falls, calls on rises.
- Right: predicted vs actual loss on real big days. Blue (normal big days) lies on the dashed line, so the test works. Orange (report days) lies above it: smaller loss thanks to the vol drop.

**`Lkurva.png` (backup): stabilisation trade-off**
- Left: next-day fit vs penalty strength (flat, so the penalty costs nothing).
- Right: daily jumps in log λ fall with penalty strength. Chosen value: 3.

---

## 7. What was removed and why

- BS with historical vol and Merton with guessed parameters: straw men.
- Merton fitted per expiry as a competitor: 99% fit from 4 numbers on about 10 prices is overfitting. It is kept only for the Kou test.
- λ and μ_J values: not identified.
- Hold-to-expiry bar chart and long-gamma market-maker numbers: not significant or confusing.
- Spread event-study line chart: too noisy.
- Old files: `main.py` (the earlier Monte Carlo/QMC single-option script), old logs, the zip duplicate, `__pycache__`.

---

## 8. Limitations

- 8 months, 2 stocks, 4 reports, a handful of jump days: event results are case studies.
- Daily closing data only; SEB works intraday.
- Constant-parameter jump models cannot fit the whole maturity structure; stochastic volatility with jumps (Bates) is the next step, done in `jump_test/` (see its README).
- American exercise ignored (per SEB); negligible for OTM options.
- Book in the gamma test is "one of every option", not SEB's real client-driven book.

---

## 9. Presentation plan (Friday)

**7 slides:**
1. SEB's question + data.
2. Where BS goes wrong (`felkarta`).
3. Merton fixes most of it: 85% vs 55% (BOL), 68% vs 24% (VOLVB) inside bid/ask (`smile_exempel`).
4. Short gamma = selling insurance: slow earnings, jump losses (`gamma_kumulativ`).
5. Report days: 3 of 4 moved less than priced (`Rapport_prisad_vs_faktisk`).
6. A 10% overnight jump costs 40–60 days of income, validated on real days (`gap_test`).
7. Takeaways for the desk + limits.

**Takeaways for the desk:**
1. Use a jump model for pricing, marking thin strikes and sizing jump-risk reserves, not for delta hedging.
2. Before a report, compare the priced move with the stock's history of report moves.
3. The gap test turns jump risk into "days of income lost", a way to judge whether spreads and reserves are big enough.
4. Jump risk is charged in the price (skew), not in the spread width.

**Backup slides:** pricer validation and bug fix, Kou decision, calibration stability (only the jump share of variance is reliable), gamma confidence intervals, spread regression.

**Likely questions and short answers:**
- *"We price with a vol surface, why compare to flat BS?"* BS here gets its own vol for every expiry. The point is how much of the market's pattern a jump model explains with 4 numbers per day, which is useful for marking thin strikes and understanding where the skew comes from.
- *"Why Merton and not Kou or Bates?"* Kou: a formal test showed no need; spreads are too wide to tell them apart. Bates would fix the maturity structure but needs more data and time.
- *"How stable are the parameters?"* Prices and the jump share of variance are stable. λ and μ_J individually are not identified, so we don't quote them.
- *"Why does Merton hedge worse?"* Merton's delta builds in protection against jumps, so on quiet days it over-hedges. Jumps cannot be delta-hedged anyway.
- *"Is it significant?"* Pricing results yes, clearly. Gamma and event results are case studies: few events, overlapping positions.
- *"What about early exercise?"* Per SEB's note, no dividends in the period; binomial check shows under 1% of price for OTM puts.

**Questions to ask SEB:** What does your book usually look like (net short gamma, which strikes)? How do you set spreads before reports? Would you use a jump model for marking?

**Team roles (group contract):** one member builds and leads the presentation; one presents the models; one presents results and handles technical questions; one gives the introduction and conclusions; one the desk takeaways; one keeps the report consistent.

---

## 10. Glossary

- **Option, call, put:** the right to buy (call) or sell (put) a stock at the strike price K by the expiry date.
- **OTM / ATM:** out of the money (would pay nothing if exercised today) / at the money (strike near today's price). Deep OTM puts act as crash protection.
- **K/F:** strike divided by forward price; below 1 = downside strikes, above 1 = upside strikes.
- **Implied volatility (IV):** the volatility that makes BS match a price. Traders quote options in IV. 1 vol point = 1 percentage point of IV.
- **Skew / smile:** the pattern of IV across strikes. For stocks, downside strikes have higher IV (crash fear).
- **Bid / ask / spread / half-spread:** buy price / sell price / difference / half the difference, the market maker's margin on each side.
- **Delta:** how much the option price moves per 1 SEK stock move. Delta hedging means holding that many shares against the option.
- **Gamma:** how fast delta changes. Long gamma = owning options (gains from big moves); short gamma = having sold options (loses on big moves).
- **Theta:** time value lost per day by an option owner, i.e. earned per day by a seller: the "insurance premium".
- **Vega:** option price change per 1 vol point.
- **Vol crush:** the drop in implied vol right after a scheduled event like a report.
- **λ, μ_J, σ_J:** Merton's jump frequency per year, average jump size and jump-size spread (in log terms).

---

## 11. Sources

- Merton (1976), "Option pricing when underlying stock returns are discontinuous", Journal of Financial Economics.
- Kou (2002), "A jump-diffusion model for option pricing", Management Science.
- Cont & Tankov (2004), "Non-parametric calibration of jump-diffusion option pricing models", Journal of Computational Finance: least-squares calibration of jump models is ill-posed.
- Das & Sundaram (1999), "Of smiles and smirks: a term-structure perspective", JFQA: jump models' skew fades too fast with maturity.
- Dubinsky, Johannes, Kaeck & Seeger (2019), "Option pricing of earnings announcement risks", Review of Financial Studies: the report-move estimator.
- Lee (2004), "The moment formula for implied volatility at extreme strikes", Mathematical Finance.
- Nasdaq: Stock options on Swedish shares are American-style with physical delivery.
- Report dates: Volvo Q2 17 Jul 2025, Q3 17 Oct 2025 (volvogroup.com); Boliden Q2 18 Jul 2025, Q3 22 Oct 2025 (boliden.com).
