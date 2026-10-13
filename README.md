# Jump risk in option pricing: Black-Scholes, Merton and Bates on Swedish equity options

Student project for SEB's equity-derivatives desk (2025-2026). The question from SEB:

> Can jump risk improve the understanding and pricing of the parts of the option market where Black-Scholes deviates most from
> market prices?

The data are daily closing bid/ask quotes for Boliden (BOL) and Volvo B (VOLVB) options and the two stocks, 2 May to 30 Dec 2025
(167 trading days). Every model is fitted once per day to all expiries, with errors measured in half-spreads (|z| <= 1 means the
model price is inside bid/ask).

## Main results

| | Boliden | Volvo B |
|---|---|---|
| Options priced inside bid/ask: Black-Scholes (one vol per expiry) | 55 % | 24 % |
| Options priced inside bid/ask: Merton jump-diffusion | 85 % | 68 % |
| Share of variance from jumps (Merton) | about 41 % | about 47 % |
| Skew-decay exponent across maturities: market | -0.36 | -0.41 |
| Skew-decay exponent: Merton | -0.60 | -0.73 |
| Skew-decay exponent: Bates (Heston + Merton jumps) | -0.46 | -0.43 |
| Next-day pricing error, RMSE in half-spreads: Merton / Bates | 0.87 / 0.60 | 1.41 / 0.90 |

- Merton fixes most of what Black-Scholes misses, but its skew fades too fast with maturity. Bates keeps the market's skew decay,
  mostly through stochastic variance (Heston alone gives 80-90 % of the improvement); the jumps add a small gain up to about
  5 trading days.
- Out of sample, Bates beats Merton for about two weeks (10 trading days); after about 20 days no model with old parameters is
  usable, so parameters are re-fitted daily.
- Report days: 3 of the 4 reports moved less than the options priced. An overnight 10 % jump costs a short-gamma book roughly
  40-60 days of option income.
- Kou (double-exponential jumps) behaves like Merton; Merton's lambda and mu_J are not separately identified, only the jump share
  of variance is.

Full write-ups: [docs/project_summary.md](docs/project_summary.md) (main pipeline, gamma, report days, presentation notes) and
[jump_test/README.md](jump_test/README.md) (Kou, Heston and Bates, the overnight out-of-sample run, figures).

## Repository layout

```
.
├── seb_merton_analys.py   main pipeline: cleaning, implied vols, Black-Scholes vs Merton, Kou test, heatmaps, smiles
├── seb_gamma_analys.py    long vs short gamma: delta-hedged P&L, market-maker simulation, bootstrap CIs
├── seb_handelsrisk.py     report days, overnight gap test, spreads vs jump risk
├── kor_allt.ps1           runs the three scripts above and writes a log (Windows PowerShell)
├── RESULTAT/              results of the main pipeline (Excel workbooks and figures)
├── data/
│   ├── raw/               SEB's raw export (semicolon CSV, options and stock)
│   └── clean/             the same quotes in cleaned form (Excel); what the scripts read
├── jump_test/             Merton vs Kou, Heston and Bates; overnight out-of-sample run; 3D page and slide figure
└── docs/                  project summary and SEB's data note
```

## How to run

Python 3.10 or newer.

```
pip install -r requirements.txt

# main pipeline: about 5 minutes; add --full to seb_merton_analys.py for the REG_WEIGHT trade-off curve and
# the forward/backward identification test (about 15 minutes more)
python seb_merton_analys.py
python seb_gamma_analys.py
python seb_handelsrisk.py
# or, on Windows: powershell -ExecutionPolicy Bypass -File .\kor_allt.ps1 [-Quick]

# jump_test (needs the main pipeline's results for its tests)
cd jump_test
python run_all.py --jobs 2         # tests, calibration on all days, analysis, figures: about 20 minutes
python overnight.py                # out-of-sample run on all CPU cores: about 1 hour on 5 worker processes
```

`jump_test` imports `seb_merton_analys.py` from the folder above, so keep the layout. Caches (`jump_test/cache/`, the overnight
checkpoints) are rebuilt on the first run and are not in the repository.

## Data

The quotes were provided by SEB for this project; see [data/README.md](data/README.md). Keep this repository private unless SEB has
agreed to the data being published.
