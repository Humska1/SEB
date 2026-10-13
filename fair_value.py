"""
fair_value.py  -  fair value of Boliden and Volvo B options with Black-Scholes, Merton and Bates
=================================================================================================

For every trading day (2 May - 30 Dec 2025) and both stocks:

  1. Read the closing bid/ask quotes in data/ and keep the usable out-of-the-money options.
  2. Market price = mid = (bid + ask) / 2. A model price counts as right if it lies between bid and ask, so the error
     is measured in half-spreads:   z = (model - mid) / ((ask - bid) / 2),   |z| <= 1  <=>  inside bid/ask.
  3. Fit three models to all options of the day by least squares on z:
        Black-Scholes  one volatility per expiry (the baseline; it cannot produce a skew)
        Merton         Black-Scholes + randomly arriving, normally distributed jumps          (4 parameters per day)
        Bates          stochastic variance (Heston) + the same jumps as Merton               (8 parameters per day)
  4. Write each option's fair value under every model next to its bid/ask, the daily parameters, summary tables (all
     options, by maturity) to csv/ and three charts to results/. The analysis by moneyness is in moneyness.py.

    python fair_value.py                 (about 10 minutes on a 6-core laptop; --jobs N sets the number of processes)
    python fair_value.py --report-only   (only redo the tables and charts from the saved results, seconds)
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import ndtr

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RESULTS_DIR = os.path.join(HERE, "results")      # charts (.png)
CSV_DIR = os.path.join(HERE, "csv")              # tables (.csv)
STOCKS = {"BOL": "Boliden", "VOLVB": "Volvo B"}

R = 0.02                 # risk-free rate (put-call parity in the data implies 1.6-2.0 %)
Q = 0.0                  # dividend yield: no dividend in the period
DAYS_PER_YEAR = 365.25
MIN_DAYS = 5             # drop options with fewer days left (noisy)
MIN_BID = 0.01           # a bid of 0.01 means there is no real buyer
MAX_REL_SPREAD = 1.0     # drop quotes whose spread is larger than the mid
SPREAD_FLOOR = 0.05      # smallest half-spread used in z (SEK)
N_FIT, N_REPORT = 384, 2048   # COS terms: fast grid inside the optimiser, fine grid for reported prices
MAT_BINS = [0, 30, 60, 90, 150, 230, 400]
MAT_LABELS = ["0-30", "30-60", "60-90", "90-150", "150-230", "230-400"]


# =============================================================================
# 1. Black-Scholes and implied volatility
# =============================================================================
def bs_price(S, K, T, sigma, is_call):
    T, sigma = np.maximum(T, 1e-8), np.maximum(sigma, 1e-8)
    d1 = (np.log(S / K) + (R - Q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * np.exp(-Q * T) * ndtr(d1) - K * np.exp(-R * T) * ndtr(d2)
    put = K * np.exp(-R * T) * ndtr(-d2) - S * np.exp(-Q * T) * ndtr(-d1)
    return np.where(is_call, call, put)


def implied_vol(price, S, K, T, is_call, lo=1e-4, hi=5.0, n_iter=80):
    """Black-Scholes implied volatility by bisection (vectorised); NaN if the price is outside the possible range."""
    price = np.asarray(price, float)
    ok = (price > bs_price(S, K, T, lo, is_call)) & (price < bs_price(S, K, T, hi, is_call))
    a, b = np.full(price.shape, lo), np.full(price.shape, hi)
    for _ in range(n_iter):
        m = 0.5 * (a + b)
        too_high = bs_price(S, K, T, m, is_call) > price
        a, b = np.where(too_high, a, m), np.where(too_high, m, b)
    return np.where(ok, 0.5 * (a + b), np.nan)


# =============================================================================
# 2. Data
# =============================================================================
def load_quotes(stock):
    """Usable out-of-the-money option quotes of one stock, with the stock price, forward, mid, half-spread and IVs."""
    df = pd.read_csv(os.path.join(DATA_DIR, f"{stock}.csv"), parse_dates=["Date", "Expiry"])
    st = df[df["Type"] == "Stock"]
    spot = pd.Series(((st["Bid"] + st["Ask"]) / 2).values, index=st["Date"], name="S")
    o = df[df["Type"] == "Option"].join(spot, on="Date", how="inner")
    o["Days"] = (o["Expiry"] - o["Date"]).dt.days
    o["T"] = o["Days"] / DAYS_PER_YEAR
    o["IsCall"] = o["CallPut"].str.upper().str.startswith("C")
    o["F"] = o["S"] * np.exp((R - Q) * o["T"])
    o["k"] = np.log(o["Strike"] / o["F"])                     # log-moneyness: < 0 below the forward
    o = o.dropna(subset=["Bid", "Ask"])
    o = o[(o["Bid"] > MIN_BID) & (o["Ask"] >= o["Bid"]) & (o["Days"] >= MIN_DAYS)]
    o = o[np.where(o["IsCall"], o["k"] >= 0, o["k"] < 0)]     # out of the money: puts below F, calls above
    o["Mid"] = (o["Bid"] + o["Ask"]) / 2
    o = o[(o["Ask"] - o["Bid"]) <= MAX_REL_SPREAD * o["Mid"]].copy()
    o["HalfSpread"] = np.maximum((o["Ask"] - o["Bid"]) / 2, SPREAD_FLOOR)
    for side in ("Bid", "Mid", "Ask"):
        o[f"IV_{side}"] = implied_vol(o[side].values, o["S"].values, o["Strike"].values, o["T"].values, o["IsCall"].values)
    o = o.dropna(subset=["IV_Mid"])
    cols = ["Date", "Expiry", "Days", "T", "CallPut", "IsCall", "Strike", "S", "F", "k", "Bid", "Ask", "Mid", "HalfSpread",
            "IV_Bid", "IV_Mid", "IV_Ask"]
    return o.sort_values(["Date", "Expiry", "Strike"], kind="stable")[cols].reset_index(drop=True)


def arrays(d):
    return d["S"].values, d["Strike"].values, d["T"].values, d["IsCall"].values, d["Mid"].values, d["HalfSpread"].values


# =============================================================================
# 3. Merton and Bates: characteristic functions and the COS pricer
#    X = ln(S_T / F_T) with E[exp(X)] = 1; logcf(u) = ln E[exp(iuX)]
# =============================================================================
def jump_term(u, T, lam, mu_j, sigma_j):
    """Merton jumps: Poisson rate lam, log-jump ~ N(mu_j, sigma_j^2), compensated so the forward is unchanged."""
    kappa = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0
    return T * lam * (np.exp(1j * u * mu_j - 0.5 * sigma_j ** 2 * u * u) - 1.0 - 1j * u * kappa)


def logcf_merton(u, T, th):
    sigma, lam, mu_j, sigma_j = th
    return -0.5 * sigma ** 2 * T * (1j * u + u * u) + jump_term(u, T, lam, mu_j, sigma_j)


def logcf_bates(u, T, th):
    """Heston variance dv = kappa (theta - v) dt + xi sqrt(v) dW, corr(dW, dW_S) = rho ('little trap' form) + Merton jumps."""
    v0, kappa, theta, xi, rho, lam, mu_j, sigma_j = th
    iu = 1j * u
    b = kappa - rho * xi * iu
    d = np.sqrt(b * b + xi * xi * (iu + u * u))
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    C = kappa * theta / xi ** 2 * ((b - d) * T - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
    D = (b - d) / xi ** 2 * (1.0 - e) / (1.0 - g * e)
    return C + D * v0 + jump_term(u, T, lam, mu_j, sigma_j)


@dataclass
class Model:
    name: str
    params: tuple
    lb: np.ndarray
    ub: np.ndarray
    logcf: Callable
    std: Callable            # rough standard deviation of X over T (only sets the COS range)


MERTON = Model("Merton", ("sigma", "lam", "mu_j", "sigma_j"),
               np.array([0.01, 0.001, -0.60, 0.01]), np.array([1.50, 10.0, 0.30, 0.80]), logcf_merton,
               lambda T, th: np.sqrt(T * (th[0] ** 2 + th[1] * (th[2] ** 2 + th[3] ** 2))))
BATES = Model("Bates", ("v0", "kappa", "theta", "xi", "rho", "lam", "mu_j", "sigma_j"),
              np.array([1e-4, 0.1, 1e-4, 0.05, -0.99, 0.001, -0.60, 0.01]),
              np.array([1.00, 20.0, 1.00, 3.00, 0.99, 10.0, 0.30, 0.80]), logcf_bates,
              lambda T, th: np.sqrt(T * (max(th[0], th[2]) + th[5] * (th[6] ** 2 + th[7] ** 2))))


def cos_price(model, th, S, K, T, is_call, N=N_REPORT, L=12.0):
    """
    European option prices with the COS method (Fang & Oosterlee 2008). The integration range [a, b] is centred on the
    mean of X and is L standard deviations wide, using cumulants read off the characteristic function.
    """
    S, K, T = (np.atleast_1d(np.asarray(x, float)) for x in (S, K, T))
    S, K, T, is_call = np.broadcast_arrays(S, K, T, np.atleast_1d(np.asarray(is_call, bool)))
    k = np.log(K / (S * np.exp((R - Q) * T)))
    Tu, inv = np.unique(T, return_inverse=True)
    # cumulants c1, c2, c4 by finite differences of logcf at u = h, 2h
    s = np.maximum(model.std(Tu, th), 1e-6)
    h = 0.1 / s
    l1, l2 = model.logcf(h, Tu, th), model.logcf(2 * h, Tu, th)
    c4 = 2.0 * (l2.real - 4.0 * l1.real) / h ** 4
    c2 = -2.0 * l1.real / h ** 2 + c4 * h ** 2 / 12.0
    c1 = (8.0 * l1.imag - l2.imag) / (6.0 * h)
    width = np.maximum(L * np.sqrt(np.maximum(c2 + np.sqrt(np.maximum(c4, 0.0)), 0.0)), 0.6 * L * s)
    width = np.maximum(width, np.abs(c1))
    a_u, b_u = c1 - width, c1 + width
    n = np.arange(N, dtype=float)
    w_u = n[None, :] * np.pi / (b_u - a_u)[:, None]
    Fphi = (np.exp(model.logcf(w_u, Tu[:, None], th)) * np.exp(-1j * w_u * a_u[:, None])).real
    Fphi[:, 0] *= 0.5
    sgn = np.where(n % 2 == 0, 1.0, -1.0)[None, :]
    out = np.empty(len(S))
    for call in (True, False):
        j = np.flatnonzero(is_call == call)
        if not len(j):
            continue
        e = inv[j]
        a, b, w, kk = a_u[e][:, None], b_u[e][:, None], w_u[e], k[j][:, None]
        kc = np.clip(kk, a, b)
        cs, sn = np.cos(w * (kc - a)), np.sin(w * (kc - a))
        with np.errstate(divide="ignore", invalid="ignore"):
            sin_over_w = sn / w
        if call:      # payoff (e^X - e^k)^+ on [k, b]
            chi = (sgn * np.exp(b) - np.exp(kc) * (cs + w * sn)) / (1.0 + w * w)
            psi = -sin_over_w
            psi[:, 0] = (b - kc)[:, 0]
            V = chi - np.exp(kk) * psi
        else:         # payoff (e^k - e^X)^+ on [a, k]
            chi = (np.exp(kc) * (cs + w * sn) - np.exp(a)) / (1.0 + w * w)
            psi = sin_over_w
            psi[:, 0] = (kc - a)[:, 0]
            V = np.exp(kk) * psi - chi
        out[j] = (2.0 / (b - a))[:, 0] * np.sum(Fphi[e] * V, axis=1)
    return S * np.exp(-Q * T) * out


# =============================================================================
# 4. Calibration of one day
# =============================================================================
def merton_starts(atm):
    return [[0.85 * atm, 0.5, -0.10, 0.15], [0.60 * atm, 1.5, -0.05, 0.10], [0.90 * atm, 0.2, -0.25, 0.20]]


def bates_starts(atm, merton_theta):
    v, (s, lam, mu, sj) = atm ** 2, merton_theta
    return [[s ** 2, 2.0, s ** 2, 0.3, -0.3, lam, mu, sj],          # Bates is Merton when xi -> 0: start next to it
            [v, 2.0, v, 0.5, -0.6, 0.3, -0.08, 0.10],
            [(0.8 * atm) ** 2, 1.0, v, 0.8, -0.8, 0.8, -0.10, 0.08],
            [v, 5.0, (0.9 * atm) ** 2, 1.0, -0.3, 0.1, -0.15, 0.15]]


def fit(model, d, starts):
    """Least squares on z over all options of the day; the best of several start points is kept."""
    S, K, T, ic, mid, w = arrays(d)

    def resid(th):
        z = (cos_price(model, th, S, K, T, ic, N=N_FIT) - mid) / w
        return np.where(np.isfinite(z), z, 1e3)

    best = None
    for x0 in starts:
        x0 = np.clip(np.asarray(x0, float), model.lb + 1e-6, model.ub - 1e-6)
        try:
            res = least_squares(resid, x0, bounds=(model.lb, model.ub), method="trf", x_scale="jac", max_nfev=300,
                                diff_step=1e-6)
        except Exception:
            continue
        if best is None or res.cost < best.cost:
            best = res
    return best.x


def bs_vol_per_expiry(d):
    """Black-Scholes baseline: for each expiry the one volatility that minimises the sum of z^2."""
    vols = {}
    for exp, g in d.groupby("Expiry"):
        S, K, T, ic, mid, w = arrays(g)
        x0 = np.nanmedian(g["IV_Mid"])
        vols[exp] = minimize_scalar(lambda s: np.sum(((bs_price(S, K, T, s, ic) - mid) / w) ** 2),
                                    bounds=(max(0.01, 0.3 * x0), min(3.0, 3 * x0)), method="bounded",
                                    options={"xatol": 1e-6}).x
    return d["Expiry"].map(vols).values


def fit_day(d):
    """Fit Black-Scholes, Merton and Bates to one day's quotes. Returns (fair values per option, parameter rows)."""
    S, K, T, ic, mid, w = arrays(d)
    near = d[d["k"].abs() < 0.05]
    atm = float(np.nanmedian((near if len(near) else d)["IV_Mid"]))
    out = d.copy()
    out["BS_Vol"] = bs_vol_per_expiry(d)
    out["BS_Price"] = bs_price(S, K, T, out["BS_Vol"].values, ic)
    th_m = fit(MERTON, d, merton_starts(atm))
    th_b = fit(BATES, d, bates_starts(atm, th_m))
    rows = []
    for model, th in ((MERTON, th_m), (BATES, th_b)):
        out[f"{model.name}_Price"] = cos_price(model, th, S, K, T, ic)
        z = (out[f"{model.name}_Price"].values - mid) / w
        rows.append(dict(Date=d["Date"].iloc[0], Model=model.name, N_options=len(d), RMSE_z=np.sqrt(np.mean(z ** 2)),
                         Inside_pct=100 * np.mean(np.abs(z) <= 1), **dict(zip(model.params, th))))
    for m in ("BS", "Merton", "Bates"):
        out[f"{m}_z"] = (out[f"{m}_Price"] - out["Mid"]) / out["HalfSpread"]
        out[f"{m}_IV"] = implied_vol(out[f"{m}_Price"].values, S, K, T, ic)
    return out.drop(columns="IsCall"), rows


# =============================================================================
# 5. Summaries: fit and skew decay
# =============================================================================
MODELS = {"BS": "Black-Scholes", "Merton": "Merton", "Bates": "Bates"}


def fit_summary(fv, by=()):
    rows = []
    for key, g in (fv.groupby(list(by), observed=True) if by else [((), fv)]):
        for m, label in MODELS.items():
            z = g[f"{m}_z"].dropna()
            iv_err = 100 * (g[f"{m}_IV"] - g["IV_Mid"])            # > 0: model too expensive, < 0: too cheap
            inside = (g[f"{m}_Price"] >= g["Bid"]) & (g[f"{m}_Price"] <= g["Ask"])
            rows.append({**dict(zip(by, key if isinstance(key, tuple) else (key,))), "Model": label, "Options": len(z),
                         "RMSE_z": np.sqrt(np.mean(z ** 2)), "Inside_bid_ask_pct": 100 * inside.mean(),
                         "Mean_IV_error_volpts": iv_err.mean(), "Median_abs_IV_error_volpts": iv_err.abs().median()})
    return pd.DataFrame(rows)


def skew_table(fv):
    """
    Skew of each (date, expiry): slope of implied vol against log-moneyness, |k| < 0.2, at least 5 quotes on both sides
    of the forward, weighted by 1 / half-spread in vol units. Market (IV_Mid), Merton and Bates.
    """
    rows = []
    for (dt, ex), g in fv.groupby(["Date", "Expiry"]):
        g = g[g["k"].abs() < 0.2]
        if len(g) < 5 or g["k"].min() > -0.03 or g["k"].max() < 0.03:
            continue
        w = 1.0 / np.maximum((g["IV_Ask"] - g["IV_Bid"]).values / 2, 1e-3)
        row = dict(Date=dt, Expiry=ex, Days=int(g["Days"].iloc[0]), T=float(g["T"].iloc[0]))
        for name, col in (("Market", "IV_Mid"), ("Merton", "Merton_IV"), ("Bates", "Bates_IV")):
            ok = np.isfinite(g[col].values)
            row[name] = np.polyfit(g["k"].values[ok], g[col].values[ok], 1, w=w[ok])[0] if ok.sum() >= 5 else np.nan
        rows.append(row)
    sk = pd.DataFrame(rows)
    return sk[(sk["Market"] < 0) & sk[["Merton", "Bates"]].notna().all(axis=1)].reset_index(drop=True)


def skew_decay(sk, n_boot=1000, seed=0):
    """Exponent b in |skew| ~ T^b (OLS of ln|skew| on ln T), 95 % CI by resampling dates."""
    x = np.log(sk["T"].values)
    ys = {s: np.log(-np.clip(sk[s].values, None, -1e-4)) for s in ("Market", "Merton", "Bates")}
    slope = lambda i, y: np.polyfit(x[i], y[i], 1)[0]
    groups = [np.flatnonzero((sk["Date"] == dt).values) for dt in sk["Date"].unique()]
    rng = np.random.default_rng(seed)
    boot = {s: [] for s in ys}
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        for s, y in ys.items():
            boot[s].append(slope(idx, y))
    allidx = np.arange(len(x))
    return pd.DataFrame([dict(Series=s, Exponent=slope(allidx, y), CI_low=np.percentile(boot[s], 2.5),
                              CI_high=np.percentile(boot[s], 97.5), Pairs=len(x)) for s, y in ys.items()])


JUMP_EDGES = np.arange(-70, 42, 2)          # jump sizes in % (bins of 2 percentage points)


def jump_sizes(params):
    """
    Distribution of the size of ONE jump, in % of the stock price, averaged over all days. On each day the log-jump is
    N(mu_j, sigma_j^2), so the share of jumps in a bin [a, b] is N((ln(1+b) - mu_j)/sigma_j) - N((ln(1+a) - mu_j)/sigma_j).
    Returns (bin table, one summary row per model).
    """
    lo, hi = np.log1p(JUMP_EDGES[:-1] / 100), np.log1p(JUMP_EDGES[1:] / 100)
    bins, rows = {}, []
    for model, g in params.groupby("Model"):
        mu, sj = g["mu_j"].values[:, None], g["sigma_j"].values[:, None]
        bins[model] = 100 * (ndtr((hi - mu) / sj) - ndtr((lo - mu) / sj)).mean(axis=0)
        rows.append(dict(Model=model, Median_jump_pct=100 * np.median(np.expm1(g["mu_j"])),
                         Share_of_jumps_down_pct=100 * ndtr(-g["mu_j"] / g["sigma_j"]).mean(),
                         Median_jumps_per_year=g["lam"].median()))
    table = pd.DataFrame(bins, index=pd.Index((JUMP_EDGES[:-1] + JUMP_EDGES[1:]) / 2, name="Jump_size_pct_bin_mid"))
    return table, pd.DataFrame(rows)


# =============================================================================
# 6. Charts
# =============================================================================
COLOR = {"Black-Scholes": "#898781", "Merton": "#2a78d6", "Bates": "#1baf7a", "Market": "#eb6834"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9)


def chart_inside(by, col, labels, ticklabels, xlabel, title, path):
    """Share of options priced inside bid/ask per bucket (maturity or moneyness), one panel per stock."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    for ax, (stock, name) in zip(axes, STOCKS.items()):
        _style(ax)
        g = by[by["Stock"] == stock]
        for model in MODELS.values():
            gm = g[g["Model"] == model].set_index(col).reindex(labels)
            ax.plot(range(len(labels)), gm["Inside_bid_ask_pct"], color=COLOR[model], lw=2, marker="o", ms=6, label=model)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(ticklabels, fontsize=8.5)
        ax.set_xlabel(xlabel, fontsize=9, color=INK2)
        ax.set_title(name, loc="left", fontsize=11, color=INK)
        ax.set_ylim(0, 102)
    axes[0].set_ylabel("options priced inside bid/ask (%)", fontsize=9, color=INK2)
    axes[0].legend(frameon=False, fontsize=9, loc="lower center")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_jump_sizes(jumps, path):
    """Size of one jump implied by the option prices, Merton vs Bates, averaged over all days."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
    for ax, (stock, name) in zip(axes, STOCKS.items()):
        _style(ax)
        table, summ = jumps[stock]
        summ = summ.set_index("Model")
        x = table.index.values
        for model in ("Merton", "Bates"):
            r = summ.loc[model]
            ax.fill_between(x, table[model], step="mid", color=COLOR[model], alpha=0.18, lw=0)
            ax.step(x, table[model], where="mid", color=COLOR[model], lw=1.8,
                    label=f"{model}: median jump {r['Median_jump_pct']:+.0f} %, {r['Share_of_jumps_down_pct']:.0f} % "
                          f"of jumps down, about {r['Median_jumps_per_year']:.1f} jumps a year")
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_xlim(-60, 30)
        ax.set_xlabel("size of one jump (% of the stock price)", fontsize=9, color=INK2)
        ax.set_title(name, loc="left", fontsize=11, color=INK)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0, -0.2))
    axes[0].set_ylabel("share of jumps (% per 2-point bin)", fontsize=9, color=INK2)
    fig.suptitle("What one jump looks like according to the option prices (average over all days)", x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.text(0.01, 0.905, "Merton needs large crashes to create the skew; Bates only small drops, because its moving "
             "variance creates the rest. The spike near -45 % is days where the average jump sits at its lower limit.",
             ha="left", fontsize=9, color=INK2)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.27, wspace=0.08)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_skew_decay(skews, decay, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, (stock, name) in zip(axes, STOCKS.items()):
        _style(ax)
        sk, ex = skews[stock], decay[decay["Stock"] == stock].set_index("Series")
        ax.scatter(sk["Days"], -sk["Market"], s=7, color=COLOR["Market"], alpha=0.2, lw=0)
        days = np.geomspace(sk["Days"].min(), sk["Days"].max(), 50)
        for s in ("Market", "Merton", "Bates"):
            b, a = np.polyfit(np.log(sk["T"]), np.log(-np.clip(sk[s], None, -1e-4)), 1)
            ax.plot(days, np.exp(a) * (days / DAYS_PER_YEAR) ** b, color=COLOR[s], lw=2.2,
                    ls="--" if s == "Market" else "-", label=f"{s}: {ex.loc[s, 'Exponent']:+.2f}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([7, 14, 30, 60, 100, 200, 330], ["7", "14", "30", "60", "100", "200", "330"])
        ax.set_yticks([0.1, 0.2, 0.5, 1.0], ["0.1", "0.2", "0.5", "1.0"])
        ax.minorticks_off()
        ax.set_xlabel("days to expiry (log scale)", fontsize=9, color=INK2)
        ax.set_ylabel("|skew| = |d implied vol / d log-moneyness| (log scale)", fontsize=9, color=INK2)
        ax.set_title(name, loc="left", fontsize=11, color=INK)
        ax.legend(frameon=False, fontsize=9, title="|skew| ~ T^b, b =", title_fontsize=9)
    fig.suptitle("How fast the skew fades with maturity: market (dots and dashed line) vs Merton and Bates. "
                 "Black-Scholes has no skew.", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# 7. Main
# =============================================================================
def report():
    """All tables and charts, from the saved fair values and parameters in csv/."""
    summaries, by_mat, skews, decay, jumps, jump_rows = [], [], {}, [], {}, []
    for stock in STOCKS:
        fv = pd.read_csv(os.path.join(CSV_DIR, f"fair_values_{stock}.csv"), parse_dates=["Date", "Expiry"])
        params = pd.read_csv(os.path.join(CSV_DIR, f"parameters_{stock}.csv"), parse_dates=["Date"])
        fv["Maturity"] = pd.cut(fv["Days"], MAT_BINS, labels=MAT_LABELS)
        summaries.append(fit_summary(fv).assign(Stock=stock))
        by_mat.append(fit_summary(fv, by=["Maturity"]).assign(Stock=stock))
        skews[stock] = skew_table(fv)
        decay.append(skew_decay(skews[stock]).assign(Stock=stock))
        jumps[stock] = jump_sizes(params)
        jump_rows.append(jumps[stock][1].assign(Stock=stock))

    first = lambda frames: (lambda df: df[["Stock"] + [c for c in df.columns if c != "Stock"]])(
        pd.concat(frames, ignore_index=True))
    out = {"summary": first(summaries), "summary_by_maturity": first(by_mat), "skew_decay": first(decay),
           "jump_sizes": first(jump_rows)}
    for name, df in out.items():
        df.to_csv(os.path.join(CSV_DIR, f"{name}.csv"), index=False, float_format="%.4f")
    chart_inside(out["summary_by_maturity"], "Maturity", MAT_LABELS, MAT_LABELS, "days to expiry",
                 "Share of options each model prices inside the bid/ask spread, by maturity",
                 os.path.join(RESULTS_DIR, "fit_by_maturity.png"))
    chart_skew_decay(skews, out["skew_decay"], os.path.join(RESULTS_DIR, "skew_decay.png"))
    chart_jump_sizes(jumps, os.path.join(RESULTS_DIR, "jump_sizes.png"))

    pd.set_option("display.width", 220)
    print("\nFit (all days): RMSE in half-spreads and share of options priced inside bid/ask")
    print(out["summary"].round(3).to_string(index=False))
    print("\nSkew decay exponent b in |skew| ~ T^b (95 % CI)")
    print(out["skew_decay"].round(3).to_string(index=False))
    print("\nSize of one jump")
    print(out["jump_sizes"].round(2).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="parallel processes")
    ap.add_argument("--max-days", type=int, default=None, help="only the first n trading days (quick test)")
    ap.add_argument("--report-only", action="store_true", help="skip the fitting, redo tables and charts")
    a = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)
    t0 = time.time()
    if not a.report_only:
        for stock, name in STOCKS.items():
            quotes = load_quotes(stock)
            days = [g for _, g in quotes.groupby("Date")][: a.max_days]
            print(f"{name}: {len(quotes)} option quotes on {len(days)} days, fitting with {a.jobs} processes ...", flush=True)
            with ProcessPoolExecutor(max_workers=a.jobs) as ex:
                res = list(ex.map(fit_day, days, chunksize=2))
            pd.concat([r[0] for r in res], ignore_index=True).to_csv(
                os.path.join(CSV_DIR, f"fair_values_{stock}.csv"), index=False, float_format="%.6g")
            pd.DataFrame([row for r in res for row in r[1]]).to_csv(
                os.path.join(CSV_DIR, f"parameters_{stock}.csv"), index=False, float_format="%.6g")
            print(f"  done ({time.time() - t0:.0f} s)", flush=True)
    report()
    print(f"\nDone in {time.time() - t0:.0f} s. Tables in {CSV_DIR}, charts in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
