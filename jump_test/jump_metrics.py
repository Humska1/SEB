"""
jump_metrics.py  -  the three metrics (fit, skew term structure, implied-vol error grid) and the Diebold-Mariano t
==================================================================================================================

Metric 1  fit:        RMSE of z and share of options priced inside bid-ask (|z| <= 1), per model / stock / maturity bucket
Metric 2  skew:       per (date, expiry) slope of implied vol against log-moneyness for the market (IV_Mid) and for each
                      model's implied vols; then ln|skew| regressed on ln T -> decay exponent, bootstrap CI over dates:
                        * quotes with |k| < 0.2 and a valid IV_Mid
                        * at least 5 quotes, and the quotes must span k = 0 (min k <= -0.03 and max k >= +0.03)
                        * weighted least squares, weight = 1 / half-spread in IV units (floor 0.1 vol point)
                        * only (date, expiry) pairs where the market skew is negative enter the log-log regression
                        * model skews are clipped at -1e-4 before the log (the number clipped is reported)
Metric 3  IV error:   mean (model IV - market IV) in vol points on a maturity x log-moneyness grid.
"""

import numpy as np
import pandas as pd

MAT_BINS = [0, 30, 60, 90, 150, 230, 400]
MAT_LABELS = ["0-30", "30-60", "60-90", "90-150", "150-230", "230-400"]
K_BINS = [-0.6, -0.25, -0.15, -0.07, 0, 0.07, 0.15, 0.5]
K_LABELS = ["<-.25", "-.25..-.15", "-.15..-.07", "-.07..0", "0..07", ".07..15", ">.15"]


def add_buckets(opt):
    opt = opt.copy()
    opt["MatBucket"] = pd.cut(opt["Days"], MAT_BINS, labels=MAT_LABELS)
    opt["KBucket"] = pd.cut(opt["LogMoneyness"], K_BINS, labels=K_LABELS)
    return opt


# =============================================================================
# Metric 1: fit
# =============================================================================
def fit_table(opt, models, by=None):
    """RMSE_z, share |z|<=1, share strictly inside bid/ask, IV errors; optionally grouped by `by` (list of columns)."""
    groups = [((), opt)] if not by else opt.groupby(by, observed=True)
    rows = []
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        for m in models:
            z = g[f"{m}_z"].dropna()
            if len(z) == 0:
                continue
            e = g[f"{m}_IVerr"].dropna() * 100
            inside_ba = ((g[f"{m}_Price"] >= g["Bid"]) & (g[f"{m}_Price"] <= g["Ask"])).mean() * 100
            r = dict(zip(by or [], key))
            r.update(Model=m, N=len(z), RMSE_z=np.sqrt(np.mean(z ** 2)), Inside_pct=100 * np.mean(np.abs(z) <= 1),
                     InsideBidAsk_pct=inside_ba, MeanZ=z.mean(), MeanIVerr_vp=e.mean(),
                     RMSE_IV_vp=np.sqrt(np.mean(e ** 2)), MedianAbsIV_vp=np.median(np.abs(e)))
            rows.append(r)
    return pd.DataFrame(rows)


# =============================================================================
# Metric 2: skew term structure
# =============================================================================
def skew_table(opt, series, kmax=0.2, min_n=5, span=0.03):
    """
    series: dict  name -> column holding that series' implied vols (e.g. {"Market": "IV_Mid", "Merton": "Merton_IV"}).
    Returns one row per (date, expiry) with the skew of every series (NaN if < min_n valid quotes).
    """
    rows = []
    for (dt, ex), g in opt.groupby(["Date", "Expiry"]):
        g = g[(g["LogMoneyness"].abs() < kmax) & g["IV_Mid"].notna()]
        if len(g) < min_n or g["LogMoneyness"].min() > -span or g["LogMoneyness"].max() < span:
            continue
        k = g["LogMoneyness"].values
        w = 1.0 / np.maximum((g["IV_Ask"] - g["IV_Bid"]).values / 2, 1e-3)
        row = dict(Date=dt, Expiry=ex, Days=int(g["Days"].iloc[0]), T=float(g["T"].iloc[0]), n=len(g))
        for name, col in series.items():
            y = g[col].values
            ok = np.isfinite(y)
            row[name] = np.polyfit(k[ok], y[ok], 1, w=w[ok])[0] if ok.sum() >= min_n else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _ols_slope(x, y):
    xm, ym = x.mean(), y.mean()
    return np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2)


def _logskew(s):
    return np.log(-np.clip(s, None, -1e-4))


SUBSETS = {"all expiries (>=5d)": lambda d: np.ones(len(d), bool),
           ">=20 days": lambda d: d["Days"].values >= 20,
           "20-200 days": (lambda d: (d["Days"].values >= 20) & (d["Days"].values <= 200))}


def decay_exponents(sk, names, n_boot=1000, seed=0, subsets=SUBSETS):
    """
    Decay exponent b in |skew| ~ T^b for every series in `names` (first entry must be the market), with a
    cluster bootstrap over dates. Also returns bootstrap CIs for the differences that matter for the hypotheses.
    Only (date, expiry) pairs with a negative market skew and a valid skew in *every* series are used.
    """
    market = names[0]
    rows, diffs = [], []
    for sname, sel in subsets.items():
        d = sk[sel(sk)]
        d = d[(d[market] < 0) & d[names].notna().all(axis=1)].reset_index(drop=True)
        if len(d) < 20:
            continue
        x = np.log(d["T"].values)
        Y = {nm: _logskew(d[nm].values) for nm in names}
        n_clip = {nm: int((d[nm] > -1e-4).sum()) for nm in names}
        dates = d["Date"].unique()
        groups = {dt: np.flatnonzero((d["Date"] == dt).values) for dt in dates}
        rng = np.random.default_rng(seed)
        boot = {nm: np.empty(n_boot) for nm in names}
        for b in range(n_boot):
            pick = rng.choice(len(dates), len(dates))
            idx = np.concatenate([groups[dates[i]] for i in pick])
            for nm in names:
                boot[nm][b] = _ols_slope(x[idx], Y[nm][idx])
        for nm in names:
            rows.append(dict(Subset=sname, Series=nm, N_pairs=len(d), N_dates=len(dates), N_clipped=n_clip[nm],
                             Exponent=_ols_slope(x, Y[nm]), CI_low=np.percentile(boot[nm], 2.5),
                             CI_high=np.percentile(boot[nm], 97.5)))
        est = {nm: _ols_slope(x, Y[nm]) for nm in names}
        pairs = [(nm, market) for nm in names[1:]] + [(a, b) for a in names[1:] for b in names[1:] if a < b]
        for a, b in pairs:
            dd = boot[a] - boot[b]
            diffs.append(dict(Subset=sname, Difference=f"{a} - {b}", Estimate=est[a] - est[b],
                              CI_low=np.percentile(dd, 2.5), CI_high=np.percentile(dd, 97.5)))
        # hypothesis (ii): is a model's exponent closer to the market's than Merton's?
        for nm in names[1:]:
            if nm == "Merton":
                continue
            closer = np.abs(boot[nm] - boot[market]) < np.abs(boot["Merton"] - boot[market])
            diffs.append(dict(Subset=sname, Difference=f"P(|{nm}-Market| < |Merton-Market|)", Estimate=np.mean(
                np.abs(est[nm] - est[market]) < np.abs(est["Merton"] - est[market])),
                CI_low=np.nan, CI_high=np.nan, Boot_share=float(closer.mean())))
    return pd.DataFrame(rows), pd.DataFrame(diffs)


def skew_by_bucket(sk, names):
    """Mean skew (vol per unit log-moneyness) per maturity bucket and series; ratio to market in the last columns."""
    d = sk.copy()
    d["MatBucket"] = pd.cut(d["Days"], MAT_BINS, labels=MAT_LABELS)
    d = d[d[names[0]] < 0]
    rows = []
    for b, g in d.groupby("MatBucket", observed=True):
        r = dict(MatBucket=b, N_pairs=len(g))
        for nm in names:
            r[f"Skew_{nm}"] = g[nm].mean()
        for nm in names[1:]:
            r[f"Ratio_{nm}/Market"] = g[nm].mean() / g[names[0]].mean()
        rows.append(r)
    return pd.DataFrame(rows)


# =============================================================================
# Metric 3: implied-vol error grid
# =============================================================================
def iv_error_grid(opt, model, min_n=10, stat="mean"):
    """Maturity x log-moneyness grid of (model IV - market IV) in vol points; cells with < min_n quotes are NaN."""
    d = add_buckets(opt)
    d = d.assign(err=d[f"{model}_IVerr"] * 100)
    g = d.groupby(["MatBucket", "KBucket"], observed=False)["err"]
    h = (g.mean() if stat == "mean" else g.median()).unstack()
    n = g.size().unstack()
    return h.where(n >= min_n), n


# =============================================================================
# Diebold-Mariano: Newey-West t statistic for the mean of a loss difference
# =============================================================================
def nw_t(d, lag):
    """Mean of d, its Newey-West (Bartlett) standard error and t statistic (H0: mean 0)."""
    d = np.asarray(d, float)
    T = len(d)
    u = d - d.mean()
    lrv = np.sum(u * u) / T
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * np.sum(u[k:] * u[:-k]) / T
    se = np.sqrt(max(lrv, 1e-300) / T)
    return d.mean(), se, d.mean() / se
