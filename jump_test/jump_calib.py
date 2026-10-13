"""
jump_calib.py  -  data loading and the daily calibrator
========================================================

Reused from ../seb_merton_analys.py (imported, not copied):
  * prepare_data()  : reading, filters (bid > 0.01, >= 5 days, OTM only, spread <= mid), implied vols, K/F, log-moneyness
  * HalfSpreadW     : half-spread with a 0.05 SEK floor  ->  z = (model - mid) / HalfSpreadW
  * implied_vol / bs_price / fit_bs_slice, RISK_FREE_RATE = 2 %, DIVIDEND_YIELD = 0, REG_WEIGHT = 3

Calibration objective for one day (the same as Merton_dag in seb_merton_analys.py), all expiries at once:

    sum_i z_i^2  +  reg_weight * sum_p ((c_p(theta) - c_p(theta_prev)) / scale_p)^2 ,

    z_i = (model price_i - mid_i) / half-spread_i ,  c_p = the model's penalty coordinates (jump_models.py).
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))           # seb_merton_analys.py lives one folder up
sys.path.insert(0, HERE)

import seb_merton_analys as seb                      # noqa: E402
import jump_models as jm                             # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

R, Q = seb.RISK_FREE_RATE, seb.DIVIDEND_YIELD
REG_WEIGHT = seb.REG_WEIGHT
COS_GAP_POLISH = 1e-3        # SEK: if the N_FIT and N_COS prices differ by more than this, re-optimise on a finer grid
CACHE_DIR = os.path.join(HERE, "cache")
OUT_DIR = os.path.join(HERE, "output")
STOCKS = list(seb.FILES)                             # ["BOL", "VOLVB"]
MODEL_ORDER = ["Merton", "Kou", "Bates"]
ALL_PARAMS = ["sigma", "lam", "mu_j", "sigma_j", "p", "eta1", "eta2",
              "v0", "kappa", "theta", "xi", "rho"]   # union of the models' parameter names
MARKET_COLS = ["Date", "Expiry", "Days", "T", "CallPut", "Strike", "StockMid", "Forward", "K_over_F",
               "LogMoneyness", "Bid", "Ask", "Mid", "Spread", "HalfSpreadW", "IV_Bid", "IV_Mid", "IV_Ask"]


def load_options(tag, use_cache=True):
    """seb.prepare_data() for one stock, cached as a pickle so repeated runs skip the slow Excel read."""
    src = os.path.join(seb.DATA_DIR, seb.FILES[tag])
    os.makedirs(CACHE_DIR, exist_ok=True)
    pkl = os.path.join(CACHE_DIR, f"{tag}_prepared.pkl")
    if use_cache and os.path.exists(pkl) and os.path.getmtime(pkl) > os.path.getmtime(src):
        return pd.read_pickle(pkl)
    options, _ = seb.prepare_data(src)
    options.to_pickle(pkl)
    return options


def atm_vol_of(d):
    near = d[np.abs(d["LogMoneyness"]) < 0.05]
    return float(np.nanmedian((near if len(near) else d)["IV_Mid"]))


def model_prices(model, th, d):
    S, K, T, ic, _, _ = seb._arrays(d)
    return jm.cos_price(model, th, S, K, T, ic, R, Q)


def fit_model(model, d, prev=None, merton_th=None, reg_weight=REG_WEIGHT, max_nfev=300,
              diff_step=1e-6, n_fit=jm.N_FIT, n_report=jm.N_COS):
    """
    Calibrate `model` to all options of one day (DataFrame d from load_options()).
      prev       : the previous day's parameters (warm start + penalty), or None
      merton_th  : today's Merton fit, used as one extra start for Kou and Bates
    Returns a dict with theta, z, price, rss, penalty, nfev, bound flags, conditioning and runtime.
    """
    t0 = time.time()
    S, K, T, ic, mid, w = seb._arrays(d)
    n = len(mid)
    starts = model.starts(atm_vol_of(d), merton_th)
    if prev is not None:
        starts = [np.asarray(prev, float)] + starts[:3]
    use_pen = prev is not None and reg_weight > 0
    sq = np.sqrt(reg_weight)
    pc_prev = model.pen_coords(np.asarray(prev, float)) if use_pen else None

    def make_resid(N):
        def resid(th):
            z = (jm.cos_price(model, th, S, K, T, ic, R, Q, N=N) - mid) / w
            z = np.where(np.isfinite(z), z, 1e3)                   # failed pricing -> huge, finite residual
            return np.concatenate([z, sq * (model.pen_coords(th) - pc_prev) / model.pen_scale]) if use_pen else z
        return resid

    def solve(x0, N):
        return least_squares(make_resid(N), x0, bounds=(model.lb, model.ub), method="trf", x_scale="jac",
                             max_nfev=max_nfev, diff_step=diff_step)

    best, best_i, nfev = None, None, 0
    for i, x0 in enumerate(starts):
        try:
            res = solve(np.clip(np.asarray(x0, float), model.lb + 1e-6, model.ub - 1e-6), n_fit)
        except Exception:
            continue
        nfev += res.nfev
        if best is None or res.cost < best.cost:
            best, best_i = res, i
    if best is None:
        raise RuntimeError(f"{model.name}: all starts failed")

    # the optimiser used the fast grid (n_fit); reported prices use the finer grid (n_report)
    th = best.x
    p_fit = jm.cos_price(model, th, S, K, T, ic, R, Q, N=n_fit)
    p_rep = jm.cos_price(model, th, S, K, T, ic, R, Q, N=n_report)
    cos_gap = float(np.nanmax(np.abs(p_rep - p_fit)))
    if cos_gap > COS_GAP_POLISH:            # fast grid not accurate at this solution: re-optimise on a 4x finer grid
        best = solve(th, 4 * n_fit)
        nfev += best.nfev
        th = best.x
        p_fit = jm.cos_price(model, th, S, K, T, ic, R, Q, N=4 * n_fit)
        p_rep = jm.cos_price(model, th, S, K, T, ic, R, Q, N=n_report)
        cos_gap = float(np.nanmax(np.abs(p_rep - p_fit)))
    z = (p_rep - mid) / w
    try:
        logcond = float(np.log10(np.linalg.cond(best.jac[:n].T @ best.jac[:n])))
    except np.linalg.LinAlgError:
        logcond = np.inf
    span = model.ub - model.lb
    at_bound = [nm for nm, v, lo, hi, sp in zip(model.pnames, th, model.lb, model.ub, span)
                if v - lo < 1e-3 * sp or hi - v < 1e-3 * sp]
    return dict(theta=th, z=z, price=p_rep, rss=float(np.sum(z ** 2)),
                pen=float(np.sum(best.fun[n:] ** 2)) if use_pen else 0.0, nfev=nfev, best_start=best_i,
                at_bound=",".join(at_bound), logcond=logcond, cos_gap=cos_gap, runtime=time.time() - t0)


def run_stock(tag, day_step=1, models=tuple(MODEL_ORDER), reg_weight=REG_WEIGHT, max_days=None, out_dir=OUT_DIR):
    """
    Calibrate the requested models on every `day_step`-th trading day of one stock, warm-starting each day from
    the previously calibrated day. Writes to out_dir:
      params_daily_<tag>.csv   one row per (date, model): parameters, fit, next-day (out-of-sample) fit, runtime
      options_fit_<tag>.csv    one row per (date, option): market data + price, z, IV, IV error per model
      options_oos_<tag>.csv    the next trading day's quotes priced with today's parameters
    Returns (params_df, options_df, runtimes_dict).
    """
    os.makedirs(out_dir, exist_ok=True)
    o = load_options(tag)
    dates = np.sort(o["Date"].unique())
    fit_idx = np.arange(0, len(dates), day_step)
    if max_days:
        fit_idx = fit_idx[:max_days]
    by_date = {pd.Timestamp(dt): g for dt, g in o.groupby("Date")}
    prev = {m: None for m in models}
    p_rows, o_rows, oos_rows = [], [], []
    rt = {m: 0.0 for m in models}
    t_start = time.time()
    for c, ii in enumerate(fit_idx):
        dt = pd.Timestamp(dates[ii])
        d = by_date[dt]
        dn = by_date.get(pd.Timestamp(dates[ii + 1])) if ii + 1 < len(dates) else None
        S, K, T, ic, mid, w = seb._arrays(d)
        orow = d[MARKET_COLS].copy().reset_index(drop=True)

        # reference: Black-Scholes with one sigma per expiry; its "parameter count" is the number of expiries
        bs_p = np.empty(len(d))
        for _, g in d.groupby("Expiry"):
            pos = np.flatnonzero((d["Expiry"] == g["Expiry"].iloc[0]).values)
            bs_p[pos] = seb.bs_price(g["StockMid"].values, g["Strike"].values, g["T"].values, R, Q, seb.fit_bs_slice(g),
                                     g["IsCall"].values)
        orow["BS_slice_Price"] = bs_p
        orow["BS_slice_z"] = (bs_p - mid) / w
        orow["BS_slice_IV"] = seb.implied_vol(bs_p, S, K, T, R, Q, ic)
        orow["BS_slice_IVerr"] = orow["BS_slice_IV"] - orow["IV_Mid"]
        p_rows.append(dict(Date=dt, Model="BS_slice", NPar=d["Expiry"].nunique(), N_options=len(d),
                           N_expiries=d["Expiry"].nunique(), RMSE_z=np.sqrt(np.mean(orow["BS_slice_z"] ** 2)),
                           Inside_pct=100 * np.mean(np.abs(orow["BS_slice_z"]) <= 1),
                           RSS=float(np.sum(orow["BS_slice_z"] ** 2))))

        merton_th, orow_n = None, None
        if dn is not None:
            orow_n = dn[MARKET_COLS].copy().reset_index(drop=True)
            orow_n.insert(0, "FitDate", dt)
            Sn, Kn, Tn, icn, midn, wn = seb._arrays(dn)
        for m in models:
            model = jm.MODELS[m]
            f = fit_model(model, d, prev=prev[m], merton_th=merton_th if m != "Merton" else None, reg_weight=reg_weight)
            th, z, price = f["theta"], f["z"], f["price"]
            prev[m] = th
            if m == "Merton":
                merton_th = th
            rt[m] += f["runtime"]
            iv = seb.implied_vol(price, S, K, T, R, Q, ic)
            orow[f"{m}_Price"], orow[f"{m}_z"] = price, z
            orow[f"{m}_IV"], orow[f"{m}_IVerr"] = iv, iv - d["IV_Mid"].values
            row = dict(Date=dt, Model=m, NPar=model.n_par, N_options=len(d), N_expiries=d["Expiry"].nunique(),
                       RMSE_z=np.sqrt(np.mean(z ** 2)), Inside_pct=100 * np.mean(np.abs(z) <= 1),
                       InsideBidAsk_pct=100 * np.mean((price >= d["Bid"].values) & (price <= d["Ask"].values)),
                       RSS=f["rss"], Penalty=f["pen"], NFev=f["nfev"], BestStart=f["best_start"],
                       AtBound=f["at_bound"], LogCondJtJ=f["logcond"], COS_gap=f["cos_gap"], Runtime_s=f["runtime"],
                       ATM_IV=atm_vol_of(d), Stock=float(d["StockMid"].iloc[0]))
            row.update(dict(zip(model.pnames, th)))
            row.update(model.diagnostics(th))
            if dn is not None:      # out-of-sample: today's parameters on the next trading day's quotes
                pn = jm.cos_price(model, th, Sn, Kn, Tn, icn, R, Q)
                ivn = seb.implied_vol(pn, Sn, Kn, Tn, R, Q, icn)
                zn = (pn - midn) / wn
                orow_n[f"{m}_Price"], orow_n[f"{m}_z"] = pn, zn
                orow_n[f"{m}_IV"], orow_n[f"{m}_IVerr"] = ivn, ivn - dn["IV_Mid"].values
                zn = zn[np.isfinite(zn)]
                row.update(OOS_N=len(zn), OOS_RMSE_z=np.sqrt(np.mean(zn ** 2)), OOS_Inside_pct=100 * np.mean(np.abs(zn) <= 1))
            p_rows.append(row)
        o_rows.append(orow)
        if orow_n is not None:
            oos_rows.append(orow_n)
        if (c + 1) % 5 == 0 or c == len(fit_idx) - 1 or c == 0:
            el = time.time() - t_start
            last = {r["Model"]: r for r in p_rows[-(len(models) + 1):]}
            msg = " | ".join(f"{m} {last[m]['RMSE_z']:.2f}/{last[m]['Inside_pct']:.0f}%" for m in ["BS_slice"] + list(models))
            print(f"  {tag}: {c + 1}/{len(fit_idx)} days  {dt.date()}  RMSE_z/inside: {msg}  "
                  f"[{el:.0f} s, ~{el / (c + 1) * (len(fit_idx) - c - 1):.0f} s left]", flush=True)

    params = pd.DataFrame(p_rows)
    cols = ["Date", "Model", "NPar", "N_options", "N_expiries", "RMSE_z", "Inside_pct", "InsideBidAsk_pct", "RSS",
            "Penalty", "OOS_N", "OOS_RMSE_z", "OOS_Inside_pct"] + ALL_PARAMS + ["TotalVol", "JumpVarShare", "Feller",
            "NFev", "BestStart", "AtBound", "LogCondJtJ", "COS_gap", "Runtime_s", "ATM_IV", "Stock"]
    params = params.reindex(columns=[c for c in cols if c in params.columns or c in ALL_PARAMS])
    options = pd.concat(o_rows, ignore_index=True)
    params.to_csv(os.path.join(out_dir, f"params_daily_{tag}.csv"), index=False)
    options.to_csv(os.path.join(out_dir, f"options_fit_{tag}.csv"), index=False)
    if oos_rows:
        pd.concat(oos_rows, ignore_index=True).to_csv(os.path.join(out_dir, f"options_oos_{tag}.csv"), index=False)
    rt["total_wall"] = time.time() - t_start
    return params, options, rt
