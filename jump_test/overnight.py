"""
overnight.py  -  the big all-days run: out-of-sample windows in time, in maturity and for the "slow" parameters
==============================================================================================================

Models: Merton, Kou, Heston (no jumps), Bates.  Data and objective exactly as in the other steps
(z = (model - mid) / half-spread, COS pricing, same filters).  Every trading day of both stocks, no day-to-day penalty
unless said otherwise, so every day is an independent calibration (no chain, no path dependence, runs in parallel).

Three experiments, each answers "which window of information predicts what the model has NOT seen?"

  PHASE 1  Time horizon.   Fit all options of day t.  Price the quotes of trading day t+h with day t's parameters,
           h = 1, 2, 5, 10, 20, 40 (h = 0 is the in-sample fit).  Metrics per model and h: RMSE_z, share |z| <= 1, skew-decay
           exponent of the out-of-sample quotes, Diebold-Mariano tests between models.
           Every fit gets the normal start points plus RESTARTS extra restarts (perturbed copies of the best solution and
           one random start), which can only lower the objective, so the fits are close to the global optimum.

  PHASE 2  Maturity window.  Fit day t on the SHORT expiries only (Days <= s, s = 45, 60, 75, 90, 120) and price the longer expiries
           (Days > s) the model never saw.  This is the skew-decay question in its sharpest form: does the model's own
           dynamics carry the short-maturity information out to long maturities?  Two variants:
              free   : only the short expiries of day t (nothing else), as a pure test of the model structure
              prior  : same, plus a penalty towards yesterday's FULL-surface parameters (leak-free: it uses only
                       information up to day t-1 plus today's short quotes), which is how a desk would actually update
           Reported on all longer expiries and on "far" expiries (Days >= max(90, s + 60)), with the skew of the far
           expiries compared with the market's.  (s = 30 is not used: no trading day has two expiries within 30 days.)

  PHASE 3  Window for the slow parameters.  For window W = 1, 3, 5, 10, 20, 40, 60 days: take the structural parameters
           (everything except the state: sigma for Merton/Kou, v0 for Heston/Bates) from the fits of the previous W days
           (the day whose structural vector is closest to the median, so the vector is a real fitted one), re-fit ONLY
           the state to today's quotes, and predict t+h.  W = 0 in the tables is the daily full re-fit of phase 1.
           If a model's parameters really are constant apart from the state, a long window should do at least as well as
           a daily re-fit; if they drift, a short window or the daily re-fit wins.

The script is resumable: every finished task is stored in output_overnight/checkpoints_<hash>/ and skipped on a restart.
Work is ordered so that an interrupted run still covers the whole sample (every 5th day first, then the days in between).

    python overnight.py                      # everything, all cores but one (about 1 h on 5 worker processes)
    python overnight.py --jobs 6             # fixed number of worker processes
    python overnight.py --hours 10           # stop starting new tasks after 10 h, then write the tables
    python overnight.py --smoke              # 12 days per stock, minutes (own folder output_overnight_smoke/)
    python overnight.py --analysis-only      # rebuild tables and charts from the checkpoints
    python overnight.py --phases 1 2         # only some phases

Output (output_overnight/):
    fits_full_<STOCK>.csv, horizon_options_<STOCK>.csv.gz, extrap_params.csv, extrap_options_<STOCK>.csv.gz,
    window_rows.csv, horizon_summary.csv, horizon_dm.csv, horizon_exponents.csv, extrap_summary.csv, extrap_dm.csv,
    window_summary.csv, window_dm.csv, scoreboard.csv, OVERNIGHT_SUMMARY.txt, horizons.png, extrapolation.png,
    structural_window.png, run.log
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")                    # one BLAS thread per worker process, the parallelism is across tasks

import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import pickle
import time
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import least_squares

import jump_calib as jc
import jump_metrics as jmx
import jump_models as jm

seb = jc.seb
HERE = jc.HERE
MODELS = ("Merton", "Kou", "Heston", "Bates")
HORIZONS = (1, 2, 5, 10, 20, 40)
SPLITS = (45, 60, 75, 90, 120)
WINDOWS = (1, 3, 5, 10, 20, 40, 60)
PRIOR_WEIGHT = 3.0
RESTARTS = 4
PAIRS = [("Merton", "Kou"), ("Merton", "Heston"), ("Merton", "Bates"), ("Heston", "Bates")]
COLORS = {"Merton": "#2a78d6", "Kou": "#eb6834", "Heston": "#7d7bd0", "Bates": "#1baf7a"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
_LOG_PATH = None


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _LOG_PATH:
        with open(_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


# =============================================================================
# helpers shared by the three phases (these run inside the worker processes)
# =============================================================================
_CACHE = {}


def stock_data(tag):
    if tag not in _CACHE:
        o = jc.load_options(tag)
        dates = [pd.Timestamp(x) for x in np.sort(o["Date"].unique())]
        by = {pd.Timestamp(dt): g.reset_index(drop=True) for dt, g in o.groupby("Date")}
        _CACHE[tag] = (dates, by)
    return _CACHE[tag]


class OnlyStarts:
    """Wrap a model so that the optimiser uses exactly these start vectors (everything else is delegated unchanged)."""

    def __init__(self, model, starts):
        self._m = model
        self._s = [np.asarray(s, float) for s in starts]

    def __getattr__(self, k):
        return getattr(self._m, k)

    def starts(self, atm_vol, merton_th=None):
        return self._s


def restart_vectors(model, th, rng, k):
    """k restart vectors: perturbed copies of th (log-normal for positive parameters, additive for the others);
    the last one is a random start over the whole box."""
    lb, ub = model.lb, model.ub
    out = []
    for r in range(k):
        x = np.array(th, float)
        for j in range(len(x)):
            if r == k - 1 and k >= 2:
                x[j] = np.exp(rng.uniform(np.log(lb[j]), np.log(ub[j]))) if lb[j] > 0 else rng.uniform(lb[j], ub[j])
            elif lb[j] >= 0 and x[j] > 1e-8:
                x[j] *= np.exp(0.3 * rng.standard_normal())
            else:
                x[j] += 0.05 * (ub[j] - lb[j]) * rng.standard_normal()
        out.append(np.clip(x, lb + 1e-6, ub - 1e-6))
    return out


def fit_free(name, d, merton_th, rng, restarts):
    """Calibrate without any day-to-day penalty, normal starts + `restarts` extra restarts; keep the lowest RSS."""
    model = jm.MODELS[name]
    f = jc.fit_model(model, d, prev=None, merton_th=merton_th if name in ("Kou", "Bates") else None, reg_weight=0.0)
    f["restart_gain_pct"] = 0.0
    if restarts > 0:
        g = jc.fit_model(OnlyStarts(model, restart_vectors(model, f["theta"], rng, restarts)), d, prev=None, reg_weight=0.0)
        nfev, rt, rss0 = f["nfev"] + g["nfev"], f["runtime"] + g["runtime"], f["rss"]
        if g["rss"] < f["rss"] - 1e-9:
            f = g
            f["restart_gain_pct"] = 100 * (1 - g["rss"] / rss0)
        f["nfev"], f["runtime"] = nfev, rt
    return f


def price_on(model, th, d):
    """Price, z and implied vol of the options in d under parameters th."""
    S, K, T, ic, mid, w = seb._arrays(d)
    p = jm.cos_price(model, th, S, K, T, ic, jc.R, jc.Q)
    z = (p - mid) / w
    iv = seb.implied_vol(p, S, K, T, jc.R, jc.Q, ic)
    return p, z, iv


def base_frame(d, fit_date, **meta):
    f = d[jc.MARKET_COLS].copy().reset_index(drop=True)
    f.insert(0, "FitDate", fit_date)
    for k, v in meta.items():
        f[k] = v
    return f


def put_model(fr, d, name, p, z, iv):
    fr[f"{name}_Price"], fr[f"{name}_z"], fr[f"{name}_IV"] = p, z, iv
    fr[f"{name}_IVerr"] = iv - d["IV_Mid"].values


def param_row(model, th):
    r = dict(zip(model.pnames, th))
    r.update(model.diagnostics(th))
    return r


def task_rng(cfg, tag, i, salt):
    return np.random.default_rng(cfg["seed"] + 7919 * i + 104729 * salt + zlib.crc32(tag.encode()) % 100003)


def ckpt(cfg, phase, tag, i):
    return os.path.join(cfg["chk"], f"p{phase}_{tag}_{i:04d}.pkl")


_P1 = {}


def p1_thetas(cfg, tag):
    """Phase-1 parameter vectors of a stock: {day index: {model: theta}} (cached per process)."""
    key = (cfg["chk"], tag)
    if key not in _P1:
        out = {}
        for fpath in glob.glob(os.path.join(cfg["chk"], f"p1_{tag}_*.pkl")):
            i = int(os.path.basename(fpath).rsplit("_", 1)[1].split(".")[0])
            with open(fpath, "rb") as fh:
                par = pickle.load(fh)["params"]
            out[i] = {m: g[list(jm.MODELS[m].pnames)].values[0].astype(float) for m, g in par.groupby("Model")}
        _P1[key] = out
    return _P1[key]


# =============================================================================
# PHASE 1: full fit of day t, quotes of day t+h priced with its parameters
# =============================================================================
def phase1(task):
    tag, i, cfg = task["tag"], task["i"], task["cfg"]
    dates, by = stock_data(tag)
    dt, d = dates[i], by[dates[i]]
    rng = task_rng(cfg, tag, i, 1)
    hz = [h for h in (0,) + tuple(cfg["horizons"]) if i + h < len(dates)]
    frames = {h: base_frame(by[dates[i + h]], dt, Stock=tag, H=h) for h in hz}
    prow, merton_th = [], None
    for name in cfg["models"]:
        model = jm.MODELS[name]
        try:
            f = fit_free(name, d, merton_th, rng, cfg["restarts"])
        except Exception:
            continue
        th = f["theta"]
        if name == "Merton":
            merton_th = th
        z = f["z"]
        row = dict(Date=dt, Stock=tag, Model=name, NPar=model.n_par, N_options=len(d), N_expiries=d["Expiry"].nunique(),
                   RMSE_z=np.sqrt(np.mean(z ** 2)), Inside_pct=100 * np.mean(np.abs(z) <= 1), RSS=f["rss"],
                   NFev=f["nfev"], AtBound=f["at_bound"], Restart_gain_pct=f["restart_gain_pct"], Runtime_s=f["runtime"])
        row.update(param_row(model, th))
        prow.append(row)
        for h in hz:
            dn = by[dates[i + h]]
            if h == 0:
                S, K, T, ic, _, _ = seb._arrays(d)
                p, zz, iv = f["price"], f["z"], seb.implied_vol(f["price"], S, K, T, jc.R, jc.Q, ic)
            else:
                p, zz, iv = price_on(model, th, dn)
            put_model(frames[h], dn, name, p, zz, iv)
    return dict(params=pd.DataFrame(prow), options=pd.concat(frames.values(), ignore_index=True))


# =============================================================================
# PHASE 2: fit on the short expiries only, predict the longer ones
# =============================================================================
def phase2(task):
    tag, i, cfg = task["tag"], task["i"], task["cfg"]
    dates, by = stock_data(tag)
    dt, d = dates[i], by[dates[i]]
    rng = task_rng(cfg, tag, i, 2)
    prev_full = p1_thetas(cfg, tag).get(i - 1) if i > 0 else None
    prow, frames = [], []
    for s in cfg["splits"]:
        tr = d[d["Days"] <= s].reset_index(drop=True)
        te = d[d["Days"] > s].reset_index(drop=True)
        if tr["Expiry"].nunique() < 2 or len(tr) < 12 or len(te) < 8:
            continue
        for variant in ("free", "prior"):
            if variant == "prior" and prev_full is None:
                continue
            fr, merton_th = base_frame(te, dt, Stock=tag, Variant=variant, Split=s), None
            for name in cfg["models"]:
                model = jm.MODELS[name]
                try:
                    if variant == "free":
                        f = fit_free(name, tr, merton_th, rng, cfg["restarts"])
                    else:
                        if name not in prev_full:
                            continue
                        f = jc.fit_model(model, tr, prev=prev_full[name], merton_th=merton_th if name in ("Kou", "Bates") else None,
                                         reg_weight=cfg["prior_weight"])
                except Exception:
                    continue
                th = f["theta"]
                if name == "Merton":
                    merton_th = th
                p, z, iv = price_on(model, th, te)
                put_model(fr, te, name, p, z, iv)
                row = dict(Date=dt, Stock=tag, Variant=variant, Split=s, Model=name, N_train=len(tr), N_test=len(te),
                           N_train_expiries=tr["Expiry"].nunique(), Train_RMSE_z=np.sqrt(np.mean(f["z"] ** 2)),
                           AtBound=f["at_bound"], Runtime_s=f["runtime"])
                row.update(param_row(model, th))
                prow.append(row)
            frames.append(fr)
    return dict(params=pd.DataFrame(prow), options=pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())


# =============================================================================
# PHASE 3: structural parameters from a rolling window, only the state re-fitted
# =============================================================================
def medoid_structure(model, thetas):
    """The fitted vector (state excluded) closest to the componentwise median of the structural coordinates."""
    X = np.array([model.pen_coords(t)[1:] / model.pen_scale[1:] for t in thetas])
    m = np.median(X, axis=0)
    return thetas[int(np.argmin(np.sum((X - m) ** 2, axis=1)))]


def refit_state(model, th_struct, d):
    """Re-fit parameter 0 (sigma for Merton/Kou, v0 for Heston/Bates) to the options in d, all other parameters fixed."""
    S, K, T, ic, mid, w = seb._arrays(d)
    lo, hi = model.lb[0], model.ub[0]

    def resid(x):
        th = th_struct.copy()
        th[0] = x[0]
        z = (jm.cos_price(model, th, S, K, T, ic, jc.R, jc.Q, N=jm.N_FIT) - mid) / w
        return np.where(np.isfinite(z), z, 1e3)

    atm = jc.atm_vol_of(d)
    st = model.starts(atm, None)
    best = None
    for x0 in (th_struct[0], st[0][0], st[1][0]):
        x0 = float(np.clip(x0, lo + 1e-6, hi - 1e-6))
        try:
            r = least_squares(resid, [x0], bounds=([lo], [hi]), method="trf", x_scale="jac", max_nfev=60, diff_step=1e-6)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        raise RuntimeError("state fit failed")
    th = th_struct.copy()
    th[0] = best.x[0]
    return th


def phase3(task):
    tag, i, cfg = task["tag"], task["i"], task["cfg"]
    dates, by = stock_data(tag)
    dt, d = dates[i], by[dates[i]]
    P1 = p1_thetas(cfg, tag)
    hz = [h for h in (0,) + tuple(cfg["horizons"]) if i + h < len(dates)]
    rows = []
    for name in cfg["models"]:
        model = jm.MODELS[name]
        for W in cfg["windows"]:
            past = [P1[j][name] for j in range(max(0, i - W), i) if j in P1 and name in P1[j]]
            if len(past) < min(W, 3):
                continue
            try:
                th = refit_state(model, medoid_structure(model, past), d)
            except Exception:
                continue
            for h in hz:
                dn = by[dates[i + h]]
                _, z, iv = price_on(model, th, dn)
                ok = np.isfinite(z)
                e = (iv - dn["IV_Mid"].values)[np.isfinite(iv - dn["IV_Mid"].values)] * 100
                rows.append(dict(Date=dt, Stock=tag, Model=name, W=W, H=h, N=int(ok.sum()), RSS=float(np.sum(z[ok] ** 2)),
                                 N_inside=int(np.sum(np.abs(z[ok]) <= 1)), MeanIVerr_vp=float(np.mean(e)) if len(e) else np.nan,
                                 MeanAbsIVerr_vp=float(np.mean(np.abs(e))) if len(e) else np.nan, N_past=len(past)))
    return dict(rows=pd.DataFrame(rows))


# =============================================================================
# ORCHESTRATION
# =============================================================================
def safe_run(args):
    phase, task = args
    try:
        t0 = time.time()
        res = {1: phase1, 2: phase2, 3: phase3}[phase](task)
        res["seconds"] = time.time() - t0
        return phase, task, res
    except Exception:
        return phase, task, {"error": traceback.format_exc()}


def day_order(n, stride=5):
    return [i for r in range(stride) for i in range(r, n, stride)]


def run_phase(phase, stocks, ndays, cfg, jobs, deadline, label):
    tasks = []
    for i in day_order(max(ndays.values())):
        for tag in stocks:
            if i < ndays[tag] and not os.path.exists(ckpt(cfg, phase, tag, i)):
                tasks.append(dict(tag=tag, i=i, cfg=cfg))
    total = sum(ndays[t] for t in stocks)
    already = total - len(tasks)
    if not tasks:
        log(f"{label}: nothing to do ({already}/{total} already done)")
        return True
    log(f"{label}: {len(tasks)} tasks to run ({already}/{total} already done), {jobs} worker processes")
    t0, done, errors = time.time(), 0, 0
    complete = True
    it = iter(tasks)
    pending = {}
    with ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("spawn")) as ex:
        while True:
            while len(pending) < 2 * jobs and time.time() < deadline:
                t = next(it, None)
                if t is None:
                    break
                pending[ex.submit(safe_run, (phase, t))] = t
            if not pending:
                break
            finished, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for fut in finished:
                t = pending.pop(fut)
                try:
                    ph, t, res = fut.result()
                except Exception:                      # a worker process died (e.g. out of memory): stop cleanly, rerun resumes
                    log(f"  worker pool broke ({label}); finished work is saved, rerun the script to continue")
                    return False
                if "error" in res:
                    errors += 1
                    log(f"  ERROR {label} {t['tag']} day {t['i']}: {res['error'].strip().splitlines()[-1]}")
                    with open(os.path.join(os.path.dirname(cfg["chk"]), "errors.log"), "a", encoding="utf-8") as fh:
                        fh.write(f"{label} {t['tag']} {t['i']}\n{res['error']}\n")
                    continue
                tmp = ckpt(cfg, ph, t["tag"], t["i"]) + ".tmp"
                with open(tmp, "wb") as fh:
                    pickle.dump(res, fh, protocol=4)
                os.replace(tmp, ckpt(cfg, ph, t["tag"], t["i"]))
                done += 1
                el = time.time() - t0
                if done % max(1, jobs) == 0 or done == len(tasks):
                    log(f"  {label} {already + done}/{total}  last {t['tag']} day {t['i']} ({res['seconds']:.0f} s)  "
                        f"elapsed {el / 60:.1f} min, about {el / done * (len(tasks) - done) / 60:.0f} min left")
    if done + errors < len(tasks):
        complete = False
        log(f"{label}: time budget reached, {done + errors}/{len(tasks)} tasks were run; rerun the script to continue")
    return complete and errors == 0


# =============================================================================
# ANALYSIS
# =============================================================================
def load_phase(cfg, phase, tag, key):
    out = []
    for fpath in sorted(glob.glob(os.path.join(cfg["chk"], f"p{phase}_{tag}_*.pkl"))):
        with open(fpath, "rb") as fh:
            r = pickle.load(fh)
        if key in r and len(r[key]):
            out.append(r[key])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def loss_by_date(fr, models, mask=None):
    f = fr if mask is None else fr[mask]
    return pd.concat({m: (f[f"{m}_z"] ** 2).groupby(f["FitDate"]).mean() for m in models if f"{m}_z" in f}, axis=1)


def dm_pair(La, Lb, lag):
    """A is the less flexible model; positive RMSE difference / DM t means B has the smaller loss."""
    L = pd.concat([La, Lb], axis=1, keys=["a", "b"]).dropna()
    L = L[(L["a"] > 0) & (L["b"] > 0)]
    if len(L) < 10:
        return None
    lag = int(min(lag, len(L) // 4))                    # a long lag on few days only adds noise
    d = (L["a"] - L["b"]).values
    _, _, t_l = jmx.nw_t(np.log(L["a"].values / L["b"].values), lag)
    return dict(Days=len(L), RMSE_z_A=np.sqrt(L["a"].mean()), RMSE_z_B=np.sqrt(L["b"].mean()), DM_t_logratio=t_l,
                DM_p_logratio=2 * stats.norm.sf(abs(t_l)), B_wins_pct=100 * np.mean(d > 0))


def dm_table(L, lag, meta):
    rows = []
    for a, b in PAIRS:
        if a in L and b in L:
            r = dm_pair(L[a], L[b], lag)
            if r:
                rows.append(dict(meta, A=a, B=b, NW_lag=lag, **r))
    return rows


def add_median(ft, g, models):
    """Median |z| per model next to the RMSE (RMSE is pulled up by a few failed fits, the median is not)."""
    ft["MedAbs_z"] = ft["Model"].map({m: float(np.nanmedian(np.abs(g[f"{m}_z"]))) for m in models})
    return ft


def skew_series(models):
    return {"Market": "IV_Mid", **{m: f"{m}_IV" for m in models}}


def available(fr, models):
    return [m for m in models if f"{m}_IV" in fr.columns and fr[f"{m}_IV"].notna().any()]


def analyse(cfg, stocks, n_boot):
    out = os.path.dirname(cfg["chk"])
    models = list(cfg["models"])
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 400)
    hz_all = (0,) + tuple(cfg["horizons"])
    summ, dms, exps = [], [], []
    p1_opts, p1_pars = {}, {}
    for tag in stocks:
        o = load_phase(cfg, 1, tag, "options")
        p = load_phase(cfg, 1, tag, "params")
        if o.empty:
            continue
        p1_opts[tag], p1_pars[tag] = o, p
        p.to_csv(os.path.join(out, f"fits_full_{tag}.csv"), index=False)
        o[o["H"] > 0].to_csv(os.path.join(out, f"horizon_options_{tag}.csv.gz"), index=False, compression="gzip")
        for h in hz_all:
            sub = o[o["H"] == h]
            if sub.empty:
                continue
            ms = available(sub, models)
            ft = add_median(jmx.fit_table(sub, ms), sub, ms)
            ft.insert(0, "Fit_dates", sub["FitDate"].nunique())
            ft.insert(0, "H", h)
            ft.insert(0, "Stock", tag)
            summ.append(ft)
            lag = max(5, h)
            dms += dm_table(loss_by_date(sub, ms), lag, dict(Stock=tag, H=h))
            sk = jmx.skew_table(sub, skew_series(ms))
            if len(sk) >= 20:
                ex, _ = jmx.decay_exponents(sk, ["Market"] + ms, n_boot=n_boot)
                if len(ex):
                    ex = ex[ex["Subset"].str.startswith("all")]
                    ex.insert(0, "H", h)
                    ex.insert(0, "Stock", tag)
                    exps.append(ex)
    hs = pd.concat(summ, ignore_index=True) if summ else pd.DataFrame()
    hdm = pd.DataFrame(dms)
    hex_ = pd.concat(exps, ignore_index=True) if exps else pd.DataFrame()
    hs.to_csv(os.path.join(out, "horizon_summary.csv"), index=False)
    hdm.to_csv(os.path.join(out, "horizon_dm.csv"), index=False)
    hex_.to_csv(os.path.join(out, "horizon_exponents.csv"), index=False)

    # ----- phase 2
    es, edm, epar, egap = [], [], [], []
    for tag in stocks:
        o = load_phase(cfg, 2, tag, "options")
        p = load_phase(cfg, 2, tag, "params")
        if o.empty:
            continue
        epar.append(p)
        o.to_csv(os.path.join(out, f"extrap_options_{tag}.csv.gz"), index=False, compression="gzip")
        for (variant, s), sub in o.groupby(["Variant", "Split"]):
            ms = available(sub, models)
            far_from = max(90, s + 60)
            subsets = {f"all test (>{s} d)": sub, f"far (>={far_from} d)": sub[sub["Days"] >= far_from]}
            for sname, g in subsets.items():
                if g.empty:
                    continue
                ft = add_median(jmx.fit_table(g, ms), g, ms)
                ft.insert(0, "Fit_dates", g["FitDate"].nunique())
                for k, v in (("Subset", sname), ("Split", s), ("Variant", variant), ("Stock", tag)):
                    ft.insert(0, k, v)
                if sname.startswith("far"):
                    sk = jmx.skew_table(g, skew_series(ms))
                    sk = sk[sk["Market"] < 0] if len(sk) else sk
                    for m in ms:
                        sel = ft["Model"] == m
                        if len(sk) and m in sk and sk[m].notna().any():
                            ft.loc[sel, "SkewRatio_to_market"] = sk[m].mean() / sk["Market"].mean()
                            ft.loc[sel, "SkewLogGap"] = np.mean(np.log(-np.clip(sk[m], None, -1e-4)) - np.log(-sk["Market"]))
                            ft.loc[sel, "Skew_pairs"] = len(sk)
                    edm += dm_table(loss_by_date(g, ms), 5, dict(Stock=tag, Variant=variant, Split=s, Subset=sname))
                es.append(ft)
            gb = jmx.fit_table(jmx.add_buckets(sub), ms, by=["MatBucket"])
            for k, v in (("Split", s), ("Variant", variant), ("Stock", tag)):
                gb.insert(0, k, v)
            egap.append(gb)
    esum = pd.concat(es, ignore_index=True) if es else pd.DataFrame()
    edmt = pd.DataFrame(edm)
    esum.to_csv(os.path.join(out, "extrap_summary.csv"), index=False)
    edmt.to_csv(os.path.join(out, "extrap_dm.csv"), index=False)
    if egap:
        pd.concat(egap, ignore_index=True).to_csv(os.path.join(out, "extrap_by_maturity.csv"), index=False)
    if epar:
        pd.concat(epar, ignore_index=True).to_csv(os.path.join(out, "extrap_params.csv"), index=False)

    # ----- phase 3
    wrows = [load_phase(cfg, 3, tag, "rows") for tag in stocks]
    wrows = pd.concat([w for w in wrows if len(w)], ignore_index=True) if any(len(w) for w in wrows) else pd.DataFrame()
    wsum, wdm = pd.DataFrame(), pd.DataFrame()
    if len(wrows):
        wrows.to_csv(os.path.join(out, "window_rows.csv"), index=False)
        wrows["Loss"] = wrows["RSS"] / wrows["N"].clip(lower=1)
        rows, dmr = [], []
        for (tag, W, h), g in wrows.groupby(["Stock", "W", "H"]):
            for m, gm in g.groupby("Model"):
                rows.append(dict(Stock=tag, W=W, H=h, Model=m, Days=len(gm), N=int(gm["N"].sum()),
                                 RMSE_z=np.sqrt(gm["RSS"].sum() / gm["N"].sum()), Inside_pct=100 * gm["N_inside"].sum() / gm["N"].sum(),
                                 MeanAbsIVerr_vp=gm["MeanAbsIVerr_vp"].mean()))
            piv = g.pivot(index="Date", columns="Model", values="Loss")
            dmr += dm_table(piv, max(5, h), dict(Stock=tag, W=W, H=h))
        # W = 0: the daily full re-fit of phase 1 (same dates), for reference
        for tag in stocks:
            if tag not in p1_opts:
                continue
            for h in hz_all:
                sub = p1_opts[tag][p1_opts[tag]["H"] == h]
                ms = available(sub, models)
                dates = set(wrows[(wrows["Stock"] == tag) & (wrows["H"] == h)]["Date"])
                sub = sub[sub["FitDate"].isin(dates)] if dates else sub
                for m in ms:
                    z = sub[f"{m}_z"].dropna()
                    if len(z):
                        rows.append(dict(Stock=tag, W=0, H=h, Model=m, Days=sub["FitDate"].nunique(), N=len(z),
                                         RMSE_z=np.sqrt(np.mean(z ** 2)), Inside_pct=100 * np.mean(np.abs(z) <= 1)))
                dmr += dm_table(loss_by_date(sub, ms), max(5, h), dict(Stock=tag, W=0, H=h))
        wsum, wdm = pd.DataFrame(rows).sort_values(["Stock", "H", "W", "Model"]), pd.DataFrame(dmr)
        wsum.to_csv(os.path.join(out, "window_summary.csv"), index=False)
        wdm.to_csv(os.path.join(out, "window_dm.csv"), index=False)
    return dict(hs=hs, hdm=hdm, hex=hex_, esum=esum, edm=edmt, wsum=wsum, wdm=wdm, p1=p1_pars)


def scoreboard(res):
    """Count, per model pair and experiment, the cells where B is significantly better / worse / not different (DM, p < 0.05)."""
    rows = []
    for name, df in (("horizons (stock x h)", res["hdm"]), ("maturity extrapolation, far (stock x variant x split)",
                     res["edm"][res["edm"]["Subset"].str.startswith("far")] if len(res["edm"]) else res["edm"]),
                     ("frozen-structure windows (stock x W x h)", res["wdm"])):
        if df is None or df.empty:
            continue
        d = df[df["H"] > 0] if "H" in df else df
        for (a, b), g in d.groupby(["A", "B"]):
            better = int(((g["DM_p_logratio"] < 0.05) & (g["DM_t_logratio"] > 0)).sum())
            worse = int(((g["DM_p_logratio"] < 0.05) & (g["DM_t_logratio"] < 0)).sum())
            rows.append(dict(Experiment=name, A=a, B=b, Cells=len(g), B_better=better, B_worse=worse, No_difference=len(g) - better - worse))
    return pd.DataFrame(rows)


def write_summary(res, cfg, out, n_done):
    L = []
    pr = L.append
    pr("OVERNIGHT SUMMARY")
    pr("=" * 78)
    pr(f"models {list(cfg['models'])}; horizons {list(cfg['horizons'])}; fit windows (days) {list(cfg['splits'])}; "
       f"structure windows {list(cfg['windows'])}; restarts {cfg['restarts']}")
    pr(f"finished tasks: {n_done}")
    pr("")
    pr("How to read: RMSE_z is the root mean squared (model - mid) / half-spread; 1.0 means the typical error is one half-spread.")
    pr("DM p is a Diebold-Mariano test on the per-day loss (log ratio, Newey-West); 'B better' means the second model has the smaller loss.")
    pr("")
    sc = scoreboard(res)
    if len(sc):
        pr("SCOREBOARD (cells where the second model is significantly better / worse at p < 0.05)")
        pr(sc.to_string(index=False))
        pr("")
    hs = res["hs"]
    if len(hs):
        pr("EXPERIMENT 1: next-trading-day-and-beyond pricing with today's parameters (RMSE_z, all days)")
        pv = hs.pivot_table(index=["Stock", "Model"], columns="H", values="RMSE_z")
        pr(pv.round(3).to_string())
        pr("")
        pr("median |z|")
        pr(hs.pivot_table(index=["Stock", "Model"], columns="H", values="MedAbs_z").round(3).to_string())
        pr("")
        pv = hs.pivot_table(index=["Stock", "Model"], columns="H", values="Inside_pct")
        pr("share of options with |z| <= 1 (%)")
        pr(pv.round(1).to_string())
        pr("")
    he = res["hex"]
    if len(he):
        pr("skew-decay exponent of the quotes h days after the fit (market vs models)")
        pv = he.pivot_table(index=["Stock", "Series"], columns="H", values="Exponent")
        pr(pv.round(3).to_string())
        pr("")
    es = res["esum"]
    if len(es):
        pr("EXPERIMENT 2: fit on expiries up to s days only, price the far expiries (RMSE_z; Days >= max(90, s + 60))")
        far = es[es["Subset"].str.startswith("far")]
        pr(far.pivot_table(index=["Stock", "Variant", "Split"], columns="Model", values="RMSE_z").round(3).to_string())
        pr("")
        pr("same, median |z| (not pulled up by the occasional failed fit)")
        pr(far.pivot_table(index=["Stock", "Variant", "Split"], columns="Model", values="MedAbs_z").round(3).to_string())
        pr("")
        pr("skew of the far expiries relative to the market's (1.0 = right, > 1 too steep)")
        if "SkewRatio_to_market" in far:
            pr(far.pivot_table(index=["Stock", "Variant", "Split"], columns="Model", values="SkewRatio_to_market").round(2).to_string())
            pr("")
        pr("best fit window per model (lowest far-expiry RMSE_z):")
        for (tag, variant, m), g in far.groupby(["Stock", "Variant", "Model"]):
            b = g.loc[g["RMSE_z"].idxmin()]
            pr(f"  {tag} {variant:5s} {m:7s} best s = {int(b['Split'])} d (RMSE_z {b['RMSE_z']:.3f})")
        pr("")
    ws = res["wsum"]
    if len(ws):
        pr("EXPERIMENT 3: structural parameters frozen over a window of W days (W = 0: full re-fit every day), RMSE_z")
        for h in (1, 5, 10):
            g = ws[ws["H"] == h]
            if len(g):
                pr(f"  quotes {h} day(s) ahead")
                pr(g.pivot_table(index=["Stock", "Model"], columns="W", values="RMSE_z").round(3).to_string())
        pr("")
        pr("best window per model at h = 5 (lowest RMSE_z):")
        g5 = ws[ws["H"] == 5]
        for (tag, m), g in g5.groupby(["Stock", "Model"]):
            b = g.loc[g["RMSE_z"].idxmin()]
            pr(f"  {tag} {m:7s} best W = {int(b['W'])} (RMSE_z {b['RMSE_z']:.3f}; daily re-fit {g[g['W'] == 0]['RMSE_z'].min():.3f})")
        pr("")
    txt = "\n".join(L)
    with open(os.path.join(out, "OVERNIGHT_SUMMARY.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    sc.to_csv(os.path.join(out, "scoreboard.csv"), index=False)
    return txt


# ----------------------------------------------------------------------------- charts
def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, colors=INK2)


def _end_labels(ax, xs, series, fontsize=8):
    """Direct labels at the right end of each line, nudged apart so they do not overlap."""
    ys = sorted([(y, m) for m, y in series.items() if np.isfinite(y)])
    lo, hi = ax.get_ylim()
    gap = 0.045 * (hi - lo)
    pos, last = {}, -np.inf
    for y, m in ys:
        p = max(y, last + gap)
        pos[m], last = p, p
    for m, p in pos.items():
        ax.annotate(m, (xs, series[m]), xytext=(xs + 0.12, p), textcoords="data", fontsize=fontsize, color=INK2,
                    va="center", annotation_clip=False)


def charts(res, cfg, out, stocks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hs, es, ws = res["hs"], res["esum"], res["wsum"]
    models = list(cfg["models"])

    if len(hs):
        fig, axes = plt.subplots(1, len(stocks), figsize=(6.4 * len(stocks), 4.2), facecolor=SURF, squeeze=False)
        hz = sorted(hs["H"].unique())
        for ax, tag in zip(axes[0], stocks):
            _style(ax)
            g = hs[hs["Stock"] == tag]
            ends = {}
            for m in models:
                gm = g[g["Model"] == m].set_index("H")["RMSE_z"].reindex(hz)
                ax.plot(range(len(hz)), gm.values, color=COLORS[m], lw=2, marker="o", ms=6, mec=SURF, mew=1.5, label=m)
                ends[m] = gm.values[-1]
            ax.set_xticks(range(len(hz)))
            ax.set_xticklabels(["in-sample" if h == 0 else f"+{h} d" for h in hz])
            ax.set_xlim(-0.3, len(hz) - 1 + 0.9)
            ax.set_ylabel("RMSE_z (error in half-spreads)", fontsize=9, color=INK2)
            ax.set_title(f"{tag}: quotes h trading days after the fit, today's parameters", fontsize=10, loc="left", color=INK)
            _end_labels(ax, len(hz) - 1, ends)
            ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=4)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "horizons.png"), dpi=130, facecolor=SURF)
        plt.close(fig)

    far = es[es["Subset"].str.startswith("far")] if len(es) else es
    if len(far):
        variants = [v for v in ("free", "prior") if v in set(far["Variant"])]
        fig, axes = plt.subplots(len(stocks), len(variants), figsize=(7.2 * len(variants), 3.8 * len(stocks)), facecolor=SURF, squeeze=False)
        splits = sorted(far["Split"].unique())
        w = 0.19
        for r, tag in enumerate(stocks):
            for c, v in enumerate(variants):
                ax = axes[r][c]
                _style(ax)
                g = far[(far["Stock"] == tag) & (far["Variant"] == v)]
                allv = np.array([g[(g["Model"] == m) & (g["Split"] == s)]["RMSE_z"].mean() for m in models for s in splits], float)
                cap = min(np.nanmax(allv), 3 * np.nanmedian(allv))
                for k, m in enumerate(models):
                    vals = np.array([g[(g["Model"] == m) & (g["Split"] == s)]["RMSE_z"].mean() for s in splits], float)
                    xs = np.arange(len(splits)) + (k - 1.5) * w
                    ax.bar(xs, np.minimum(vals, cap), width=w - 0.02, color=COLORS[m], label=m)
                    for x_, val in zip(xs, vals):                    # bars cut at the cap carry their value as a label
                        if np.isfinite(val) and val > cap:
                            ax.text(x_, cap, f"{val:.1f}", ha="center", va="bottom", fontsize=7, color=INK2)
                ax.set_ylim(0, cap * 1.12)
                ax.set_xticks(range(len(splits)))
                ax.set_xticklabels([f"fit up to {s} d" for s in splits])
                ax.set_ylabel("far-expiry RMSE_z", fontsize=9, color=INK2)
                what = "short expiries only" if v == "free" else "short expiries + yesterday's surface as prior"
                ax.set_title(f"{tag}, {what}", fontsize=9.5, loc="left", color=INK)
                if r == 0 and c == 0:
                    ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "extrapolation.png"), dpi=130, facecolor=SURF)
        plt.close(fig)

    if len(ws):
        h = 5 if 5 in set(ws["H"]) else sorted(ws["H"].unique())[-1]
        fig, axes = plt.subplots(1, len(stocks), figsize=(6.4 * len(stocks), 4.2), facecolor=SURF, squeeze=False)
        for ax, tag in zip(axes[0], stocks):
            _style(ax)
            g = ws[(ws["Stock"] == tag) & (ws["H"] == h)]
            Ws = sorted(g["W"].unique())
            ends = {}
            for m in models:
                gm = g[g["Model"] == m].set_index("W")["RMSE_z"].reindex(Ws)
                ax.plot(range(len(Ws)), gm.values, color=COLORS[m], lw=2, marker="o", ms=6, mec=SURF, mew=1.5, label=m)
                ends[m] = gm.values[-1]
            ax.set_xticks(range(len(Ws)))
            ax.set_xticklabels(["daily re-fit" if x == 0 else f"{x} d" for x in Ws])
            ax.set_xlim(-0.3, len(Ws) - 1 + 0.9)
            ax.set_ylabel(f"RMSE_z, quotes {h} days ahead", fontsize=9, color=INK2)
            ax.set_title(f"{tag}: slow parameters frozen over W days, only the state re-fitted", fontsize=10, loc="left", color=INK)
            _end_labels(ax, len(Ws) - 1, ends)
            ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=4)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "structural_window.png"), dpi=130, facecolor=SURF)
        plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    global _LOG_PATH
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stocks", nargs="+", default=jc.STOCKS, choices=jc.STOCKS)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="worker processes (default: cores - 1)")
    ap.add_argument("--hours", type=float, default=None, help="stop starting new tasks after this many hours")
    ap.add_argument("--phases", nargs="+", type=int, default=[1, 2, 3], choices=[1, 2, 3])
    ap.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    ap.add_argument("--splits", nargs="+", type=int, default=list(SPLITS), help="fit-window ends (days to expiry) for phase 2")
    ap.add_argument("--windows", nargs="+", type=int, default=list(WINDOWS), help="structure windows (days) for phase 3")
    ap.add_argument("--restarts", type=int, default=RESTARTS, help="extra restarts per fit (0 = none)")
    ap.add_argument("--prior-weight", type=float, default=PRIOR_WEIGHT)
    ap.add_argument("--max-days", type=int, default=None, help="only the first n trading days per stock")
    ap.add_argument("--smoke", action="store_true", help="12 days per stock, 2 restarts, own output folder")
    ap.add_argument("--analysis-only", action="store_true")
    ap.add_argument("--n-boot", type=int, default=500, help="bootstrap draws for the exponent tables")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    if a.smoke:
        a.max_days, a.restarts = a.max_days or 12, min(a.restarts, 2)
    out = os.path.join(HERE, "output_overnight_smoke" if a.smoke else "output_overnight")
    cfg = dict(models=list(a.models), horizons=list(a.horizons), splits=list(a.splits), windows=list(a.windows),
               restarts=a.restarts, seed=a.seed, prior_weight=a.prior_weight)
    h = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    cfg["chk"] = os.path.join(out, f"checkpoints_{h}")
    os.makedirs(cfg["chk"], exist_ok=True)
    _LOG_PATH = os.path.join(out, "run.log")
    t0 = time.time()
    log(f"overnight: stocks {a.stocks}, models {a.models}, horizons {a.horizons}, fit windows {a.splits}, "
        f"structure windows {a.windows}, restarts {a.restarts}, jobs {a.jobs}")
    log(f"checkpoints: {cfg['chk']}")
    ndays = {}
    for tag in a.stocks:                                  # also fills the data cache once, before the workers start
        n = len(np.unique(jc.load_options(tag)["Date"]))
        ndays[tag] = min(n, a.max_days) if a.max_days else n
    log("trading days per stock: " + ", ".join(f"{t} {n}" for t, n in ndays.items()))
    if not a.analysis_only:
        budget = (a.hours * 3600) if a.hours else None
        # cumulative time limits so that a short budget still reaches every phase
        share = {1: 0.25, 2: 0.95, 3: 1.00}
        for ph, label in ((1, "phase 1 (horizons)"), (2, "phase 2 (maturity windows)"), (3, "phase 3 (structure windows)")):
            if ph not in a.phases:
                continue
            deadline = t0 + budget * share[ph] if budget else float("inf")
            ok = run_phase(ph, a.stocks, ndays, cfg, a.jobs, deadline, label)
            if not ok:
                log(f"{label} not complete; continuing with what exists")
    log("analysis ...")
    res = analyse(cfg, a.stocks, a.n_boot)
    n_done = {p: len(glob.glob(os.path.join(cfg["chk"], f"p{p}_*.pkl"))) for p in (1, 2, 3)}
    txt = write_summary(res, cfg, out, n_done)
    try:
        charts(res, cfg, out, a.stocks)
    except Exception:
        log("chart error:\n" + traceback.format_exc())
    print("\n" + txt)
    log(f"done in {(time.time() - t0) / 3600:.2f} h. Files are in {out}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
