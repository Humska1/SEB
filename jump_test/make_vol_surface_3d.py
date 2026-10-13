"""
Data for the 3D implied-volatility surface page (market quotes vs Merton vs Bates), every trading day, both stocks.

Reads output/options_fit_<STOCK>.csv and params_daily_<STOCK>.csv (made by calibrate.py) and writes
output_3d/surface_data.json and the standalone page output_3d/vol_surface_3d.html (vol_surface_3d_template.html + data):
  * market points per day: strike/forward, days to expiry, implied vol (mid, bid, ask), model IVs and z at each quote
  * model surfaces per day on a data-shaped grid: 24 maturities between the shortest and longest expiry of the day,
    25 strikes per maturity spanning the quoted strike range at that maturity (interpolated between expiries);
    out-of-the-money convention (puts below the forward, calls above), priced with the COS pricer, inverted to Black-Scholes IV
  * daily fit statistics (RMSE_z, share inside bid/ask) for BS_slice, Merton and Bates

    python make_vol_surface_3d.py      (about a minute)
"""
import json
import os
import time

import numpy as np
import pandas as pd

import jump_calib as jc
import jump_models as jm

seb = jc.seb
HERE = jc.HERE
SRC = os.path.join(HERE, "output")
OUT = os.path.join(HERE, "output_3d")
STOCKS = ("BOL", "VOLVB")
MODELS = ("Merton", "Bates")
N_T, N_K = 24, 25
REPORTS = {"BOL": ["2025-07-18", "2025-10-22"], "VOLVB": ["2025-07-17", "2025-10-17"]}


def r1(a, nd=1):
    a = np.asarray(a, float)
    return [None if not np.isfinite(v) else round(float(v), nd) for v in a.ravel()]


def theta_of(model, row):
    return np.array([row[p] for p in model.pnames], float)


def day_surface(d, params):
    S = float(d["StockMid"].iloc[0])
    ex = d.groupby("T").LogMoneyness.agg(["min", "max"]).sort_index()
    t_lo, t_hi = float(ex.index.min()), float(ex.index.max())
    T = np.linspace(np.sqrt(t_lo), np.sqrt(t_hi), N_T) ** 2
    kmin = np.interp(T, ex.index.values, ex["min"].values)
    kmax = np.interp(T, ex.index.values, ex["max"].values)
    kk = np.array([np.linspace(a, b, N_K) for a, b in zip(kmin, kmax)])          # (N_T, N_K)
    TT = np.repeat(T[:, None], N_K, axis=1)
    F = S * np.exp((jc.R - jc.Q) * TT)
    K = F * np.exp(kk)
    ic = kk >= 0
    surf = {}
    for name in MODELS:
        model = jm.MODELS[name]
        th = theta_of(model, params[name])
        p = jm.cos_price(model, th, np.full(K.size, S), K.ravel(), TT.ravel(), ic.ravel(), jc.R, jc.Q)
        iv = seb.implied_vol(p, np.full(K.size, S), K.ravel(), TT.ravel(), jc.R, jc.Q, ic.ravel())
        surf[name] = r1(100 * iv, 2)
    rows = dict(days=r1(T * 365, 2), kmin=r1(kmin, 4), kmax=r1(kmax, 4))
    return S, rows, surf


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    data = dict(meta=dict(models=list(MODELS), n_t=N_T, n_k=N_K, reports=REPORTS,
                          note="IV in %, x = strike/forward in %, y = calendar days to expiry"))
    for st in STOCKS:
        o = pd.read_csv(os.path.join(SRC, f"options_fit_{st}.csv"))
        P = pd.read_csv(os.path.join(SRC, f"params_daily_{st}.csv"))
        days = []
        for dt, d in o.groupby("Date", sort=True):
            pr = P[P.Date == dt].set_index("Model")
            if not all(m in pr.index for m in MODELS):
                continue
            d = d.copy()
            d["IsCall"] = d["CallPut"].astype(str).str.upper().str.startswith("C")
            S, rows, surf = day_surface(d, {m: pr.loc[m] for m in MODELS})
            mk = dict(x=r1(100 * d.K_over_F, 2), y=r1(d.Days, 0), iv=r1(100 * d.IV_Mid, 2),
                      ivb=r1(100 * d.IV_Bid, 2), iva=r1(100 * d.IV_Ask, 2),
                      K=r1(d.Strike, 2), cp="".join("C" if c else "P" for c in d.IsCall),
                      **{f"iv_{m}": r1(100 * d[f"{m}_IV"], 2) for m in MODELS},
                      **{f"z_{m}": r1(d[f"{m}_z"], 2) for m in MODELS})
            stats = {m: dict(rmse=round(float(pr.loc[m, "RMSE_z"]), 3),
                             inside=round(float(pr.loc[m, "InsideBidAsk_pct"] if pd.notna(pr.loc[m].get("InsideBidAsk_pct", np.nan))
                                                else pr.loc[m, "Inside_pct"]), 1))
                     for m in ("BS_slice",) + MODELS if m in pr.index}
            days.append(dict(date=dt, S=round(S, 2), rows=rows, surf=surf, mk=mk, stats=stats))
        data[st] = days
        print(f"{st}: {len(days)} days, {time.time() - t0:.0f} s", flush=True)
    path = os.path.join(OUT, "surface_data.json")
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB) in {time.time() - t0:.0f} s")
    write_page(path)


def write_page(json_path):
    """Standalone page output_3d/vol_surface_3d.html (open in any browser; needs internet for Plotly and the fonts)."""
    tmpl = os.path.join(HERE, "vol_surface_3d_template.html")
    if not os.path.exists(tmpl):
        return
    with open(tmpl, encoding="utf-8") as f:
        body = f.read()
    with open(json_path, encoding="utf-8") as f:
        body = body.replace("/*__DATA__*/", f.read())
    page = ("<!doctype html>\n<html lang=\"sv\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n</head>\n<body>\n"
            + body + "\n</body>\n</html>\n")
    out = os.path.join(OUT, "vol_surface_3d.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
