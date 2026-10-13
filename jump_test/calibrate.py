"""
calibrate.py  -  daily calibration of Merton, Kou and Bates on the SEB option data
==================================================================================

One parameter set per model and trading day, fitted to all expiries of that day at once by least squares on
    z = (model price - mid) / half-spread            (half-spread floored at 0.05 SEK, as in seb_merton_analys.py)
with bounds, several start points (the previous day's parameters, a Merton-based start for Kou/Bates and fixed starts)
and the same mild day-to-day penalty as Merton_dag (REG_WEIGHT = 3, per-parameter scales in jump_models.py).

    python calibrate.py                    # every trading day, both stocks (about 12 min on 2 cores with --jobs 2)
    python calibrate.py --jobs 2           # the two stocks in parallel
    python calibrate.py --day-step 5       # every 5th day, quick look (output_step5/)
    python calibrate.py --stocks BOL --max-days 3          # smoke test

Output (output/): params_daily_<STOCK>.csv, options_fit_<STOCK>.csv, options_oos_<STOCK>.csv (see jump_calib.run_stock)
and runtimes.csv.
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

import jump_calib as jc


def _job(args):
    tag, kw = args
    return tag, jc.run_stock(tag, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", nargs="+", default=jc.STOCKS, choices=jc.STOCKS)
    ap.add_argument("--day-step", type=int, default=1, help="calibrate every n-th trading day (default 1 = all days)")
    ap.add_argument("--max-days", type=int, default=None, help="only the first n calibrated days (smoke test)")
    ap.add_argument("--models", nargs="+", default=jc.MODEL_ORDER, choices=list(jc.jm.MODELS))
    ap.add_argument("--reg-weight", type=float, default=jc.REG_WEIGHT)
    ap.add_argument("--jobs", type=int, default=1, help="processes (one per stock); 2 = both stocks in parallel")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    out_dir = a.out_dir or (jc.OUT_DIR if a.day_step == 1 else f"{jc.OUT_DIR}_step{a.day_step}")
    kw = dict(day_step=a.day_step, models=tuple(a.models), reg_weight=a.reg_weight, max_days=a.max_days, out_dir=out_dir)
    print(f"Calibrating {a.models} on {a.stocks}, every {a.day_step}. trading day, reg weight {a.reg_weight}, "
          f"jobs {a.jobs}. Output: {out_dir}")
    t0 = time.time()
    if a.jobs > 1 and len(a.stocks) > 1:
        with ProcessPoolExecutor(max_workers=min(a.jobs, len(a.stocks))) as ex:
            res = dict(ex.map(_job, [(t, kw) for t in a.stocks]))
    else:
        res = dict(_job((t, kw)) for t in a.stocks)
    rows = []
    for tag, (params, _, rt) in res.items():
        n_days = params["Date"].nunique()
        for m in a.models:
            rows.append(dict(Stock=tag, Model=m, Days=n_days, Seconds=rt[m], Seconds_per_day=rt[m] / n_days))
        rows.append(dict(Stock=tag, Model="(whole stock incl. BS, OOS, IV)", Days=n_days, Seconds=rt["total_wall"],
                         Seconds_per_day=rt["total_wall"] / n_days))
    rtdf = pd.DataFrame(rows)
    rtdf.to_csv(os.path.join(out_dir, "runtimes.csv"), index=False)
    print("\nRuntimes (calibration only, per model; last row per stock = wall time of the whole stock):")
    print(rtdf.round(2).to_string(index=False))
    print(f"\nTotal wall time: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
