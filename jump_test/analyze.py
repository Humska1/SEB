"""
analyze.py  -  metrics, tables and charts from the calibration output
=====================================================================

Reads  <in-dir>/params_daily_<STOCK>.csv, options_fit_<STOCK>.csv and options_oos_<STOCK>.csv (written by calibrate.py)
and writes to the same folder

  summary_table.csv            one row per stock x model: #parameters, fit, out-of-sample fit, per-parameter measures,
                               bound hits, skew-decay exponent with bootstrap CI, runtime
  summary_by_maturity.csv      fit (RMSE_z, share |z|<=1, IV errors) and mean skew per maturity bucket
  skew_pairs_<STOCK>.csv       the skew of every (date, expiry): market and each model
  skew_exponents.csv           decay exponents (ln|skew| on ln T) with 95 % bootstrap CI over dates, three subsets
  skew_exponent_differences.csv  bootstrap CI of Model - Market, Kou - Merton, Bates - Merton, Bates - Kou ...
  heatmap_iv_error.csv         mean (model IV - market IV) in vol points on the maturity x log-moneyness grid
  skew_loglog.png              log-log skew versus maturity: Market, Merton, Kou, Bates, one panel per stock
  heatmaps.png                 IV-error heatmaps: rows = stocks, columns = Merton, Kou, Bates

    python analyze.py                       (reads output/)
    python analyze.py --in-dir output_step5
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

import jump_calib as jc
import jump_metrics as jmx

SERIES = ["Market", "Merton", "Kou", "Bates"]
MODELS = ["Merton", "Kou", "Bates"]
ALL = ["BS_slice"] + MODELS

# chart colours: same surface/ink as seb_merton_analys.py; blue/orange/green are colour-blind safe; Market in ink
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
COL = {"Market": INK, "Merton": "#2a78d6", "Kou": "#eb6834", "Bates": "#1baf7a"}
MARK = {"Market": "o", "Merton": "s", "Kou": "^", "Bates": "D"}   # secondary (non-colour) encoding


def load(tag, in_dir):
    p = pd.read_csv(os.path.join(in_dir, f"params_daily_{tag}.csv"), parse_dates=["Date"])
    o = pd.read_csv(os.path.join(in_dir, f"options_fit_{tag}.csv"), parse_dates=["Date", "Expiry"])
    return p, o


def pooled_oos(g):
    n = g["OOS_N"].fillna(0).values
    if n.sum() == 0:
        return np.nan, np.nan
    return (np.sqrt(np.sum(n * g["OOS_RMSE_z"].fillna(0).values ** 2) / n.sum()),
            np.sum(n * g["OOS_Inside_pct"].fillna(0).values) / n.sum())


def bound_text(g):
    s = g["AtBound"].fillna("").astype(str)
    cnt = s.str.split(",").explode().replace("", np.nan).value_counts()
    return "; ".join(f"{k}: {v}/{len(g)}" for k, v in cnt.items()), s.map(lambda x: 0 if x == "" else len(x.split(","))).mean()


def per_parameter_table(p):
    """Gaussian AIC/BIC per day with the error variance estimated from the residuals:
         AIC = n ln(RSS/n) + 2k ,   BIC = n ln(RSS/n) + k ln n ,    RSS = sum of z^2.
    (An earlier version used RSS + 2k, i.e. the error variance fixed at one half-spread^2; that over-penalises extra
    parameters whenever the residual RMSE_z is below 1, as it is for BOL.)  Still heuristic: the residuals are correlated
    across strikes within an expiry."""
    d = p.copy()
    n = d["N_options"]
    d["AIC"] = n * np.log(d["RSS"] / n) + 2 * d["NPar"]
    d["BIC"] = n * np.log(d["RSS"] / n) + d["NPar"] * np.log(n)
    piv = {c: d.pivot(index="Date", columns="Model", values=c) for c in ("RSS", "AIC", "BIC")}
    rows = {}
    for m in ["Kou", "Bates"]:
        k_extra = jc.jm.MODELS[m].n_par - jc.jm.MODELS["Merton"].n_par
        rows[m] = dict(
            RSS_reduction_vs_Merton_pct=100 * (1 - piv["RSS"][m].sum() / piv["RSS"]["Merton"].sum()),
            RSS_reduction_per_extra_param=(piv["RSS"]["Merton"] - piv["RSS"][m]).mean() / k_extra,
            dAIC_vs_Merton_mean=(piv["AIC"][m] - piv["AIC"]["Merton"]).mean(),
            Share_days_AIC_better_than_Merton=100 * (piv["AIC"][m] < piv["AIC"]["Merton"]).mean(),
            dBIC_vs_Merton_mean=(piv["BIC"][m] - piv["BIC"]["Merton"]).mean(),
            Share_days_BIC_better_than_Merton=100 * (piv["BIC"][m] < piv["BIC"]["Merton"]).mean())
    return rows


def heat_min_n(n_days, override=None):
    """At least 30 quotes per heatmap cell with all 167 days, scaled with the number of calibrated days."""
    return override if override else max(6, int(round(30 * n_days / 167)))


# =============================================================================
# charts
# =============================================================================
def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6, which="both", alpha=0.8)
    ax.set_axisbelow(True)


def _spread_labels(ys, min_gap):
    """Nudge label positions (in log space) apart so end labels do not collide."""
    order = np.argsort(ys)
    out = np.array(ys, float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    return out


def chart_loglog(results, path, n_days_note):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(results), figsize=(6.6 * len(results), 5.4), facecolor=SURF)
    axes = np.atleast_1d(axes)
    for ax, (tag, sk, ex) in zip(axes, results):
        _style(ax)
        d = sk[(sk["Market"] < 0) & sk[SERIES].notna().all(axis=1)].copy()
        d["bucket"] = pd.cut(d["Days"], jmx.MAT_BINS, labels=jmx.MAT_LABELS)
        # faint dots: the individual market skews (date x expiry)
        ax.scatter(d["Days"], -d["Market"], s=9, color=MUTED, alpha=0.28, linewidths=0, zorder=1)
        ends = {}
        e_all = ex[ex["Subset"].str.startswith("all")].set_index("Series")
        x_fit = np.geomspace(max(d["Days"].min(), 5), d["Days"].max(), 50)
        for s in SERIES:
            g = d.groupby("bucket", observed=True).agg(Days=("Days", "median"), sk=(s, lambda x: -x.median()))
            lw = 2.4 if s == "Market" else 1.9
            ax.plot(g["Days"], g["sk"], color=COL[s], linewidth=lw, marker=MARK[s], markersize=7.5, markeredgecolor=SURF,
                    markeredgewidth=1.5, zorder=3 if s != "Market" else 4)
            # fitted power law (thin): |skew| = c T^b from the pooled regression
            x = np.log(d["T"].values)
            y = np.log(-np.clip(d[s].values, None, -1e-4))
            b, a0 = np.polyfit(x, y, 1)
            ax.plot(x_fit, np.exp(a0) * (x_fit / jc.seb.DAYS_PER_YEAR) ** b, color=COL[s], linewidth=1.0, alpha=0.55, zorder=2)
            ends[s] = np.log(g["sk"].iloc[-1])
        # direct labels at the right end of the lines (nudged apart), plus a legend with exponent and 95 % CI
        names = list(ends)
        pos = _spread_labels([ends[n] for n in names], 0.11)
        xr = d["Days"].max() * 1.18
        for n, yy in zip(names, pos):
            ax.text(xr, np.exp(yy), n, color=INK2, fontsize=9.5, va="center", ha="left", fontweight="bold" if n == "Market" else "normal")
        handles = []
        for s in SERIES:
            r = e_all.loc[s]
            handles.append(plt.Line2D([0], [0], color=COL[s], marker=MARK[s], markersize=7, linewidth=2,
                                      markeredgecolor=SURF, label=f"{s}: {r['Exponent']:+.2f}  [{r['CI_low']:+.2f}, {r['CI_high']:+.2f}]"))
        leg = ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=9, title="decay exponent b in |skew| ~ T^b\n(95 % bootstrap CI over dates)",
                        title_fontsize=8.5, labelcolor=INK2)
        leg.get_title().set_color(INK2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(d["Days"].min() * 0.85, d["Days"].max() * 1.75)
        ax.set_xticks([7, 14, 30, 60, 100, 200, 330])
        ax.set_xticklabels(["7", "14", "30", "60", "100", "200", "330"])
        ax.set_yticks([0.1, 0.15, 0.2, 0.3, 0.4, 0.6])
        ax.set_yticklabels(["0.10", "0.15", "0.20", "0.30", "0.40", "0.60"])
        ax.minorticks_off()
        ax.set_xlabel("days to expiry (log scale)", fontsize=9.5, color=INK2)
        ax.set_ylabel("|skew| = |d IV / d log-moneyness| (log scale)", fontsize=9.5, color=INK2)
        ax.set_title(f"{tag}", fontsize=11.5, color=INK, loc="left")
    fig.suptitle("Skew term structure: market vs Merton, Kou and Bates  -  markers: median skew per maturity bucket; thin lines: "
                 "pooled power-law fit; grey dots: market skew of each date x expiry", fontsize=10, color=INK2, x=0.01, ha="left", y=0.995)
    fig.text(0.01, 0.005, n_days_note, fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def chart_heatmaps(grids, path, vmax=4.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])   # same blue-grey-red as felkarta
    stocks = list(dict.fromkeys(g[0] for g in grids))
    fig, axes = plt.subplots(len(stocks), 3, figsize=(15.5, 4.6 * len(stocks)), facecolor=SURF, squeeze=False,
                             gridspec_kw=dict(hspace=0.42))
    im = None
    for (tag, m, h, n, min_n) in grids:
        ax = axes[stocks.index(tag), MODELS.index(m)]
        h = h.reindex(index=jmx.MAT_LABELS, columns=jmx.K_LABELS)
        im = ax.imshow(h.values.astype(float), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(h.shape[1]))
        ax.set_xticklabels(h.columns, fontsize=8.5, color=INK2, rotation=25)
        ax.set_yticks(range(h.shape[0]))
        ax.set_yticklabels(h.index, fontsize=9, color=INK2)
        for (i, j), v in np.ndenumerate(h.values.astype(float)):
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8.5, color=INK)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{tag}: {m}", fontsize=11, color=INK, loc="left")
        if m == "Merton":
            ax.set_ylabel("days to expiry", fontsize=9, color=INK2)
        if tag == stocks[-1]:
            ax.set_xlabel("log-moneyness k = ln(K/F)   (puts left, calls right)", fontsize=9, color=INK2)
    cb = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.015)
    cb.set_label("mean IV error, model - market (vol points)\nblue: model too low, red: too high", fontsize=9, color=INK2)
    cb.ax.tick_params(colors=INK2, labelsize=8.5)
    fig.suptitle(f"Implied-vol error by maturity and moneyness (cells with fewer than {grids[0][4]} quotes left blank; "
                 f"colour scale clipped at +-{vmax:.0f} vol points)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


# =============================================================================
# main
# =============================================================================
def analyze(d_in, n_boot=1000, min_cell=None, make_charts=True, verbose=True):
    """Everything in this file for one output folder. Returns dict(summary, exponents, differences)."""
    t0 = time.time()
    summ_rows, mat_rows, ex_rows, diff_rows, heat_rows, chart_in, grids = [], [], [], [], [], [], []
    for tag in jc.STOCKS:
        if not os.path.exists(os.path.join(d_in, f"params_daily_{tag}.csv")):
            continue
        p, o = load(tag, d_in)
        n_days = p["Date"].nunique()
        o = jmx.add_buckets(o)

        # ---- metric 1: fit
        fit = jmx.fit_table(o, ALL).set_index("Model")
        by_mat = jmx.fit_table(o, ALL, by=["MatBucket"])
        ppt = per_parameter_table(p)

        # ---- metric 2: skew term structure (in-sample: same quotes the models were fitted to)
        series = {"Market": "IV_Mid", **{m: f"{m}_IV" for m in MODELS}}
        sk = jmx.skew_table(o, series)
        sk.to_csv(os.path.join(d_in, f"skew_pairs_{tag}.csv"), index=False)
        ex, df = jmx.decay_exponents(sk, SERIES, n_boot=n_boot)
        ex.insert(0, "Sample", "in-sample")
        df.insert(0, "Sample", "in-sample")
        # ... and out-of-sample: the next trading day's quotes priced with today's parameters
        f_oos = os.path.join(d_in, f"options_oos_{tag}.csv")
        e_oos = None
        if os.path.exists(f_oos):
            oo = pd.read_csv(f_oos, parse_dates=["Date", "Expiry", "FitDate"])
            sk_o = jmx.skew_table(oo, series)
            sk_o.to_csv(os.path.join(d_in, f"skew_pairs_oos_{tag}.csv"), index=False)
            ex_o, df_o = jmx.decay_exponents(sk_o, SERIES, n_boot=n_boot)
            ex_o.insert(0, "Sample", "next-day out-of-sample")
            df_o.insert(0, "Sample", "next-day out-of-sample")
            ex, df = pd.concat([ex, ex_o]), pd.concat([df, df_o])
            e_oos = ex_o[ex_o["Subset"].str.startswith("all")].set_index("Series")
        ex.insert(0, "Stock", tag)
        df.insert(0, "Stock", tag)
        ex_rows.append(ex)
        diff_rows.append(df)
        e_all = ex[(ex["Sample"] == "in-sample") & ex["Subset"].str.startswith("all")].set_index("Series")
        chart_in.append((tag, sk, ex[ex["Sample"] == "in-sample"]))
        sb = jmx.skew_by_bucket(sk, SERIES)
        sb.insert(0, "Stock", tag)

        # ---- metric 3: IV-error grids
        mn = heat_min_n(n_days, min_cell)
        for m in MODELS:
            h, n = jmx.iv_error_grid(o, m, min_n=mn)
            h_all, _ = jmx.iv_error_grid(o, m, min_n=1)
            grids.append((tag, m, h, n, mn))
            for mb in h_all.index:
                for kb in h_all.columns:
                    heat_rows.append(dict(Stock=tag, Model=m, MatBucket=mb, KBucket=kb, MeanIVerr_vp=h_all.loc[mb, kb],
                                          N=int(n.loc[mb, kb]), Shown=bool(np.isfinite(h.loc[mb, kb]))))

        # ---- summary rows
        for m in ALL:
            pm = p[p["Model"] == m]
            r = dict(Stock=tag, Model=m, Days_calibrated=n_days, Params_per_day=pm["NPar"].mean(), N_options=int(fit.loc[m, "N"]),
                     RMSE_z=fit.loc[m, "RMSE_z"], Inside_pct=fit.loc[m, "Inside_pct"], InsideBidAsk_pct=fit.loc[m, "InsideBidAsk_pct"],
                     MedianAbsIVerr_vp=fit.loc[m, "MedianAbsIV_vp"], RMSE_IV_vp=fit.loc[m, "RMSE_IV_vp"],
                     MeanIVerr_vp=fit.loc[m, "MeanIVerr_vp"])
            if m != "BS_slice":
                r["OOS_RMSE_z"], r["OOS_Inside_pct"] = pooled_oos(pm)
                txt, nb = bound_text(pm)
                r.update(Params_at_bound_per_day=nb, Params_free_per_day=pm["NPar"].mean() - nb, Bound_hits=txt,
                         Runtime_s_per_day=pm["Runtime_s"].mean(), Max_COS_gap_SEK=pm["COS_gap"].max())
                if m in ppt:
                    r.update(ppt[m])
                r.update(Skew_exponent=e_all.loc[m, "Exponent"], Skew_exp_CI_low=e_all.loc[m, "CI_low"],
                         Skew_exp_CI_high=e_all.loc[m, "CI_high"])
                if e_oos is not None:
                    r.update(Skew_exponent_OOS=e_oos.loc[m, "Exponent"], Skew_exp_OOS_CI_low=e_oos.loc[m, "CI_low"],
                             Skew_exp_OOS_CI_high=e_oos.loc[m, "CI_high"])
            summ_rows.append(r)
        mrow = dict(Stock=tag, Model="Market", Days_calibrated=n_days, Skew_exponent=e_all.loc["Market", "Exponent"],
                    Skew_exp_CI_low=e_all.loc["Market", "CI_low"], Skew_exp_CI_high=e_all.loc["Market", "CI_high"])
        if e_oos is not None:
            mrow.update(Skew_exponent_OOS=e_oos.loc["Market", "Exponent"], Skew_exp_OOS_CI_low=e_oos.loc["Market", "CI_low"],
                        Skew_exp_OOS_CI_high=e_oos.loc["Market", "CI_high"])
        summ_rows.append(mrow)

        # ---- by maturity: fit + skew level
        bm = by_mat.merge(sb.drop(columns="Stock"), on="MatBucket", how="left")
        bm.insert(0, "Stock", tag)
        bm["Skew_model"] = bm.apply(lambda r: r.get(f"Skew_{r['Model']}", np.nan), axis=1)
        bm["Skew_ratio_model_to_market"] = bm["Skew_model"] / bm["Skew_Market"]
        mat_rows.append(bm[["Stock", "MatBucket", "Model", "N", "RMSE_z", "Inside_pct", "InsideBidAsk_pct", "MeanZ", "MeanIVerr_vp",
                            "RMSE_IV_vp", "MedianAbsIV_vp", "N_pairs", "Skew_Market", "Skew_model", "Skew_ratio_model_to_market"]])

    summ = pd.DataFrame(summ_rows)
    summ.to_csv(os.path.join(d_in, "summary_table.csv"), index=False)
    pd.concat(mat_rows).to_csv(os.path.join(d_in, "summary_by_maturity.csv"), index=False)
    ex_all = pd.concat(ex_rows)
    ex_all.to_csv(os.path.join(d_in, "skew_exponents.csv"), index=False)
    diff_all = pd.concat(diff_rows)
    diff_all.to_csv(os.path.join(d_in, "skew_exponent_differences.csv"), index=False)
    pd.DataFrame(heat_rows).to_csv(os.path.join(d_in, "heatmap_iv_error.csv"), index=False)

    if make_charts:
        n_note = (f"{summ['Days_calibrated'].dropna().iloc[0]} calibrated days per stock; skew from (date, expiry) pairs with >= 5 quotes "
                  f"spanning k = 0 and a negative market skew.")
        chart_loglog(chart_in, os.path.join(d_in, "skew_loglog.png"), n_note)
        chart_heatmaps(grids, os.path.join(d_in, "heatmaps.png"))

    if verbose:
        pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_colwidth", 60)
        print("\n================ SUMMARY (one row per stock x model) ================")
        cols = ["Stock", "Model", "Params_per_day", "Params_free_per_day", "RMSE_z", "Inside_pct", "OOS_RMSE_z", "OOS_Inside_pct",
                "MedianAbsIVerr_vp", "Skew_exponent", "Skew_exp_CI_low", "Skew_exp_CI_high", "Skew_exponent_OOS",
                "Skew_exp_OOS_CI_low", "Skew_exp_OOS_CI_high", "Runtime_s_per_day"]
        print(summ[[c for c in cols if c in summ]].round(3).to_string(index=False))
        print("\n---- per extra parameter (vs Merton) ----")
        pc = ["Stock", "Model", "RSS_reduction_vs_Merton_pct", "RSS_reduction_per_extra_param", "dAIC_vs_Merton_mean",
              "Share_days_AIC_better_than_Merton", "dBIC_vs_Merton_mean", "Share_days_BIC_better_than_Merton"]
        print(summ[summ["Model"].isin(["Kou", "Bates"])][pc].round(2).to_string(index=False))
        print("\n---- parameters sitting at their bounds (days with hit / calibrated days) ----")
        for _, r in summ[summ["Model"].isin(MODELS)].iterrows():
            print(f"  {r['Stock']:6s} {r['Model']:7s} {r['Bound_hits'] or '-'}")
        print("\n---- skew decay exponents (ln|skew| on ln T), 95 % bootstrap CI over dates ----")
        print(ex_all.round(3).to_string(index=False))
        print("\n---- differences (bootstrap) ----")
        print(diff_all.round(3).to_string(index=False))
        print(f"\nAnalysis runtime {time.time() - t0:.0f} s. Files written to {d_in}")
    return dict(summary=summ, exponents=ex_all, differences=diff_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=jc.OUT_DIR)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--min-cell", type=int, default=None, help="minimum quotes per heatmap cell (default: scaled with #days)")
    a = ap.parse_args()
    analyze(a.in_dir, a.n_boot, a.min_cell)


if __name__ == "__main__":
    main()
