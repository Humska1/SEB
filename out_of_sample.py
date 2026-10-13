"""
out_of_sample.py  -  does Bates price FUTURE quotes better than Merton?
=======================================================================

A model with more parameters always fits today's quotes at least as well. The fair test is out of sample: take the
parameters fitted on day t (csv/detail/parameters_<STOCK>.csv from fair_value.py) and price the quotes of trading day t + h,
which the fit never saw, for h = 1, 2, 5, 10, 20 and 40 trading days.

For each horizon:
  * RMSE_z and the share of options priced inside bid/ask, pooled over all options and days
  * Diebold-Mariano test, Merton vs Bates: per fit day the loss is the mean z^2 of the future quotes; the test asks whether
    the log of Merton's loss over Bates' loss is zero on average (Newey-West standard error, because neighbouring days
    overlap). p < 0.05 means the difference is not luck.

    python out_of_sample.py          (run fair_value.py first; takes about a minute)
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

import fair_value as fv

HORIZONS = (1, 2, 5, 10, 20, 40)
MODELS = {"Merton": fv.MERTON, "Bates": fv.BATES}


def newey_west_t(d, lag):
    """t statistic for mean(d) = 0 with a Newey-West (Bartlett) long-run variance."""
    d = np.asarray(d, float)
    u = d - d.mean()
    lrv = (np.sum(u * u) + sum(2 * (1 - k / (lag + 1)) * np.sum(u[k:] * u[:-k]) for k in range(1, lag + 1))) / len(d)
    return d.mean() / np.sqrt(max(lrv, 1e-300) / len(d))


def run_stock(stock):
    quotes = fv.load_quotes(stock)
    days = {dt: g for dt, g in quotes.groupby("Date")}
    dates = sorted(days)
    params = pd.read_csv(os.path.join(fv.DETAIL_DIR, f"parameters_{stock}.csv"), parse_dates=["Date"])
    theta = {(r.Date, r.Model): np.array([getattr(r, p) for p in MODELS[r.Model].params]) for r in params.itertuples()}
    rows = []
    for i, dt in enumerate(dates):
        for h in HORIZONS:
            if i + h >= len(dates):
                continue
            d = days[dates[i + h]]
            S, K, T, ic, mid, w = fv.arrays(d)
            row = dict(Stock=stock, FitDate=dt, H=h, N=len(d))
            for name, model in MODELS.items():
                if (dt, name) not in theta:
                    continue
                price = fv.cos_price(model, theta[(dt, name)], S, K, T, ic)
                z = (price - mid) / w
                row[f"{name}_SSE"] = np.nansum(z ** 2)
                row[f"{name}_inside"] = np.sum((price >= d["Bid"].values) & (price <= d["Ask"].values))
            rows.append(row)
    return pd.DataFrame(rows)


def summarise(per_day):
    summary, tests = [], []
    for (stock, h), g in per_day.groupby(["Stock", "H"]):
        for name in MODELS:
            summary.append(dict(Stock=stock, H=h, Model=name, Fit_days=len(g), Options=int(g["N"].sum()),
                                RMSE_z=np.sqrt(g[f"{name}_SSE"].sum() / g["N"].sum()),
                                Inside_bid_ask_pct=100 * g[f"{name}_inside"].sum() / g["N"].sum()))
        loss_m, loss_b = g["Merton_SSE"] / g["N"], g["Bates_SSE"] / g["N"]
        lag = int(min(max(5, h), len(g) // 4))
        t = newey_west_t(np.log(loss_m / loss_b), lag)
        tests.append(dict(Stock=stock, H=h, Fit_days=len(g), RMSE_z_Merton=np.sqrt(loss_m.mean()),
                          RMSE_z_Bates=np.sqrt(loss_b.mean()), Bates_better_on_pct_of_days=100 * np.mean(loss_b < loss_m),
                          DM_t=t, p_value=2 * stats.norm.sf(abs(t)),
                          Verdict="Bates better" if (t > 0 and 2 * stats.norm.sf(abs(t)) < 0.05)
                          else "Merton better" if (t < 0 and 2 * stats.norm.sf(abs(t)) < 0.05) else "no significant difference"))
    return pd.DataFrame(summary), pd.DataFrame(tests)


def chart(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for ax, (stock, name) in zip(axes, fv.STOCKS.items()):
        fv._style(ax)
        for model in MODELS:
            g = summary[(summary["Stock"] == stock) & (summary["Model"] == model)].sort_values("H")
            ax.plot(range(len(g)), g["RMSE_z"], color=fv.COLOR[model], lw=2, marker="o", ms=6, label=model)
        ax.set_xticks(range(len(HORIZONS)))
        ax.set_xticklabels([f"+{h}" for h in HORIZONS])
        ax.set_xlabel("trading days after the fit", fontsize=9, color=fv.INK2)
        ax.set_title(name, loc="left", fontsize=11, color=fv.INK)
    axes[0].set_ylabel("pricing error, RMSE in half-spreads", fontsize=9, color=fv.INK2)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Out of sample: parameters fitted on day t, quotes of day t + h (lower is better)", x=0.01, ha="left",
                 fontsize=12, color=fv.INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    per_day = pd.concat([run_stock(s) for s in fv.STOCKS], ignore_index=True)
    summary, tests = summarise(per_day)
    summary.to_csv(os.path.join(fv.SUMMARY_DIR, "out_of_sample.csv"), index=False, float_format="%.4f")
    tests.to_csv(os.path.join(fv.SUMMARY_DIR, "out_of_sample_test.csv"), index=False, float_format="%.4g")
    chart(summary, os.path.join(fv.RESULTS_DIR, "out_of_sample.png"))
    pd.set_option("display.width", 200)
    print("Pricing error h trading days after the fit (RMSE in half-spreads)")
    print(summary.pivot_table(index=["Stock", "Model"], columns="H", values="RMSE_z").round(3).to_string())
    print("\nDiebold-Mariano test, Merton vs Bates")
    print(tests[["Stock", "H", "Fit_days", "Bates_better_on_pct_of_days", "DM_t", "p_value", "Verdict"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
