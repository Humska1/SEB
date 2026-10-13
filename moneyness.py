"""
moneyness.py  -  how well each model prices options in each moneyness bucket (deep OTM puts ... deep OTM calls)
================================================================================================================

Moneyness = strike / forward, where the forward F = S * exp(r * T) is the stock price grown at the 2 % rate until expiry.

    bucket           strike / forward    what it is
    Deep OTM put     below 0.85          crash insurance: pays only if the stock falls more than 15 %
    OTM put          0.85 - 0.95         pays if the stock falls 5-15 %
    ATM put          0.95 - 1.00         strike just below the forward
    ATM call         1.00 - 1.05         strike just above the forward
    OTM call         1.05 - 1.15         pays if the stock rises 5-15 %
    Deep OTM call    above 1.15          pays only if the stock rises more than 15 %

Only out-of-the-money options are used (puts below the forward, calls above). An in-the-money option carries the same
information as the out-of-the-money option with the same strike, by put-call parity: call - put = S - K * exp(-r T).

Two uses:

    python moneyness.py                          from csv/fair_values_*.csv (run fair_value.py first):
                                                   csv/moneyness_options_<STOCK>.csv  every option with its bucket
                                                   csv/summary_by_moneyness.csv       table per bucket and model
                                                   results/fit_by_moneyness.png, results/moneyness_bias.png
    python moneyness.py --option 280 250 90 put  classify one option: stock price, strike, days to expiry, call/put;
                                                 the result is also added to csv/classified_options.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

import fair_value as fv

BINS = [0, 0.85, 0.95, 1.00, 1.05, 1.15, np.inf]
LABELS = ["Deep OTM put", "OTM put", "ATM put", "ATM call", "OTM call", "Deep OTM call"]
RANGES = ["< 0.85", "0.85-0.95", "0.95-1.00", "1.00-1.05", "1.05-1.15", "> 1.15"]


def bucket(k_over_f):
    """Moneyness bucket of strike / forward (one value or an array)."""
    return pd.cut(np.atleast_1d(k_over_f), BINS, right=False, labels=LABELS)


def classify(S, K, days, call_put):
    """Forward, strike / forward, in/out of the money and bucket of one option."""
    is_call = call_put.lower().startswith("c")
    F = S * np.exp((fv.R - fv.Q) * days / fv.DAYS_PER_YEAR)
    m = K / F
    itm = (K < F) if is_call else (K > F)
    twin = f"{'put' if is_call else 'call'} with the same strike"
    return dict(Forward=F, Strike_over_forward=m, Log_moneyness=np.log(m),
                Status="in the money" if itm else "out of the money",
                Bucket=bucket(m)[0] if not itm else f"(in the money; used via the out-of-the-money {twin})",
                Distance_pct=100 * (m - 1))


def table(stock):
    """Per bucket and model: number of options, spread, share inside bid/ask, average and typical IV error."""
    f = pd.read_csv(os.path.join(fv.CSV_DIR, f"fair_values_{stock}.csv"))
    f["Strike_over_forward"] = np.exp(f["k"])
    f["Bucket"] = bucket(f["Strike_over_forward"])
    per_option = f[["Date", "Expiry", "Days", "CallPut", "Strike", "S", "F", "Strike_over_forward", "Bucket", "Bid", "Ask",
                    "Mid", "IV_Mid"]].rename(columns={"S": "Stock_price", "F": "Forward", "IV_Mid": "Market_IV"})
    for m, label in fv.MODELS.items():
        per_option[f"{label}_price"] = f[f"{m}_Price"]
        per_option[f"{label}_IV_error_volpts"] = 100 * (f[f"{m}_IV"] - f["IV_Mid"])
        per_option[f"{label}_inside_bid_ask"] = (f[f"{m}_Price"] >= f["Bid"]) & (f[f"{m}_Price"] <= f["Ask"])
    per_option.to_csv(os.path.join(fv.CSV_DIR, f"moneyness_options_{stock}.csv"), index=False, float_format="%.6g")
    rows = []
    for b, g in f.groupby("Bucket", observed=True):
        for m, label in fv.MODELS.items():
            err = 100 * (g[f"{m}_IV"] - g["IV_Mid"])
            rows.append(dict(Stock=stock, Bucket=b, Range=RANGES[LABELS.index(b)], Model=label, Options=len(g),
                             Half_spread_volpts=100 * ((g["IV_Ask"] - g["IV_Bid"]) / 2).median(),
                             Inside_bid_ask_pct=100 * ((g[f"{m}_Price"] >= g["Bid"]) & (g[f"{m}_Price"] <= g["Ask"])).mean(),
                             Mean_IV_error_volpts=err.mean(), Median_abs_IV_error_volpts=err.abs().median(),
                             RMSE_z=np.sqrt(np.mean(g[f"{m}_z"] ** 2))))
    return pd.DataFrame(rows)


def chart(t, value, ylabel, title, path, zero_line=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    ticks = [f"{l}\n({r})" for l, r in zip(LABELS, RANGES)]
    for ax, (stock, name) in zip(axes, fv.STOCKS.items()):
        fv._style(ax)
        for model in fv.MODELS.values():
            g = t[(t["Stock"] == stock) & (t["Model"] == model)].set_index("Bucket").reindex(LABELS)
            ax.plot(range(len(LABELS)), g[value], color=fv.COLOR[model], lw=2, marker="o", ms=6, label=model)
        if zero_line:
            ax.axhline(0, color=fv.INK2, lw=0.8)
        ax.set_xticks(range(len(LABELS)))
        ax.set_xticklabels(ticks, fontsize=8.5)
        ax.set_xlabel("strike / forward", fontsize=9, color=fv.INK2)
        ax.set_title(name, loc="left", fontsize=11, color=fv.INK)
    axes[0].set_ylabel(ylabel, fontsize=9, color=fv.INK2)
    axes[0].legend(frameon=False, fontsize=9, loc="lower center")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=fv.INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--option", nargs=4, metavar=("STOCK_PRICE", "STRIKE", "DAYS", "CALL_PUT"),
                    help="classify one option instead of analysing the results")
    a = ap.parse_args()
    if a.option:
        S, K, days, cp = float(a.option[0]), float(a.option[1]), float(a.option[2]), a.option[3]
        res = classify(S, K, days, cp)
        for key, v in res.items():
            print(f"  {key:20s} {v:.4f}" if isinstance(v, float) else f"  {key:20s} {v}")
        path = os.path.join(fv.CSV_DIR, "classified_options.csv")
        row = pd.DataFrame([dict(Stock_price=S, Strike=K, Days=days, CallPut=cp, **res)])
        row.to_csv(path, mode="a", header=not os.path.exists(path), index=False, float_format="%.6g")
        print(f"  (added to {path})")
        return

    t = pd.concat([table(s) for s in fv.STOCKS], ignore_index=True)
    t.to_csv(os.path.join(fv.CSV_DIR, "summary_by_moneyness.csv"), index=False, float_format="%.4f")
    chart(t, "Inside_bid_ask_pct", "options priced inside bid/ask (%)",
          "Share of options each model prices inside the bid/ask spread, by moneyness",
          os.path.join(fv.RESULTS_DIR, "fit_by_moneyness.png"))
    chart(t, "Mean_IV_error_volpts", "model IV - market IV (vol points)",
          "Average pricing error by moneyness: below 0 the model is too cheap, above 0 too expensive",
          os.path.join(fv.RESULTS_DIR, "moneyness_bias.png"), zero_line=True)

    pd.set_option("display.width", 200)
    for col, what in (("Inside_bid_ask_pct", "Share priced inside bid/ask (%)"),
                      ("Mean_IV_error_volpts", "Average error in vol points (< 0: model too cheap)")):
        print(f"\n{what}")
        print(t.pivot_table(index=["Stock", "Bucket", "Range"], columns="Model", values=col, sort=False)
              [list(fv.MODELS.values())].round(1).to_string())
    print(f"\nTables in {fv.CSV_DIR}: moneyness_options_<STOCK>.csv, summary_by_moneyness.csv")
    print(f"Charts in {fv.RESULTS_DIR}: fit_by_moneyness.png, moneyness_bias.png")


if __name__ == "__main__":
    main()
