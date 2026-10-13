"""
Slide figure: the skew power law. |skew| against days to expiry on log-log axes, market vs Merton vs Bates,
with the band of exponents -0.3 to -0.5 documented for S&P 500 ATM skew (Gatheral et al.).

Reads output/skew_pairs_<STOCK>.csv and skew_exponents.csv (made by calibrate.py and analyze.py)
and writes output_figures/skew_powerlaw_light.png and skew_powerlaw_dark.png (16:9, 3200 x 1800 px).
Uses the Archivo and IBM Plex Mono fonts if their .ttf files are in jump_test/fonts/, otherwise DejaVu Sans.
Skew = slope of implied vol on log-moneyness per (day, expiry), shown as the vol points the implied vol rises
when the strike drops 10 %. Lines: OLS of log skew on log maturity over the same (day, expiry) pairs as the
exponent table (negative market skew, every series valid), so the slopes equal the tabulated exponents.

    python plot_skew_powerlaw.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "output")
OUT = os.path.join(HERE, "output_figures")
FONT_DIR = os.path.join(HERE, "fonts")
STOCKS = (("VOLVB", "Volvo B"), ("BOL", "Boliden"))
SERIES = ("Market", "Merton", "Kou", "Bates")
TO_VOLPTS = 100 * np.log(1 / 0.9)          # slope per unit log-moneyness -> vol points per 10 % lower strike
BAND = (-0.3, -0.5)

THEMES = {
    "light": dict(bg="#ffffff", ink="#10151c", ink2="#4b5664", rule="#dfe4ea", band="#e6eaef",
                  Market="#eb6834", Merton="#2a78d6", Bates="#1baf7a"),
    "dark": dict(bg="#0c1117", ink="#e8edf3", ink2="#9aa7b6", rule="#1f2833", band="#1e2a37",
                 Market="#d95926", Merton="#3987e5", Bates="#199e70"),
}


def font(name, fallback_weight="normal", size=12):
    path = os.path.join(FONT_DIR, name)
    if os.path.exists(path):
        return fm.FontProperties(fname=path, size=size)
    return fm.FontProperties(family="DejaVu Sans", weight=fallback_weight, size=size)


def load(stock):
    sk = pd.read_csv(os.path.join(SRC, f"skew_pairs_{stock}.csv"))
    sk = sk[(sk["Market"] < 0) & sk[list(SERIES)].notna().all(axis=1)].reset_index(drop=True)
    ex = pd.read_csv(os.path.join(SRC, "skew_exponents.csv"))
    ex = ex[(ex.Stock == stock) & (ex.Sample == "in-sample") & (ex.Subset == "all expiries (>=5d)")].set_index("Series")
    x = np.log(sk["T"].values)
    fits = {}
    for s in SERIES:
        y = np.log(np.maximum(-sk[s].values, 1e-4))
        b, a = np.polyfit(x, y, 1)
        fits[s] = dict(b=b, a=a, ci=(ex.loc[s, "CI_low"], ex.loc[s, "CI_high"]), tab=ex.loc[s, "Exponent"])
        assert abs(b - ex.loc[s, "Exponent"]) < 1e-6, (stock, s, b, ex.loc[s, "Exponent"])
    return sk, fits


def curve(fit, days):
    T = days / 365.0
    return np.exp(fit["a"] + fit["b"] * np.log(T)) * TO_VOLPTS


def spread_labels(ys, min_ratio=1.24):
    """Nudge label heights apart on a log axis so they do not overlap; returns new heights in the same order."""
    order = np.argsort(ys)
    out = np.array(ys, float)
    for k in range(1, len(order)):
        lo, hi = order[k - 1], order[k]
        if out[hi] < out[lo] * min_ratio:
            out[hi] = out[lo] * min_ratio
    return out


def draw(theme):
    C = THEMES[theme]
    f_title = font("Archivo-ExpandedExtraBold.ttf", "bold", 25)
    f_sub = font("Archivo-Regular.ttf", "normal", 13.5)
    f_panel = font("Archivo-SemiBold.ttf", "bold", 17)
    f_lab = font("IBMPlexMono-Medium.ttf", "normal", 12)
    f_tick = font("IBMPlexMono-Regular.ttf", "normal", 11)
    f_note = font("IBMPlexMono-Regular.ttf", "normal", 9.6)
    f_axis = font("IBMPlexMono-Regular.ttf", "normal", 11.5)

    fig = plt.figure(figsize=(16, 9), dpi=200, facecolor=C["bg"])
    fig.text(0.045, 0.935, "Merton's skew fades too fast. Bates keeps the market's power law.", color=C["ink"], fontproperties=f_title)
    fig.text(0.045, 0.893, "Skew per expiry and day, May to December 2025, measured as the vol points the implied vol rises when the strike drops 10 %. "
             "Log scales: a straight line is a power law,\nand its slope is the decay exponent. Shaded: exponents from −0.3 to −0.5, "
             "the range documented for S&P 500 index options.", color=C["ink2"], fontproperties=f_sub, va="top", linespacing=1.45)

    axes = [fig.add_axes([0.065 + i * 0.49, 0.235, 0.345, 0.56]) for i in range(2)]
    for ax, (stock, label) in zip(axes, STOCKS):
        sk, fits = load(stock)
        ax.set_facecolor(C["bg"])
        ax.set_xscale("log"); ax.set_yscale("log")
        d = sk["Days"].values
        lo_d, hi_d = max(5, d.min()), d.max()
        xs = np.geomspace(lo_d, hi_d, 120)
        # band through the market's fit at the geometric-mean maturity
        T0 = np.exp(np.mean(np.log(d)))
        y0 = curve(fits["Market"], np.array([T0]))[0]
        band_hi = y0 * (xs / T0) ** BAND[1]
        band_lo = y0 * (xs / T0) ** BAND[0]
        ax.fill_between(xs, np.minimum(band_lo, band_hi), np.maximum(band_lo, band_hi), color=C["band"], lw=0, zorder=0)
        # market cloud and binned medians
        mk = -sk["Market"].values * TO_VOLPTS
        ax.scatter(d, mk, s=7, color=C["Market"], alpha=0.16, lw=0, zorder=1)
        edges = np.geomspace(lo_d, hi_d * 1.0001, 10)
        bx, by = [], []
        for e0, e1 in zip(edges[:-1], edges[1:]):
            m = (d >= e0) & (d < e1)
            if m.sum() >= 15:
                bx.append(np.exp(np.mean(np.log(d[m])))); by.append(np.median(mk[m]))
        ax.scatter(bx, by, s=70, color=C["Market"], edgecolor=C["bg"], linewidth=1.6, zorder=5)
        # fitted power laws
        ends = {}
        for s, lw in (("Merton", 3.0), ("Bates", 3.0), ("Market", 1.6)):
            ys = curve(fits[s], xs)
            ax.plot(xs, ys, color=C[s], lw=lw, zorder=4 if s != "Market" else 3,
                    ls="-" if s != "Market" else (0, (5, 3)), solid_capstyle="round")
            ends[s] = ys[-1]
        # direct labels: coloured marker + ink text, nudged apart
        names = ["Market", "Bates", "Merton"]
        ylab = spread_labels([ends[n] for n in names])
        for n, yl in zip(names, ylab):
            ax.plot([hi_d * 1.08, hi_d * 1.22], [yl, yl], color=C[n], lw=3 if n != "Market" else 1.8,
                    ls="-" if n != "Market" else (0, (3, 2)), clip_on=False, solid_capstyle="round")
            ax.text(hi_d * 1.27, yl, f"{n} {fits[n]['b']:+.2f}".replace("-", "−"), color=C["ink"],
                    fontproperties=f_lab, va="center", ha="left", clip_on=False)
        ax.add_patch(Rectangle((0.655, 1.032), 0.03, 0.028, transform=ax.transAxes, color=C["band"], clip_on=False, lw=0))
        ax.text(0.695, 1.046, "shaded: S&P 500 range", transform=ax.transAxes, color=C["ink2"], fontproperties=f_note, va="center", ha="left")
        # axes styling
        ax.set_xlim(lo_d * 0.92, hi_d * 1.04)
        ax.set_ylim(0.35, 22)
        ax.xaxis.set_major_locator(FixedLocator([5, 10, 20, 50, 100, 200, 300]))
        ax.yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10, 20]))
        ax.xaxis.set_minor_locator(NullLocator()); ax.yaxis.set_minor_locator(NullLocator())
        fmt = FuncFormatter(lambda v, _: f"{v:g}")
        ax.xaxis.set_major_formatter(fmt); ax.yaxis.set_major_formatter(fmt)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_fontproperties(f_tick); t.set_color(C["ink2"])
        ax.tick_params(colors=C["rule"], length=0, pad=6)
        ax.grid(True, color=C["rule"], lw=0.8, zorder=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel("days to expiry", color=C["ink2"], fontproperties=f_axis, labelpad=8)
        ax.set_ylabel("skew (vol points per 10 % lower strike)", color=C["ink2"], fontproperties=f_axis, labelpad=8)
        ax.set_title(label, loc="left", color=C["ink"], fontproperties=f_panel, pad=10)
        ci = {n: fits[n]["ci"] for n in ("Market", "Bates", "Merton")}
        ax.text(0.0, -0.165, "95 % CI:  " + "   ".join(f"{n} [{ci[n][0]:.2f}, {ci[n][1]:.2f}]" for n in ci).replace("-", "−"),
                transform=ax.transAxes, color=C["ink2"], fontproperties=f_note)
    fig.text(0.045, 0.018, "Data: SEB daily closing bid/ask quotes for Volvo B and Boliden options, out of the money, 2 May to 30 Dec 2025. Models fitted once per day to all expiries.\n"
             "Skew = weighted slope of implied vol on log-moneyness (|k| < 0.2) per (day, expiry). Lines: OLS of log skew on log maturity; CIs by date-cluster bootstrap.\n"
             "Dots: market medians by maturity bin; faint dots: every (day, expiry). Kou (not shown) is close to Merton.\n"
             "Band: Gatheral, Rough Volatility (2018): S&P 500 ATM skew falls like maturity to a power between −0.3 and −0.5.",
             color=C["ink2"], fontproperties=f_note, va="bottom", linespacing=1.5)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"skew_powerlaw_{theme}.png")
    fig.savefig(path, facecolor=C["bg"])
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    for th in THEMES:
        draw(th)
