"""
SEB-projekt: tre market making-frågor
=====================================

Kör EFTER seb_merton_analys.py och seb_gamma_analys.py (läser deras resultat).

  1. RAPPORTDAGAR: hur stor rörelse prisade marknaden in inför rapporten, och hur
     stor blev den? Säger om "försäkringen" (optionerna) var för dyr eller för billig.
       - Terminsstruktur (bara info FÖRE rapporten): två löptider som båda täcker
         rapporten, total varians = sigma_d^2 * T + sigma_E^2  =>  sigma_E
         (Dubinsky, Johannes, Kaeck & Seeger 2019, Review of Financial Studies).
       - IV-kollaps (kontroll): samma löptid dagen före och dagen efter rapporten,
         sigma_E^2 = IV_före^2 * T_före - IV_efter^2 * T_efter.
     sigma_E är standardavvikelsen för rapportdagens rörelse; förväntad |rörelse| = 0.8 * sigma_E.

  2. GAP-TEST: en deltahedgad SHORT-position (short gamma) chockas med ett
     omedelbart hopp i aktien (-10 % ... +10 %). Förlusten räknas med
       - BS "sticky strike" (varje option behåller sin implicita vol),
       - BS med vol-spik (+5 vol-punkter vid nedgång, som vid en krasch),
       - Merton (dagens kalibrerade parametrar, Merton-delta).
     Förlusten anges i SEK, i halvspreadar och i "dagar av theta" (hur många dagars
     tidsvärde som försvinner). Testet valideras mot de faktiska stora rörelsedagarna.

  3. SPREADAR vs HOPPRISK: breddar market makers bid-ask där hopprisken är störst?
     Spreaden mäts i vol-punkter (IV_ask - IV_bid), som på en optionsdesk.
       - median-spread per löptid och moneyness,
       - samma optioner före vs efter rapporten (parat test),
       - regression mot Mertons hopprisk per vega, med standardfel klustrade per dag.

Kör:  python seb_handelsrisk.py
"""

import os

import numpy as np
import pandas as pd
from scipy.special import ndtr

import seb_merton_analys as sm
import seb_gamma_analys as ga

R, Q = sm.RISK_FREE_RATE, sm.DIVIDEND_YIELD
GAPS = [-0.10, -0.075, -0.05, -0.025, 0.025, 0.05, 0.075, 0.10]
VOL_SPIKE = 0.05            # +5 vol-punkter vid nedgångsgap
EVENT_WINDOW = 10           # handelsdagar före/efter rapport i spread-studien
PAIR_WINDOW = 3             # dagar före resp. efter i det parade testet
MIN_VEGA_VP = ga.MIN_VEGA_VP
N_GH = 20                   # Gauss-Hermite-noder för Mertons förväntade hoppkostnad


# =============================================================================
# GEMENSAMT
# =============================================================================
def bs_price_theta(S, K, T, sigma, is_call, r=R):
    """BS-pris och theta per KALENDERDAG (negativt för en lång option)."""
    T = np.maximum(T, 1e-8)
    sigma = np.maximum(sigma, 1e-8)
    sq = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sq)
    d2 = d1 - sigma * sq
    pdf = np.exp(-0.5 * d1 ** 2) / np.sqrt(2 * np.pi)
    call = S * ndtr(d1) - K * np.exp(-r * T) * ndtr(d2)
    put = K * np.exp(-r * T) * ndtr(-d2) - S * ndtr(-d1)
    th_c = -S * pdf * sigma / (2 * sq) - r * K * np.exp(-r * T) * ndtr(d2)
    th_p = -S * pdf * sigma / (2 * sq) + r * K * np.exp(-r * T) * ndtr(-d2)
    return np.where(is_call, call, put), np.where(is_call, th_c, th_p) / 365.0


def load_common(tag):
    params = ga.load_params(tag)
    op, st, dates = ga.build_panel(tag)
    op["F"] = op["S"] * np.exp((R - Q) * op["T"])
    op["LogM"] = np.log(op["Strike"] / op["F"])
    op["OTM"] = np.where(op["IsCall"], op["LogM"] >= 0, op["LogM"] < 0)
    return params, op, st, pd.DatetimeIndex(dates)


# =============================================================================
# 1. RAPPORTDAGAR
# =============================================================================
def atm_iv(g):
    """ATM-vol för en (dag, löptid): linjär interpolation i log-moneyness av OTM-optionernas IV."""
    g = g[g["OTM"] & np.isfinite(g["IV"]) & (np.abs(g["LogM"]) < 0.10)].sort_values("LogM")
    if g.empty:
        return np.nan
    below, above = g[g["LogM"] < 0], g[g["LogM"] >= 0]
    if len(below) and len(above):
        b, a = below.iloc[-1], above.iloc[0]
        w = (0 - b["LogM"]) / (a["LogM"] - b["LogM"])
        return (1 - w) * b["IV"] + w * a["IV"]
    return g.iloc[np.argmin(np.abs(g["LogM"].values))]["IV"]


def report_moves(tag, op, st, dates):
    spot = st.set_index("Date")["S"]
    rows = []
    for rd in pd.to_datetime(ga.REPORT_DATES[tag]):
        d0 = dates[dates < rd].max()                    # sista handelsdag före rapporten
        d1 = dates[dates >= rd].min()                   # rapportdagen (stängning efter rapporten)
        o0, o1 = op[op["Date"] == d0], op[op["Date"] == d1]
        exps = sorted(e for e in o0["Expiry"].unique() if e >= rd)
        row = dict(Aktie=tag, Rapportdatum=rd.date(), Dag_före=d0.date(), Rapportdag=d1.date(),
                   Faktisk_rörelse=spot[d1] / spot[d0] - 1)            # enkel procentuell förändring
        # (a) terminsstruktur, bara info före rapporten
        if len(exps) >= 2:
            e1, e2 = exps[0], exps[1]
            s1, s2 = atm_iv(o0[o0["Expiry"] == e1]), atm_iv(o0[o0["Expiry"] == e2])
            T1, T2 = (e1 - d0).days / sm.DAYS_PER_YEAR, (e2 - d0).days / sm.DAYS_PER_YEAR
            var_d = (s2 ** 2 * T2 - s1 ** 2 * T1) / (T2 - T1)
            var_E = s1 ** 2 * T1 - var_d * T1
            row.update(Löptid_1=pd.Timestamp(e1).date(), Löptid_2=pd.Timestamp(e2).date(), ATM_IV_1=s1, ATM_IV_2=s2,
                       Vol_utan_rapport=np.sqrt(var_d) if var_d > 0 else np.nan,
                       Prisad_rörelse_sigma_terminsstruktur=np.sqrt(var_E) if var_E > 0 else np.nan)
        # (b) IV-kollaps: första löptid som lever efter rapportdagen
        alive = [e for e in exps if e > d1]
        if alive:
            e = alive[0]
            pre, post = atm_iv(o0[o0["Expiry"] == e]), atm_iv(o1[o1["Expiry"] == e])
            Tp, Tq = (e - d0).days / sm.DAYS_PER_YEAR, (e - d1).days / sm.DAYS_PER_YEAR
            var_E2 = pre ** 2 * Tp - post ** 2 * Tq
            row.update(Kollaps_löptid=pd.Timestamp(e).date(), IV_före=pre, IV_efter=post,
                       IV_kollaps_vp=100 * (post - pre),
                       Prisad_rörelse_sigma_kollaps=np.sqrt(var_E2) if var_E2 > 0 else np.nan)
        rows.append(row)
    R_ = pd.DataFrame(rows)
    main = R_["Prisad_rörelse_sigma_terminsstruktur"].fillna(R_.get("Prisad_rörelse_sigma_kollaps"))
    R_["Prisad_rörelse_sigma"] = main
    R_["Förväntad_abs_rörelse"] = main * np.sqrt(2 / np.pi)
    R_["Faktisk_abs_rörelse"] = R_["Faktisk_rörelse"].abs()
    R_["Faktisk_delat_med_prisad"] = R_["Faktisk_abs_rörelse"] / R_["Prisad_rörelse_sigma"]
    R_["Tolkning"] = np.where(
        R_["Faktisk_abs_rörelse"] > R_["Förväntad_abs_rörelse"],
        "Större än väntat: försäkringen var billig (long gamma gynnas av rörelsen)",
        "Mindre än väntat: försäkringen var dyr (short gamma gynnas)")
    # Short gamma-P&L på rapportdagen från gamma-analysen (om den finns)
    gf = os.path.join(sm.OUT_DIR, f"SEB_Gamma_{tag}.xlsx")
    if os.path.exists(gf):
        gp = pd.read_excel(gf, sheet_name="Daglig_positioner",
                           usecols=["Date", "NextDate", "PnL_long_BS", "PnL_short_BS_halvspr", "GammaPnL", "VegaPnL"])
        gp["NextDate"] = pd.to_datetime(gp["NextDate"])
        agg = gp.groupby("NextDate").agg(Short_gamma_PnL_SEK=("PnL_long_BS", lambda x: -x.mean()),
                                         Short_gamma_PnL_halvspr=("PnL_short_BS_halvspr", "mean"),
                                         Short_varav_rörelse=("GammaPnL", lambda x: -x.mean()),
                                         Short_varav_volfall=("VegaPnL", lambda x: -x.mean()))
        R_ = R_.merge(agg, left_on=pd.to_datetime(R_["Rapportdag"]), right_index=True, how="left")
        R_ = R_.drop(columns=[c for c in R_.columns if c == "key_0"])
    return R_


# =============================================================================
# 2. GAP-TEST
# =============================================================================
def merton_jump_cost_per_day(S, K, T, ic, sig, lam, mu, sj, V0, dM):
    """
    Mertons förväntade hoppkostnad per kalenderdag för en deltahedgad SHORT-position:
      lambda * E_J[ V(S e^J) - V(S) - delta*S*(e^J - 1) ] / 365,   J ~ N(mu, sj^2)
    (Gauss-Hermite). Det är den del av optionspriset som betalar för hopprisk.
    """
    x, w = np.polynomial.hermite.hermgauss(N_GH)
    tot = np.zeros_like(S, dtype=float)
    for xi, wi in zip(x, w):
        y = mu + np.sqrt(2.0) * sj * xi
        Sj = S * np.exp(y)
        Vj = sm.merton_price(Sj, K, T, R, Q, sig, lam, mu, sj, ic)
        tot += wi / np.sqrt(np.pi) * (Vj - V0 - dM * (Sj - S))
    return lam * tot / 365.0


def gap_ladder(tag, params):
    e = sm.load_stock(tag).merge(params, on="Date", how="left")      # samma OTM-urval som huvudanalysen
    S, K, T, ic, iv = (e["StockMid"].values, e["Strike"].values, e["T"].values, e["IsCall"].values,
                       e["IV_Mid"].values)
    sig, lam, mu, sj = (e[c].values for c in ("Sigma", "Lambda", "Mu_J", "Sigma_J"))
    V0, theta_day = bs_price_theta(S, K, T, iv, ic)
    dBS = ga.bs_greeks(S, K, T, iv, ic)[0]
    V0M = sm.merton_price(S, K, T, R, Q, sig, lam, mu, sj, ic)
    dM = ga.merton_delta(S, K, T, ic, sig, lam, mu, sj)
    hs = e["HalfSpreadW"].values
    base = e[["Date", "Isin", "CallPut", "Strike", "StockMid", "Days", "MaturityBucket", "MoneynessBucket"]].copy()
    base["Theta_short_per_dag"] = -theta_day                      # tidsvärdet short-positionen tjänar per dag
    base["Halvspread"] = hs
    base["Merton_hoppkostnad_per_dag"] = merton_jump_cost_per_day(S, K, T, ic, sig, lam, mu, sj, V0M, dM)
    rows = []
    for g in GAPS:
        S1 = S * (1 + g)
        loss_bs = -(bs_price_theta(S1, K, T, iv, ic)[0] - V0 - dBS * S * g)          # short: -(dV - delta dS)
        iv_sp = iv + (VOL_SPIKE if g < 0 else 0.0)
        loss_sp = -(bs_price_theta(S1, K, T, iv_sp, ic)[0] - V0 - dBS * S * g)
        loss_m = -(sm.merton_price(S1, K, T, R, Q, sig, lam, mu, sj, ic) - V0M - dM * S * g)
        b = base.copy()
        b["Gap"] = g
        b["PnL_short_BS"] = loss_bs
        b["PnL_short_BS_volspik"] = loss_sp
        b["PnL_short_Merton"] = loss_m
        rows.append(b)
    L = pd.concat(rows, ignore_index=True)
    L["PnL_BS_halvspr"] = L["PnL_short_BS"] / L["Halvspread"]
    L["PnL_BS_thetadagar"] = L["PnL_short_BS"] / L["Theta_short_per_dag"]
    return L, base


def summarize_ladder(L, by):
    return (L.groupby(by + ["Gap"], observed=True)
            .agg(Antal=("PnL_short_BS", "size"),
                 PnL_BS_median_SEK=("PnL_short_BS", "median"),
                 PnL_BS_volspik_median_SEK=("PnL_short_BS_volspik", "median"),
                 PnL_Merton_median_SEK=("PnL_short_Merton", "median"),
                 PnL_BS_median_halvspr=("PnL_BS_halvspr", "median"),
                 PnL_BS_median_thetadagar=("PnL_BS_thetadagar", "median"))
            .reset_index())


def summarize_premium(base, by):
    g = base.groupby(by, observed=True).agg(
        Antal=("Theta_short_per_dag", "size"),
        Theta_short_per_dag_median=("Theta_short_per_dag", "median"),
        Merton_hoppkostnad_per_dag_median=("Merton_hoppkostnad_per_dag", "median"),
        Halvspread_median=("Halvspread", "median")).reset_index()
    g["Hoppdel_av_theta_pct"] = 100 * g["Merton_hoppkostnad_per_dag_median"] / g["Theta_short_per_dag_median"]
    return g


def validate_gaps(tag):
    """Jämför gap-testets förutsägelse (BS sticky strike, faktisk rörelse) med utfallet på stora rörelsedagar."""
    gf = os.path.join(sm.OUT_DIR, f"SEB_Gamma_{tag}.xlsx")
    if not os.path.exists(gf):
        return pd.DataFrame()
    dt = pd.read_excel(gf, sheet_name="Dagtyper")
    gp = pd.read_excel(gf, sheet_name="Daglig_positioner",
                       usecols=["Date", "Isin", "S0", "Strike", "T", "IsCall", "IV0", "PnL_long_BS", "GammaPnL", "VegaPnL"])
    big = dt[(dt["DayType_alt"] != "Normal")][["Date", "NextDate", "Ret", "DayType_alt"]]
    rows = []
    for _, d in big.iterrows():
        g = gp[gp["Date"] == d["Date"]]
        if g.empty:
            continue
        gret = np.exp(d["Ret"]) - 1
        S, K, T, ic, iv = g["S0"].values, g["Strike"].values, g["T"].values, g["IsCall"].values.astype(bool), g["IV0"].values
        V0 = bs_price_theta(S, K, T, iv, ic)[0]
        dBS = ga.bs_greeks(S, K, T, iv, ic)[0]
        pred = -(bs_price_theta(S * (1 + gret), K, T, iv, ic)[0] - V0 - dBS * S * gret)
        rows.append(dict(Datum=pd.Timestamp(d["NextDate"]).date(), Typ=d["DayType_alt"], Rörelse=gret,
                         Antal_optioner=len(g), Förutsagt_short_PnL_SEK=pred.mean(),
                         Faktiskt_short_PnL_SEK=-g["PnL_long_BS"].mean(),
                         Faktiskt_varav_rörelse=-g["GammaPnL"].mean(), Faktiskt_varav_volförändring=-g["VegaPnL"].mean()))
    return pd.DataFrame(rows)


# =============================================================================
# 3. SPREADAR vs HOPPRISK
# =============================================================================
def spread_panel(tag, op, params, dates):
    s = op[op["OTM"] & (op["Days"] >= sm.MIN_DAYS)].copy()
    a = (s["S"].values, s["Strike"].values, s["T"].values, R, Q)
    ic = s["IsCall"].values
    s["IV_bid"] = sm.implied_vol(s["Bid"].values, *a, ic)
    s["IV_ask"] = sm.implied_vol(s["Ask"].values, *a, ic)
    _, _, vega = ga.bs_greeks(s["S"].values, s["Strike"].values, s["T"].values, s["IV"].values, ic)
    s["Vega_vp"] = vega * 0.01
    s = s[np.isfinite(s["IV_bid"]) & np.isfinite(s["IV_ask"]) & (s["Vega_vp"] >= MIN_VEGA_VP)].copy()
    s["Spread_vp"] = 100 * (s["IV_ask"] - s["IV_bid"])
    s["Spread_SEK"] = s["Ask"] - s["Bid"]
    s = s[s["Spread_vp"] > 0].copy()
    s["MaturityBucket"] = sm.maturity_bucket(s["Days"])
    s["MoneynessBucket"] = sm.moneyness_bucket(np.exp(s["LogM"]))
    # Mertons hopprisk per option: förväntad hoppkostnad per dag, per vol-punkt vega
    s = s.merge(params, on="Date", how="left")
    S, K, T = s["S"].values, s["Strike"].values, s["T"].values
    ic = s["IsCall"].values
    sig, lam, mu, sj = (s[c].values for c in ("Sigma", "Lambda", "Mu_J", "Sigma_J"))
    V0M = sm.merton_price(S, K, T, R, Q, sig, lam, mu, sj, ic)
    dM = ga.merton_delta(S, K, T, ic, sig, lam, mu, sj)
    s["Hoppkostnad_per_dag"] = merton_jump_cost_per_day(S, K, T, ic, sig, lam, mu, sj, V0M, dM)
    s["Hopprisk_per_vega"] = s["Hoppkostnad_per_dag"] / s["Vega_vp"]
    # Rapporttiming i handelsdagar
    didx = {d: i for i, d in enumerate(dates)}
    s["DagIdx"] = s["Date"].map(didx)
    reps = [(pd.Timestamp(r), didx[dates[dates >= pd.Timestamp(r)].min()]) for r in ga.REPORT_DATES[tag]]
    s["Offset"] = np.nan
    s["Rapport"] = ""
    for r, ridx in reps:
        m = (s["DagIdx"] - ridx).abs() <= EVENT_WINDOW
        s.loc[m, "Offset"] = s.loc[m, "DagIdx"] - ridx
        s.loc[m, "Rapport"] = str(r.date())
        s.loc[m, "Exponerad"] = s.loc[m, "Expiry"] > r    # optionen lever över rapporten
    s["Före_rapport"] = (s["Offset"] >= -5) & (s["Offset"] <= -1) & (s["Exponerad"] == True)
    s["Efter_rapport"] = (s["Offset"] >= 0) & (s["Offset"] <= 4)
    return s


def paired_report_test(s, seed=3):
    """Samma option: medel-spread 3 dagar före rapporten vs 3 dagar efter (rapportdagen räknas som efter)."""
    rng = np.random.default_rng(seed)
    rows = []
    x = s[s["Exponerad"] == True]
    for r, g in x.groupby("Rapport"):
        pre_g = g[(g["Offset"] >= -PAIR_WINDOW) & (g["Offset"] <= -1)].groupby("Isin")
        post_g = g[(g["Offset"] >= 0) & (g["Offset"] <= PAIR_WINDOW - 1)].groupby("Isin")
        pre, post = pre_g["Spread_vp"].mean(), post_g["Spread_vp"].mean()
        both = pd.concat([pre.rename("före"), post.rename("efter"), pre_g["Spread_SEK"].mean().rename("före_SEK"),
                          post_g["Spread_SEK"].mean().rename("efter_SEK")], axis=1).dropna()
        if len(both) < 5:
            continue
        diff = both["efter"] - both["före"]
        boot = np.array([np.median(rng.choice(diff.values, len(diff))) for _ in range(2000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append(dict(Rapport=r, Antal_optioner=len(both), Median_spread_före_vp=both["före"].median(),
                         Median_spread_efter_vp=both["efter"].median(), Median_förändring_vp=diff.median(),
                         KI_2_5=lo, KI_97_5=hi, Andel_smalare_efter=np.mean(diff < 0),
                         Median_spread_före_SEK=both["före_SEK"].median(), Median_spread_efter_SEK=both["efter_SEK"].median(),
                         Median_förändring_SEK=(both["efter_SEK"] - both["före_SEK"]).median()))
    return pd.DataFrame(rows)


def ols_cluster(y, X, groups):
    """OLS med standardfel klustrade per grupp (här: handelsdag)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b
    meat = np.zeros((X.shape[1], X.shape[1]))
    for gval in np.unique(groups):
        idx = groups == gval
        sg = X[idx].T @ u[idx]
        meat += np.outer(sg, sg)
    G, n, k = len(np.unique(groups)), len(y), X.shape[1]
    c = G / (G - 1) * (n - 1) / (n - k)
    V = c * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(V))
    r2 = 1 - np.sum(u ** 2) / np.sum((y - y.mean()) ** 2)
    return b, se, r2


def spread_regression(s):
    """
    log(spread i vol-punkter) mot hopprisk och kontroller.
    Spec A: hopprisk + löptid + rapportfönster + log(vega) + log(mid)
    Spec B: som A plus moneyness-hinkar (kollar om hopprisk förklarar något UTÖVER put/call-läge)
    """
    d = s.copy()
    d["logSpread"] = np.log(d["Spread_vp"])
    d["Hopprisk_z"] = (d["Hopprisk_per_vega"] - d["Hopprisk_per_vega"].mean()) / d["Hopprisk_per_vega"].std()
    d["log_vega"] = np.log(d["Vega_vp"])
    d["log_mid"] = np.log(d["Mid"])
    base_cols = ["Hopprisk_z", "Före_rapport", "Efter_rapport", "log_vega", "log_mid"]
    mat = pd.get_dummies(d["MaturityBucket"].astype(str), prefix="Löptid", drop_first=True)
    mon = pd.get_dummies(d["MoneynessBucket"].astype(str), prefix="Moneyness", drop_first=True)
    out = []
    groups = d["Date"].values
    for spec, extra in [("A", [mat]), ("B", [mat, mon])]:
        X = pd.concat([pd.Series(1.0, index=d.index, name="Konstant"), d[base_cols].astype(float)] + extra, axis=1)
        b, se, r2 = ols_cluster(d["logSpread"].values, X.values, groups)
        for name, bi, si in zip(X.columns, b, se):
            out.append(dict(Spec=spec, Variabel=name, Koefficient=bi, Std_fel=si, t=bi / si,
                            Effekt_pct=100 * (np.exp(bi) - 1), R2=r2, N=len(d)))
    return pd.DataFrame(out)


# =============================================================================
# FIGURER
# =============================================================================
_plt = sm._plt
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]    # kategorisk palett, fast ordning


def plot_report_moves(allrep):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 4.4), facecolor=sm.SURF)
    sm._style(ax)
    lab = [f"{r.Aktie}\n{r.Rapportdatum}" for r in allrep.itertuples()]
    x = np.arange(len(allrep))
    ax.bar(x - 0.19, 100 * allrep["Förväntad_abs_rörelse"], width=0.36, color=SERIES[0],
           label="Prisad förväntad |rörelse| (från optionerna dagen före)")
    ax.bar(x + 0.19, 100 * allrep["Faktisk_abs_rörelse"], width=0.36, color=SERIES[1], label="Faktisk |rörelse|")
    for i, r in enumerate(allrep.itertuples()):
        ax.text(x[i] + 0.19, 100 * r.Faktisk_abs_rörelse + 0.15, f"{100 * r.Faktisk_rörelse:+.1f} %",
                ha="center", fontsize=8, color=sm.INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(lab, fontsize=9, color=sm.INK2)
    ax.set_ylabel("Rörelse på rapportdagen (%)", fontsize=9, color=sm.INK2)
    ax.legend(frameon=False, fontsize=9, labelcolor=sm.INK2, loc="upper left")
    ax.set_title("Rapportdagar: vad optionerna prisade in vs vad som hände", fontsize=11, color=sm.INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(sm.OUT_DIR, "Rapport_prisad_vs_faktisk.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_gap(tag, L, val):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=sm.SURF)
    ax = axes[0]
    sm._style(ax)
    picks = [("1. <1M", "2. 0.85-0.95 (OTM put)", "<1 mån, OTM put"), ("1. <1M", "5. 1.05-1.15 (OTM call)", "<1 mån, OTM call"),
             ("2. 1-3M", "2. 0.85-0.95 (OTM put)", "1-3 mån, OTM put"), ("2. 1-3M", "5. 1.05-1.15 (OTM call)", "1-3 mån, OTM call")]
    for (mb, mo, lab), c in zip(picks, SERIES):
        g = L[(L["MaturityBucket"].astype(str) == mb) & (L["MoneynessBucket"].astype(str) == mo)]
        m = g.groupby("Gap")["PnL_BS_thetadagar"].median()
        ax.plot(100 * m.index, m.values, color=c, linewidth=2, marker="o", markersize=5, label=lab)
    ax.axhline(0, color=sm.AXIS, linewidth=1)
    ax.set_xlabel("Omedelbart hopp i aktien (%)", fontsize=9, color=sm.INK2)
    ax.set_ylabel("Förlust i antal dagars theta", fontsize=9, color=sm.INK2)
    ax.set_title("Deltahedgad short-position: hur många dagars premie ett hopp kostar", fontsize=10,
                 color=sm.INK, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=sm.INK2)
    ax = axes[1]
    sm._style(ax)
    if len(val):
        for typ, c in [("Stor rörelse", SERIES[0]), ("Rapport", SERIES[1])]:
            v = val[val["Typ"] == typ]
            ax.scatter(v["Förutsagt_short_PnL_SEK"], v["Faktiskt_short_PnL_SEK"], s=40, color=c, label=typ,
                       edgecolors=sm.SURF, linewidths=0.6, zorder=3)
            for r in v.itertuples():
                # etikett bara på rapportdagar och den största rörelsedagen, så att texterna inte krockar
                if typ == "Rapport" or abs(r.Rörelse) == val.loc[val["Typ"] == "Stor rörelse", "Rörelse"].abs().max():
                    ax.annotate(f"{r.Datum} ({100 * r.Rörelse:+.1f} %)",
                                (r.Förutsagt_short_PnL_SEK, r.Faktiskt_short_PnL_SEK),
                                xytext=(6, 3), textcoords="offset points", fontsize=7, color=sm.INK2)
        lim = [min(val["Förutsagt_short_PnL_SEK"].min(), val["Faktiskt_short_PnL_SEK"].min()) - 0.2,
               max(val["Förutsagt_short_PnL_SEK"].max(), val["Faktiskt_short_PnL_SEK"].max()) + 0.2]
        ax.plot(lim, lim, color=sm.MUTED, linestyle="--", linewidth=1)
        ax.legend(frameon=False, fontsize=8, labelcolor=sm.INK2, loc="upper left")
    ax.set_xlabel("Gap-testets förutsägelse (SEK per option)", fontsize=9, color=sm.INK2)
    ax.set_ylabel("Faktiskt utfall (SEK per option)", fontsize=9, color=sm.INK2)
    ax.set_title("Kontroll mot verkliga stora dagar (streckad = perfekt)", fontsize=10, color=sm.INK, loc="left")
    fig.suptitle(f"{tag}: gap-test för short gamma", fontsize=12, color=sm.INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(sm.OUT_DIR, f"{tag}_gap_test.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# HUVUDPROGRAM
# =============================================================================
def run_one(tag):
    params, op, st, dates = load_common(tag)

    rep = report_moves(tag, op, st, dates)

    L, base = gap_ladder(tag, params)
    gap_bucket = summarize_ladder(L, ["MaturityBucket", "MoneynessBucket"])
    gap_all = (L.groupby("Gap").agg(PnL_BS_medel_SEK=("PnL_short_BS", "mean"),
                                     PnL_BS_volspik_medel_SEK=("PnL_short_BS_volspik", "mean"),
                                     PnL_Merton_medel_SEK=("PnL_short_Merton", "mean"),
                                     PnL_BS_median_halvspr=("PnL_BS_halvspr", "median"),
                                     PnL_BS_median_thetadagar=("PnL_BS_thetadagar", "median")).reset_index())
    prem = summarize_premium(base, ["MaturityBucket", "MoneynessBucket"])
    prem_all = summarize_premium(base.assign(Alla="Alla OTM"), ["Alla"])
    val = validate_gaps(tag)

    s = spread_panel(tag, op, params, dates)
    spread_grid = s.pivot_table(index="MaturityBucket", columns="MoneynessBucket", values="Spread_vp",
                                aggfunc="median", observed=True)
    paired = paired_report_test(s)
    event = s[s["Exponerad"] == True].groupby(["Rapport", "Offset"])["Spread_vp"].median().unstack(0)
    reg = spread_regression(s)

    out = os.path.join(sm.OUT_DIR, f"SEB_Handelsrisk_{tag}.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        rep.to_excel(xw, sheet_name="1_Rapport_rörelse", index=False)
        gap_all.to_excel(xw, sheet_name="2_Gap_alla_OTM", index=False)
        gap_bucket.to_excel(xw, sheet_name="2_Gap_per_hink", index=False)
        prem.to_excel(xw, sheet_name="2_Premie_och_hoppdel", index=False)
        val.to_excel(xw, sheet_name="2_Gap_kontroll", index=False)
        spread_grid.to_excel(xw, sheet_name="3_Spread_per_hink_vp")
        paired.to_excel(xw, sheet_name="3_Spread_före_efter", index=False)
        event.to_excel(xw, sheet_name="3_Spread_eventstudie")
        reg.to_excel(xw, sheet_name="3_Spread_regression", index=False)
    plot_gap(tag, L, val)

    print(f"\n  1. RAPPORTDAGAR ({tag}):")
    for r in rep.itertuples():
        print(f"    {r.Rapportdatum}: prisad förväntad |rörelse| {100 * r.Förväntad_abs_rörelse:.1f} % "
              f"(sigma {100 * r.Prisad_rörelse_sigma:.1f} %), faktisk {100 * r.Faktisk_rörelse:+.1f} %  -> {r.Tolkning}"
              + (f" | short gamma {r.Short_gamma_PnL_SEK:+.2f} SEK/option (rörelse {r.Short_varav_rörelse:+.2f}, "
                 f"volfall {r.Short_varav_volfall:+.2f})" if hasattr(r, "Short_gamma_PnL_SEK") else ""))
    print(f"  2. GAP-TEST ({tag}), deltahedgad short, alla OTM-optioner:")
    for r in gap_all.itertuples():
        print(f"    hopp {100 * r.Gap:+5.1f} %: medel {r.PnL_BS_medel_SEK:+.2f} SEK (med vol-spik {r.PnL_BS_volspik_medel_SEK:+.2f}, "
              f"Merton {r.PnL_Merton_medel_SEK:+.2f}) | median {r.PnL_BS_median_halvspr:+.1f} halvspreadar, "
              f"{r.PnL_BS_median_thetadagar:+.0f} dagars theta")
    pa = prem_all.iloc[0]
    print(f"    Merton: {pa.Hoppdel_av_theta_pct:.0f} % av den dagliga premien (theta) är betalning för hopprisk (median, alla OTM)")
    if len(val):
        print("    Kontroll mot verkliga stora dagar (förutsagt vs faktiskt short P&L, SEK/option):")
        for r in val.itertuples():
            print(f"      {r.Datum} {r.Typ:12s} {100 * r.Rörelse:+5.1f} %: förutsagt {r.Förutsagt_short_PnL_SEK:+.2f}, "
                  f"faktiskt {r.Faktiskt_short_PnL_SEK:+.2f}")
    print(f"  3. SPREADAR ({tag}), {len(s)} OTM-noteringar, spread i vol-punkter:")
    for r in paired.itertuples():
        print(f"    Rapport {r.Rapport}: samma {r.Antal_optioner} optioner, median {r.Median_spread_före_vp:.2f} vp före -> "
              f"{r.Median_spread_efter_vp:.2f} vp efter (förändring {r.Median_förändring_vp:+.2f}, KI [{r.KI_2_5:+.2f}, {r.KI_97_5:+.2f}])"
              f" | i SEK {r.Median_spread_före_SEK:.2f} -> {r.Median_spread_efter_SEK:.2f}")
    for spec in ("A", "B"):
        rr = reg[(reg["Spec"] == spec) & reg["Variabel"].isin(["Hopprisk_z", "Före_rapport", "Efter_rapport"])]
        print(f"    Regression {spec}: " + ", ".join(f"{x.Variabel} {x.Effekt_pct:+.1f} % (t={x.t:.1f})" for x in rr.itertuples())
              + f"  R2={rr['R2'].iloc[0]:.2f}")
    print(f"  Sparat: {out}")
    return rep


if __name__ == "__main__":
    os.makedirs(sm.OUT_DIR, exist_ok=True)
    reps = []
    for tag in sm.FILES:
        print(f"\n=================== {tag}: rapporter, gap-test, spreadar ===================")
        reps.append(run_one(tag))
    allrep = pd.concat(reps, ignore_index=True)
    allrep.to_excel(os.path.join(sm.OUT_DIR, "SEB_Rapportdagar_alla.xlsx"), index=False)
    plot_report_moves(allrep)
    print("\nFÄRDIG.")
