"""
SEB-projekt: long vs short gamma med deltahedge (BS-delta vs Merton-delta)
=========================================================================

Kör EFTER seb_merton_analys.py (läser dagens Merton-parametrar från RESULTAT).

Två delar:

  A. Daglig hedgad P&L. Varje OTM-option (samma urval som huvudanalysen) hålls
     LÅNG en dag (long gamma) och deltahedgas med aktien. Short gamma = minus detta.
       P&L_long = dV - delta*dS - r*dt*(V - delta*S)
     delta tas antingen från BS (optionens egen implicita vol, som en market maker
     gör) eller från Merton (dagens kalibrerade parametrar). Uppdelning enligt BS:
       gamma-P&L = 0.5*Gamma*S^2*((dS/S)^2 - IV^2*dt)   (rörelse vs prissatt varians)
       vega-P&L  = Vega*(IV_imorgon - IV_idag)
     Dagar delas in i: rapportdag, stor rörelse (|avkastning| > 3 dagliga ATM-sigma), normal.

  B. Market maker till lösen. Var 5:e handelsdag "säljer" MM varje OTM-option med
     14-60 dagar kvar på ASK (short gamma) och deltahedgar dagligen till lösen;
     spegelbilden köper på BID (long gamma). Jämför implicit vol vid start med
     realiserad vol över livslängden, uppdelat på om en rapport ligger i livslängden.

Alla P&L är per option på EN aktie, i SEK, och i halvspread-enheter (P&L / halva
bid-ask-spreaden vid start) så att de kan jämföras med vad MM tjänar på spreaden.

OBS vid tolkning: positionerna överlappar (samma löptid och samma aktiebana), så det
effektiva urvalet är litet - i praktiken 2 rapporter per aktie och ett fåtal löptider.
Se resultaten som fallstudier, inte som statistiskt säkerställda effekter.

Kör:  python seb_gamma_analys.py
"""

import os
import time

import numpy as np
import pandas as pd
from scipy.special import ndtr

import seb_merton_analys as sm

# =============================================================================
# INSTÄLLNINGAR
# =============================================================================
# Rapportdatum i perioden (publicerade före börsöppning). Källor:
#   Volvo Q2 2025-07-17, Q3 2025-10-17 (volvogroup.com); Boliden Q2 2025-07-18, Q3 2025-10-22 (boliden.com)
REPORT_DATES = {
    "BOL": ["2025-07-18", "2025-10-22"],
    "VOLVB": ["2025-07-17", "2025-10-17"],
}
BIG_MOVE_SIGMAS = 3.0       # "stor rörelse" = |log-avkastning| > 3 * ATM-IV/sqrt(252)
BIG_MOVE_SIGMAS_ALT = 2.5   # känslighetstest av gränsen
N_BOOT = 2000               # bootstrap-dragningar för konfidensintervall
MIN_VEGA_VP = 0.02          # P&L i vol-punkter bara för optioner med vega >= 0.02 SEK per vol-punkt
HOLD_ENTRY_EVERY = 5        # nya MM-positioner var 5:e handelsdag
HOLD_MIN_DAYS, HOLD_MAX_DAYS = 14, 60
FD_BUMP = 1e-3              # relativ aktiebump för Merton-delta (central differens)
R = sm.RISK_FREE_RATE
Q = sm.DIVIDEND_YIELD


# =============================================================================
# GREKER
# =============================================================================
def bs_greeks(S, K, T, sigma, is_call, r=R):
    T = np.maximum(T, 1e-8)
    sigma = np.maximum(sigma, 1e-8)
    sq = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sq)
    pdf = np.exp(-0.5 * d1 ** 2) / np.sqrt(2 * np.pi)
    delta = np.where(is_call, ndtr(d1), ndtr(d1) - 1.0)
    gamma = pdf / (S * sigma * sq)
    vega = S * pdf * sq
    return delta, gamma, vega


def merton_delta(S, K, T, is_call, sig, lam, mu, sj):
    h = FD_BUMP * S
    up = sm.merton_price(S + h, K, T, R, Q, sig, lam, mu, sj, is_call)
    dn = sm.merton_price(S - h, K, T, R, Q, sig, lam, mu, sj, is_call)
    return (up - dn) / (2 * h)


# =============================================================================
# DATA
# =============================================================================
def load_params(tag):
    f = os.path.join(sm.OUT_DIR, f"SEB_Merton_vs_BS_{tag}.xlsx")
    if not os.path.exists(f):
        raise FileNotFoundError(f"{f} saknas - kör seb_merton_analys.py först.")
    p = pd.read_excel(f, sheet_name="Merton_param_dag")[["Date", "Sigma", "Lambda", "Mu_J", "Sigma_J", "ATM_IV"]]
    p["Date"] = pd.to_datetime(p["Date"])
    return p


def build_panel(tag):
    """Alla giltiga optionsnoteringar (inte bara OTM) - behövs för att värdera positionen nästa dag."""
    raw = sm.read_file(os.path.join(sm.DATA_DIR, sm.FILES[tag]))
    st = raw[raw["InsType"] == "Stock"].copy()
    st["S"] = (st["Bid"] + st["Ask"]) / 2
    st["HS_aktie"] = (st["Ask"] - st["Bid"]) / 2        # halva aktiespreaden = kostnad per handlad aktie
    st = st.sort_values("Date").drop_duplicates("Date")[["Date", "S", "HS_aktie"]]
    op = raw[raw["InsType"] == "Option"].dropna(subset=["Bid", "Ask"])
    op = op[(op["Bid"] > sm.MIN_BID) & (op["Ask"] >= op["Bid"])].merge(st[["Date", "S"]], on="Date")
    op["Days"] = (op["Expiry"] - op["Date"]).dt.days
    op = op[op["Days"] >= 1].copy()
    op["T"] = op["Days"] / sm.DAYS_PER_YEAR
    op["IsCall"] = op["CallPut"].str.upper().str.startswith("C")
    op["Mid"] = (op["Bid"] + op["Ask"]) / 2
    op["IV"] = sm.implied_vol(op["Mid"].values, op["S"].values, op["Strike"].values, op["T"].values, R, Q,
                              op["IsCall"].values)
    dates = np.sort(op["Date"].unique())
    return op, st, dates


def day_types(tag, st, params, dates):
    """Klassar varje intervall (t, t+1] som rapport / stor rörelse / normal."""
    s = st.set_index("Date")["S"].reindex(dates)
    ret = np.log(s.shift(-1) / s)                       # avkastning från t till nästa handelsdag
    atm = params.set_index("Date")["ATM_IV"].reindex(dates).ffill()
    thr = BIG_MOVE_SIGMAS * atm / np.sqrt(252)
    rep = pd.to_datetime(REPORT_DATES[tag])
    nxt = pd.Series(dates, index=dates).shift(-1)
    is_rep = [any((r_ > d) & (r_ <= n) for r_ in rep) if pd.notna(n) else False for d, n in zip(dates, nxt)]
    out = pd.DataFrame({"Date": dates, "NextDate": nxt.values, "Ret": ret.values, "Thr": thr.values,
                        "Report": is_rep})
    out["DayType"] = np.where(out["Report"], "Rapport",
                              np.where(np.abs(out["Ret"]) > out["Thr"], "Stor rörelse", "Normal"))
    thr2 = out["Thr"] * BIG_MOVE_SIGMAS_ALT / BIG_MOVE_SIGMAS
    out["DayType_alt"] = np.where(out["Report"], "Rapport",
                                  np.where(np.abs(out["Ret"]) > thr2, "Stor rörelse", "Normal"))
    return out


# =============================================================================
# A. DAGLIG HEDGAD P&L
# =============================================================================
def daily_pnl(tag, entries, op, st, dates, params):
    dt_tab = day_types(tag, st, params, dates)

    e = entries[["Date", "Isin", "Expiry", "Days", "T", "CallPut", "IsCall", "Strike", "StockMid", "Bid", "Ask",
                 "Mid", "HalfSpreadW", "IV_Mid", "K_over_F", "MaturityBucket", "MoneynessBucket"]].copy()
    e = e.rename(columns={"StockMid": "S0", "Mid": "V0", "IV_Mid": "IV0"})
    e = e.merge(dt_tab, on="Date", how="left")
    nxt = op[["Date", "Isin", "Mid", "S", "IV", "Days"]].rename(
        columns={"Date": "NextDate", "Mid": "V1", "S": "S1", "IV": "IV1", "Days": "Days1"})
    e = e.merge(nxt, on=["NextDate", "Isin"], how="inner")
    e = e[e["Days1"] >= 1].copy()                        # inga lösendagar som "nästa dag"
    e = e.merge(params, on="Date", how="left")
    p1 = params.drop(columns="ATM_IV").rename(columns=lambda c: c if c == "Date" else c + "_1")
    e = e.merge(p1.rename(columns={"Date": "NextDate"}), on="NextDate", how="left")
    e = e.merge(st[["Date", "HS_aktie"]].rename(columns={"Date": "NextDate"}), on="NextDate", how="left")

    S0, K, T, ic = e["S0"].values, e["Strike"].values, e["T"].values, e["IsCall"].values
    dBS, gam, veg = bs_greeks(S0, K, T, e["IV0"].values, ic)
    dM = merton_delta(S0, K, T, ic, e["Sigma"].values, e["Lambda"].values, e["Mu_J"].values, e["Sigma_J"].values)
    dt = (e["NextDate"] - e["Date"]).dt.days.values / 365.0
    dS = e["S1"].values - S0
    dV = e["V1"].values - e["V0"].values
    V0 = e["V0"].values
    e["Delta_BS"], e["Delta_Merton"], e["Gamma_BS"], e["Vega_BS"] = dBS, dM, gam, veg
    e["PnL_long_ohedgad"] = dV - R * dt * V0
    e["PnL_long_BS"] = dV - dBS * dS - R * dt * (V0 - dBS * S0)
    e["PnL_long_Merton"] = dV - dM * dS - R * dt * (V0 - dM * S0)
    # Kostnad för att justera aktiehedgen nästa dag: |delta_imorgon - delta_idag| * halva aktiespreaden
    S1, T1 = e["S1"].values, e["Days1"].values / sm.DAYS_PER_YEAR
    dBS1 = bs_greeks(S1, K, T1, e["IV1"].fillna(e["IV0"]).values, ic)[0]
    dM1 = merton_delta(S1, K, T1, ic, e["Sigma_1"].values, e["Lambda_1"].values, e["Mu_J_1"].values,
                       e["Sigma_J_1"].values)
    hs = e["HS_aktie"].fillna(0).values
    e["Hedgekostnad_BS"] = np.abs(dBS1 - dBS) * hs
    e["Hedgekostnad_Merton"] = np.abs(dM1 - dM) * hs
    e["GammaPnL"] = 0.5 * gam * S0 ** 2 * ((dS / S0) ** 2 - e["IV0"].values ** 2 * dt)
    e["VegaPnL"] = veg * (e["IV1"].fillna(e["IV0"]).values - e["IV0"].values)
    e["RestPnL"] = e["PnL_long_BS"] - e["GammaPnL"] - e["VegaPnL"]
    for c in ("PnL_long_BS", "PnL_long_Merton"):
        e[c.replace("PnL_long", "PnL_short") + "_halvspr"] = -e[c] / e["HalfSpreadW"]
    # Short gamma efter hedgekostnad (kostnaden drabbar både long och short)
    e["PnL_short_BS_netto"] = -e["PnL_long_BS"] - e["Hedgekostnad_BS"]
    # P&L i vol-punkter: P&L / (vega per vol-punkt). Gör djupa OTM och ATM jämförbara (lika vega-risk).
    vvp = e["Vega_BS"] * 0.01
    e["PnL_short_BS_vp"] = np.where(vvp >= MIN_VEGA_VP, -e["PnL_long_BS"] / vvp, np.nan)
    return e, dt_tab


def summarize_daily(e, by):
    rows = []
    for key, g in e.groupby(by, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        sBS, sM = -g["PnL_long_BS"], -g["PnL_long_Merton"]
        rows.append(dict(**dict(zip(by, key)), Antal=len(g),
                         Short_BS_medel_SEK=sBS.mean(),
                         Short_BS_medel_halvspr=g["PnL_short_BS_halvspr"].mean(),
                         Antal_dagar=g["Date"].nunique(),
                         Short_BS_netto_medel_SEK=g["PnL_short_BS_netto"].mean(),
                         Hedgekostnad_BS_medel_SEK=g["Hedgekostnad_BS"].mean(),
                         Short_BS_medel_volpunkter=g["PnL_short_BS_vp"].mean(),
                         Short_BS_median_volpunkter=g["PnL_short_BS_vp"].median(),
                         Short_BS_1pct_SEK=sBS.quantile(0.01),
                         Short_BS_andel_förlustdagar=np.mean(sBS < 0),
                         Short_Merton_medel_SEK=sM.mean(),
                         Hedgefel_std_BS=g["PnL_long_BS"].std(),
                         Hedgefel_std_Merton=g["PnL_long_Merton"].std(),
                         Varians_minskning_Merton_pct=100 * (1 - g["PnL_long_Merton"].var() / g["PnL_long_BS"].var()),
                         Std_ohedgad=g["PnL_long_ohedgad"].std(),
                         Long_GammaPnL_medel=g["GammaPnL"].mean(), Long_VegaPnL_medel=g["VegaPnL"].mean(),
                         Long_RestPnL_medel=g["RestPnL"].mean(),
                         Medel_delta_BS=g["Delta_BS"].mean(), Medel_delta_Merton=g["Delta_Merton"].mean()))
    return pd.DataFrame(rows)


# =============================================================================
# B. MARKET MAKER TILL LÖSEN
# =============================================================================
def hold_to_expiry(tag, entries, op, st, dates, params):
    dates = pd.DatetimeIndex(dates)
    last = dates.max()
    spot = st.set_index("Date")["S"]
    hs_st = st.set_index("Date")["HS_aktie"]
    iv_by = op.set_index(["Isin", "Date"])["IV"]
    par = params.set_index("Date").reindex(dates).ffill()
    rep = pd.to_datetime(REPORT_DATES[tag])

    entry_dates = dates[::HOLD_ENTRY_EVERY]
    E = entries[entries["Date"].isin(entry_dates) & entries["Days"].between(HOLD_MIN_DAYS, HOLD_MAX_DAYS)
                & (entries["Expiry"] <= last)].copy()
    rows = []
    for _, x in E.iterrows():
        path = [d for d in dates if x["Date"] <= d < x["Expiry"]]
        exp = x["Expiry"]
        S_path = spot.reindex(path + [exp]).values
        if np.any(~np.isfinite(S_path)):
            continue
        K, ic = x["Strike"], bool(x["IsCall"])
        HS_path = hs_st.reindex(path + [exp]).fillna(0).values
        res, cost = {}, {}
        for hedge in ("BS", "Merton"):
            cash, h, c = x["Mid"], 0.0, 0.0              # short 1 option till mid; kassan får premien
            iv_last = x["IV_Mid"]
            for k, d in enumerate(path):
                S = S_path[k]
                T = (exp - d).days / sm.DAYS_PER_YEAR
                if hedge == "BS":
                    iv = iv_by.get((x["Isin"], d), np.nan)
                    iv_last = iv if np.isfinite(iv) else iv_last
                    delta = bs_greeks(np.array([S]), np.array([K]), np.array([T]), np.array([iv_last]),
                                      np.array([ic]))[0][0]
                else:
                    p = par.loc[d]
                    delta = merton_delta(np.array([S]), np.array([K]), np.array([T]), np.array([ic]),
                                         p["Sigma"], p["Lambda"], p["Mu_J"], p["Sigma_J"])[0]
                cash -= (delta - h) * S                  # köp/sälj aktier så att innehavet = delta (short option)
                c += abs(delta - h) * HS_path[k]         # handelskostnad i aktien
                h = delta
                nxt = path[k + 1] if k + 1 < len(path) else exp
                cash *= np.exp(R * (nxt - d).days / 365.0)
            S_T = S_path[-1]
            payoff = max(S_T - K, 0.0) if ic else max(K - S_T, 0.0)
            res[hedge] = cash + h * S_T - payoff         # stäng aktier, betala ut lösen
            cost[hedge] = c + abs(h) * HS_path[-1]
        rets = np.diff(np.log(S_path))
        rows.append(dict(Entry=x["Date"], Expiry=exp, Isin=x["Isin"], CallPut=x["CallPut"], Strike=K,
                         S0=x["StockMid"], K_over_F=x["K_over_F"], Days=x["Days"], IV_start=x["IV_Mid"],
                         RV_livslängd=np.sqrt(252 * np.mean(rets ** 2)) if len(rets) else np.nan,
                         Rapport_i_livslängd=bool(np.any((rep > x["Date"]) & (rep <= exp))),
                         Mid=x["Mid"], Halvspread=x["HalfSpreadW"],
                         Short_mid_BS=res["BS"], Short_mid_Merton=res["Merton"],
                         Hedgekostnad_BS=cost["BS"], Hedgekostnad_Merton=cost["Merton"],
                         Short_MM_BS=res["BS"] + x["HalfSpreadW"],          # såld på ask
                         Long_MM_BS=-res["BS"] + x["HalfSpreadW"],          # köpt på bid
                         MoneynessBucket=x["MoneynessBucket"]))
    H = pd.DataFrame(rows)
    H["IV_minus_RV"] = H["IV_start"] - H["RV_livslängd"]
    H["Short_MM_BS_halvspr"] = H["Short_MM_BS"] / H["Halvspread"]
    H["Short_MM_BS_netto"] = H["Short_MM_BS"] - H["Hedgekostnad_BS"]
    return H


def summarize_hold(H, by):
    rows = []
    for key, g in H.groupby(by, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append(dict(**dict(zip(by, key)), Antal=len(g),
                         IV_start_medel=g["IV_start"].mean(), RV_medel=g["RV_livslängd"].mean(),
                         IV_minus_RV_medel=g["IV_minus_RV"].mean(),
                         Short_mid_medel_SEK=g["Short_mid_BS"].mean(),      # utan spreadintäkt
                         Halvspread_medel_SEK=g["Halvspread"].mean(),
                         Hedgekostnad_medel_SEK=g["Hedgekostnad_BS"].mean(),
                         Short_MM_netto_medel_SEK=g["Short_MM_BS_netto"].mean(),
                         Antal_löptider=g["Expiry"].nunique(),
                         Short_MM_medel_SEK=g["Short_MM_BS"].mean(), Short_MM_median_SEK=g["Short_MM_BS"].median(),
                         Short_MM_5pct_SEK=g["Short_MM_BS"].quantile(0.05),
                         Short_MM_andel_vinst=np.mean(g["Short_MM_BS"] > 0),
                         Short_MM_medel_halvspr=g["Short_MM_BS_halvspr"].mean(),
                         Long_MM_medel_SEK=g["Long_MM_BS"].mean(), Long_MM_andel_vinst=np.mean(g["Long_MM_BS"] > 0),
                         Short_mid_BS_std=g["Short_mid_BS"].std(), Short_mid_Merton_std=g["Short_mid_Merton"].std()))
    return pd.DataFrame(rows)


# =============================================================================
# KONFIDENSINTERVALL (bootstrap)
# =============================================================================
def boot_daily(g, label, seed=1):
    """
    Block-bootstrap över DAGAR: alla optioner samma dag dras tillsammans (de rör sig ihop),
    så intervallen tar hänsyn till att optionerna samma dag inte är oberoende.
    """
    rng = np.random.default_rng(seed)
    d = g.groupby("Date").agg(n=("PnL_long_BS", "size"), s=("PnL_long_BS", lambda x: -x.sum()),
                              sn=("PnL_short_BS_netto", "sum"),
                              vb=("PnL_long_BS", lambda x: (x ** 2).sum()),
                              vm=("PnL_long_Merton", lambda x: (x ** 2).sum()),
                              svp=("PnL_short_BS_vp", "sum"), nvp=("PnL_short_BS_vp", "count"))
    k = len(d)
    est = dict(Short_medel_SEK=d["s"].sum() / d["n"].sum(), Short_netto_medel_SEK=d["sn"].sum() / d["n"].sum(),
               Short_medel_volpunkter=d["svp"].sum() / max(d["nvp"].sum(), 1),
               Merton_minskning_hedgefel_pct=100 * (1 - d["vm"].sum() / d["vb"].sum()))
    rows = []
    if k < 10:
        for m, v in est.items():
            rows.append(dict(Del="Daglig", Grupp=label, Mått=m, Skattning=v, KI_2_5=np.nan, KI_97_5=np.nan,
                             Antal_dagar=k, Kommentar="för få dagar för intervall"))
        return rows
    idx = rng.integers(0, k, (N_BOOT, k))
    S = {c: d[c].values[idx].sum(axis=1) for c in d.columns}
    draws = dict(Short_medel_SEK=S["s"] / S["n"], Short_netto_medel_SEK=S["sn"] / S["n"],
                 Short_medel_volpunkter=S["svp"] / np.maximum(S["nvp"], 1),
                 Merton_minskning_hedgefel_pct=100 * (1 - S["vm"] / S["vb"]))
    for m, v in est.items():
        lo, hi = np.percentile(draws[m], [2.5, 97.5])
        rows.append(dict(Del="Daglig", Grupp=label, Mått=m, Skattning=v, KI_2_5=lo, KI_97_5=hi, Antal_dagar=k,
                         Kommentar="signifikant" if (lo > 0 or hi < 0) else "ej signifikant (0 i intervallet)"))
    return rows


def boot_hold(g, label, seed=2):
    """Kluster-bootstrap över LÖPTIDER: positioner med samma lösen delar aktiebana och dras tillsammans."""
    rng = np.random.default_rng(seed)
    c = g.groupby("Expiry").agg(n=("Short_mid_BS", "size"), s=("Short_mid_BS", "sum"),
                                sn=("Short_MM_BS_netto", "sum"))
    k = len(c)
    est = dict(Short_utan_spread_medel_SEK=c["s"].sum() / c["n"].sum(),
               Short_MM_netto_medel_SEK=c["sn"].sum() / c["n"].sum())
    rows = []
    if k < 4:
        for m, v in est.items():
            rows.append(dict(Del="Till lösen", Grupp=label, Mått=m, Skattning=v, KI_2_5=np.nan, KI_97_5=np.nan,
                             Antal_dagar=np.nan, Antal_löptider=k, Kommentar="för få löptider för intervall"))
        return rows
    idx = rng.integers(0, k, (N_BOOT, k))
    n, s_, sn = (c[col].values[idx].sum(axis=1) for col in ("n", "s", "sn"))
    draws = dict(Short_utan_spread_medel_SEK=s_ / n, Short_MM_netto_medel_SEK=sn / n)
    for m, v in est.items():
        lo, hi = np.percentile(draws[m], [2.5, 97.5])
        rows.append(dict(Del="Till lösen", Grupp=label, Mått=m, Skattning=v, KI_2_5=lo, KI_97_5=hi,
                         Antal_dagar=np.nan, Antal_löptider=k,
                         Kommentar="signifikant" if (lo > 0 or hi < 0) else "ej signifikant (0 i intervallet)"))
    return rows


def confidence_table(e, H):
    rows = []
    rows += boot_daily(e, "Alla dagar")
    rows += boot_daily(e[e["DayType"] == "Normal"], f"Normala dagar ({BIG_MOVE_SIGMAS:g}σ-gräns)")
    rows += boot_daily(e[e["DayType_alt"] == "Normal"], f"Normala dagar ({BIG_MOVE_SIGMAS_ALT:g}σ-gräns)")
    for cp in ("Put", "Call"):
        rows += boot_daily(e[(e["DayType"] == "Normal") & (e["CallPut"] == cp)], f"Normala dagar, {cp}")
    for t in ("Rapport", "Stor rörelse"):
        if (e["DayType"] == t).any():
            rows += boot_daily(e[e["DayType"] == t], t)
    if len(H):
        rows += boot_hold(H, "Alla positioner")
        rows += boot_hold(H[~H["Rapport_i_livslängd"]], "Ingen rapport i livslängden")
        rows += boot_hold(H[H["Rapport_i_livslängd"]], "Rapport i livslängden")
    return pd.DataFrame(rows)


# =============================================================================
# FIGURER
# =============================================================================
def make_plots(tag, e):
    """Kumulativ daglig P&L för en short-gamma-bok (short 1 av varje OTM-option, BS-hedgad)."""
    plt = sm._plt()
    INK, INK2, MUTED, SURF = sm.INK, sm.INK2, sm.MUTED, sm.SURF
    book = e.groupby("Date").agg(pnl=("PnL_long_BS", lambda x: -x.sum()), typ=("DayType", "first"))
    fig, ax = plt.subplots(figsize=(12, 4.4), facecolor=SURF)
    sm._style(ax)
    ax.plot(book.index, book["pnl"].cumsum(), color=sm.C_MERTON, linewidth=2, label="Short gamma, BS-deltahedgad")
    for d, r in book.iterrows():
        if r["typ"] != "Normal":
            ax.axvline(d, color=sm.C_BS if r["typ"] == "Rapport" else MUTED, linewidth=1,
                       linestyle="-" if r["typ"] == "Rapport" else ":", alpha=0.9)
    ax.plot([], [], color=sm.C_BS, linewidth=1, label="Dag med rapport")
    ax.plot([], [], color=MUTED, linewidth=1, linestyle=":", label=f"Stor rörelse (>{BIG_MOVE_SIGMAS:.0f}σ)")
    ax.axhline(0, color=sm.AXIS, linewidth=1)
    ax.set_ylabel("Kumulativ P&L (SEK, summa över optioner)", fontsize=9, color=INK2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    ax.set_title(f"{tag}: bok som är short 1 av varje noterad OTM-option, hedgad dagligen (mid till mid, "
                 f"utan spreadintäkt)", fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(sm.OUT_DIR, f"{tag}_gamma_kumulativ.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# HUVUDPROGRAM
# =============================================================================
def run_one(tag):
    t0 = time.time()
    params = load_params(tag)
    entries = sm.load_stock(tag)                    # samma OTM-urval som huvudanalysen
    op, st, dates = build_panel(tag)
    e, dt_tab = daily_pnl(tag, entries, op, st, dates, params)
    H = hold_to_expiry(tag, entries, op, st, dates, params)

    s_type = summarize_daily(e, ["DayType"])
    s_type_alt = summarize_daily(e, ["DayType_alt"])
    ci = confidence_table(e, H)
    s_all = summarize_daily(e.assign(Alla="Alla"), ["Alla"])
    s_bucket = summarize_daily(e, ["MaturityBucket", "MoneynessBucket"])
    s_cp_type = summarize_daily(e, ["CallPut", "DayType"])
    h_all = summarize_hold(H.assign(Alla="Alla"), ["Alla"])
    h_rep = summarize_hold(H, ["Rapport_i_livslängd"])
    h_rep_cp = summarize_hold(H, ["Rapport_i_livslängd", "CallPut"])
    h_mon = summarize_hold(H, ["MoneynessBucket", "Rapport_i_livslängd"])

    out = os.path.join(sm.OUT_DIR, f"SEB_Gamma_{tag}.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        s_all.to_excel(xw, sheet_name="Daglig_totalt", index=False)
        ci.to_excel(xw, sheet_name="Konfidensintervall", index=False)
        s_type.to_excel(xw, sheet_name="Daglig_per_dagtyp", index=False)
        s_type_alt.to_excel(xw, sheet_name=f"Daglig_dagtyp_{BIG_MOVE_SIGMAS_ALT:g}sigma", index=False)
        s_cp_type.to_excel(xw, sheet_name="Daglig_callput_dagtyp", index=False)
        s_bucket.to_excel(xw, sheet_name="Daglig_per_hink", index=False)
        h_all.to_excel(xw, sheet_name="Lösen_totalt", index=False)
        h_rep.to_excel(xw, sheet_name="Lösen_rapport", index=False)
        h_rep_cp.to_excel(xw, sheet_name="Lösen_rapport_callput", index=False)
        h_mon.to_excel(xw, sheet_name="Lösen_moneyness", index=False)
        dt_tab.to_excel(xw, sheet_name="Dagtyper", index=False)
        H.to_excel(xw, sheet_name="Lösen_positioner", index=False)
        e.to_excel(xw, sheet_name="Daglig_positioner", index=False)
    make_plots(tag, e)

    print(f"\n  {tag}: {len(e)} option-dagar, {len(H)} positioner till lösen ({time.time() - t0:.0f} s)")
    print(f"  Dagtyper ({BIG_MOVE_SIGMAS:g}σ): " + ", ".join(f"{k}={v}" for k, v in dt_tab["DayType"].value_counts().items())
          + f"   |  med {BIG_MOVE_SIGMAS_ALT:g}σ: " + ", ".join(f"{k}={v}" for k, v in dt_tab["DayType_alt"].value_counts().items()))
    print("  A. Daglig P&L för SHORT gamma (deltahedgad, mid-mid), per option och dag:")
    for _, r in pd.concat([s_all.rename(columns={"Alla": "DayType"}), s_type]).iterrows():
        print(f"    {str(r['DayType']):13s} n={r['Antal']:6d}  medel {r['Short_BS_medel_SEK']:+.3f} SEK "
              f"({r['Short_BS_medel_halvspr']:+.3f} halvspr)  1%-sämsta {r['Short_BS_1pct_SEK']:+.2f} SEK  "
              f"förlustdagar {100 * r['Short_BS_andel_förlustdagar']:.0f} %  | hedgefel std BS {r['Hedgefel_std_BS']:.3f} "
              f"vs Merton {r['Hedgefel_std_Merton']:.3f} ({r['Varians_minskning_Merton_pct']:+.0f} % varians)")
    print("  B. Market maker till lösen (säljer på ask = short gamma), BS-hedge:")
    for _, r in pd.concat([h_all.rename(columns={"Alla": "Rapport_i_livslängd"}), h_rep]).iterrows():
        print(f"    rapport i livslängd={str(r['Rapport_i_livslängd']):5s} n={r['Antal']:4d}  IV {100 * r['IV_start_medel']:.1f} % "
              f"vs RV {100 * r['RV_medel']:.1f} %  | short utan spread {r['Short_mid_medel_SEK']:+.2f}, "
              f"halvspread {r['Halvspread_medel_SEK']:.2f} | short MM: medel {r['Short_MM_medel_SEK']:+.2f} SEK, vinst "
              f"{100 * r['Short_MM_andel_vinst']:.0f} %, 5%-sämsta {r['Short_MM_5pct_SEK']:+.2f}")
    print("  C. 95 %-konfidensintervall (bootstrap över dagar / löptider):")
    for _, r in ci.iterrows():
        ki = "" if pd.isna(r["KI_2_5"]) else f"[{r['KI_2_5']:+.3f}, {r['KI_97_5']:+.3f}]"
        print(f"    {r['Del']:10s} {r['Grupp']:34s} {r['Mått']:30s} {r['Skattning']:+8.3f} {ki:20s} {r['Kommentar']}")
    print(f"  Sparat: {out}")


if __name__ == "__main__":
    os.makedirs(sm.OUT_DIR, exist_ok=True)
    for tag in sm.FILES:
        print(f"\n=================== {tag}: long/short gamma ===================")
        run_one(tag)
    print("\nFÄRDIG.")
