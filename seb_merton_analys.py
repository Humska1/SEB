"""
SEB-projekt: Black-Scholes vs Merton jump-diffusion på BOL och VOLVB
======================================================================

Modeller som kalibreras mot marknaden (OTM-optioner, dagliga stängningskurser):
    BS_slice      Black-Scholes med EN sigma per (dag, löptid)
    Merton_dag    Merton med EN parameteruppsättning per dag, alla löptider,
                  stabiliserad med ett straff mot gårdagens parametrar (REG_WEIGHT)
    Merton_slice  Merton per (dag, löptid), bara korta löptider (underlag för Kou-testet)

Fel mäts i halvspread-enheter: z = (modell - mid)/(spread/2). |z| <= 1 <=> priset ligger inom bid-ask.

Enligt SEB är optionerna amerikanska men perioden saknar utdelning, så europeisk värdering
används med utdelning 0.

Kör:  python seb_merton_analys.py          huvudanalysen (ca 2-3 min)
      python seb_merton_analys.py --full   + avvägningskurvan för REG_WEIGHT (lcurve) och
                                             fram/bak-testet (fram_bak_test), ca 15 min till
"""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import ndtr
from scipy.stats import poisson

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# 1. INSTÄLLNINGAR
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "clean")
OUT_DIR = os.path.join(BASE_DIR, "RESULTAT")
FILES = {"BOL": "BOL_SS_rensad.xlsx", "VOLVB": "VOLVB_SS_rensad.xlsx"}   # .xlsx eller rå .csv (semikolon)

RISK_FREE_RATE = 0.02   # put-call-paritet i datan ger ca 1.6-2.0 %
DIVIDEND_YIELD = 0.00   # enligt SEB: ingen utdelning i perioden
DAYS_PER_YEAR = 365.25

# Datafilter
MIN_DAYS = 5            # släng optioner med < 5 dagar kvar (mikrostruktur-brus)
MIN_BID = 0.01          # bid <= 0.01 = ingen riktig köpare
MAX_REL_SPREAD = 1.0    # släng om spread > mid
SPREAD_FLOOR = 0.05     # minsta halvspread i viktningen (SEK)

# Merton: theta = (sigma, lambda, mu_j, sigma_j)
MERTON_LB = np.array([0.01, 0.001, -0.60, 0.01])
MERTON_UB = np.array([1.50, 10.0, 0.30, 0.80])
MAX_NFEV = 300

# Straff mot gårdagens parametrar (bara Merton_dag):
#   REG_WEIGHT * sum(((theta - theta_igår)/REG_SCALE)^2), lambda jämförs i log.
# REG_WEIGHT = 1: att flytta en parameter en skalenhet kostar som en option en halvspread fel.
# Värdet 3 är valt med lcurve(): största vikt som kostar < 1 procentenhet "inom bid-ask".
REG_WEIGHT = 3.0
REG_SCALE = np.array([0.02, 0.5, 0.05, 0.05])   # sigma, log(lambda), mu_j, sigma_j

KOU_TEST_MAX_DAYS = 45  # "korta" löptider i Kou-testet
SLICE_MIN_QUOTES = 6    # minst så många optioner för att kalibrera 4 parametrar per löptid

MODELS = ["BS_slice", "Merton_dag", "Merton_slice"]


# =============================================================================
# 2. PRISSÄTTNING (vektoriserad)
# =============================================================================
def bs_price(S, K, T, r, q, sigma, is_call):
    """Black-Scholes för arrayer. is_call är en bool-array."""
    T = np.maximum(T, 1e-8)
    sigma = np.maximum(sigma, 1e-8)
    sqT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqT)
    d2 = d1 - sigma * sqT
    call = S * np.exp(-q * T) * ndtr(d1) - K * np.exp(-r * T) * ndtr(d2)
    put = K * np.exp(-r * T) * ndtr(-d2) - S * np.exp(-q * T) * ndtr(-d1)
    return np.where(is_call, call, put)


def merton_price(S, K, T, r, q, sigma, lamb, mu_j, sigma_j, is_call, tol=1e-12):
    """
    Mertons (1976) serie: pris = sum_n P(N=n; lambda'*T) * BS(r_n, sigma_n), där
      kappa = exp(mu_j + sigma_j^2/2) - 1,  lambda' = lambda*(1+kappa),
      sigma_n^2 = sigma^2 + n*sigma_j^2/T,  r_n = r - lambda*kappa + n*ln(1+kappa)/T.
    Antalet termer väljs så att Poisson-svansen < tol.
    """
    T = np.maximum(T, 1e-8)
    kappa = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0
    lam_T = lamb * (1.0 + kappa) * T
    n_max = int(poisson.ppf(1.0 - tol, float(np.max(lam_T)))) + 2
    weight = np.exp(-lam_T)
    price = np.zeros(np.broadcast(S, K, T).shape, dtype=float)
    for n in range(n_max + 1):
        if n > 0:
            weight = weight * lam_T / n
        sigma_n = np.sqrt(sigma ** 2 + n * sigma_j ** 2 / T)
        r_n = r - lamb * kappa + n * np.log1p(kappa) / T
        price = price + weight * bs_price(S, K, T, r_n, q, sigma_n, is_call)
    return price


def implied_vol(price, S, K, T, r, q, is_call, lo=1e-4, hi=5.0, n_iter=80):
    """Vektoriserad bisektion. NaN om priset ligger utanför [BS(lo), BS(hi)]."""
    price = np.asarray(price, dtype=float)
    p_lo = bs_price(S, K, T, r, q, lo, is_call)
    p_hi = bs_price(S, K, T, r, q, hi, is_call)
    ok = (price > p_lo) & (price < p_hi)
    a = np.full(price.shape, lo)
    b = np.full(price.shape, hi)
    for _ in range(n_iter):
        m = 0.5 * (a + b)
        too_high = bs_price(S, K, T, r, q, m, is_call) > price
        b = np.where(too_high, m, b)
        a = np.where(too_high, a, m)
    iv = 0.5 * (a + b)
    iv[~ok] = np.nan
    return iv


def sjalvtest():
    """Put-call-paritet och Monte Carlo-kontroll av Merton-serien, samt IV-inversionen."""
    print("\n--- Självtest av Merton-prissättaren ---")
    S, r, q, T, sig = 100.0, RISK_FREE_RATE, 0.0, 0.25, 0.25
    lam, mu, sj = 0.75, -0.10, 0.30
    K = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    c = merton_price(S, K, T, r, q, sig, lam, mu, sj, np.ones(5, bool))
    p = merton_price(S, K, T, r, q, sig, lam, mu, sj, np.zeros(5, bool))
    parity = np.max(np.abs(c - p - (S * np.exp(-q * T) - K * np.exp(-r * T))))
    print(f"  Put-call-paritet, max fel: {parity:.2e} SEK")

    rng = np.random.default_rng(42)          # Monte Carlo med antitetiska variabler
    M = 1_000_000
    kappa = np.exp(mu + 0.5 * sj ** 2) - 1
    Z, Nj, Zj = rng.standard_normal(M), rng.poisson(lam * T, M), rng.standard_normal(M)
    drift = (r - q - lam * kappa - 0.5 * sig ** 2) * T
    ST1 = S * np.exp(drift + sig * np.sqrt(T) * Z + Nj * mu + np.sqrt(Nj) * sj * Zj)
    ST2 = S * np.exp(drift - sig * np.sqrt(T) * Z + Nj * mu - np.sqrt(Nj) * sj * Zj)
    print("  Call, serie vs MC (1M antitetiska):")
    for i, k in enumerate(K):
        pay = 0.5 * (np.maximum(ST1 - k, 0) + np.maximum(ST2 - k, 0)) * np.exp(-r * T)
        print(f"    K={k:5.0f}: serie={c[i]:8.4f}  MC={pay.mean():8.4f} ± {1.96 * pay.std() / np.sqrt(M):.4f}")
    iv_back = implied_vol(bs_price(S, K, T, r, q, 0.3, np.ones(5, bool)), S, K, T, r, q, np.ones(5, bool))
    print(f"  Implicit vol-inversion, max fel: {np.max(np.abs(iv_back - 0.3)):.1e}")


# =============================================================================
# 3. DATA
# =============================================================================
def read_file(path):
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path, sep=";")
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    else:
        df = pd.read_excel(path)
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df["Expiry"] = pd.to_datetime(df["Expiry"]).dt.normalize()
    return df


def maturity_bucket(days):
    return pd.cut(days, [0, 30, 90, 180, 10_000], right=False,
                  labels=["1. <1M", "2. 1-3M", "3. 3-6M", "4. >6M"])


def moneyness_bucket(k_over_f):
    return pd.cut(k_over_f, [0, 0.85, 0.95, 1.0, 1.05, 1.15, 100], right=False,
                  labels=["1. K/F<0.85 (djup OTM put)", "2. 0.85-0.95 (OTM put)",
                          "3. 0.95-1.00 (ATM put)", "4. 1.00-1.05 (ATM call)",
                          "5. 1.05-1.15 (OTM call)", "6. K/F>1.15 (djup OTM call)"])


def prepare_data(path):
    """Läser en fil, filtrerar och räknar implicit vol. Returnerar (optioner, filterlogg)."""
    print(f"Läser in {os.path.basename(path)} ...")
    df = read_file(path)
    stock = df[df["InsType"] == "Stock"].copy()
    options = df[df["InsType"] == "Option"].copy()
    log = [("Optionsrader i filen", len(options))]

    stock["StockMid"] = (stock["Bid"] + stock["Ask"]) / 2
    stock = stock.sort_values("Date").drop_duplicates("Date")
    options = options.merge(stock[["Date", "StockMid"]], on="Date", how="inner")

    options["Days"] = (options["Expiry"] - options["Date"]).dt.days
    options["T"] = options["Days"] / DAYS_PER_YEAR
    options["IsCall"] = options["CallPut"].str.upper().str.startswith("C")
    options["Forward"] = options["StockMid"] * np.exp((RISK_FREE_RATE - DIVIDEND_YIELD) * options["T"])
    options["K_over_F"] = options["Strike"] / options["Forward"]
    options["LogMoneyness"] = np.log(options["K_over_F"])

    # Filter (antalen loggas för rapporten)
    options = options.dropna(subset=["Bid", "Ask"])
    log.append(("Efter borttag av saknad bid/ask", len(options)))
    options = options[(options["Bid"] > MIN_BID) & (options["Ask"] >= options["Bid"])]
    log.append((f"Efter bid > {MIN_BID} och ask >= bid", len(options)))
    options = options[options["Days"] >= MIN_DAYS]
    log.append((f"Efter löptid >= {MIN_DAYS} dagar", len(options)))
    otm = (options["IsCall"] & (options["K_over_F"] >= 1.0)) | (~options["IsCall"] & (options["K_over_F"] < 1.0))
    options = options[otm].copy()
    log.append(("Efter bara OTM (put om K<F, call om K>=F)", len(options)))

    options["Mid"] = (options["Bid"] + options["Ask"]) / 2
    options["Spread"] = options["Ask"] - options["Bid"]
    options["RelSpread"] = options["Spread"] / options["Mid"]
    options = options[options["RelSpread"] <= MAX_REL_SPREAD].copy()
    log.append((f"Efter spread/mid <= {MAX_REL_SPREAD}", len(options)))
    options["HalfSpreadW"] = np.maximum(options["Spread"] / 2, SPREAD_FLOOR)

    a = [options[c].values for c in ("StockMid", "Strike", "T")]
    ic = options["IsCall"].values
    for side in ("Bid", "Mid", "Ask"):
        options[f"IV_{side}"] = implied_vol(options[side].values, *a, RISK_FREE_RATE, DIVIDEND_YIELD, ic)
    options = options.dropna(subset=["IV_Mid"]).copy()
    log.append(("Efter giltig implicit vol (mid)", len(options)))

    options["MaturityBucket"] = maturity_bucket(options["Days"])
    options["MoneynessBucket"] = moneyness_bucket(options["K_over_F"])
    options = options.sort_values(["Date", "Expiry", "Strike"]).reset_index(drop=True)
    for name, n in log:
        print(f"  {name:45s} {n:7d}")
    return options, pd.DataFrame(log, columns=["Steg", "Antal optioner"])


def load_stock(tag):
    return prepare_data(os.path.join(DATA_DIR, FILES[tag]))[0]


# =============================================================================
# 4. KALIBRERING
# =============================================================================
def _arrays(d):
    return (d["StockMid"].values, d["Strike"].values, d["T"].values,
            d["IsCall"].values, d["Mid"].values, d["HalfSpreadW"].values)


def fit_bs_slice(d):
    """En sigma per (dag, löptid) som minimerar sum(((BS - mid)/halvspread)^2)."""
    S, K, T, ic, mid, w = _arrays(d)
    f = lambda s: np.sum(((bs_price(S, K, T, RISK_FREE_RATE, DIVIDEND_YIELD, s, ic) - mid) / w) ** 2)
    x0 = np.nanmedian(d["IV_Mid"].values)
    res = minimize_scalar(f, bounds=(max(0.01, 0.3 * x0), min(3.0, 3 * x0)), method="bounded",
                          options={"xatol": 1e-6})
    return res.x


def _penalty_vec(th, prior):
    """(theta - prior)/skala, med lambda i log så att 0.5->1 och 5->10 räknas lika."""
    a = np.array([th[0], np.log(th[1]), th[2], th[3]])
    b = np.array([prior[0], np.log(prior[1]), prior[2], prior[3]])
    return (a - b) / REG_SCALE


def fit_merton(d, starts, prior=None, reg_weight=0.0):
    """
    Minsta kvadrat i halvspread-enheter, flera startpunkter, bästa behålls. Med prior och
    reg_weight > 0 läggs straffet mot prior (gårdagens värden) till.
    Returnerar (theta, z, se, cond, vid_gräns, bästa_start, straffkostnad).
    """
    S, K, T, ic, mid, w = _arrays(d)
    use_pen = prior is not None and reg_weight > 0
    sq = np.sqrt(reg_weight)

    def resid(th):
        p = merton_price(S, K, T, RISK_FREE_RATE, DIVIDEND_YIELD, th[0], th[1], th[2], th[3], ic)
        z = (p - mid) / w
        return np.concatenate([z, sq * _penalty_vec(th, prior)]) if use_pen else z

    best, best_start = None, None
    for i, x0 in enumerate(starts):
        x0 = np.clip(np.asarray(x0, float), MERTON_LB + 1e-6, MERTON_UB - 1e-6)
        try:
            res = least_squares(resid, x0, bounds=(MERTON_LB, MERTON_UB), method="trf",
                                x_scale="jac", max_nfev=MAX_NFEV)
        except Exception:
            continue
        if best is None or res.cost < best.cost:
            best, best_start = res, i
    n = len(mid)
    z = best.fun[:n]
    pen = float(np.sum(best.fun[n:] ** 2)) if use_pen else 0.0
    # standardfel ur fitdelens Jacobian (grovt mått på hur väl datan bestämmer parametrarna)
    J = best.jac[:n]
    s2 = np.sum(z ** 2) / max(n - 4, 1)
    try:
        JtJ = J.T @ J
        se = np.sqrt(np.clip(np.diag(s2 * np.linalg.inv(JtJ)), 0, None))
        cond = np.linalg.cond(JtJ)
    except np.linalg.LinAlgError:
        se, cond = np.full(4, np.nan), np.inf
    at_bound = bool(np.any(np.isclose(best.x, MERTON_LB, rtol=0, atol=1e-4) |
                           np.isclose(best.x, MERTON_UB, rtol=0, atol=1e-4)))
    return best.x, z, se, cond, at_bound, best_start, pen


def default_starts(atm_vol):
    return [[0.85 * atm_vol, 0.5, -0.10, 0.15],
            [0.60 * atm_vol, 1.5, -0.05, 0.10],
            [0.90 * atm_vol, 0.2, -0.25, 0.20]]


def merton_day_fit(d, prev, reg_weight):
    """En dags Merton-kalibrering över alla löptider. prev = gårdagens theta (eller None)."""
    near = d[np.abs(d["LogMoneyness"]) < 0.05]
    atm_vol = float(np.nanmedian((near if len(near) else d)["IV_Mid"]))
    starts = default_starts(atm_vol)
    if prev is not None:
        starts = [prev] + starts[:2]
    th, z, se, cond, at_b, bstart, pen = fit_merton(d, starts, prior=prev, reg_weight=reg_weight)
    return th, z, se, cond, at_b, bstart, pen, atm_vol


def jump_var_share(th):
    jv = th[1] * (th[2] ** 2 + th[3] ** 2)
    return jv / (th[0] ** 2 + jv)


def calibrate(options, tag):
    dates = np.sort(options["Date"].unique())
    for m in MODELS:
        options[f"{m}_Price"] = np.nan
    options["BS_slice_Sigma"] = np.nan

    day_rows, slice_rows = [], []
    prev = None
    t0 = time.time()
    for i, dt in enumerate(dates):
        d = options.iloc[np.flatnonzero((options["Date"] == dt).values)]

        # BS: en sigma per löptid
        for _, g in d.groupby("Expiry"):
            s = fit_bs_slice(g)
            options.loc[g.index, "BS_slice_Sigma"] = s
            options.loc[g.index, "BS_slice_Price"] = bs_price(g["StockMid"].values, g["Strike"].values, g["T"].values,
                                                              RISK_FREE_RATE, DIVIDEND_YIELD, s, g["IsCall"].values)

        # Merton: en uppsättning per dag, alla löptider, stabiliserad mot gårdagen
        th, z, se, cond, at_b, bstart, pen, atm_vol = merton_day_fit(d, prev, REG_WEIGHT)
        prev = th
        options.loc[d.index, "Merton_dag_Price"] = merton_price(
            d["StockMid"].values, d["Strike"].values, d["T"].values, RISK_FREE_RATE, DIVIDEND_YIELD, *th,
            d["IsCall"].values)
        day_rows.append(dict(Date=pd.Timestamp(dt), Sigma=th[0], Lambda=th[1], Mu_J=th[2], Sigma_J=th[3],
                             SE_Sigma=se[0], SE_Lambda=se[1], SE_Mu_J=se[2], SE_Sigma_J=se[3],
                             Cond_JtJ=cond, VidGräns=at_b, BästaStart=bstart, Straff=pen,
                             N_optioner=len(d), N_löptider=d["Expiry"].nunique(),
                             RMSE_z=np.sqrt(np.mean(z ** 2)), Andel_inom_spread=np.mean(np.abs(z) <= 1),
                             ATM_IV=atm_vol, Aktie=d["StockMid"].iloc[0],
                             TotalVol=np.sqrt(th[0] ** 2 + th[1] * (th[2] ** 2 + th[3] ** 2)),
                             Hoppvariansandel=jump_var_share(th)))

        # Merton per löptid, bara korta löptider (underlag för Kou-testet)
        for exp, g in d.groupby("Expiry"):
            row = dict(Date=pd.Timestamp(dt), Expiry=exp, Days=int(g["Days"].iloc[0]),
                       N_put=int((~g["IsCall"]).sum()), N_call=int(g["IsCall"].sum()),
                       BS_Sigma=g["BS_slice_Sigma"].iloc[0])
            if row["Days"] <= KOU_TEST_MAX_DAYS and len(g) >= SLICE_MIN_QUOTES:
                ths, zs, _, conds, atbs, _, _ = fit_merton(g, [th, default_starts(atm_vol)[0]])
                options.loc[g.index, "Merton_slice_Price"] = merton_price(
                    g["StockMid"].values, g["Strike"].values, g["T"].values, RISK_FREE_RATE, DIVIDEND_YIELD, *ths,
                    g["IsCall"].values)
                row.update(Sigma=ths[0], Lambda=ths[1], Mu_J=ths[2], Sigma_J=ths[3], Cond_JtJ=conds,
                           VidGräns=atbs, RMSE_z=np.sqrt(np.mean(zs ** 2)))
            slice_rows.append(row)

        if (i + 1) % 20 == 0 or i == len(dates) - 1:
            print(f"  {tag}: {i + 1}/{len(dates)} dagar kalibrerade ({time.time() - t0:.0f} s)")

    return options, pd.DataFrame(day_rows), pd.DataFrame(slice_rows)


# =============================================================================
# 5. VAL AV REG_WEIGHT (lcurve) OCH IDENTIFIERING (fram_bak_test), körs med --full
# =============================================================================
def lcurve_one(options, reg_weight):
    """
    Merton_dag för en straffvikt: passform samma dag, passform NÄSTA dag med gårdagens parametrar
    (som en market maker som prissätter idag med gårdagens kalibrering), och parameterhopp mellan dagar.
    """
    prev, rows, z_in, z_out = None, [], [], []
    for dt in np.sort(options["Date"].unique()):
        d = options[options["Date"] == dt]
        if prev is not None:
            p = merton_price(d["StockMid"].values, d["Strike"].values, d["T"].values, RISK_FREE_RATE,
                             DIVIDEND_YIELD, *prev, d["IsCall"].values)
            z_out.append((p - d["Mid"].values) / d["HalfSpreadW"].values)
        th, z, _, _, at_b, _, _, _ = merton_day_fit(d, prev, reg_weight)
        z_in.append(z)
        rows.append(dict(Sigma=th[0], LogLambda=np.log(th[1]), Mu_J=th[2], Sigma_J=th[3], VidGräns=at_b,
                         LambdaVidTak=th[1] > MERTON_UB[1] - 1e-3))
        prev = th
    P = pd.DataFrame(rows)
    zi, zo = np.concatenate(z_in), np.concatenate(z_out)
    dP = P[["Sigma", "LogLambda", "Mu_J", "Sigma_J"]].diff().abs().median()
    return dict(REG_WEIGHT=reg_weight,
                Inom_spread_samma_dag_pct=100 * np.mean(np.abs(zi) <= 1), RMSE_z_samma_dag=np.sqrt(np.mean(zi ** 2)),
                Inom_spread_nästa_dag_pct=100 * np.mean(np.abs(zo) <= 1), RMSE_z_nästa_dag=np.sqrt(np.mean(zo ** 2)),
                Median_dSigma=dP["Sigma"], Median_dLogLambda=dP["LogLambda"], Median_dMu_J=dP["Mu_J"],
                Median_dSigma_J=dP["Sigma_J"], Andel_dagar_vid_gräns_pct=100 * P["VidGräns"].mean(),
                Andel_dagar_lambda_vid_tak_pct=100 * P["LambdaVidTak"].mean())


def lcurve(tag, weights=(0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)):
    """Avvägningskurva för REG_WEIGHT. Sparar RESULTAT/Lkurva_<tag>.xlsx."""
    options = load_stock(tag)
    rows = []
    for w in weights:
        t0 = time.time()
        r = lcurve_one(options, w)
        rows.append(r)
        print(f"  {tag} REG_WEIGHT={w:5.1f} ({time.time() - t0:.0f} s): inom spread samma dag "
              f"{r['Inom_spread_samma_dag_pct']:.1f} %, nästa dag {r['Inom_spread_nästa_dag_pct']:.1f} % | "
              f"median |dlogLambda| {r['Median_dLogLambda']:.3f} | dagar vid gräns "
              f"{r['Andel_dagar_vid_gräns_pct']:.0f} %")
    out = pd.DataFrame(rows)
    out.to_excel(os.path.join(OUT_DIR, f"Lkurva_{tag}.xlsx"), index=False)
    return out


def fram_bak_test(tag):
    """
    Kör den stabiliserade dagskalibreringen framåt OCH bakåt i tiden. Parametrar som datan bestämmer
    får samma bana åt båda hållen; parametrar som bara följer start/straff får olika banor.
    Sparar RESULTAT/Fram_bak_<tag>.xlsx.
    """
    o = load_stock(tag)
    dates = np.sort(o["Date"].unique())
    paths = {}
    for name, seq in [("fram", dates), ("bak", dates[::-1])]:
        prev, rows = None, []
        for dt in seq:
            th, z, *_ = merton_day_fit(o[o["Date"] == dt], prev, REG_WEIGHT)
            prev = th
            rows.append(dict(Date=pd.Timestamp(dt), Sigma=th[0], Lambda=th[1], Mu_J=th[2], Sigma_J=th[3],
                             Hoppvariansandel=jump_var_share(th), Inom_spread=np.mean(np.abs(z) <= 1)))
        paths[name] = pd.DataFrame(rows).set_index("Date").sort_index()
    f, b = paths["fram"], paths["bak"]
    summ = pd.DataFrame([dict(Parameter=c, Median_abs_diff=(f[c] - b[c]).abs().median(),
                              Korrelation=np.corrcoef(f[c], b[c])[0, 1],
                              Median_fram=f[c].median(), Median_bak=b[c].median())
                         for c in ["Sigma", "Lambda", "Mu_J", "Sigma_J", "Hoppvariansandel", "Inom_spread"]])
    with pd.ExcelWriter(os.path.join(OUT_DIR, f"Fram_bak_{tag}.xlsx"), engine="openpyxl") as xw:
        summ.to_excel(xw, sheet_name="Sammanfattning", index=False)
        pd.concat([f.add_suffix("_fram"), b.add_suffix("_bak")], axis=1).to_excel(xw, sheet_name="Banor")
    print(summ.round(3).to_string(index=False))
    return summ


# =============================================================================
# 6. MÄTMÅTT OCH KOU-TEST
# =============================================================================
def add_metrics(o):
    a = (o["StockMid"].values, o["Strike"].values, o["T"].values, RISK_FREE_RATE, DIVIDEND_YIELD)
    ic = o["IsCall"].values
    for m in MODELS:
        p = o[f"{m}_Price"]
        o[f"{m}_z"] = (p - o["Mid"]) / o["HalfSpreadW"]
        o[f"{m}_InSpread"] = np.where(p.isna(), np.nan, ((p >= o["Bid"]) & (p <= o["Ask"])).astype(float))
        o[f"{m}_IVerr"] = implied_vol(p.values, *a, ic) - o["IV_Mid"].values  # i vol-enheter
    return o


def summarize(o, by):
    rows = []
    for key, g in o.groupby(by, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        r = dict(zip(by, key))
        r["Antal"] = len(g)
        r["Median_halvspread_IV_vp"] = 100 * np.nanmedian((g["IV_Ask"] - g["IV_Bid"]) / 2)
        for m in MODELS:
            z = g[f"{m}_z"].dropna()
            if len(z) == 0:
                continue
            e = g[f"{m}_IVerr"].dropna()
            r[f"{m}_Medel_z"] = z.mean()
            r[f"{m}_RMSE_z"] = np.sqrt(np.mean(z ** 2))
            r[f"{m}_Inom_spread_%"] = 100 * g[f"{m}_InSpread"].mean()
            r[f"{m}_RMSE_IV_vp"] = 100 * np.sqrt(np.mean(e ** 2)) if len(e) else np.nan
            r[f"{m}_MedianAbs_IV_vp"] = 100 * np.median(np.abs(e)) if len(e) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def kou_test(o, slices):
    """
    Villkoren för att byta till Kou (korta löptider, Merton per löptid):
      A: andel OTM-puts med |z| > 1 (utanför bid-ask)
      B: vingmönster: puts underprissatta (z<0) OCH calls överprissatta (z>0) samtidigt
      C: antal OTM-calls per kort löptid (kan Kous uppsidesparametrar identifieras?)
    Tröskelvärdena är bedömningar, inte litteraturvärden.
    """
    s = o[(o["Days"] <= KOU_TEST_MAX_DAYS) & o["Merton_slice_z"].notna()]
    puts = s[~s["IsCall"]]
    A = np.mean(np.abs(puts["Merton_slice_z"]) > 1) if len(puts) else np.nan
    far_put = s[~s["IsCall"] & (s["K_over_F"] < 0.95)].groupby(["Date", "Expiry"])["Merton_slice_z"].mean()
    far_call = s[s["IsCall"] & (s["K_over_F"] > 1.05)].groupby(["Date", "Expiry"])["Merton_slice_z"].mean()
    both = pd.concat([far_put.rename("put"), far_call.rename("call")], axis=1).dropna()
    B_kou = np.mean((both["put"] < 0) & (both["call"] > 0)) if len(both) else np.nan
    B_rev = np.mean((both["put"] > 0) & (both["call"] < 0)) if len(both) else np.nan
    short = slices[slices["Days"] <= KOU_TEST_MAX_DAYS]
    C = short["N_call"].median() if len(short) else np.nan
    return pd.DataFrame([
        dict(Villkor="A: andel korta OTM-puts utanför bid-ask (Merton per löptid)", Värde=A, Förslag_tröskel="> 0.25"),
        dict(Villkor="B: andel löptider med puts under + calls över (Kou-mönstret)", Värde=B_kou, Förslag_tröskel="> 0.50"),
        dict(Villkor="B (omvänt): puts över + calls under", Värde=B_rev, Förslag_tröskel="(jämförelse)"),
        dict(Villkor="C: median antal OTM-calls per kort löptid", Värde=C, Förslag_tröskel=">= 3"),
        dict(Villkor="Antal korta löptider i testet", Värde=len(both), Förslag_tröskel=""),
    ])


# =============================================================================
# 7. FIGURER
# =============================================================================
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
C_MERTON, C_BS = "#2a78d6", "#eb6834"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def make_plots(o, days, tag):
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap

    # (a) medel-z per hink: BS_slice vs Merton_dag
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=SURF, sharey=True)
    for ax, m, title in zip(axes, ["BS_slice", "Merton_dag"],
                            ["BS, en σ per löptid", "Merton, en parameteruppsättning per dag"]):
        piv = o.pivot_table(index="MaturityBucket", columns="MoneynessBucket", values=f"{m}_z",
                            aggfunc="mean", observed=True)
        cnt = o.pivot_table(index="MaturityBucket", columns="MoneynessBucket", values=f"{m}_z",
                            aggfunc="count", observed=True)
        im = ax.imshow(piv.values, cmap=cmap, vmin=-3, vmax=3, aspect="auto")
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels([c.split(" ", 1)[1].replace(" (", "\n(") for c in piv.columns], fontsize=8, color=INK2)
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels([r.split(" ", 1)[1] for r in piv.index], fontsize=9, color=INK2)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:+.2f}\nn={int(cnt.values[i, j])}", ha="center", va="center",
                            fontsize=8, color=INK)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("medel z = (modell − mid)/halvspread\n<0: modellen för billig, >0: för dyr", fontsize=9, color=INK2)
    fig.suptitle(f"{tag}: medelfel i halvspread-enheter per löptid och moneyness", fontsize=12,
                 color=INK, x=0.01, ha="left")
    fig.savefig(os.path.join(OUT_DIR, f"{tag}_felkarta.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # (b) parametrar över tid; lambda och mu_J visas inte (identifieras inte var för sig, se fram_bak_test)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), facecolor=SURF, sharex=True)
    spec = [("Sigma", "σ (diffusionsvol)"), ("Hoppvariansandel", "Hoppens andel av total varians"),
            ("Sigma_J", "σ_J (spridning i hoppstorlek)"), ("TotalVol", "Total vol √(σ² + λ(μ_J² + σ_J²))"),
            ("Andel_inom_spread", "Andel inom bid-ask"), ("Aktie", "Aktiekurs (SEK)")]
    for ax, (col, lab) in zip(axes.ravel(), spec):
        _style(ax)
        ax.plot(days["Date"], days[col], color=C_MERTON if col != "Aktie" else INK2, linewidth=2)
        ax.set_title(lab, fontsize=10, color=INK, loc="left")
    for ax in axes[1]:
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle(f"{tag}: kalibrerad Merton per dag (λ och μ_J var för sig finns i Excel, men är inte "
                 f"identifierade – se fram/bak-testet)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{tag}_parametrar.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # (c) smile på kortaste löptiden (>= 14 dagar) på tre dagar
    all_dates = np.sort(o["Date"].unique())
    pick = [all_dates[int(q * (len(all_dates) - 1))] for q in (0.1, 0.5, 0.9)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), facecolor=SURF)
    for ax, dt in zip(axes, pick):
        _style(ax)
        d = o[(o["Date"] == dt) & (o["Days"] >= 14)]
        if d.empty:
            continue
        g = d[d["Expiry"] == d["Expiry"].min()].sort_values("Strike")
        x = g["K_over_F"].values
        ax.vlines(x, 100 * g["IV_Bid"], 100 * g["IV_Ask"], color=MUTED, linewidth=3, alpha=0.6,
                  label="Marknad bid–ask")
        a = (g["StockMid"].values, g["Strike"].values, g["T"].values, RISK_FREE_RATE, DIVIDEND_YIELD)
        for m, c, lab in [("BS_slice", C_BS, "BS (en σ)"), ("Merton_dag", C_MERTON, "Merton (dag)")]:
            iv = implied_vol(g[f"{m}_Price"].values, *a, g["IsCall"].values)
            ax.plot(x, 100 * iv, color=c, linewidth=2, marker="o", markersize=4, label=lab)
        ax.set_title(f"{pd.Timestamp(dt).date()}, löptid {int(g['Days'].iloc[0])} d", fontsize=10,
                     color=INK, loc="left")
        ax.set_xlabel("K/F", fontsize=9, color=INK2)
        ax.set_ylabel("Implicit vol (%)", fontsize=9, color=INK2)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.suptitle(f"{tag}: smile på kortaste löptiden – marknad vs modeller", fontsize=12, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{tag}_smile_exempel.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_lcurve():
    """Figur av avvägningskurvan från RESULTAT/Lkurva_*.xlsx (kör lcurve() först)."""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURF)
    for tag, c in zip(FILES, [C_MERTON, C_BS]):
        f = os.path.join(OUT_DIR, f"Lkurva_{tag}.xlsx")
        if not os.path.exists(f):
            continue
        L = pd.read_excel(f)
        x = L["REG_WEIGHT"].replace(0, 0.03)   # 0 ritas längst till vänster på log-axeln
        for ax, col in zip(axes, ["Inom_spread_nästa_dag_pct", "Median_dLogLambda"]):
            ax.plot(x, L[col], color=c, linewidth=2, marker="o", markersize=5, label=tag)
        axes[0].annotate(tag, (x.iloc[-1], L["Inom_spread_nästa_dag_pct"].iloc[-1]), xytext=(6, 0),
                         textcoords="offset points", va="center", fontsize=9, color=INK2)
    for ax, t in zip(axes, ["Andel inom bid-ask nästa dag (%), gårdagens parametrar",
                            "Median dagsförändring i log(λ)"]):
        _style(ax)
        ax.set_xscale("log")
        ax.axvline(REG_WEIGHT, color=MUTED, linewidth=1, linestyle="--")
        ax.set_xlabel("REG_WEIGHT (0 ritad vid 0.03), streckad = valt värde", fontsize=9, color=INK2)
        ax.set_title(t, fontsize=10, color=INK, loc="left")
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.suptitle("Stabilisering av Merton-kalibreringen: passform vs parameterhopp", fontsize=12, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "Lkurva.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 8. HUVUDPROGRAM
# =============================================================================
def run_one(tag):
    options, filter_log = prepare_data(os.path.join(DATA_DIR, FILES[tag]))
    options, days, slices = calibrate(options, tag)
    options = add_metrics(options)

    s_all = summarize(options.assign(Alla="Alla"), ["Alla"])
    kou = kou_test(options, slices)
    keep = ["Date", "Expiry", "Days", "T", "Name", "CallPut", "Strike", "StockMid", "Forward", "K_over_F",
            "LogMoneyness", "Bid", "Ask", "Mid", "Spread", "IV_Bid", "IV_Mid", "IV_Ask",
            "BS_slice_Sigma", "MaturityBucket", "MoneynessBucket"]
    keep += [f"{m}_{s}" for m in MODELS for s in ("Price", "z", "InSpread", "IVerr")]
    out = os.path.join(OUT_DIR, f"SEB_Merton_vs_BS_{tag}.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        s_all.to_excel(xw, sheet_name="Totalt", index=False)
        summarize(options, ["MaturityBucket"]).to_excel(xw, sheet_name="Per_löptid", index=False)
        summarize(options, ["MoneynessBucket"]).to_excel(xw, sheet_name="Per_moneyness", index=False)
        summarize(options, ["MaturityBucket", "MoneynessBucket"]).to_excel(xw, sheet_name="Löptid_x_moneyness",
                                                                           index=False)
        kou.to_excel(xw, sheet_name="Kou_test", index=False)
        days.to_excel(xw, sheet_name="Merton_param_dag", index=False)
        slices.to_excel(xw, sheet_name="Param_per_löptid", index=False)
        filter_log.to_excel(xw, sheet_name="Datafilter", index=False)
        options[keep].to_excel(xw, sheet_name="Optioner", index=False)
    print(f"  Sparat: {out}")
    make_plots(options, days, tag)

    print(f"\n  {tag} – totalt (RMSE i halvspread-enheter, andel inom bid-ask):")
    for m in MODELS:
        print(f"    {m:16s} RMSE_z={s_all[f'{m}_RMSE_z'].iloc[0]:7.2f}   "
              f"inom spread={s_all[f'{m}_Inom_spread_%'].iloc[0]:5.1f} %   "
              f"RMSE_IV={s_all[f'{m}_RMSE_IV_vp'].iloc[0]:5.2f} vp   "
              f"median |IV-fel|={s_all[f'{m}_MedianAbs_IV_vp'].iloc[0]:5.2f} vp")
    print("  Kou-test:")
    for _, r in kou.iterrows():
        print(f"    {r['Villkor']:62s} {r['Värde']:.3f}   (tröskel {r['Förslag_tröskel']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="kör även lcurve och fram_bak_test (ca 15 min till)")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    sjalvtest()
    for tag in FILES:
        print(f"\n=================== {tag} ===================")
        run_one(tag)
    if args.full:
        for tag in FILES:
            print(f"\n=================== {tag}: avvägningskurva och fram/bak-test ===================")
            lcurve(tag)
            fram_bak_test(tag)
        plot_lcurve()
    print("\nFÄRDIG. Resultat i:", OUT_DIR)
