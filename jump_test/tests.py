"""
tests.py  -  tests of the pricing engine, run before any calibration
====================================================================

  A  martingale: ln phi(-i) = 0 (forward reproduced) and phi(0) = 1, many random parameter sets
  B  COS vs Merton's series (the pricer in seb_merton_analys.py), calls and puts
  C  COS vs an independent Fourier inversion (Lewis 2001 integral with scipy.quad) for all three models
  D  COS vs brute-force Monte Carlo for Merton, Kou and Bates (and Bates with lam = 0 = pure Heston)
  E  Bates(lam = 0) vs an independent Heston-1993 pricer (original two-probability formula)
  F  Bates(xi -> 0) vs Merton (constant variance, and deterministic time-varying variance)
  G  Kou(lam = 0) vs Black-Scholes
  H  put-call parity and forward, all three models, with q = 0 and q > 0
  I  COS convergence in N and in L (is N_FIT / N_COS / L_COS big enough?)
  J  truncation range: numeric cumulants vs closed forms; MC paths outside [a, b]
  K  reproduction of the main pipeline's Merton_dag numbers (RESULTAT workbooks, run seb_merton_analys.py first):
     prices, z, RMSE, share inside, a fresh COS-based re-fit of the first days, and the skew exponents

    python tests.py            (about 1 minute; --quick: smaller Monte Carlo in J and fewer re-fit days in K)
Writes output/test_report.csv. Exit code 1 if any test fails.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.integrate import quad

import jump_calib as jc
import jump_models as jm
import jump_metrics as jmx
from jump_calib import seb

R = seb.RISK_FREE_RATE
RESULTS = []          # (block, name, value, tolerance, passed, note)
PRICERS = {k: jm.MODELS[k] for k in ("Merton", "Kou", "Bates")}     # Heston = Bates with lam = 0 (blocks D, E)


def check(block, name, value, tol, note="", lower_is_better=True):
    ok = bool(value <= tol) if lower_is_better else bool(value >= tol)
    RESULTS.append((block, name, float(value), float(tol), ok, note))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:68s} {value: .3e}  (tol {tol:.1e}) {note}")
    return ok


# ----------------------------------------------------------------------------- parameter sets used throughout
TH = {"Merton": np.array([0.17, 1.0, -0.10, 0.10]),
      "Kou": np.array([0.17, 1.0, 0.35, 15.0, 8.0]),
      "Bates": np.array([0.04, 2.0, 0.05, 0.5, -0.6, 0.3, -0.08, 0.10])}
TH_HEAVY = {"Merton": np.array([0.05, 3.0, -0.15, 0.25]),
            "Kou": np.array([0.05, 3.0, 0.3, 40.0, 5.0]),
            "Bates": np.array([0.079, 0.687, 0.062, 0.191, -0.99, 0.445, -0.128, 0.321])}


# ----------------------------------------------------------------------------- independent references
def lewis_price(model, th, S, K, T, r, q=0.0, is_call=True):
    """
    Lewis (2001) / Carr-Madan-type single integral, completely independent of the COS code:
      C / (S e^{-qT}) = 1 - e^{k/2}/pi * int_0^inf Re[ e^{-iuk} phi(u - i/2) ] / (u^2 + 1/4) du
    Puts follow from parity.
    """
    F = S * np.exp((r - q) * T)
    k = np.log(K / F)
    f = lambda u: (np.exp(-1j * u * k) * np.exp(model.logcf(u - 0.5j, T, th))).real / (u * u + 0.25)
    edges = np.concatenate([[0.0], np.geomspace(0.5, 4000.0, 45)])
    tot = sum(quad(f, a, b, limit=200, epsabs=1e-13, epsrel=1e-12)[0] for a, b in zip(edges[:-1], edges[1:]))
    call = S * np.exp(-q * T) * (1.0 - np.exp(k / 2) / np.pi * tot)
    return call if is_call else call - (S * np.exp(-q * T) - K * np.exp(-r * T))


def heston1993_call(S, K, T, r, v0, kappa, theta, xi, rho):
    """Heston (1993) original formulation: C = S P1 - K e^{-rT} P2, P_j from Gil-Pelaez inversion."""
    def cf(u, j):
        uj, b = (0.5, kappa - rho * xi) if j == 1 else (-0.5, kappa)
        iu = 1j * u
        d = np.sqrt((rho * xi * iu - b) ** 2 - xi ** 2 * (2 * uj * iu - u * u))
        g = (b - rho * xi * iu + d) / (b - rho * xi * iu - d)
        C = r * iu * T + kappa * theta / xi ** 2 * ((b - rho * xi * iu + d) * T - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g)))
        D = (b - rho * xi * iu + d) / xi ** 2 * (1 - np.exp(d * T)) / (1 - g * np.exp(d * T))
        return np.exp(C + D * v0 + iu * np.log(S))

    def P(j):
        f = lambda u: (np.exp(-1j * u * np.log(K)) * cf(u, j) / (1j * u)).real
        edges = np.concatenate([[1e-12], np.geomspace(0.5, 800.0, 30)])
        return 0.5 + sum(quad(f, a, b, limit=200)[0] for a, b in zip(edges[:-1], edges[1:])) / np.pi
    return S * P(1) - K * np.exp(-r * T) * P(2)


def mc_jump_sum(model_name, th, T, n, rng):
    """Exact simulation of X_T for Merton / Kou (diffusion + compound Poisson), antithetic in the diffusion."""
    if model_name == "Merton":
        sg, lam, mu, sj = th
        kap = np.exp(mu + 0.5 * sj ** 2) - 1
        Nj = rng.poisson(lam * T, n)
        J = mu * Nj + np.sqrt(Nj) * sj * rng.standard_normal(n)
    else:
        sg, lam, p, e1, e2 = th
        kap = p * e1 / (e1 - 1) + (1 - p) * e2 / (e2 + 1) - 1
        Nj = rng.poisson(lam * T, n)
        nu = rng.binomial(Nj, p)
        J = rng.gamma(np.maximum(nu, 1e-300), 1 / e1) * (nu > 0) - rng.gamma(np.maximum(Nj - nu, 1e-300), 1 / e2) * (Nj - nu > 0)
    Z = rng.standard_normal(n)
    drift = (-0.5 * sg ** 2 - lam * kap) * T
    return np.concatenate([drift + sg * np.sqrt(T) * Z + J, drift - sg * np.sqrt(T) * Z + J])


def mc_heston_qe(th, T, n, steps, rng):
    """Andersen (2008) QE scheme for ln(S_T/F_T) in the Heston model (+ Merton jumps if lam > 0)."""
    v0, kap, theta, xi, rho, lam, mu, sj = th
    dt = T / steps
    ek = np.exp(-kap * dt)
    K0 = -rho * kap * theta * dt / xi
    K1 = 0.5 * dt * (kap * rho / xi - 0.5) - rho / xi
    K2 = 0.5 * dt * (kap * rho / xi - 0.5) + rho / xi
    K3 = K4 = 0.5 * dt * (1 - rho ** 2)
    v = np.full(n, v0)
    X = np.zeros(n)
    for _ in range(steps):
        m = theta + (v - theta) * ek
        s2 = v * xi ** 2 * ek * (1 - ek) / kap + theta * xi ** 2 * (1 - ek) ** 2 / (2 * kap)
        psi = s2 / m ** 2
        U, Zv, Z = rng.random(n), rng.standard_normal(n), rng.standard_normal(n)
        lo = psi <= 1.5
        inv = 2.0 / np.where(lo, psi, 1.0)
        b2 = inv - 1 + np.sqrt(inv) * np.sqrt(np.maximum(inv - 1, 0))
        v_lo = m / (1 + b2) * (np.sqrt(b2) + Zv) ** 2
        pp = (psi - 1) / (psi + 1)
        beta = (1 - pp) / m
        v_hi = np.where(U <= pp, 0.0, np.log((1 - pp) / np.maximum(1 - U, 1e-300)) / beta)
        vn = np.where(lo, v_lo, v_hi)
        X += K0 + K1 * v + K2 * vn + np.sqrt(np.maximum(K3 * v + K4 * vn, 0)) * Z
        v = vn
    if lam > 0:
        kJ = np.exp(mu + 0.5 * sj ** 2) - 1
        Nj = rng.poisson(lam * T, n)
        X += mu * Nj + np.sqrt(Nj) * sj * rng.standard_normal(n) - lam * kJ * T
    return X


def mc_price(X, S, K, T, r, is_call):
    F = S * np.exp(r * T)
    ST = F * np.exp(X)
    pay = np.maximum(ST - K, 0) if is_call else np.maximum(K - ST, 0)
    return np.exp(-r * T) * pay.mean(), np.exp(-r * T) * pay.std() / np.sqrt(len(pay))


# ----------------------------------------------------------------------------- tests
def test_A():
    print("\nA. martingale / forward: ln phi(-i) = 0 and phi(0) = 1")
    rng = np.random.default_rng(1)
    for name, m in PRICERS.items():
        worst, worst0 = 0.0, 0.0
        for _ in range(300):
            th = m.lb + (m.ub - m.lb) * rng.uniform(0.02, 0.98, m.n_par)
            if name == "Bates":
                th[1] = rng.uniform(0.2, 8); th[3] = rng.uniform(0.1, 1.5)   # keep Heston in a sane (non-exploding) region
                th[0] = rng.uniform(0.01, 0.2); th[2] = rng.uniform(0.01, 0.2)
            T = rng.uniform(0.01, 1.5)
            worst = max(worst, abs(np.exp(m.logcf(-1j, T, th)) - 1))
            worst0 = max(worst0, abs(np.exp(m.logcf(0.0, T, th)) - 1))
        check("A", f"{name}: max |E[exp(X)] - 1| over 300 random parameter sets", worst, 1e-9)
        check("A", f"{name}: max |phi(0) - 1|", worst0, 1e-12)


def test_B():
    print("\nB. COS vs Merton series (seb_merton_analys.merton_price)")
    S, m = 100.0, jm.MODELS["Merton"]
    for lab, th in (("base", TH["Merton"]), ("heavy", TH_HEAVY["Merton"])):
        worst = 0
        for T in (5 / 365.25, 14 / 365.25, 30 / 365.25, 90 / 365.25, 0.5, 1.0, 1.5):
            F = S * np.exp(R * T)
            K = F * np.exp(np.linspace(-0.4, 0.4, 21))
            for ic in (True, False):
                ps = seb.merton_price(S, K, T, R, 0.0, *th, np.full(len(K), ic))
                pc = jm.cos_price(m, th, S, K, T, np.full(len(K), ic), R)
                worst = max(worst, np.max(np.abs(pc - ps)))
        check("B", f"Merton COS vs series, {lab} params, max |diff| (SEK, S = 100)", worst, 5e-6)


def test_C():
    print("\nC. COS vs independent Lewis integral (scipy.quad)")
    S = 100.0
    for name, m in PRICERS.items():
        for lab, th in (("base", TH[name]), ("heavy", TH_HEAVY[name])):
            worst = 0
            for T in (10 / 365.25, 60 / 365.25, 0.5, 1.0):
                F = S * np.exp(R * T)
                for k in (-0.3, -0.1, 0.0, 0.1, 0.3):
                    K = F * np.exp(k)
                    ref = lewis_price(m, th, S, K, T, R, is_call=(k >= 0))
                    pc = jm.cos_price(m, th, S, K, T, k >= 0, R)[0]
                    worst = max(worst, abs(pc - ref))
            check("C", f"{name} COS vs Lewis integral, {lab} params, max |diff| (SEK)", worst, 5e-6)


def test_D(quick):
    print("\nD. COS vs brute-force Monte Carlo (price +- 4 standard errors)")
    S = 100.0
    rng = np.random.default_rng(7)
    n_exact, n_qe = 1_000_000, 300_000
    cases = [("Merton", TH["Merton"]), ("Kou", TH["Kou"]), ("Bates", TH["Bates"]),
             ("Bates (lam=0, pure Heston)", np.r_[TH["Bates"][:5], 0.0, -0.08, 0.10])]
    for label, th in cases:
        name = label.split(" ")[0]
        m = jm.MODELS[name]
        worst_z = 0.0
        for T in (0.25, 1.0):
            if name == "Bates":
                X = mc_heston_qe(th, T, n_qe, int(T * 100), rng)
            else:
                X = mc_jump_sum(name, th, T, n_exact // 2, rng)
            F = S * np.exp(R * T)
            for k in (-0.3, -0.15, 0.0, 0.15, 0.3):
                K = F * np.exp(k)
                ic = k >= 0
                mc, se = mc_price(X, S, K, T, R, ic)
                pc = jm.cos_price(m, th, S, K, T, ic, R)[0]
                # QE has a tiny weak-discretisation bias: allow 0.1% of the ATM price on top of 4 SE
                allow = 4 * se + (1e-3 * 4.0 if name == "Bates" else 0.0)
                worst_z = max(worst_z, abs(pc - mc) / allow)
        check("D", f"{label}: max |COS - MC| / (4 SE) over 2 maturities x 5 strikes", worst_z, 1.0,
              note="(<= 1 means inside 4 SE)")


def test_E():
    print("\nE. Bates(lam = 0) vs independent Heston-1993 pricer")
    S, m = 100.0, jm.MODELS["Bates"]
    for lab, hp in (("base", TH["Bates"][:5]), ("high vol-of-vol", np.array([0.05, 1.5, 0.06, 1.0, -0.7]))):
        th = np.r_[hp, 0.0, 0.0, 0.1]
        worst = 0
        for T in (0.1, 0.5, 1.0):
            F = S * np.exp(R * T)
            for k in (-0.3, -0.1, 0.0, 0.1, 0.3):
                K = F * np.exp(k)
                ref = heston1993_call(S, K, T, R, *hp)           # a call
                pc = jm.cos_price(m, th, S, K, T, True, R)[0]
                worst = max(worst, abs(pc - ref))
        check("E", f"Bates(lam=0) vs Heston 1993 calls, {lab}, max |diff| (SEK)", worst, 5e-6)


def test_F():
    print("\nF. Bates(xi -> 0) vs Merton")
    S, mB, mM = 100.0, jm.MODELS["Bates"], jm.MODELS["Merton"]
    sig, lam, mu, sj = TH["Merton"]
    grid = [(T, S * np.exp(R * T) * np.exp(np.linspace(-0.3, 0.3, 13))) for T in (30 / 365.25, 0.25, 1.0)]

    def gap(thB):
        worst = 0
        for T, K in grid:
            F = S * np.exp(R * T); ic = K >= F
            worst = max(worst, np.max(np.abs(jm.cos_price(mB, thB, S, K, T, ic, R) - jm.cos_price(mM, TH["Merton"], S, K, T, ic, R))))
        return worst
    # (a) rho = 0: the only effect of a small vol-of-vol is a convexity term of order xi^2
    e = {xi: gap(np.array([sig ** 2, 2.0, sig ** 2, xi, 0.0, lam, mu, sj])) for xi in (1e-2, 1e-3, 1e-4)}
    check("F", "Bates(v0=theta=sigma^2, rho=0, xi=1e-4) vs Merton, max |diff| (SEK)", e[1e-4], 5e-7)
    check("F", "  error slope in log10(xi) between 1e-2 and 1e-3 (expect 2: convexity ~ xi^2)", abs(np.log10(e[1e-2] / e[1e-3]) - 2.0), 0.3)
    # (b) rho != 0: the leverage effect (skew) is first order in rho*xi, so the error must fall LINEARLY in xi
    e = {xi: gap(np.array([sig ** 2, 2.0, sig ** 2, xi, -0.6, lam, mu, sj])) for xi in (1e-2, 1e-3, 1e-4)}
    check("F", "Bates(rho=-0.6, xi=1e-4) vs Merton, max |diff| (SEK)", e[1e-4], 5e-4,
          note="(first order in rho*xi: small but not zero)")
    check("F", "  error slope in log10(xi) between 1e-2 and 1e-3 (expect 1: skew ~ rho*xi)", abs(np.log10(e[1e-2] / e[1e-3]) - 1.0), 0.3)
    # (c) deterministic time-varying variance: v0 != theta, xi -> 0, rho = 0  ==>  Merton with sigma_eff^2 T = int v dt
    v0, kap, thet = 0.09, 3.0, 0.03
    thB = np.array([v0, kap, thet, 1e-4, 0.0, lam, mu, sj])
    worst = 0
    for T, K in grid:
        w = thet * T + (v0 - thet) * (1 - np.exp(-kap * T)) / kap
        thM = np.array([np.sqrt(w / T), lam, mu, sj])
        F = S * np.exp(R * T); ic = K >= F
        worst = max(worst, np.max(np.abs(jm.cos_price(mB, thB, S, K, T, ic, R) - jm.cos_price(mM, thM, S, K, T, ic, R))))
    check("F", "Bates(v0 != theta, xi=1e-4, rho=0) vs Merton with integrated-variance sigma, max |diff|", worst, 5e-6)


def test_G():
    print("\nG. Kou(lam = 0) vs Black-Scholes")
    S, m = 100.0, jm.MODELS["Kou"]
    th = np.array([0.22, 0.0, 0.35, 15.0, 8.0])
    worst = 0
    for T in (30 / 365.25, 0.5, 1.0):
        F = S * np.exp(R * T); K = F * np.exp(np.linspace(-0.3, 0.3, 13)); ic = K >= F
        worst = max(worst, np.max(np.abs(jm.cos_price(m, th, S, K, T, ic, R) - seb.bs_price(S, K, T, R, 0.0, 0.22, ic))))
    check("G", "Kou(lam=0) vs Black-Scholes, max |diff| (SEK)", worst, 1e-6)


def test_H():
    print("\nH. put-call parity and forward")
    S = 100.0
    for q in (0.0, 0.015):
        for name, m in PRICERS.items():
            worst = 0
            for T in (7 / 365.25, 30 / 365.25, 90 / 365.25, 0.5, 1.0):
                F = S * np.exp((R - q) * T)
                K = F * np.exp(np.linspace(-0.4, 0.4, 21))
                c = jm.cos_price(m, TH[name], S, K, T, np.ones(len(K), bool), R, q)
                p = jm.cos_price(m, TH[name], S, K, T, np.zeros(len(K), bool), R, q)
                worst = max(worst, np.max(np.abs(c - p - (S * np.exp(-q * T) - K * np.exp(-R * T)))))
            check("H", f"{name}: put-call parity max error (SEK), q = {q:g}", worst, 5e-6)


def test_I():
    print("\nI. COS convergence (reference: N = 16384, L = 16)")
    S = 100.0
    for name, m in PRICERS.items():
        for lab, th, tol_fit in (("base", TH[name], 1e-5), ("heavy", TH_HEAVY[name], 5e-2)):
            e_fit = e_rep = 0
            for T in (5 / 365.25, 14 / 365.25, 30 / 365.25, 90 / 365.25, 0.5, 330 / 365.25):
                F = S * np.exp(R * T); K = F * np.exp(np.linspace(-0.3, 0.3, 13)); ic = K >= F
                ref = jm.cos_price(m, th, S, K, T, ic, R, N=16384, L=16)
                e_fit = max(e_fit, np.max(np.abs(jm.cos_price(m, th, S, K, T, ic, R, N=jm.N_FIT) - ref)))
                e_rep = max(e_rep, np.max(np.abs(jm.cos_price(m, th, S, K, T, ic, R, N=jm.N_COS) - ref)))
            check("I", f"{name} {lab}: error at N_FIT = {jm.N_FIT} (optimiser grid), SEK", e_fit, tol_fit,
                  note="(extreme corner: sigma = 0.05 at 5 days; the calibrator re-polishes on a finer grid if the gap > 1e-3)" if lab == "heavy" else "")
            check("I", f"{name} {lab}: error at N_COS = {jm.N_COS} (reported prices), SEK", e_rep, 1e-6 if lab == "base" else 5e-6)


def test_J(quick):
    print("\nJ. truncation range: numeric cumulants vs closed forms; MC paths outside [a, b]")
    T = np.array([30 / 365.25, 0.5])
    th = TH["Merton"]; sg, lam, mu, sj = th
    m = jm.MODELS["Merton"]
    a, b = jm.cos_range(m, th, T, L=1.0)                      # L = 1 -> width = sqrt(c2 + sqrt(c4))
    EJ2, EJ4 = mu ** 2 + sj ** 2, mu ** 4 + 6 * mu ** 2 * sj ** 2 + 3 * sj ** 4
    c1 = T * (-0.5 * sg ** 2 - lam * (np.exp(mu + 0.5 * sj ** 2) - 1) + lam * mu)
    c2, c4 = T * (sg ** 2 + lam * EJ2), T * lam * EJ4
    exact_w = np.sqrt(c2 + np.sqrt(c4))
    check("J", "Merton: numeric vs closed-form half-width sqrt(c2+sqrt(c4)), max rel. error", np.max(np.abs((b - a) / 2 / exact_w - 1)), 0.02)
    check("J", "Merton: numeric vs closed-form c1 (centre), max abs error", np.max(np.abs((a + b) / 2 - c1)), 1e-3)
    th = TH["Kou"]; sg, lam, p, e1, e2 = th
    m = jm.MODELS["Kou"]
    _keep = jm.Kou.extra_width
    jm.Kou.extra_width = lambda self, T_, th_: np.zeros_like(np.asarray(T_, float))   # isolate the cumulant part
    a, b = jm.cos_range(m, th, T, L=1.0)
    jm.Kou.extra_width = _keep
    zeta = p * e1 / (e1 - 1) + (1 - p) * e2 / (e2 + 1) - 1
    EJ1 = p / e1 - (1 - p) / e2
    EJ2 = 2 * p / e1 ** 2 + 2 * (1 - p) / e2 ** 2
    EJ4 = 24 * p / e1 ** 4 + 24 * (1 - p) / e2 ** 4
    c1 = T * (-0.5 * sg ** 2 - lam * zeta + lam * EJ1)
    exact_w = np.sqrt(T * (sg ** 2 + lam * EJ2) + np.sqrt(T * lam * EJ4))
    check("J", "Kou: numeric vs closed-form half-width, max rel. error", np.max(np.abs((b - a) / 2 / exact_w - 1)), 0.10,
          note="(only the order of magnitude matters: L = 12 and the tail rule add a large margin)")
    check("J", "Kou: numeric vs closed-form c1 (centre), max abs error", np.max(np.abs((a + b) / 2 - c1)), 1e-3)
    # Bates: no closed form here; check that essentially no Monte-Carlo path leaves [a, b] at the production L
    rng = np.random.default_rng(3)
    for lab, thb in (("base", TH["Bates"]), ("heavy", TH_HEAVY["Bates"])):
        for Tm in (30 / 365.25, 0.9):
            X = mc_heston_qe(thb, Tm, 200_000 if not quick else 50_000, 60, rng)
            a, b = jm.cos_range(jm.MODELS["Bates"], thb, np.array([Tm]))
            outside = np.mean((X < a[0]) | (X > b[0]))
            check("J", f"Bates {lab}, T = {Tm * 365.25:.0f} d: share of MC paths outside [a, b]", outside, 1e-6)


def test_K(quick):
    print("\nK. reproduction of the main pipeline's Merton_dag numbers (workbooks in RESULTAT)")
    doc = {"BOL": dict(inside=85.2, rmse=0.81, mkt=-0.36, mer=-0.60), "VOLVB": dict(inside=68.4, rmse=1.28, mkt=-0.41, mer=-0.73)}
    m = jm.MODELS["Merton"]
    for tag in jc.STOCKS:
        f = os.path.join(seb.OUT_DIR, f"SEB_Merton_vs_BS_{tag}.xlsx")
        if not os.path.exists(f):
            print(f"  [SKIP] {f} not found - run seb_merton_analys.py first to reproduce the baseline")
            continue
        xl = pd.ExcelFile(f)
        par, stored = xl.parse("Merton_param_dag"), xl.parse("Optioner")
        o = jc.load_options(tag)
        same = (len(o) == len(stored)) and np.allclose(o["Strike"].values, stored["Strike"].values) and \
            (o["Date"].values == pd.to_datetime(stored["Date"]).values).all()
        check("K", f"{tag}: option rows identical to stored workbook (1 = yes)", float(same), 1.0, lower_is_better=False,
              note=f"({len(o)} rows)")
        if not same:
            continue
        parmap = {pd.Timestamp(r.Date): np.array([r.Sigma, r.Lambda, r.Mu_J, r.Sigma_J]) for r in par.itertuples()}
        z = np.empty(len(o)); price = np.empty(len(o))
        for dt, idx in o.groupby("Date").indices.items():
            d = o.iloc[idx]
            price[idx] = jc.model_prices(m, parmap[pd.Timestamp(dt)], d)
            z[idx] = (price[idx] - d["Mid"].values) / d["HalfSpreadW"].values
        check("K", f"{tag}: COS-Merton at stored daily params vs stored Merton_dag_Price, max |diff| (SEK)",
              np.max(np.abs(price - stored["Merton_dag_Price"].values)), 5e-6)
        check("K", f"{tag}: ... vs stored Merton_dag_z, max |diff|", np.max(np.abs(z - stored["Merton_dag_z"].values)), 1e-3)
        rmse, inside = np.sqrt(np.mean(z ** 2)), 100 * np.mean(np.abs(z) <= 1)
        ba = 100 * np.mean((price >= o["Bid"].values) & (price <= o["Ask"].values))
        check("K", f"{tag}: RMSE_z = {rmse:.3f} vs project summary {doc[tag]['rmse']}", abs(rmse - doc[tag]["rmse"]), 0.006)
        check("K", f"{tag}: share inside bid-ask = {ba:.1f} % vs project summary {doc[tag]['inside']} %", abs(ba - doc[tag]["inside"]), 0.06)
        print(f"        ({tag}: share with |z| <= 1 is {inside:.1f} % - the floor of 0.05 SEK on the half-spread makes it a bit larger)")
        # fresh COS re-fit of the first days with the same warm-start chain (day-step 1)
        n_days = 4 if quick else 12
        dates = np.sort(o["Date"].unique())
        prev, worst_rmse, worst_par = None, 0, 0
        for i in range(n_days):
            d = o[o["Date"] == dates[i]]
            fit = jc.fit_model(m, d, prev=prev)
            prev = fit["theta"]
            worst_rmse = max(worst_rmse, abs(np.sqrt(np.mean(fit["z"] ** 2)) - par["RMSE_z"].iloc[i]))
            worst_par = max(worst_par, np.max(np.abs(fit["theta"] - parmap[pd.Timestamp(dates[i])])))
        check("K", f"{tag}: re-fit first {n_days} days (COS, same chain): max |RMSE_z - stored|", worst_rmse, 2e-3)
        print(f"        (largest parameter difference to stored fit: {worst_par:.1e})")
        # skew exponents from the stored implied-vol errors (all days), as in the project summary
        s = stored.copy()
        s["Merton_IV"] = s["IV_Mid"] + s["Merton_dag_IVerr"]
        sk = jmx.skew_table(s, {"Market": "IV_Mid", "Merton": "Merton_IV"})
        ex, _ = jmx.decay_exponents(sk, ["Market", "Merton"], n_boot=200)
        e = ex[ex["Subset"].str.startswith("all")].set_index("Series")["Exponent"]
        check("K", f"{tag}: market skew exponent {e['Market']:.3f} vs project summary {doc[tag]['mkt']}", abs(e["Market"] - doc[tag]["mkt"]), 0.01)
        check("K", f"{tag}: Merton skew exponent {e['Merton']:.3f} vs project summary {doc[tag]['mer']}", abs(e["Merton"] - doc[tag]["mer"]), 0.01)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller Monte Carlo and fewer refit days")
    ap.add_argument("--out-dir", default=jc.OUT_DIR)
    a = ap.parse_args()
    t0 = time.time()
    blocks = [("A", test_A), ("B", test_B), ("C", test_C), ("D", lambda: test_D(a.quick)), ("E", test_E), ("F", test_F),
              ("G", test_G), ("H", test_H), ("I", test_I), ("J", lambda: test_J(a.quick)), ("K", lambda: test_K(a.quick))]
    for name, fn in blocks:
        t1 = time.time()
        fn()
        print(f"     block {name}: {time.time() - t1:.1f} s")
    df = pd.DataFrame(RESULTS, columns=["Block", "Test", "Value", "Tolerance", "Passed", "Note"])
    os.makedirs(a.out_dir, exist_ok=True)
    df.to_csv(os.path.join(a.out_dir, "test_report.csv"), index=False)
    nfail = int((~df["Passed"]).sum())
    print(f"\n{len(df) - nfail}/{len(df)} checks passed; {nfail} failed.  Runtime {time.time() - t0:.0f} s.  "
          f"Saved {os.path.join(a.out_dir, 'test_report.csv')}")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
