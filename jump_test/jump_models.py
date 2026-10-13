"""
jump_models.py  -  Merton, Kou, Heston and Bates priced with characteristic functions + COS inversion
======================================================================================================

Everything here is forward-based and European (as agreed with SEB: no dividends in the sample,
American premium on OTM options is below the spread).

Notation
--------
  X = ln(S_T / F_T)      F_T = S_0 * exp((r - q) * T)        so E[exp(X)] = 1 (martingale)
  logcf(u, T, theta)     = ln E[exp(i u X)]                  (the characteristic *exponent*)
  k = ln(K / F_T)        log-moneyness (same as 'LogMoneyness' in the SEB data)

Models (parameter order = the order of the arrays everywhere in this folder)
--------------------------------------------------------------------------
  Merton : (sigma, lam, mu_j, sigma_j)                       4 parameters
           jumps ~ Poisson(lam); ln-jump ~ N(mu_j, sigma_j^2); same parametrisation as
           seb_merton_analys.py so the numbers are directly comparable.
  Kou    : (sigma, lam, p, eta1, eta2)                       5 parameters
           double-exponential jumps: up with prob p, size ~ Exp(rate eta1); down with prob 1-p,
           size ~ -Exp(rate eta2). Needs eta1 > 1 (finite E[e^J]).
  Bates  : (v0, kappa, theta, xi, rho, lam, mu_j, sigma_j)   8 parameters
           Heston variance dv = kappa (theta - v) dt + xi sqrt(v) dW_v, corr(dW_S, dW_v) = rho,
           plus Merton jumps with the same jump-compensator as in Merton.
  Heston : (v0, kappa, theta, xi, rho)                       5 parameters = Bates without jumps (control model)

Compensators ("drift so the forward is a martingale")
-----------------------------------------------------
  Merton : omega = -sigma^2/2 - lam*kappa_J,           kappa_J = exp(mu_j + sigma_j^2/2) - 1
  Kou    : omega = -sigma^2/2 - lam*zeta,              zeta    = p*eta1/(eta1-1) + (1-p)*eta2/(eta2+1) - 1
  Bates  : Heston part is already a martingale in forward terms (the -iu... terms in C, D below),
           jump part:  T * lam * ( phi_J(u) - 1 - i u kappa_J ).
  Check (done in tests.py): logcf(-i) = 0 for every parameter set  <=>  E[exp(X)] = 1.

COS method (Fang & Oosterlee 2008)
----------------------------------
  E[g(X)] ~= (2/(b-a)) * sum'_{n=0}^{N-1} Re[ phi(w_n) e^{-i w_n a} ] * V_n ,   w_n = n*pi/(b-a)
  with V_n the cosine coefficients of the payoff g(X) = (e^X - e^k)^+ (call) or (e^k - e^X)^+ (put),
  and  price = S * exp(-q T) * E[g(X)]   (because exp(-rT) F = S exp(-qT)).
  The truncation interval [a, b] is c1 +- L*sqrt(c2 + sqrt(c4)) from the first, second and fourth
  cumulants of X, which are obtained numerically from the characteristic exponent (so the same code
  works for all models; checked against closed forms in tests.py).

Only numpy is needed.
"""

import numpy as np

# COS grid. N_FIT is used inside the optimiser (fast), N_COS for every price that is reported.
# Convergence in N and the effect of L are checked in tests.py (error far below 1e-5 SEK
# for S = 100 across 5-400 day maturities, i.e. far below the 0.05 SEK half-spread floor).
N_FIT = 384
N_COS = 2048
L_COS = 12.0


# =============================================================================
# 1. CHARACTERISTIC EXPONENTS
# =============================================================================
def _merton_jump_term(u, T, lam, mu_j, sigma_j):
    """Compensated Merton jump exponent: T*lam*(phi_J(u) - 1 - i u kappa_J)."""
    kJ = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0
    return T * lam * (np.exp(1j * u * mu_j - 0.5 * sigma_j ** 2 * u * u) - 1.0 - 1j * u * kJ)


def logcf_merton(u, T, th):
    sigma, lam, mu_j, sigma_j = th
    diff = T * (-0.5 * sigma ** 2 * (1j * u + u * u))   # -sigma^2/2 * (iu + u^2): drift + variance
    return diff + _merton_jump_term(u, T, lam, mu_j, sigma_j)


def logcf_kou(u, T, th):
    sigma, lam, p, eta1, eta2 = th
    q = 1.0 - p
    zeta = p * eta1 / (eta1 - 1.0) + q * eta2 / (eta2 + 1.0) - 1.0
    phiJ = p * eta1 / (eta1 - 1j * u) + q * eta2 / (eta2 + 1j * u)
    diff = T * (-0.5 * sigma ** 2 * (1j * u + u * u))
    return diff + T * lam * (phiJ - 1.0 - 1j * u * zeta)


def logcf_heston(u, T, v0, kappa, theta, xi, rho):
    """
    Heston exponent for X = ln(S_T/F_T) ("little Heston trap" form of Albrecher et al. 2007:
    exp(-dT) instead of exp(+dT), continuous branch of the logarithm for all u in the strip).
    """
    iu = 1j * u
    b = kappa - rho * xi * iu
    d = np.sqrt(b * b + xi * xi * (iu + u * u))
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    C = (kappa * theta / xi ** 2) * ((b - d) * T - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
    D = ((b - d) / xi ** 2) * (1.0 - e) / (1.0 - g * e)
    return C + D * v0


def logcf_bates(u, T, th):
    v0, kappa, theta, xi, rho, lam, mu_j, sigma_j = th
    return logcf_heston(u, T, v0, kappa, theta, xi, rho) + _merton_jump_term(u, T, lam, mu_j, sigma_j)


# =============================================================================
# 2. MODEL OBJECTS (bounds, start values, penalty coordinates, diagnostics)
# =============================================================================
class JumpModel:
    """Container for everything the calibrator needs to know about one model."""
    name = ""
    pnames = ()
    lb = ub = pen_scale = None

    def logcf(self, u, T, th):
        raise NotImplementedError

    def std_guess(self, T, th):
        """Rough std-dev of X over horizon T (only used to pick the finite-difference step)."""
        raise NotImplementedError

    def extra_width(self, T, th):
        """Minimum half-width of the COS interval beyond the cumulant rule (needed for exponential jump tails)."""
        return np.zeros_like(np.asarray(T, float))

    def starts(self, atm_vol, merton_th=None):
        raise NotImplementedError

    def pen_coords(self, th):
        """Coordinates in which the day-to-day penalty is measured (logs for rates, vols for variances)."""
        raise NotImplementedError

    def diagnostics(self, th):
        """Extra columns for the daily CSV: total vol (diffusion + jumps) and jump share of variance."""
        raise NotImplementedError

    @property
    def n_par(self):
        return len(self.pnames)


class Merton(JumpModel):
    name = "Merton"
    pnames = ("sigma", "lam", "mu_j", "sigma_j")
    # identical to seb_merton_analys.py (MERTON_LB/UB, REG_SCALE) so the baseline is the same problem
    lb = np.array([0.01, 0.001, -0.60, 0.01])
    ub = np.array([1.50, 10.0, 0.30, 0.80])
    pen_scale = np.array([0.02, 0.5, 0.05, 0.05])   # sigma, log(lam), mu_j, sigma_j

    def logcf(self, u, T, th):
        return logcf_merton(u, T, th)

    def std_guess(self, T, th):
        s, lam, mu, sj = th
        return np.sqrt(T * (s ** 2 + lam * (mu ** 2 + sj ** 2)))

    def starts(self, atm_vol, merton_th=None):          # same three starts as default_starts()
        return [np.array([0.85 * atm_vol, 0.5, -0.10, 0.15]),
                np.array([0.60 * atm_vol, 1.5, -0.05, 0.10]),
                np.array([0.90 * atm_vol, 0.2, -0.25, 0.20])]

    def pen_coords(self, th):
        return np.array([th[0], np.log(th[1]), th[2], th[3]])

    def diagnostics(self, th):
        s, lam, mu, sj = th
        jv = lam * (mu ** 2 + sj ** 2)
        return dict(TotalVol=np.sqrt(s ** 2 + jv), JumpVarShare=jv / (s ** 2 + jv))


class Kou(JumpModel):
    name = "Kou"
    pnames = ("sigma", "lam", "p", "eta1", "eta2")
    # eta1 > 1 is needed for E[e^J] < inf; mean up-jump = 1/eta1, mean down-jump = 1/eta2
    lb = np.array([0.01, 0.001, 0.01, 1.5, 1.5])
    ub = np.array([1.50, 10.0, 0.99, 80.0, 80.0])
    pen_scale = np.array([0.02, 0.5, 0.10, 0.3, 0.3])   # sigma, log(lam), p, log(eta1), log(eta2)

    def logcf(self, u, T, th):
        return logcf_kou(u, T, th)

    def std_guess(self, T, th):
        s, lam, p, e1, e2 = th
        return np.sqrt(T * (s ** 2 + lam * (2 * p / e1 ** 2 + 2 * (1 - p) / e2 ** 2)))

    def extra_width(self, T, th):
        """
        Double-exponential jumps have tails exp(-eta x), much heavier than the Gaussian-like tails the cumulant rule
        assumes at short maturities. Require lam*T*prob*exp(-eta*w) < 1e-9 for both sides (x 1.2 for multi-jump sums)
        plus 8 diffusion standard deviations.
        """
        s, lam, p, e1, e2 = th
        T = np.asarray(T, float)
        lt = np.maximum(lam * T, 1e-3)
        w_up = np.log(np.maximum(lt * p, 1e-300) / 1e-9) / e1
        w_dn = np.log(np.maximum(lt * (1 - p), 1e-300) / 1e-9) / e2
        return 1.2 * np.maximum(np.maximum(w_up, w_dn), 0.0) + 8.0 * s * np.sqrt(T)

    def starts(self, atm_vol, merton_th=None):
        st = [np.array([0.85 * atm_vol, 0.5, 0.3, 15.0, 8.0]),
              np.array([0.60 * atm_vol, 2.0, 0.4, 20.0, 15.0]),
              np.array([0.90 * atm_vol, 0.2, 0.2, 10.0, 5.0])]
        if merton_th is not None:   # Kou with Merton-like jump mix: mostly down jumps of mean size |mu_j|
            s, lam, mu, sj = merton_th
            st.insert(0, np.array([s, lam, 0.25, 1 / max(0.5 * sj + 0.02, 0.02), 1 / max(abs(mu) + 0.5 * sj, 0.03)]))
        return st

    def pen_coords(self, th):
        return np.array([th[0], np.log(th[1]), th[2], np.log(th[3]), np.log(th[4])])

    def diagnostics(self, th):
        s, lam, p, e1, e2 = th
        jv = lam * (2 * p / e1 ** 2 + 2 * (1 - p) / e2 ** 2)
        return dict(TotalVol=np.sqrt(s ** 2 + jv), JumpVarShare=jv / (s ** 2 + jv))


class Bates(JumpModel):
    name = "Bates"
    pnames = ("v0", "kappa", "theta", "xi", "rho", "lam", "mu_j", "sigma_j")
    lb = np.array([1e-4, 0.1, 1e-4, 0.05, -0.99, 0.001, -0.60, 0.01])
    ub = np.array([1.00, 20.0, 1.00, 3.00, 0.99, 10.0, 0.30, 0.80])
    # v0, theta penalised in vol units (sqrt), kappa and xi in logs
    pen_scale = np.array([0.02, 0.5, 0.02, 0.3, 0.1, 0.5, 0.05, 0.05])

    def logcf(self, u, T, th):
        return logcf_bates(u, T, th)

    def std_guess(self, T, th):
        v0, kap, thb, xi, rho, lam, mu, sj = th
        return np.sqrt(T * (max(v0, thb) + lam * (mu ** 2 + sj ** 2)))

    def starts(self, atm_vol, merton_th=None):
        a2 = atm_vol ** 2
        st = [np.array([a2, 2.0, a2, 0.5, -0.6, 0.3, -0.08, 0.10]),
              np.array([(0.8 * atm_vol) ** 2, 1.0, a2, 0.8, -0.8, 0.8, -0.10, 0.08]),
              np.array([a2, 5.0, (0.9 * atm_vol) ** 2, 1.0, -0.3, 0.1, -0.15, 0.15])]
        if merton_th is not None:   # Bates nests Merton as xi -> 0: start from the Merton fit of the same day
            s, lam, mu, sj = merton_th
            st.insert(0, np.array([s ** 2, 2.0, s ** 2, 0.3, -0.3, lam, mu, sj]))
        return st

    def pen_coords(self, th):
        return np.array([np.sqrt(th[0]), np.log(th[1]), np.sqrt(th[2]), np.log(th[3]), th[4],
                         np.log(th[5]), th[6], th[7]])

    def diagnostics(self, th):
        v0, kap, thb, xi, rho, lam, mu, sj = th
        jv = lam * (mu ** 2 + sj ** 2)
        # long-run total vol (theta + jumps); the term-structure of Heston variance is in between v0 and theta
        return dict(TotalVol=np.sqrt(thb + jv), JumpVarShare=jv / (thb + jv),
                    Feller=2 * kap * thb / xi ** 2)


class Heston(JumpModel):
    """Heston stochastic variance without jumps; same bounds, penalty scales and starts as the diffusion part of Bates."""
    name = "Heston"
    pnames = ("v0", "kappa", "theta", "xi", "rho")
    lb = Bates.lb[:5].copy()
    ub = Bates.ub[:5].copy()
    pen_scale = Bates.pen_scale[:5].copy()

    def logcf(self, u, T, th):
        return logcf_heston(u, T, *th)

    def std_guess(self, T, th):
        v0, kap, thb, xi, rho = th
        return np.sqrt(T * max(v0, thb))

    def starts(self, atm_vol, merton_th=None):
        return [s[:5] for s in Bates().starts(atm_vol, None)]

    def pen_coords(self, th):
        return np.array([np.sqrt(th[0]), np.log(th[1]), np.sqrt(th[2]), np.log(th[3]), th[4]])

    def diagnostics(self, th):
        v0, kap, thb, xi, rho = th
        return dict(TotalVol=np.sqrt(thb), JumpVarShare=0.0, Feller=2 * kap * thb / xi ** 2)


MODELS = {"Merton": Merton(), "Kou": Kou(), "Bates": Bates(), "Heston": Heston()}


# =============================================================================
# 3. COS PRICER
# =============================================================================
def cos_range(model, th, Tu, L=L_COS):
    """
    Truncation interval [a, b] for X at each (unique) maturity in Tu.
    Cumulants c1, c2, c4 come from a 4-point finite-difference stencil of the characteristic exponent
    on the real axis at u = h, 2h with h = 0.1/std (Re ln phi = -c2 u^2/2 + c4 u^4/24 - ..., Im = c1 u - c3 u^3/6).
    Only an approximate size of the cumulants matters here (L = 12 gives a large safety margin).
    """
    Tu = np.asarray(Tu, float)
    s = np.maximum(model.std_guess(Tu, th), 1e-6)
    h = 0.1 / s
    l1, l2 = model.logcf(h, Tu, th), model.logcf(2 * h, Tu, th)
    A, B, I1, I2 = l1.real, l2.real, l1.imag, l2.imag
    c4 = 2.0 * (B - 4.0 * A) / h ** 4
    c2 = -2.0 * A / h ** 2 + c4 * h ** 2 / 12.0
    c1 = (8.0 * I1 - I2) / (6.0 * h)
    width = L * np.sqrt(np.maximum(c2 + np.sqrt(np.maximum(c4, 0.0)), 0.0))
    width = np.maximum(width, 0.6 * L * s)           # floor in case of numerical trouble
    width = np.maximum(width, model.extra_width(Tu, th) + np.abs(c1))
    return c1 - width, c1 + width


def cos_price(model, th, S, K, T, is_call, r, q=0.0, N=N_COS, L=L_COS):
    """
    European option prices (arrays S, K, T, is_call of equal length) under `model` with parameters th.
    Returns an array of prices (NaN where the computation failed).
    """
    S, K, T = (np.atleast_1d(np.asarray(x, float)) for x in (S, K, T))
    is_call = np.atleast_1d(np.asarray(is_call, bool))
    S, K, T, is_call = np.broadcast_arrays(S, K, T, is_call)
    Fwd = S * np.exp((r - q) * T)
    k = np.log(K / Fwd)
    Tu, inv = np.unique(T, return_inverse=True)
    a_u, b_u = cos_range(model, th, Tu, L)
    n = np.arange(N, dtype=float)
    w_u = n[None, :] * np.pi / (b_u - a_u)[:, None]                       # (nu, N)
    phi = np.exp(model.logcf(w_u, Tu[:, None], th))                       # (nu, N)
    Fphi = (phi * np.exp(-1j * w_u * a_u[:, None])).real
    Fphi[:, 0] *= 0.5                                                      # sum' halves the n = 0 term
    out = np.empty(len(S))
    sgn = np.where(n % 2 == 0, 1.0, -1.0)[None, :]                         # cos(n pi)
    for mask, call in ((is_call, True), (~is_call, False)):
        if not mask.any():
            continue
        j = np.flatnonzero(mask)
        e = inv[j]
        a, b = a_u[e][:, None], b_u[e][:, None]
        w = w_u[e]
        kk = k[j][:, None]
        kc = np.clip(kk, a, b)
        th_n = w * (kc - a)                                                # = n pi (kc - a)/(b - a)
        cs, sn = np.cos(th_n), np.sin(th_n)
        ekc = np.exp(kc)
        with np.errstate(divide="ignore", invalid="ignore"):
            sin_over_w = sn / w
        sin_over_w[:, 0] = (kc - a)[:, 0] if not call else 0.0             # psi n = 0 limits
        if call:
            # chi(kc,b) - e^k psi(kc,b);  upper limit b has w(b-a) = n pi, so cos = (-1)^n, sin = 0
            chi = (sgn * np.exp(b) - ekc * (cs + w * sn)) / (1.0 + w * w)
            psi = -sin_over_w
            psi[:, 0] = (b - kc)[:, 0]
            V = chi - np.exp(kk) * psi
        else:
            # e^k psi(a,kc) - chi(a,kc);  lower limit a has w*0 = 0, so cos = 1, sin = 0
            chi = (ekc * (cs + w * sn) - np.exp(a)) / (1.0 + w * w)
            psi = sin_over_w
            V = np.exp(kk) * psi - chi
        out[j] = (2.0 / (b - a))[:, 0] * np.sum(Fphi[e] * V, axis=1)
    return S * np.exp(-q * T) * out
