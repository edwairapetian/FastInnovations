"""Topic life cycles with catch-up costs, and the cross-section they generate.

A topic is born with flow P0 and an empty stock.  While entry is privately
profitable its flow grows at a topic-specific diffusion rate g; the stock of
relevant literature follows dS/dt = P - delta S.  An entrant must first read
the stock at speed b_c; with career turnover rho her lifetime value relative
to the frontier is

    E(P, S) = zeta v(P) exp(-kappa_s S) / F,        kappa_s = rho / b_c.

Phase G (growth):        dP/dt = g P            while E > 1
Phase C (consolidation): dP/dt = -rho P         no entry; the stock catches up
Phase M (mature):        P = P_m(zeta)          replacement entry, w(P_m) = F/zeta

with w(P) = v(P) exp(-kappa P), kappa = kappa_s / delta, the entrant's payoff
once the stock has caught up (S = P / delta).  Phase C uses this fast-stock
approximation: entry resumes when zeta w(P) = F.  Topics become obsolete at
rate xi.  Returns of active researchers are zeta v(P): within a topic they
trace the payoff curve, which is what the author-fixed-effects design of
Test 4 recovers.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import brentq

from .payoff import GridCurve


@dataclass(frozen=True)
class LifeCycleParams:
    F: float = 0.95           # frontier payoff (per unit of writing speed)
    kappa_s: float = 0.0035   # catch-up cost per paper of stock, rho / b_c
    delta: float = 0.25       # decay rate of the stock's relevance
    rho: float = 0.10         # rate at which researchers leave a topic
    xi: float = 0.04          # topic obsolescence rate
    P0: float = 1.0           # flow of a newly opened topic
    g_med: float = 0.30       # median diffusion rate of new topics
    g_disp: float = 0.8       # log-sd of the diffusion rate
    zeta_sd: float = 0.0      # log-sd of topic fertility
    horizon: float = 120.0
    dt: float = 0.05
    n_g: int = 25
    n_z: int = 9

    @property
    def kappa(self) -> float:
        return self.kappa_s / self.delta

    def with_(self, **kw) -> "LifeCycleParams":
        return replace(self, **kw)


def _nodes(n, sd, trunc: float = 2.5):
    """Normal(0, sd^2) truncated at +-trunc sd, on n equally spaced nodes (midpoint rule)."""
    if sd <= 0 or n == 1:
        return np.zeros(1), np.ones(1)
    h = 2 * trunc / n
    x = -trunc + h * (np.arange(n) + 0.5)
    w = np.exp(-0.5 * x ** 2)
    return sd * x, w / w.sum()


def g_types(lc: LifeCycleParams):
    """Lognormal diffusion rates, truncated at +-2.5 log-sd."""
    x, w = _nodes(lc.n_g, lc.g_disp)
    return lc.g_med * np.exp(x), w


def mature_density(gc: GridCurve, lc: LifeCycleParams, lz: float = 0.0):
    """P_m(zeta): zeta w(P) = F on the falling branch of w."""
    lnF = np.log(lc.F)
    f = lambda P: lz + gc.lnv(P) - lc.kappa * P - lnF
    grid = np.exp(gc.lp)
    lw = gc.lv - lc.kappa * grid
    i = int(np.argmax(lw))
    Pw = grid[i]
    if f(Pw) < 0:
        return np.nan
    hi = Pw * 2
    while f(hi) > 0:
        hi *= 2
    return brentq(f, Pw, hi, xtol=1e-10)


def stopping_density_asymptotic(gc: GridCurve, lc: LifeCycleParams, g, lz: float = 0.0):
    """P_c^inf(g): v(P) exp(-kappa delta P / (delta + g)) zeta = F (Proposition 7)."""
    lnF = np.log(lc.F)
    kg = lc.kappa * lc.delta / (lc.delta + g)
    grid = np.exp(gc.lp)
    i = int(np.argmax(gc.lv - kg * grid))
    Pw = grid[i]
    f = lambda P: lz + gc.lnv(P) - kg * P - lnF
    if f(Pw) < 0:
        return np.nan
    hi = Pw * 2
    while f(hi) > 0:
        hi *= 2
    return brentq(f, Pw, hi, xtol=1e-10)


def g_star(gc: GridCurve, lc: LifeCycleParams, lz: float = 0.0):
    """Critical diffusion rate above which a topic overshoots the peak of v."""
    rent = lz + np.log(gc.vhat) - np.log(lc.F)
    if rent <= 0:
        return np.nan
    return lc.delta * (lc.kappa * gc.Phat / rent - 1.0)


def consolidation_value(gc: GridCurve, lc: LifeCycleParams, P, lz: float = 0.0, n: int = 400):
    """U(P): expected per-period payoff over a career that starts when entry
    stops at density P, with correct expectations about consolidation.

    After entry stops the topic thins at rate rho down to the mature density
    P_m (the consolidation path of Proposition 7(iii)).  With exponential
    careers, in units u = rho * time,
        U(P) = int_0^inf e^{-u} zeta v(max(P e^{-u}, P_m)) du.
    The catch-up delay is ignored in the timing of payoffs (it only scales the
    value by the survival factor exp(-rho S / b_c)).
    """
    Pm = mature_density(gc, lc, lz)
    P = float(P)
    if P <= Pm:
        return float(np.exp(lz + gc.lnv(P)))
    um = np.log(P / Pm)
    u = np.linspace(0.0, um, n)
    f = np.exp(-u + lz + gc.lnv(P * np.exp(-u)))
    integral = float(np.sum((f[1:] + f[:-1]) * np.diff(u)) / 2)
    return integral + float(np.exp(-um + lz + gc.lnv(Pm)))


def stopping_density_re(gc: GridCurve, lc: LifeCycleParams, g, lz: float = 0.0):
    """Stopping density with forward-looking entrants (P0 -> 0).

    Entry stops at the first density along the growth path at which the value
    of joining, U(P) exp(-rho S / b_c) with S = P / (delta + g), falls to F.
    The search starts at the mature density: below it the stock is smaller
    than a mature topic's and the value exceeds F.
    """
    lnF = np.log(lc.F)
    Pm = mature_density(gc, lc, lz)
    kg = lc.kappa * lc.delta / (lc.delta + g)
    f = lambda P: np.log(consolidation_value(gc, lc, P, lz)) - kg * P - lnF
    grid = Pm * np.exp(np.linspace(0.0, np.log(np.exp(gc.lp[-1]) / Pm), 600))
    vals = np.array([f(P) for P in grid])
    neg = np.where(vals <= 0)[0]
    if len(neg) == 0:
        return np.nan
    k = neg[0]
    if k == 0:
        return float(Pm)
    return brentq(f, grid[k - 1], grid[k], xtol=1e-9)


def g_star_re(gc: GridCurve, lc: LifeCycleParams, lz: float = 0.0):
    """Growth rate above which forward-looking entry overshoots the peak:
    U(Phat) exp(-kappa delta Phat / (delta + g)) = F."""
    rent = np.log(consolidation_value(gc, lc, gc.Phat, lz)) - np.log(lc.F)
    if rent <= 0:
        return np.nan
    return lc.delta * (lc.kappa * gc.Phat / rent - 1.0)


def g_star_re_bound(gc: GridCurve, lc: LifeCycleParams, lz: float = 0.0):
    """Sufficient condition for overshooting under rational expectations:
    g > delta (Phat / P_m - 1), i.e. the stock at the peak is smaller than a
    mature topic's stock (Proposition 7, part ii)."""
    return lc.delta * (gc.Phat / mature_density(gc, lc, lz) - 1.0)


@dataclass
class CrossSection:
    lnP: np.ndarray       # (types, ages)
    y: np.ndarray         # log returns of active researchers, ln zeta + ln v(P)
    w: np.ndarray         # researcher-year weights
    phase: np.ndarray     # 0 growth, 1 consolidation, 2 mature
    age: np.ndarray
    g: np.ndarray         # (types,)
    lz: np.ndarray        # (types,)
    Pc: np.ndarray        # stopping density by type
    Pm: np.ndarray        # mature density by type
    Phat: float
    meta: dict = field(default_factory=dict)


def simulate(gc: GridCurve, lc: LifeCycleParams) -> CrossSection:
    gs, gw = g_types(lc)
    lzs, zw = _nodes(lc.n_z, lc.zeta_sd)
    G, LZ = np.meshgrid(gs, lzs, indexing="ij")
    W = np.outer(gw, zw)
    g = G.ravel(); lz = LZ.ravel(); wt = W.ravel()
    nT = g.size
    Pm = np.array([mature_density(gc, lc, l) for l in lzs])
    Pm_t = np.repeat(Pm[None, :], len(gs), axis=0).ravel()
    lnF = np.log(lc.F)
    ages = np.arange(0.0, lc.horizon, lc.dt)
    nA = ages.size
    # Growth phase in closed form: P = P0 e^{g t}, S = P (1 - e^{-(g+delta) t}) / (g + delta).
    gt = g[:, None] * ages[None, :]
    lnPg = np.minimum(np.log(lc.P0) + gt, gc.lp[-1])
    Pg = np.exp(lnPg)
    Sg = Pg * (1 - np.exp(-(g[:, None] + lc.delta) * ages[None, :])) / (g[:, None] + lc.delta)
    E = lz[:, None] + gc.lnv(Pg) - lc.kappa_s * Sg - lnF
    neg = E <= 0
    k_c = np.where(neg.any(axis=1), neg.argmax(axis=1), nA - 1)
    # interpolate the stopping time between grid points
    k0 = np.maximum(k_c - 1, 0)
    E0, E1 = E[np.arange(nT), k0], E[np.arange(nT), k_c]
    frac = np.where(k_c > 0, E0 / np.where(E0 - E1 > 0, E0 - E1, 1.0), 0.0)
    t_c = ages[k0] + np.clip(frac, 0, 1) * lc.dt * (k_c > 0)
    Pc = lc.P0 * np.exp(g * t_c)
    # Consolidation (fast-stock approximation): P = Pc e^{-rho (t - t_c)} down to P_m.
    lnPm = np.log(np.where(np.isfinite(Pm_t), Pm_t, 1e-300))
    t_m = t_c + np.where(np.isfinite(Pm_t), np.maximum(np.log(Pc) - lnPm, 0.0) / lc.rho, np.inf)
    A = ages[None, :]
    lnP = np.where(A < t_c[:, None], lnPg,
                   np.where(A < t_m[:, None], np.log(Pc)[:, None] - lc.rho * (A - t_c[:, None]),
                            lnPm[:, None]))
    ph = np.where(A < t_c[:, None], 0, np.where(A < t_m[:, None], 1, 2)).astype(np.int8)
    Pfull = np.exp(lnP)
    y = lz[:, None] + gc.lnv(Pfull)
    w = wt[:, None] * np.exp(-lc.xi * ages)[None, :] * lc.dt * Pfull / gc.x(Pfull)
    return CrossSection(lnP=lnP, y=y, w=w, phase=ph, age=np.broadcast_to(ages, lnP.shape),
                        g=g, lz=lz, Pc=Pc, Pm=Pm_t, Phat=gc.Phat,
                        meta=dict(type_weight=wt))


def _within(a, w):
    """Demean rows of a with weights w (topic fixed effects)."""
    m = (a * w).sum(axis=1, keepdims=True) / w.sum(axis=1, keepdims=True)
    return a - m


def test4_moments(cs: CrossSection, estimator: str = "pooled", noise_sd: float = 0.3, n_noise: int = 7):
    """Replicate the Test 4 regressions on the simulated cross-section.

    Density is measured with classical noise in logs (sd ``noise_sd``), as the
    arXiv density proxy is (only 58% of nearest neighbours fall in the same
    fine area); each researcher-year is spread over ``n_noise`` Gauss-Hermite
    draws of the measurement error.  Observed density is standardised with its
    researcher-weighted mean and sd.  ``estimator='pooled'`` compares
    researcher-years across topics, the analogue of researchers who move
    between denser and sparser topics (the model has no ability heterogeneity,
    so author effects are absorbed); ``estimator='within'`` adds topic fixed
    effects and uses only the life-cycle variation inside a topic.  Weighted
    least squares throughout.
    """
    w0 = cs.w / cs.w.sum()
    if noise_sd > 0 and n_noise > 1:
        e, we = np.polynomial.hermite_e.hermegauss(n_noise)
        we = we / we.sum()
    else:
        e, we = np.zeros(1), np.ones(1)
    nT = w0.shape[0]
    # observed log density and weights, shape (types, ages * draws)
    lp = (cs.lnP[:, :, None] + noise_sd * e[None, None, :]).reshape(nT, -1)
    w = (w0[:, :, None] * we[None, None, :]).reshape(nT, -1)
    yv = np.repeat(cs.y[:, :, None], e.size, axis=2).reshape(nT, -1)
    mu = (w * lp).sum()
    sd = np.sqrt((w * (lp - mu) ** 2).sum())
    z = (lp - mu) / sd
    wf = w.ravel()
    if estimator == "within":
        tr = lambda a: _within(a, w).ravel()
        const = []
    else:
        tr = lambda a: a.ravel()
        const = [np.ones(z.size)]
    y = tr(yv)

    def ols(cols):
        X = np.column_stack(const + [tr(c) for c in cols])
        Xw = X * wf[:, None]
        return np.linalg.solve(Xw.T @ X, Xw.T @ y)[len(const):]

    b = ols([z, z ** 2])
    lin = ols([z])[0]
    zpeak = float(-b[0] / (2 * b[1])) if b[1] < 0 else np.nan
    beyond = float((w0 * (cs.lnP > np.log(cs.Phat))).sum())
    beyond_est = float((w * (z > zpeak)).sum()) if np.isfinite(zpeak) else np.nan
    zf = z.ravel()
    o = np.argsort(zf); cw = np.cumsum(wf[o])
    cuts = zf[o][np.minimum(np.searchsorted(cw, [0.2, 0.4, 0.6, 0.8]), zf.size - 1)]
    q = np.digitize(z, cuts)
    bq = ols([(q == j).astype(float) for j in (0, 1, 3, 4)])
    quint = np.array([bq[0], bq[1], 0.0, bq[2], bq[3]])
    return dict(sd_lnP=float(sd), lin=float(lin), b1=float(b[0]), b2=float(b[1]),
                zpeak=zpeak, beyond=beyond, beyond_est=beyond_est, quint=quint,
                zhat=float((np.log(cs.Phat) - mu) / sd), mu_lnP=float(mu))


def composition_covariance(cs: CrossSection):
    """Researcher-weighted covariance of log density and topic age.

    If the community grows at rate g_N, topic ages in the cross-section are
    weighted by exp(-(xi + g_N) a); d E[ln P] / d g_N = -Cov(ln P, age).  A
    measured volume elasticity that compares areas whose growth accelerated by
    dg over a window in which volume rose by dlnV is biased by
    -Cov(ln P, age) dg / dlnV relative to the steady-state elasticity.
    """
    w = cs.w / cs.w.sum()
    m_l = float((w * cs.lnP).sum())
    m_a = float((w * cs.age).sum())
    return float((w * (cs.lnP - m_l) * (cs.age - m_a)).sum())


def composition_bias(cs: CrossSection, T: float = 7.0, dg: float = 1e-3):
    """Bias of a measured volume elasticity from topic-age composition.

    An area whose growth rate rises by dg for T years gains dg*T in log volume
    and, because topics born in the window are more numerous, its age weights
    for a < T scale by exp(dg (T - a)).  Returns d E[ln P] / d ln V from that
    composition shift alone; the limit dg -> 0 is Cov(ln P, (T - a)^+) / T.
    The steady-state elasticity E is the measured one minus this bias.
    """
    w0 = cs.w / cs.w.sum()
    w1 = w0 * np.exp(dg * np.clip(T - cs.age, 0, None))
    w1 = w1 / w1.sum()
    return float(((w1 - w0) * cs.lnP).sum() / (dg * T))


def phase_shares(cs: CrossSection):
    w = cs.w / cs.w.sum()
    return {name: float(w[cs.phase == k].sum()) for k, name in enumerate(["growth", "consolidation", "mature"])}
