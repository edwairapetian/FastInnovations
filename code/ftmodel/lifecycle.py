"""Topic life cycles with exact catch-up, and the cross-section they generate.

A topic is born with flow P0 (its founder's papers) and an empty stock.  While
entry pays, its flow grows at a topic-specific diffusion rate g; the stock of
relevant literature follows dS/dt = P - delta S, so along the growth path

    S(P) = P (1 - (P0 / P)^((g + delta) / g)) / (g + delta)  <=  P / (g + delta).

A newcomer must read the stock at speed b_c while new papers keep arriving
(ftmodel.catchup).  A myopic newcomer who expects the flow to stay at P values
the topic, relative to the frontier, at

    zeta v(P) Gamma(P, S) / F,   Gamma(P, S) = (1 + delta S / (b_c - P))^(-nu),  nu = (rho + xi) / delta,

which is zero for P >= b_c.  A forward-looking newcomer anticipates that the
topic thins at rate rho once entry stops and counts payoffs only after her
catch-up ends (re_value_ln).

Phase G (growth):        dP/dt = g P            while entry pays
Phase C (consolidation): dP/dt = -rho P         no entry
Phase M (mature):        P = P_m(zeta)          replacement entry, zeta w(P_m) = F

with w(P) = v(P) (1 - P / b_c)^nu the newcomer's payoff in a stationary topic.
Phase C uses the fast-stock approximation: entry resumes when zeta w(P) = F.
A topic that grows too slowly stops below P_m, in the valley where zeta w < F on
the rising side of w: its stock catches up with its flow before it is worth
joining.  It never matures; it thins and is abandoned (take-off threshold).
Topics become obsolete at rate xi.  Returns of active researchers are zeta v(P).

Policies (LifeCycleParams.subsidy, .stock_tax) change the entry rule to
zeta v Gamma - tau S > F + s: a flat subsidy s to exploring, or an entry tax
proportional to the stock a newcomer must read (zero in a newborn topic).
life_cycle_accounts aggregates output and researcher-years over the life
cycle, explorers and catch-up included; founder_fixed_point solves
F = (1 - omega) e + omega U_F(F) with founders who stay in their topics.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import brentq

from .catchup import CatchUp
from .payoff import GridCurve


@dataclass(frozen=True)
class LifeCycleParams:
    F: float = 0.95           # frontier value (flow-equivalent, per unit of production speed)
    b_c: float = 30.0         # catch-up reading speed: the speed limit
    delta: float = 0.25       # decay of the stock's relevance
    rho: float = 0.10         # rate at which careers in a topic end
    xi: float = 0.04          # topic obsolescence rate
    P0: float = 1.0           # flow of a newly founded topic
    g_med: float = 0.30       # median diffusion rate of new topics
    g_disp: float = 0.8       # log-sd of the diffusion rate
    zeta_sd: float = 0.0      # log-sd of topic fertility
    horizon: float = 120.0
    dt: float = 0.05
    n_g: int = 25
    n_z: int = 9
    subsidy: float = 0.0      # flat subsidy to exploring (raises the entry bar to F + s)
    stock_tax: float = 0.0    # entry tax per unit of stock (flow-equivalent), zero at birth

    @property
    def policy(self) -> bool:
        return self.subsidy != 0.0 or self.stock_tax != 0.0

    @property
    def cu(self) -> CatchUp:
        return CatchUp(b_c=self.b_c, delta=self.delta, rho=self.rho, xi=self.xi)

    @property
    def nu(self) -> float:
        return (self.rho + self.xi) / self.delta

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


def _lnv(gc: GridCurve, P):
    return gc.lnv(np.minimum(np.asarray(P, float), np.exp(gc.lp[-1])))


# ----------------------------------------------------------------- stationary objects
def lnw(gc: GridCurve, lc: LifeCycleParams, P, lz: float = 0.0):
    """ln of the newcomer's payoff in a stationary topic, lz + ln v(P) + nu ln(1 - P/b_c)."""
    P = np.asarray(P, float)
    return lz + _lnv(gc, P) + lc.cu.ln_gamma(P)


def peak_w(gc: GridCurve, lc: LifeCycleParams):
    grid = np.exp(gc.lp)
    grid = grid[grid < lc.b_c]
    lw = lnw(gc, lc, grid)
    i = int(np.argmax(lw))
    return float(grid[i]), float(np.exp(lw[i]))


def entry_gap(gc: GridCurve, lc: LifeCycleParams, P, S=None, lz: float = 0.0):
    """ln of a myopic newcomer's net value over the entry bar, ln(zeta v Gamma - tau S) - ln(F + s).

    S is the stock she must read (stationary P / delta if None).  Without a
    policy this is lz + ln v(P) + ln Gamma(P, S) - ln F.
    """
    P = np.asarray(P, float)
    lv = lz + _lnv(gc, P) + lc.cu.ln_gamma(P, S)
    if not lc.policy:
        return lv - np.log(lc.F)
    St = P / lc.delta if S is None else np.asarray(S, float)
    with np.errstate(over="ignore"):
        net = np.exp(lv) - lc.stock_tax * St
    with np.errstate(divide="ignore"):
        return np.where(net > 0, np.log(np.maximum(net, 1e-300)), -np.inf) - np.log(lc.F + lc.subsidy)


def mature_density(gc: GridCurve, lc: LifeCycleParams, lz: float = 0.0):
    """P_m(zeta): zeta w(P) = F (net of any policy) on the falling branch, below b_c."""
    f = lambda P: float(entry_gap(gc, lc, P, None, lz))
    Pw, _ = peak_w(gc, lc)
    if lc.policy:
        grid = np.exp(gc.lp)
        grid = grid[grid < (lc.b_c if np.isfinite(lc.b_c) else np.inf)]
        vals = entry_gap(gc, lc, grid, None, lz)
        Pw = float(grid[int(np.argmax(vals))])
    if f(Pw) < 0:
        return np.nan
    if np.isfinite(lc.b_c):
        return brentq(f, Pw, lc.b_c * (1 - 1e-12), xtol=1e-11)
    hi = Pw * 2
    while f(hi) > 0:
        hi *= 2
    return brentq(f, Pw, hi, xtol=1e-11)


# ----------------------------------------------------------------- growth phase
def growth_stock(P, g, P0, delta):
    """Stock when a topic born with flow P0 and no stock reaches flow P growing at rate g.

    P0 = 0 gives the asymptotic stock P / (g + delta).
    """
    P = np.asarray(P, float)
    if P0 <= 0:
        return P / (g + delta)
    ratio = np.clip(P0 / np.maximum(P, P0), 0.0, 1.0)
    return P * (1 - ratio ** ((g + delta) / g)) / (g + delta)


def ln_value_myopic(gc: GridCurve, lc: LifeCycleParams, P, S, lz: float = 0.0):
    """ln of a myopic newcomer's payoff: lz + ln v(P) + ln Gamma(P, S) (-inf if P >= b_c, S > 0)."""
    P = np.asarray(P, float)
    return lz + _lnv(gc, P) + lc.cu.ln_gamma(P, S)


def _first_root(f, grid):
    """First point on an increasing grid where f changes sign from + to <= 0 (nan if none)."""
    vals = np.array([f(x) for x in grid])
    if vals[0] <= 0:
        return float(grid[0])
    neg = np.where(vals <= 0)[0]
    if len(neg) == 0:
        return np.nan
    k = neg[0]
    a, b = grid[k - 1], grid[k]
    if not np.isfinite(vals[k]):
        # value drops to -inf at the speed limit: bisect on the finite side
        for _ in range(80):
            mid = 0.5 * (a + b)
            fm = f(mid)
            if np.isfinite(fm) and fm > 0:
                a = mid
            else:
                b = mid
            if np.isfinite(fm) and fm <= 0:
                return brentq(f, a, mid, xtol=1e-10) if f(a) > 0 else mid
        return 0.5 * (a + b)
    return brentq(f, a, b, xtol=1e-10)


def stopping_density_myopic(gc: GridCurve, lc: LifeCycleParams, g, lz: float = 0.0,
                            P0: float | None = None, stock: str = "exact", n: int = 400):
    """Density at which myopic entry stops along the growth path (below b_c)."""
    P0 = lc.P0 if P0 is None else P0
    s0 = 0.0 if stock == "asymptotic" else P0
    f = lambda P: float(entry_gap(gc, lc, P, growth_stock(P, g, s0, lc.delta), lz))
    top = min(lc.b_c * (1 - 1e-9), np.exp(gc.lp[-1])) if np.isfinite(lc.b_c) else np.exp(gc.lp[-1])
    grid = np.exp(np.linspace(np.log(P0), np.log(top), n))
    r = _first_root(f, grid)
    return float(top) if np.isnan(r) else r


def g_star_peak(gc: GridCurve, lc: LifeCycleParams, lz: float = 0.0, P0: float | None = None,
                stock: str = "exact"):
    """Diffusion rate at which a myopic newcomer at the peak density still beats the frontier.

    stock='asymptotic' (P0 -> 0 in the stock only, S = P / (g + delta)) has the closed form
        g_inf = delta Phat / ((b_c - Phat) ((zeta vhat / F)^(1/nu) - 1)) - delta.
    The exact stock is smaller, so the exact rate is at most g_inf.  Returns inf if the
    peak lies beyond the speed limit or above every frontier crossing, 0 if the value at
    the peak exceeds F for every g.
    """
    P0 = lc.P0 if P0 is None else P0
    Ph = gc.Phat
    lnF = np.log(lc.F)
    if np.isfinite(lc.b_c) and Ph >= lc.b_c:
        return np.inf
    vh = lz + np.log(gc.vhat)
    if vh <= lnF:
        return np.inf
    if stock == "asymptotic":
        if not np.isfinite(lc.b_c):
            return 0.0
        k = np.exp((vh - lnF) / lc.nu) - 1
        return max(lc.delta * Ph / ((lc.b_c - Ph) * k) - lc.delta, 0.0)
    f = lambda g: float(ln_value_myopic(gc, lc, Ph, growth_stock(Ph, g, P0, lc.delta), lz) - lnF)
    lo, hi = 1e-6, 1e4
    if f(lo) > 0:
        return 0.0
    if f(hi) < 0:
        return np.inf
    return brentq(f, lo, hi, xtol=1e-10)


def _g_threshold(pred, lo=1e-4, hi=1e3, iters=60):
    """Smallest g in [lo, hi] at which the monotone predicate becomes true (bisection in logs)."""
    if pred(lo):
        return 0.0
    if not pred(hi):
        return np.inf
    a, b = np.log(lo), np.log(hi)
    for _ in range(iters):
        mid = 0.5 * (a + b)
        if pred(np.exp(mid)):
            b = mid
        else:
            a = mid
    return float(np.exp(b))


def g_takeoff(gc: GridCurve, lc: LifeCycleParams, lz: float = 0.0, P0: float | None = None):
    """Take-off threshold: the smallest diffusion rate at which myopic entry carries a newborn
    topic to its mature density.  Slower topics stall in the valley where zeta w < F and are
    abandoned; inf if the first follower of a newborn topic does not beat the frontier."""
    P0 = lc.P0 if P0 is None else P0
    if float(entry_gap(gc, lc, P0, 0.0, lz)) <= 0:
        return np.inf
    Pm = mature_density(gc, lc, lz)
    if not np.isfinite(Pm):
        return np.inf
    return _g_threshold(lambda g: stopping_density_myopic(gc, lc, g, lz, P0) >= Pm * (1 - 1e-6))


def stall_recovery_gap(gc: GridCurve, lc: LifeCycleParams, g: float, lz: float = 0.0,
                       horizon: float = 200.0, n: int = 4000):
    """Largest myopic entry gap after a stall: a topic whose growth stops below P_m thins
    at rate rho while its stock follows dS/dt = P - delta S.  A negative value means the
    topic is never worth joining again (it is abandoned)."""
    Pm = mature_density(gc, lc, lz)
    Ps = stopping_density_myopic(gc, lc, g, lz)
    if not (np.isfinite(Pm) and Ps < Pm):
        return np.nan
    Ss = float(growth_stock(Ps, g, lc.P0, lc.delta))
    t = np.linspace(0.0, horizon, n)
    P = Ps * np.exp(-lc.rho * t)
    # dS/dt = P - delta S with P = Ps e^{-rho t}: closed form
    S = Ss * np.exp(-lc.delta * t) + Ps * (np.exp(-lc.rho * t) - np.exp(-lc.delta * t)) / (lc.delta - lc.rho)
    gaps = entry_gap(gc, lc, P[1:], S[1:], lz)
    return float(np.max(gaps))


def stopping_density_stock_learning(curve, lc: LifeCycleParams, g: float, chi: float,
                                    n: int = 500):
    """Myopic stopping density when learning draws on L = (1 - chi) P + chi delta S
    (Appendix: learning from the stock).  In a stationary topic L = P, so mature
    densities are unchanged; along the growth path S < P / delta and L < P."""
    lnF = np.log(lc.F)
    top = lc.b_c * (1 - 1e-9)
    P = np.exp(np.linspace(np.log(lc.P0), np.log(top), n))
    S = growth_stock(P, g, lc.P0, lc.delta)
    L = (1 - chi) * P + chi * lc.delta * S
    L = np.maximum(L, 1e-9)
    val = curve.lnv_learning_from(P, L) + lc.cu.ln_gamma(P, S) - lnF
    neg = np.where(val <= 0)[0]
    if len(neg) == 0:
        return float(top)
    k = neg[0]
    if k == 0:
        return float(P[0])
    return float(P[k - 1] + (P[k] - P[k - 1]) * val[k - 1] / (val[k - 1] - val[k]))


def chi_bar(curve, lc: LifeCycleParams):
    """Largest weight of the stock in learning at which the first follower of a newborn topic
    (empty stock, so L = (1 - chi) P0) still beats the frontier."""
    f = lambda chi: float(curve.lnv_learning_from(lc.P0, (1 - chi) * lc.P0)[0] - np.log(lc.F))
    if f(0.0) <= 0:
        return 0.0
    if f(1.0 - 1e-9) > 0:
        return 1.0
    return brentq(f, 0.0, 1.0 - 1e-9, xtol=1e-10)


def thresholds_stock_learning(curve, gc: GridCurve, lc: LifeCycleParams, chi: float):
    """Take-off and overshooting thresholds when learning draws on the stock with weight chi."""
    Pm = mature_density(gc, lc)
    t0 = _g_threshold(lambda g: stopping_density_stock_learning(curve, lc, g, chi) >= Pm * (1 - 1e-4))
    ts = _g_threshold(lambda g: stopping_density_stock_learning(curve, lc, g, chi) > gc.Phat)
    return dict(chi=chi, g_takeoff=t0, g_star=ts)


def g_star_myopic(gc: GridCurve, lc: LifeCycleParams, lz: float = 0.0, P0: float | None = None,
                  stock: str = "exact"):
    """Overshooting threshold: the smallest diffusion rate at which myopic entry carries a
    newborn topic past the peak of v.  It is the larger of the take-off threshold and the
    rate at which the value at the peak beats the frontier (the stopping density rises in g).

    stock='asymptotic' returns the closed-form rate at the peak (g_star_peak), an upper
    bound on the exact one when the topic takes off.
    """
    if stock == "asymptotic":
        return g_star_peak(gc, lc, lz, P0, stock)
    P0 = lc.P0 if P0 is None else P0
    if np.isfinite(lc.b_c) and gc.Phat >= lc.b_c:
        return np.inf
    return _g_threshold(lambda g: stopping_density_myopic(gc, lc, g, lz, P0) > gc.Phat)


# ----------------------------------------------------------------- forward-looking entry
def _backlog(tau, S, Ps, Pm, b_c, delta, rho):
    """Backlog of a newcomer who joins at the stopping date (flow Ps, stock S),
    when the flow then thins at rate rho down to Pm (closed form)."""
    tau = np.asarray(tau, float)
    tau_m = np.log(Ps / Pm) / rho if Ps > Pm else 0.0

    def first(t):
        if abs(delta - rho) > 1e-12:
            inflow = Ps * (np.exp((delta - rho) * t) - 1) / (delta - rho)
        else:
            inflow = Ps * t
        return np.exp(-delta * t) * (S + inflow - b_c * (np.exp(delta * t) - 1) / delta)

    Bm = first(tau_m)
    Bstar = (Pm - b_c) / delta
    late = Bstar + (Bm - Bstar) * np.exp(-delta * (tau - tau_m))
    return np.where(tau <= tau_m, first(np.minimum(tau, tau_m)), late), tau_m, Bm, Bstar


def catchup_time_re(S, Ps, Pm, b_c, delta, rho):
    """Catch-up time of a newcomer who joins at the stopping date (see _backlog)."""
    if S <= 0:
        return 0.0
    _, tau_m, Bm, Bstar = _backlog(0.0, S, Ps, Pm, b_c, delta, rho)
    if tau_m > 0:
        grid = np.linspace(0.0, tau_m, 400)
        B, *_ = _backlog(grid, S, Ps, Pm, b_c, delta, rho)
        neg = np.where(B <= 0)[0]
        if len(neg):
            k = neg[0]
            if k == 0:
                return 0.0
            return brentq(lambda t: float(_backlog(t, S, Ps, Pm, b_c, delta, rho)[0]),
                          grid[k - 1], grid[k], xtol=1e-12)
    if Bm <= 0:
        return tau_m
    if Bstar >= 0:
        return np.inf
    return tau_m + np.log((Bm - Bstar) / (-Bstar)) / delta


def re_value_ln(gc: GridCurve, lc: LifeCycleParams, Ps, Ss, lz: float = 0.0, Pm=None,
                n: int = 300):
    """ln of a forward-looking newcomer's flow-equivalent value if entry stops at (Ps, Ss).

    After the stopping date the flow thins at rate rho down to the mature density;
    payoffs zeta v(P(tau)) start once her backlog is read and are discounted at
    rho + xi (career end, topic death):  (rho + xi) int_T^inf e^{-(rho+xi) tau} zeta v(P(tau)) dtau.
    """
    Pm = mature_density(gc, lc, lz) if Pm is None else Pm
    r = lc.rho + lc.xi
    if not np.isfinite(Pm):
        return -np.inf
    Ps = float(Ps)
    if Ps <= Pm:
        T = float(lc.cu.time(Ps, Ss))
        return lz + float(_lnv(gc, Ps)) - r * T
    if not np.isfinite(lc.b_c):
        T = 0.0
    else:
        T = catchup_time_re(float(Ss), Ps, Pm, lc.b_c, lc.delta, lc.rho)
    if not np.isfinite(T):
        return -np.inf
    tau_m = np.log(Ps / Pm) / lc.rho
    tail = np.exp(-r * max(T, tau_m) + lz + float(_lnv(gc, Pm)))
    if T >= tau_m:
        return float(np.log(tail))
    t = np.linspace(T, tau_m, n)
    y = r * np.exp(-r * t + lz + _lnv(gc, Ps * np.exp(-lc.rho * t)))
    return float(np.log(np.sum((y[1:] + y[:-1]) * np.diff(t)) / 2 + tail))


def stopping_density_re(gc: GridCurve, lc: LifeCycleParams, g, lz: float = 0.0,
                        P0: float | None = None, stock: str = "exact", n: int = 300):
    """Density at which forward-looking entry stops along the growth path."""
    P0 = lc.P0 if P0 is None else P0
    lnF = np.log(lc.F)
    Pm = mature_density(gc, lc, lz)
    if not np.isfinite(Pm):
        return np.nan
    s0 = 0.0 if stock == "asymptotic" else P0
    f = lambda P: re_value_ln(gc, lc, P, float(growth_stock(P, g, s0, lc.delta)), lz, Pm) - lnF
    start = max(P0, Pm * (1 + 1e-9))
    top = np.exp(gc.lp[-1]) * 0.99
    if np.isfinite(lc.b_c):
        top = min(top, 4 * lc.b_c)
    grid = np.exp(np.linspace(np.log(start), np.log(top), n))
    r = _first_root(f, grid)
    return float(top) if np.isnan(r) else r


def g_star_re(gc: GridCurve, lc: LifeCycleParams, lz: float = 0.0, P0: float | None = None):
    """Diffusion rate above which forward-looking entry overshoots the peak of v."""
    P0 = lc.P0 if P0 is None else P0
    Ph = gc.Phat
    lnF = np.log(lc.F)
    Pm = mature_density(gc, lc, lz)
    if not np.isfinite(Pm) or Pm >= Ph:
        return np.nan
    f = lambda g: re_value_ln(gc, lc, Ph, float(growth_stock(Ph, g, P0, lc.delta)), lz, Pm) - lnF
    lo, hi = 1e-6, 1e4
    if f(lo) > 0:
        return 0.0
    if f(hi) < 0:
        return np.inf
    return brentq(f, lo, hi, xtol=1e-10)


def g_star_re_bound(gc: GridCurve, lc: LifeCycleParams, lz: float = 0.0):
    """Sufficient condition for overshooting with forward-looking entry (Proposition 9(ii)):
    g > delta [Phat (b_c - P_m) / (P_m (b_c - Phat)) - 1]."""
    Pm = mature_density(gc, lc, lz)
    Ph = gc.Phat
    if not np.isfinite(lc.b_c):
        return lc.delta * (Ph / Pm - 1.0)
    if Ph >= lc.b_c:
        return np.inf
    return lc.delta * (Ph * (lc.b_c - Pm) / (Pm * (lc.b_c - Ph)) - 1.0)


# ----------------------------------------------------------------- cross-section
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
    Pcap = np.exp(gc.lp[-1])
    if np.isfinite(lc.b_c):
        Pcap = min(Pcap, 1.5 * lc.b_c)
    # growth phase: P = P0 e^{g t}, S = P (1 - e^{-(g+delta) t}) / (g + delta)
    gt = g[:, None] * ages[None, :]
    lnPg = np.minimum(np.log(lc.P0) + gt, np.log(Pcap))
    Pg = np.exp(lnPg)
    Sg = Pg * (1 - np.exp(-(g[:, None] + lc.delta) * ages[None, :])) / (g[:, None] + lc.delta)
    E = entry_gap(gc, lc, Pg, Sg, lz[:, None])
    E = np.where(np.isfinite(E), E, -1e3)
    neg = E <= 0
    k_c = np.where(neg.any(axis=1), neg.argmax(axis=1), nA - 1)
    k0 = np.maximum(k_c - 1, 0)
    E0, E1 = E[np.arange(nT), k0], E[np.arange(nT), k_c]
    frac = np.where(k_c > 0, E0 / np.where(E0 - E1 > 0, E0 - E1, 1.0), 0.0)
    t_c = ages[k0] + np.clip(frac, 0, 1) * lc.dt * (k_c > 0)
    Pc = lc.P0 * np.exp(g * t_c)
    # consolidation (fast-stock approximation): P = Pc e^{-rho (t - t_c)} down to P_m
    lnPm = np.log(np.where(np.isfinite(Pm_t), Pm_t, 1e-300))
    t_m = t_c + np.where(np.isfinite(Pm_t), np.maximum(np.log(Pc) - lnPm, 0.0) / lc.rho, np.inf)
    # A topic whose entry stops below its mature density never matures.  Entry can only
    # stop there in the valley where even a stationary topic is not worth joining
    # (zeta w < F on the rising side of w): along the growth path the stock lags, so a
    # newcomer's value exceeds zeta w(P).  After the stop the flow falls and the stock
    # falls more slowly, so entry never resumes: the topic thins as its members leave
    # and is abandoned.  A newborn topic whose first follower would not beat the entry
    # bar is the special case of a stop at birth.
    t_m = np.where((E[:, 0] <= 0) | (Pc < Pm_t * (1 - 1e-9)), np.inf, t_m)
    A = ages[None, :]
    lnP = np.where(A < t_c[:, None], lnPg,
                   np.where(A < t_m[:, None], np.log(Pc)[:, None] - lc.rho * (A - t_c[:, None]),
                            lnPm[:, None]))
    ph = np.where(A < t_c[:, None], 0, np.where(A < t_m[:, None], 1, 2)).astype(np.int8)
    Pfull = np.exp(lnP)
    y = lz[:, None] + _lnv(gc, Pfull)
    w = wt[:, None] * np.exp(-lc.xi * ages)[None, :] * lc.dt * Pfull / gc.x(Pfull)
    return CrossSection(lnP=lnP, y=y, w=w, phase=ph, age=np.broadcast_to(ages, lnP.shape),
                        g=g, lz=lz, Pc=Pc, Pm=Pm_t, Phat=gc.Phat,
                        meta=dict(type_weight=wt))


def founder_value(cs: CrossSection, lc: LifeCycleParams):
    """Founders' expected tenure-average payoff U_F = E[(rho+xi) int e^{-(rho+xi) t} zeta v(P(t)) dt].

    A founder stays in the topic she opens (no catch-up) until her career ends or the
    topic dies; she does not know its diffusion rate or fertility when she explores.
    """
    r = lc.rho + lc.xi
    disc = r * np.exp(-r * cs.age[0]) * lc.dt
    per_type = (np.exp(cs.y) * disc[None, :]).sum(axis=1)
    wt = cs.meta["type_weight"]
    return float((per_type * wt).sum() / wt.sum())


def founder_fixed_point(gc: GridCurve, lc: LifeCycleParams, e: float, omega: float,
                        F_lo: float | None = None, F_hi: float | None = None):
    """Frontier value with founders who stay: F = (1 - omega) e + omega U_F(F).

    U_F is computed on the life cycle that F itself generates (entry stops and
    mature densities depend on F).  Returns (F, U_F); omega = 0 gives F = e.
    """
    if omega <= 0:
        return float(e), founder_value(simulate(gc, lc.with_(F=e)), lc.with_(F=e))

    def gap(F):
        one = lc.with_(F=F)
        return (1 - omega) * e + omega * founder_value(simulate(gc, one), one) - F

    lo = F_lo if F_lo is not None else 0.7 * lc.F
    hi = F_hi if F_hi is not None else 1.3 * lc.F
    F = brentq(gap, lo, hi, xtol=1e-9)
    one = lc.with_(F=F)
    return float(F), founder_value(simulate(gc, one), one)


def life_cycle_accounts(gc: GridCurve, lc: LifeCycleParams, a: float, e: float, s_rate: float,
                        cs: CrossSection | None = None, paths: bool = False):
    """Output per researcher-year in the life-cycle economy, in units of production speed.

    Per topic born: explorer-years 1/s (births equal deaths, s E = xi M) producing e
    each; writer-years n_w = P / (a x*(P)) producing zeta v(P) each; and catch-up years
    n_c = G (e^{rho T} - 1) / rho, where G = dn_w/dt + rho n_w is the flow of new writers
    and T the catch-up time they face (G = 0 while a topic consolidates, since no one
    enters).  Topic-years are weighted by survival e^{-xi t}.  Returns Y / (a N) and the
    shares of researcher-years in exploration, catch-up and writing.
    """
    cs = simulate(gc, lc) if cs is None else cs
    P = np.exp(cs.lnP)
    nw = P / (a * gc.x(P))
    ages = cs.age[0]
    surv = np.exp(-lc.xi * ages)[None, :]
    wt = cs.meta["type_weight"][:, None] / cs.meta["type_weight"].sum()
    # stock faced by newcomers: growth-path stock while growing, stationary when mature
    Sg = growth_stock(P, cs.g[:, None], lc.P0, lc.delta)
    S = np.where(cs.phase == 0, Sg, P / lc.delta)
    T = lc.cu.time(P, S)
    T = np.where(np.isfinite(T), T, 0.0)
    dnw = np.gradient(nw, lc.dt, axis=1)
    G = np.where(cs.phase == 1, 0.0, np.maximum(dnw + lc.rho * nw, 0.0))
    nc = G * np.expm1(lc.rho * T) / lc.rho
    out = float((wt * surv * nw * np.exp(cs.y)).sum() * lc.dt)
    writers = float((wt * surv * nw).sum() * lc.dt)
    catch = float((wt * surv * nc).sum() * lc.dt)
    expl = 1.0 / s_rate
    total = writers + catch + expl
    res = dict(Y_per_aN=(out + e * expl) / total, writers=writers / total, catchup=catch / total,
               explorers=expl / total, out_topics=out, years_topics=writers + catch)
    if paths:
        res.update(nw=nw, nc=nc, T=T)
    return res


def _within(a, w):
    """Demean rows of a with weights w (topic fixed effects)."""
    m = (a * w).sum(axis=1, keepdims=True) / w.sum(axis=1, keepdims=True)
    return a - m


def test4_moments(cs: CrossSection, estimator: str = "pooled", noise_sd: float = 0.3, n_noise: int = 7):
    """Replicate the returns-by-density regressions on the simulated cross-section.

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


def composition_covariance(cs: CrossSection):
    """Researcher-weighted covariance of log density and topic age."""
    w = cs.w / cs.w.sum()
    m_l = float((w * cs.lnP).sum())
    m_a = float((w * cs.age).sum())
    return float((w * (cs.lnP - m_l) * (cs.age - m_a)).sum())


def phase_shares(cs: CrossSection):
    w = cs.w / cs.w.sum()
    return {name: float(w[cs.phase == k].sum()) for k, name in enumerate(["growth", "consolidation", "mature"])}
