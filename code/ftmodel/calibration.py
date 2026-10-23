"""Calibration of the reading technology and topic life cycles to the arXiv facts.

Targets (docs/empirical_facts.md, Fact 3 = Test 4 with author fixed effects):

    lin      +6.7% citations per author-year per SD of topic density
    Q1,Q4,Q5 returns by fifth of density relative to the middle fifth: -11%, +2%,
             -2% (the arXiv fifths re-expressed relative to the middle fifth; the
             earlier -8%, +5%, +1% were deviations from each author's mean)
    beyond   12% of researcher-years beyond the peak of the payoff curve
    premium  incumbents' log premium over the frontier, kappa P_m = rho T
             (soft target 0.15: one and a half years of catch-up per ten-year tenure)
    aware    aware share of flagged overlap at the mature density, 0.60: the raw
             PDF-matched share (35-42%, Fact 5) divided by the 63% citation recall

The quadratic summary of Test 4 (0.094 z - 0.047 z^2) is not targeted: it
imposes a symmetric hump, while both the quintiles and the model's payoff
curve are asymmetric (a gentle concave rise and a steep fall).  We report it
as an untargeted moment.

Externally fixed (see Fixed): b = 100 papers read per year at full time (the
unit of P), the credit block (dl, q) from Fact 4 via ftmodel.credit, delta = 0.25 (relevance half-life ~2.8y), rho = 0.10
(ten-year topic tenure), xi = 0.04 (25-year topic life), P0 = 2 papers per
year at birth.

Estimated: beta, gamma, Theta (payoff curve), F (frontier), g_med, g_disp
(diffusion rates of new topics), zeta_sd (fertility dispersion), b_c
(catch-up reading speed), sigma (search precision).  The analogue of the author-fixed-effects design
is the pooled cross-section of researcher-years (researchers who move across
topics); the within-topic estimator is reported as a robustness check.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import minimize

from .credit import CreditTargets, credit_block
from .lifecycle import LifeCycleParams, g_star, mature_density, simulate, test4_moments
from .payoff import GridCurve, PayoffCurve, ReaderParams

TARGETS = dict(lin=0.067, Q1=-0.11, Q4=0.02, Q5=-0.02, beyond=0.12, premium=0.15, aware=0.60)
SCALES = dict(lin=0.008, Q1=0.015, Q4=0.012, Q5=0.012, beyond=0.03, premium=0.06, aware=0.03)

NAMES = ("beta", "gamma", "Theta", "F", "g_med", "g_disp", "zeta_sd", "b_c", "sigma")
LOWER = dict(beta=0.1, gamma=0.05, Theta=1e-4, F=0.8, g_med=0.05, g_disp=0.2, zeta_sd=0.02, b_c=20.0,
             sigma=0.3)
# g_med <= 0.5: the median new topic takes at least about 3.5 years to grow from
# P0 to mature density.  Searches with g_med up to 1 end in corners where almost
# every topic overshoots (and beta, b_c and zeta_sd sit at their lower bounds)
# without a lower loss than the interior solution.
UPPER = dict(beta=50.0, gamma=1.0, Theta=0.02, F=2.5, g_med=0.5, g_disp=1.5, zeta_sd=0.4, b_c=300.0,
             sigma=10.0)


@dataclass
class Fixed:
    b: float = 100.0
    delta: float = 0.25
    rho: float = 0.10
    xi: float = 0.04
    P0: float = 2.0
    credit: CreditTargets = field(default_factory=CreditTargets)
    estimator: str = "pooled"

    @property
    def dl(self) -> float:
        return credit_block(self.credit)["dl"]


def bounds_log():
    return np.log([LOWER[k] for k in NAMES]), np.log([UPPER[k] for k in NAMES])


def build(theta: dict, fx: Fixed, fine: bool = False):
    rp = ReaderParams(b=fx.b, beta=theta["beta"], gamma=theta["gamma"], Theta=theta["Theta"],
                      dl=fx.dl, sigma=theta["sigma"])
    gc = GridCurve(PayoffCurve(rp), n=3000 if fine else 1200)
    grid = dict(n_g=41, n_z=51, dt=0.05) if fine else dict(n_g=15, n_z=15, dt=0.1, horizon=100.0)
    lc = LifeCycleParams(F=theta["F"], kappa_s=fx.rho / theta["b_c"], delta=fx.delta, rho=fx.rho,
                         xi=fx.xi, P0=fx.P0, g_med=theta["g_med"], g_disp=theta["g_disp"],
                         zeta_sd=theta["zeta_sd"], **grid)
    return rp, gc, lc


def evaluate(theta: dict, fx: Fixed, fine: bool = False, estimator: str | None = None):
    """Model moments, or None if the parameterisation violates the model's assumptions."""
    if fx.b * theta["gamma"] * theta["beta"] <= 1.2:  # Assumption 4(a), with margin
        return None
    rp, gc, lc = build(theta, fx, fine)
    if not (5 < gc.Phat < 500):
        return None
    Pm = mature_density(gc, lc)
    if not np.isfinite(Pm) or gc.lnv(fx.P0) < np.log(theta["F"]) - 1e-3:  # newborn topics must attract followers
        return None
    cs = simulate(gc, lc)
    m = test4_moments(cs, estimator=estimator or fx.estimator)
    cv = gc.curve
    phi = float(cv.phi(cv.coverage(Pm))[0])
    q = credit_block(fx.credit)["q"]
    out = dict(lin=m["lin"], Q1=m["quint"][0], Q4=m["quint"][3], Q5=m["quint"][4],
               beyond=m["beyond"], premium=lc.kappa * Pm, aware=q * phi / (1 - (1 - q) * phi))
    info = dict(rp=rp, gc=gc, lc=lc, cs=cs, raw=m, Pm=Pm, g_star=g_star(gc, lc))
    return out, info


def loss(out: dict) -> float:
    return float(sum(((out[k] - TARGETS[k]) / SCALES[k]) ** 2 for k in TARGETS))


def _theta(x):
    return {k: float(v) for k, v in zip(NAMES, np.exp(x))}


def objective(x, fx: Fixed):
    lo, hi = bounds_log()
    x = np.clip(x, lo, hi)
    try:
        r = evaluate(_theta(x), fx)
    except Exception:
        return 1e3
    return 1e3 if r is None else loss(r[0])


def calibrate(fx: Fixed | None = None, n_scan: int = 2048, n_polish: int = 3, seed: int = 3,
              maxiter: int = 700, seeds: list | None = None, verbose: bool = True):
    """Sobol scan of the bounded parameter space, Nelder-Mead from the best scan
    points and from any supplied seed parameterisations, and selection of the
    winner by the loss on the fine grid (the search itself uses a coarse grid)."""
    from scipy.stats import qmc
    fx = fx or Fixed()
    lo, hi = bounds_log()
    U = qmc.Sobol(d=len(NAMES), seed=seed).random(n_scan)
    scored = []
    for u in U:
        x = lo + u * (hi - lo)
        L = objective(x, fx)
        if L < 1e3:
            scored.append((L, x))
    scored.sort(key=lambda t: t[0])
    starts = [x for _, x in scored[:n_polish]]
    for th in seeds or []:
        starts.append(np.clip(np.log([th[k] for k in NAMES]), lo, hi))
    best, best_fine = None, np.inf
    for x0 in starts:
        res = minimize(objective, x0, args=(fx,), method="Nelder-Mead",
                       options=dict(maxiter=maxiter, xatol=1e-4, fatol=1e-4, adaptive=True))
        th = _theta(np.clip(res.x, lo, hi))
        r = evaluate(th, fx, fine=True)
        L_fine = loss(r[0]) if r is not None else np.inf
        if verbose:
            print(f"start loss {objective(x0, fx):.2f} -> polished {res.fun:.3f} (fine {L_fine:.3f})", flush=True)
        if L_fine < best_fine:
            best, best_fine = th, L_fine
    return best, float(best_fine)


def to_calibration(theta: dict, fx: Fixed, E_target: float = 0.15, a: float = 4.0) -> dict:
    """Assemble the calibration dictionary used by ftmodel.report."""
    from .equilibrium import Economy, Stationary
    rp, gc, lc = build(theta, fx, fine=True)
    cb = credit_block(fx.credit)
    Pm = mature_density(gc, lc)
    s = Stationary(gc.curve, Economy(a=a, kappa=lc.kappa, Fbar=lc.F))
    eta = s.eta_from_E(Pm, E_target)
    return dict(
        theta={k: float(v) for k, v in theta.items()},
        reader=asdict(rp),
        lifecycle=asdict(lc),
        credit=cb,
        credit_targets=asdict(fx.credit),
        fixed=dict(b=fx.b, delta=fx.delta, rho=fx.rho, xi=fx.xi, P0=fx.P0, estimator=fx.estimator),
        a=a, E_target=E_target, eta=float(eta),
        targets=TARGETS, scales=SCALES,
    )


def save(cal: dict, path: str):
    with open(path, "w") as f:
        json.dump(cal, f, indent=2)


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
