"""Quantitative results for the paper: calibration, welfare, AI counterfactuals.

Everything is computed from a calibration dictionary (results/calibration.json)
with the keys produced by scripts/run_calibration.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .credit import CreditTargets, aware_share, credit_block, flagged_rate_index
from .equilibrium import Economy, Stationary
from .lifecycle import (LifeCycleParams, composition_bias, g_star, g_star_re,
                        g_star_re_bound, mature_density, phase_shares, simulate,
                        stopping_density_asymptotic, stopping_density_re, test4_moments)
from .payoff import GridCurve, PayoffCurve, ReaderParams


@dataclass
class Model:
    """A calibrated parameterisation and the objects built from it."""

    cal: dict

    def __post_init__(self):
        c = self.cal
        self.reader = ReaderParams(**c["reader"])
        self.curve = PayoffCurve(self.reader)
        self.grid = GridCurve(self.curve, n=3000)
        self.lc = LifeCycleParams(**c["lifecycle"])
        self.credit = c["credit"]

    # ----- stationary economy around the mature density ---------------------
    @property
    def kappa(self):
        return self.lc.kappa

    def economy(self, eta=None, **kw):
        eta = self.cal["eta"] if eta is None else eta
        base = dict(a=self.cal["a"], kappa=self.kappa, Fbar=self.lc.F, Mbar=1.0, eta=eta, N=1.0)
        base.update(kw)
        return Economy(**base)

    def baseline_N(self):
        """Choose N so that M = Mbar = 1 at the calibrated frontier payoff."""
        s = Stationary(self.curve, self.economy())
        Pm = self.mature()
        return float(s.members(Pm)[0])

    def mature(self, curve=None, kappa=None, F=None):
        curve = curve or self.curve
        kappa = self.kappa if kappa is None else kappa
        F = self.lc.F if F is None else F
        return curve.crossing(F, "falling", kappa=kappa)


def summary(m: Model) -> dict:
    """Headline numbers of the calibrated model."""
    cv, lc = m.curve, m.lc
    Ph, vh = cv.peak()
    Pw, ww = cv.peak_w(lc.kappa)
    Pm = m.mature()
    F = lc.F
    P_fric = cv.crossing(F, "falling", kappa=0.0)
    s = Stationary(cv, m.economy())
    eps_w = float(s.eps_w(Pm)[0])
    eps_v = float(cv.eps_v(Pm)[0])
    D = float(s.D(Pm)[0])
    E = m.cal["E_target"]
    eta = m.cal["eta"]
    lam = 1 - E * abs(eps_w)
    T = lc.kappa_s * (Pm / lc.delta) / lc.rho  # catch-up time S / b_c
    cs = simulate(m.grid, lc)
    ph = phase_shares(cs)
    mom_p = test4_moments(cs, "pooled")
    mom_w = test4_moments(cs, "within")
    gs = g_star(m.grid, lc)
    phi_m = float(cv.phi(cv.coverage(Pm))[0])
    q = m.credit["q"]
    return dict(
        Phat=Ph, vhat=vh, Pw=Pw, what=ww, Pm=Pm, v_Pm=float(cv.v(Pm)[0]), F=F,
        P_frictionless=P_fric, premium=lc.kappa * Pm, T=T,
        rent=np.log(vh / F), dissipated_frictionless=1 - F / vh,
        planner_gain=ww / F - 1, subsidy=(ww - F) / F,
        eps_w=eps_w, eps_v=eps_v, D=D, E=E, eta=eta, lam=lam,
        g_star=gs, x_m=float(cv.x_star(Pm)[0]), r_m=float(cv.r_star(Pm)[0]),
        coverage_m=float(cv.coverage(Pm)[0]), phi_m=phi_m,
        aware_m=float(aware_share(phi_m, q)), erosion_m=float(cv.erosion_loss(Pm)[0]),
        phases=ph, moments_pooled=mom_p, moments_within=mom_w,
        beyond_true=mom_p["beyond"],
        g_star_re=g_star_re(m.grid, lc), g_star_re_bound=g_star_re_bound(m.grid, lc),
        comp_bias_T7=composition_bias(cs, 7.0),
    )


def social_block(m: Model, taus=(0.0, 0.25, 0.5, 0.75, 1.0)) -> dict:
    """Social value versus private credit (Proposition 6).

    Uniform wedge: s = v exp(-theta (lam_s - lam) P).  The equilibrium is
    overcrowded relative to the social optimum iff the share of overlap losses
    that are transfers, tau = (lam - lam_s) / lam, is below
    |eps_w(P_m)| / (theta lam P_m).
    Proportional transfer: ln C_s = (1 - tau) ln C; the social optimum is the
    peak of ln s - kappa P, found numerically.
    """
    from scipy.optimize import minimize_scalar
    cv, k = m.curve, m.kappa
    Pm = m.mature()
    eps_w = float(cv.eps_v(Pm)[0]) - k * Pm
    ThP = cv.p.Theta * Pm
    out = dict(Pm=Pm, eps_w=eps_w, theta_lam_Pm=ThP,
               erosion_m=float(cv.erosion_loss(Pm)[0]),
               tau_crit_uniform=abs(eps_w) / ThP, tau_max_uniform=1 - cv.p.dl)
    rows = []
    for t in taus:
        f = lambda q: -(cv.ln_social(np.exp(q), tau=t)[0] - k * np.exp(q))
        r = minimize_scalar(f, bounds=(np.log(1.0), np.log(400.0)), method="bounded",
                            options={"xatol": 1e-9})
        Po = float(np.exp(r.x))
        gain = float(np.exp(-r.fun - (cv.ln_social(Pm, tau=t)[0] - k * Pm)) - 1)
        rows.append(dict(tau=t, P_social=Po, overcrowded=bool(Po < Pm), gain=gain))
    out["proportional"] = rows
    g = lambda t: (lambda q: -(cv.ln_social(np.exp(q), tau=t)[0] - k * np.exp(q)))
    def Po(t):
        r = minimize_scalar(g(t), bounds=(np.log(1.0), np.log(400.0)), method="bounded",
                            options={"xatol": 1e-9})
        return float(np.exp(r.x))
    from scipy.optimize import brentq
    try:
        out["tau_crit_proportional"] = brentq(lambda t: Po(t) - Pm, 0.0, 5.0, xtol=1e-6)
    except ValueError:
        out["tau_crit_proportional"] = float("nan")
    return out


def re_overshooting(m: Model, gs=(0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0)) -> dict:
    """Forward-looking entry (Proposition 7): thresholds and stopping densities."""
    gc, lc = m.grid, m.lc
    rows = [dict(g=g, myopic=stopping_density_asymptotic(gc, lc, g),
                 rational=stopping_density_re(gc, lc, g)) for g in gs]
    return dict(g_star=g_star(gc, lc), g_star_re=g_star_re(gc, lc),
                g_star_re_bound=g_star_re_bound(gc, lc),
                P_frictionless=m.curve.crossing(lc.F, "falling"), rows=rows)


def ai_counterfactuals(m: Model, size: float = 0.3) -> list[dict]:
    """Long-run effects of the AI channels of Proposition 8 (stationary economy)."""
    N0 = m.baseline_N()
    q0 = m.credit["q"]

    def solve(curve, kappa, a, N, q):
        eco = m.economy(kappa=kappa, a=a, N=N)
        s = Stationary(curve, eco)
        eq = s.equilibrium(N=N, explorers=False)
        P = eq["P"]
        phi = float(curve.phi(curve.coverage(P))[0])
        Pc = curve  # noqa
        return dict(P=P, M=eq["M"], papers=a * eq["x"], flagged=float(flagged_rate_index(P, phi, q)),
                    aware=float(aware_share(phi, q)), ypr=a * eq["F"], erosion=float(curve.erosion_loss(P)[0]))

    a0 = m.cal["a"]
    base = solve(m.curve, m.kappa, a0, N0, q0)

    def gstar_for(curve, kappa):
        gc = GridCurve(curve, n=2000)
        lc = m.lc.with_(kappa_s=kappa * m.lc.delta)
        return g_star(gc, lc)

    g0 = gstar_for(m.curve, m.kappa)
    rows = []

    def add(name, channel, curve=None, kappa=None, a=None, N=None, q=None):
        curve = curve or m.curve
        kappa = m.kappa if kappa is None else kappa
        a = a0 if a is None else a
        N = N0 if N is None else N
        q = q0 if q is None else q
        r = solve(curve, kappa, a, N, q)
        gs = gstar_for(curve, kappa)
        rows.append(dict(
            name=name, channel=channel,
            dP=np.log(r["P"] / base["P"]), dM=np.log(r["M"] / base["M"]),
            dpapers=np.log(r["papers"] / base["papers"]),
            dflag=np.log(r["flagged"] / base["flagged"]),
            daware=r["aware"] - base["aware"],
            dypr=np.log(r["ypr"] / base["ypr"]),
            dgstar=gs - g0))

    k = 1 + size
    p = m.reader
    add("Faster drafting", r"$a\times%.1f$" % k, a=a0 * k)
    add("Cheaper entry", r"$N\times%.1f$" % k, N=N0 * k)
    add("Faster reading", r"$b\times%.1f$" % k, curve=PayoffCurve(p.with_(b=p.b * k)))
    add("Better learning", r"$\beta\times%.1f$" % k, curve=PayoffCurve(p.with_(beta=p.beta * k)))
    add("Better search", r"$\sigma\times%.1f$" % k, curve=PayoffCurve(p.with_(sigma=p.sigma * k)))
    add("Faster catch-up", r"$b_c\times 2$", kappa=m.kappa / 2)
    ct = CreditTargets(pivot_cost=m.cal["credit_targets"]["pivot_cost"] / 2)
    cb = credit_block(CreditTargets(**{**m.cal["credit_targets"], "pivot_cost": ct.pivot_cost}))
    add("Cheaper pivots", r"$\kappa_p\div 2$", curve=PayoffCurve(p.with_(dl=cb["dl"])), q=cb["q"])
    return rows


def write_json(obj, path):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=conv)
