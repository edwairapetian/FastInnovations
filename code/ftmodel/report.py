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
from .lifecycle import (LifeCycleParams, g_star, mature_density, phase_shares,
                        simulate, stopping_density_asymptotic, test4_moments)
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
    )


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
