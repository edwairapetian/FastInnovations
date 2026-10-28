"""Stationary topic equilibrium with a frontier of founders, the planner, and
sufficient statistics.

A newcomer to a stationary topic of flow P earns, per unit of production speed
and as a flow-equivalent career value,

    w(P) = v(P) Gamma(P),        Gamma(P) = (1 - P / b_c)^nu,   nu = (rho + xi) / delta,

(ftmodel.catchup).  The frontier is a flow.  Explorers produce e per period and
found a viable topic at rate s; a founder stays in her topic as an incumbent (no
catch-up).  Topics die at rate xi, so births equal deaths, s E = xi M.  The value
of a free researcher is

    F(M, P) = (M / Mbar)^(-eta_F) [(1 - omega) e + omega U_F(P)],   omega = s / (rho + xi + s),

where U_F is the founder's tenure-average payoff and eta_F >= 0 is the frontier's
depletion as more directions are taken.  In this stationary module founders'
topics mature at once (U_F = v(P)); ftmodel.lifecycle computes U_F on the life
cycle.  Equilibrium: w(P*) = F, w'(P*) < 0.  Output per researcher is exactly a F
(renewal reward: every option a free researcher takes has value F), which is the
frontier-incidence identity.  The comparative statics use the large-topic
resource constraint N = M n(P).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .catchup import CatchUp
from .payoff import PayoffCurve


@dataclass(frozen=True)
class Economy:
    a: float = 4.0          # production speed: research output per unit of production time
    cu: CatchUp = field(default_factory=CatchUp)
    e: float = 0.9          # explorers' flow output per unit of production speed
    s: float = 0.14         # explorers' success rate (viable topics founded per explorer-year)
    eta_F: float = 0.0      # frontier depletion: F scales with (M / Mbar)^(-eta_F)
    Mbar: float = 1.0
    founders: bool = True   # founders stay in the topics they open
    N: float = 1000.0

    def with_(self, **kw) -> "Economy":
        return replace(self, **kw)

    @property
    def omega(self) -> float:
        """Probability that an explorer founds a topic before her spell ends (0 without founders)."""
        if not self.founders:
            return 0.0
        return self.s / (self.cu.rho + self.cu.xi + self.s)

    def depletion(self, M):
        return (np.asarray(M, float) / self.Mbar) ** (-self.eta_F)


class Stationary:
    """Equilibrium, planner and sufficient statistics for a given payoff curve."""

    def __init__(self, curve: PayoffCurve, eco: Economy):
        self.c = curve
        self.e = eco

    # ----- newcomer payoff curve --------------------------------------------
    def lnw(self, P):
        return self.c.lnw(P, self.e.cu)

    def eps_w(self, P):
        P = np.atleast_1d(np.asarray(P, float))
        return self.c.eps_v(P) - self.e.cu.eps_gamma(P)

    def members(self, P):
        """Members per topic, n(P) = P / (a x*(P) writing share(P))."""
        P = np.atleast_1d(np.asarray(P, float))
        return P * np.exp(-self.e.cu.ln_writing_share(P)) / (self.e.a * self.c.x_star(P))

    def D(self, P):
        """d ln n / d ln P = 1 - eps_x + nu_m (P/b_c) / (1 - P/b_c)."""
        P = np.atleast_1d(np.asarray(P, float))
        return 1.0 - self.c.eps_x(P) + self.e.cu.eps_members(P)

    def peak_w(self):
        return self.c.peak_w(self.e.cu)

    # ----- frontier -------------------------------------------------------------
    def F(self, M, P):
        """Frontier value with founders whose topics mature at once (U_F = v(P))."""
        om = self.e.omega
        UF = float(self.c.v(P)[0])
        return float(self.e.depletion(M)) * ((1 - om) * self.e.e + om * UF)

    def theta_F(self, M, P):
        """Founder share of the frontier value, omega U_F / [(1 - omega) e + omega U_F]."""
        om = self.e.omega
        UF = float(self.c.v(P)[0])
        return om * UF / ((1 - om) * self.e.e + om * UF)

    def eta_F(self, M=None, P=None):
        """Frontier elasticity -d ln F / d ln M at fixed P."""
        return self.e.eta_F

    # ----- decentralised equilibrium (large-topic limit) --------------------------
    def equilibrium(self, N: float | None = None):
        """Solve w(P) = F(M, P), M = N / n(P), on the falling branch of w."""
        N = self.e.N if N is None else N
        Pw, _ = self.peak_w()
        M_of = lambda P: N / float(self.members(P)[0])

        def gap(P):
            return float(self.lnw(P)[0] - np.log(self.F(M_of(P), P)))

        if gap(Pw) <= 0:
            raise ValueError("no topic is worth joining: peak of w below the frontier")
        top = self.e.cu.b_c * (1 - 1e-12) if not self.e.cu.frictionless else 4 * Pw
        if self.e.cu.frictionless:
            while gap(top) > 0:
                top *= 2
        P = brentq(gap, Pw, top, xtol=1e-12)
        M = M_of(P)
        F = self.F(M, P)
        return dict(P=P, M=M, n=float(self.members(P)[0]), F=F,
                    w=float(np.exp(self.lnw(P)[0])), v=float(self.c.v(P)[0]),
                    x=float(self.c.x_star(P)[0]), Y_per_N=self.e.a * F,
                    theta_F=self.theta_F(M, P), eta_F=self.eta_F(M, P))

    # ----- sufficient statistics ----------------------------------------------------
    def volume_elasticity(self, P, eta_F, theta_F):
        """E = d ln P* / d ln N = eta_F / (eta_F D + |eps_w| + theta_F eps_v) (Proposition 3)."""
        ew = abs(float(self.eps_w(P)[0]))
        ev = float(self.c.eps_v(P)[0])
        return eta_F / (eta_F * float(self.D(P)[0]) + ew + theta_F * ev)

    def eta_from_E(self, P, E, theta_F):
        ew = abs(float(self.eps_w(P)[0]))
        ev = float(self.c.eps_v(P)[0])
        return E * (ew + theta_F * ev) / (1.0 - E * float(self.D(P)[0]))

    def lambda_eq(self, P, E):
        """Equilibrium stepping-on-toes elasticity d ln Y / d ln N = 1 - E |eps_w| (Proposition 10)."""
        return 1.0 - E * abs(float(self.eps_w(P)[0]))

    def lambda_fixed_topics(self, P):
        """Elasticity of output in researchers holding open topics fixed: 1 + eps_w / D."""
        return 1.0 + float(self.eps_w(P)[0]) / float(self.D(P)[0])

    def incidence(self, P, f, h, theta_F, eta_F=0.0, eps_U=None):
        """Frontier incidence: long-run d ln(Y/N) - d ln a and d ln P* for a tool that shifts
        ln F by f and ln w by h at fixed density.  f includes the frontier's response to
        the change in the number of topics that the given researchers fill at that
        density, -eta_F d ln M|_P = eta_F d ln n|_P (for faster production, -eta_F d ln a).

            d ln(Y/N) = lam_F f + (1 - lam_F) h,  lam_F = |eps_w| / (|eps_w| + theta_F eps_U + eta_F D),
            d ln P*   = (h - f) / (|eps_w| + theta_F eps_U + eta_F D).
        """
        ew = abs(float(self.eps_w(P)[0]))
        eU = float(self.c.eps_v(P)[0]) if eps_U is None else eps_U
        den = ew + theta_F * eU + eta_F * float(self.D(P)[0])
        lam = ew / den
        return dict(dlnY=lam * f + (1 - lam) * h, dlnP=(h - f) / den, lam_F=lam)

    # ----- planner (large-topic limit) ------------------------------------------------
    def planner(self, N: float | None = None):
        """In the large-topic limit output per researcher under any density P is a w(P);
        the planner puts topics at the peak of w."""
        eq = self.equilibrium(N)
        Pw, ww = self.peak_w()
        return dict(P=Pw, w=ww, gain=ww / eq["F"] - 1.0, subsidy=ww - eq["F"],
                    tax=ww - self.F(eq["M"], Pw), eq=eq)

    def planner_finite(self, N: float | None = None):
        """Planner with explorers: births equal deaths (s E = xi M), N = E + M n(P).
        Output a [e E + M n w(P)] (founders' extra rents ignored; eta_F = 0)."""
        N = self.e.N if N is None else N
        xi, s, a = self.e.cu.xi, self.e.s, self.e.a

        def out(q):
            P = np.exp(q)
            n = float(self.members(P)[0])
            M = N / (n + xi / s)
            E = xi * M / s
            return -a * (self.e.e * E + M * n * float(np.exp(self.lnw(P)[0])))

        Pw, _ = self.peak_w()
        hi = 0.999 * self.e.cu.b_c if not self.e.cu.frictionless else 20 * Pw
        res = minimize_scalar(out, bounds=(np.log(0.2 * Pw), np.log(hi)), method="bounded",
                              options={"xatol": 1e-10})
        return dict(P=float(np.exp(res.x)), Y=-float(res.fun))
