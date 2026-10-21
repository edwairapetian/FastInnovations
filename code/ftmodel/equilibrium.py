"""Stationary topic equilibrium, the planner, and sufficient statistics.

All topics that are open are identical (the access-depletion frontier of the
paper).  A researcher who enters a topic of flow P earns, over her career and
relative to the unit of writing speed,

    w(P) = v(P) exp(-kappa P),       kappa = rho / (delta b_c),

where exp(-kappa P) is the share of a career left after reading the stock
S = P / delta at speed b_c (kappa = 0 is the frictionless benchmark).  The
frontier pays F(M) = f0 pi(M) = Fbar (M / Mbar)^(-eta): an explorer opens a
viable topic with probability pi(M), which falls with the number of topics
already open, and earns f0 per unit of writing speed when she succeeds.

Equilibrium:   w(P*) = F(M*),  w'(P*) < 0,  N = Phi(M*) + M* n(P*),
with members per topic n(P) = P exp(kappa P) / (a x*(P)) and explorers
Phi(M) = int_0^M dm / pi(m) (one successful attempt per topic).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .payoff import PayoffCurve


@dataclass(frozen=True)
class Economy:
    a: float = 4.0         # papers per period at full-time writing
    kappa: float = 0.0     # catch-up cost per unit of flow density
    Fbar: float = 0.95     # frontier payoff at M = Mbar
    Mbar: float = 1.0
    eta: float = 0.0       # frontier (access-depletion) elasticity
    N: float = 1000.0
    f0: float = 1.0        # value of a successful explorer's output (pi = F / f0)

    def with_(self, **kw) -> "Economy":
        return replace(self, **kw)

    def F(self, M):
        return self.Fbar * (np.asarray(M, float) / self.Mbar) ** (-self.eta)

    def Finv(self, F):
        if self.eta == 0:
            raise ValueError("F is constant when eta = 0")
        return self.Mbar * (F / self.Fbar) ** (-1.0 / self.eta)

    def pi(self, M):
        return self.F(M) / self.f0

    def Phi(self, M):
        """Researchers needed to open M topics: int_0^M dm / pi(m)."""
        return self.f0 * M * (M / self.Mbar) ** self.eta / (self.Fbar * (1.0 + self.eta))


class Stationary:
    """Equilibrium and planner allocations for a given payoff curve."""

    def __init__(self, curve: PayoffCurve, eco: Economy):
        self.c = curve
        self.e = eco

    # ----- entrant payoff curve -----------------------------------------
    def lnw(self, P):
        P = np.atleast_1d(np.asarray(P, float))
        return self.c.lnv(P) - self.e.kappa * P

    def eps_w(self, P):
        P = np.atleast_1d(np.asarray(P, float))
        return self.c.eps_v(P) - self.e.kappa * P

    def members(self, P):
        P = np.atleast_1d(np.asarray(P, float))
        return P * np.exp(self.e.kappa * P) / (self.e.a * self.c.x_star(P))

    def D(self, P):
        """d ln n / d ln P = 1 - eps_x + kappa P."""
        return 1.0 - self.c.eps_x(P) + self.e.kappa * np.atleast_1d(P)

    def peak_w(self):
        return self.c.peak_w(self.e.kappa)

    # ----- decentralised equilibrium ----------------------------------------
    def equilibrium(self, N: float | None = None, explorers: bool = True):
        """Solve w(P) = F(M), N = Phi(M) + M n(P) on the falling branch of w."""
        N = self.e.N if N is None else N
        Pw, ww = self.peak_w()

        def M_of(P):
            n = float(self.members(P)[0])
            if not explorers:
                return N / n
            return brentq(lambda M: self.e.Phi(M) + M * n - N, 1e-12, N / n * (1 + 1e-12) + 1e-9)

        def gap(P):
            return float(self.lnw(P)[0] - np.log(self.e.F(M_of(P))))

        if gap(Pw) <= 0:
            raise ValueError("no topic is worth joining: peak of w below the frontier")
        hi = 2 * Pw
        while gap(hi) > 0:
            hi *= 2
        P = brentq(gap, Pw, hi, xtol=1e-12)
        M = M_of(P)
        return dict(P=P, M=M, n=float(self.members(P)[0]), F=float(self.e.F(M)),
                    w=float(np.exp(self.lnw(P)[0])), v=float(self.c.v(P)[0]),
                    x=float(self.c.x_star(P)[0]))

    # ----- sufficient statistics (large-topic limit) --------------------------
    def volume_elasticity(self, P: float, eta: float | None = None):
        """E = d ln P* / d ln N = eta / (eta D + |eps_w|) (Proposition 3)."""
        eta = self.e.eta if eta is None else eta
        ew = float(self.eps_w(P)[0])
        return eta / (eta * float(self.D(P)[0]) + abs(ew))

    def eta_from_E(self, P: float, E: float):
        ew = abs(float(self.eps_w(P)[0]))
        D = float(self.D(P)[0])
        return E * ew / (1.0 - E * D)

    def stepping_on_toes(self, P: float, E: float):
        """lambda = 1 - E |eps_w| (Proposition 9)."""
        return 1.0 - E * abs(float(self.eps_w(P)[0]))

    # ----- planner ------------------------------------------------------------
    def output(self, P: float, N: float | None = None, explorers: bool = True):
        """Output when every topic has flow P and researchers keep x*(P)."""
        N = self.e.N if N is None else N
        n = float(self.members(P)[0])
        if explorers:
            M = brentq(lambda M: self.e.Phi(M) + M * n - N, 1e-12, N / n + 1e-9)
        else:
            M = N / n
        w = float(np.exp(self.lnw(P)[0]))
        Y = self.e.a * (self.e.f0 * M * explorers + M * n * w)
        return Y, M

    def planner(self, N: float | None = None, explorers: bool = True):
        """Constrained planner: chooses topic density, researchers choose x."""
        N = self.e.N if N is None else N
        eq = self.equilibrium(N, explorers)
        Pw, _ = self.peak_w()
        res = minimize_scalar(lambda q: -self.output(np.exp(q), N, explorers)[0],
                              bounds=(np.log(Pw * 0.5), np.log(eq["P"] * 1.2)),
                              method="bounded", options={"xatol": 1e-10})
        P = float(np.exp(res.x))
        Y, M = self.output(P, N, explorers)
        n = float(self.members(P)[0])
        w = float(np.exp(self.lnw(P)[0]))
        F = float(self.e.F(M))
        subsidy = w - F
        return dict(P=P, M=M, n=n, Y=Y, w=w, F=F, subsidy=subsidy,
                    Y_eq=self.output(eq["P"], N, explorers)[0], eq=eq)

    def planner_condition(self, P: float, M: float):
        """Residual of (w - F)/w = -eps_w (1 + n pi) / D (general planner FOC)."""
        n = float(self.members(P)[0])
        w = float(np.exp(self.lnw(P)[0]))
        F = float(self.e.F(M))
        return (w - F) / w + float(self.eps_w(P)[0]) * (1 + n * F / self.e.f0) / float(self.D(P)[0])

    def full_planner(self, N: float | None = None):
        """Planner who also sets the writing share x (both wedges)."""
        N = self.e.N if N is None else N
        p = self.c.p
        a, kap = self.e.a, self.e.kappa

        def out(theta):
            lnP, logit_x = theta
            P = np.exp(lnP)
            x = 1 / (1 + np.exp(-logit_x))
            n = P * np.exp(kap * P) / (a * x)
            M = brentq(lambda M: self.e.Phi(M) + M * n - N, 1e-12, N / n + 1e-9)
            lng = float(self.c.lng(1 - x, P))
            return -(a * (self.e.f0 * M + M * n * np.exp(lng - kap * P)))

        from scipy.optimize import minimize
        eq = self.equilibrium(N)
        x0 = eq["x"]
        th0 = np.array([np.log(self.peak_w()[0]), np.log(x0 / (1 - x0))])
        res = minimize(out, th0, method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-12, maxiter=4000))
        P = float(np.exp(res.x[0])); x = float(1 / (1 + np.exp(-res.x[1])))
        return dict(P=P, x=x, Y=-res.fun, x_private=float(self.c.x_star(P)[0]))
