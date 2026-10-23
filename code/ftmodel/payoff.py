"""The researcher's problem and the payoff curve v(P).

A researcher in a topic whose literature flow is P papers per period splits one
unit of time between writing (share x) and reading (share r = 1 - x).  She reads
R = b r papers.  Reading teaches (quality Q(R, P)) and protects (each covering
paper is detected with probability phi(R / P)).  Expected credit per paper is

    C(R, P) = exp(-theta P [lam - Delta phi(R / P)]),

the Poisson generating function of multiplicative erosion: an undetected
covering paper removes a share lam of the credit, a detected one a share
ell = lam - Delta.  Only Theta = theta * lam and the ratio dl = Delta / lam
enter the payoff curve.

The researcher's per-period payoff per unit of writing speed is

    g(r, P) = (1 - r) Q(b r, P) C(b r, P),     v(P) = max_r g(r, P).

Functional forms (the quantitative section of the paper):

    Q(R, P) = (1 + beta h(R, P))**gamma,  h = R P / (R + P)   (harmonic aggregator)
    phi(c)  = 1 - exp(-sigma c)                                 (search precision sigma)

Theorem 1 of the paper shows that ln g is jointly concave in (r, P) under these
forms, so ln v is concave and v is single peaked.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import brentq, minimize_scalar


@dataclass(frozen=True)
class ReaderParams:
    """Primitives of the researcher's problem.

    b      papers read per period at full-time reading (sets the unit of P)
    beta   learning efficiency of what is read
    gamma  curvature of the return to learning, in (0, 1]
    Theta  erosion intensity per earlier paper, theta * lam
    dl     share of erosion averted when a covering paper is detected, Delta / lam
    sigma  search precision; sigma = 1 is random reading with replacement
    """

    b: float = 100.0
    beta: float = 0.2
    gamma: float = 0.3
    Theta: float = 0.004
    dl: float = 0.42
    sigma: float = 3.0

    def with_(self, **kw) -> "ReaderParams":
        return replace(self, **kw)


def _h(R, P):
    S = R + P
    return np.where(S > 0, R * P / np.where(S > 0, S, 1.0), 0.0)


class PayoffCurve:
    """Vectorised evaluation of r*(P), v(P) and its derivatives."""

    def __init__(self, prm: ReaderParams):
        self.p = prm

    # ----- primitives -------------------------------------------------
    def phi(self, c):
        return 1.0 - np.exp(-self.p.sigma * c)

    def dphi(self, c):
        return self.p.sigma * np.exp(-self.p.sigma * c)

    def lng(self, r, P):
        """ln g(r, P); r in [0, 1), P > 0 (arrays broadcast)."""
        p = self.p
        r = np.asarray(r, float)
        P = np.asarray(P, float)
        R = p.b * r
        h = _h(R, P)
        c = np.where(P > 0, R / np.where(P > 0, P, 1.0), 0.0)
        return (np.log1p(-r) + p.gamma * np.log1p(p.beta * h)
                - p.Theta * P + p.dl * p.Theta * P * self.phi(c))

    def dlng_dr(self, r, P):
        p = self.p
        R = p.b * r
        h = _h(R, P)
        hR = P ** 2 / (R + P) ** 2
        return (-1.0 / (1.0 - r)
                + p.b * (p.gamma * p.beta * hR / (1.0 + p.beta * h)
                         + p.dl * p.Theta * self.dphi(R / P)))

    def d2lng_dr2(self, r, P):
        p = self.p
        R = p.b * r
        h = _h(R, P)
        hR = P ** 2 / (R + P) ** 2
        hRR = -2.0 * P ** 2 / (R + P) ** 3
        lq = p.gamma * p.beta * (hRR * (1 + p.beta * h) - p.beta * hR ** 2) / (1 + p.beta * h) ** 2
        lc = -p.dl * p.Theta * p.sigma ** 2 * np.exp(-p.sigma * R / P) / P
        return -1.0 / (1.0 - r) ** 2 + p.b ** 2 * (lq + lc)

    # ----- optimal reading ----------------------------------------------
    def r_star(self, P, iters: int = 60):
        """Optimal reading share; unique because ln g is strictly concave in r."""
        P = np.atleast_1d(np.asarray(P, float))
        out = np.zeros_like(P)
        pos = P > 0
        if not pos.any():
            return out
        Pp = P[pos]
        interior = self.dlng_dr(np.zeros_like(Pp), Pp) > 0
        lo = np.zeros_like(Pp)
        hi = np.full_like(Pp, 1.0 - 1e-12)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            f = self.dlng_dr(mid, Pp)
            lo = np.where(f > 0, mid, lo)
            hi = np.where(f > 0, hi, mid)
        r = 0.5 * (lo + hi)
        for _ in range(3):  # Newton polish
            r_new = r - self.dlng_dr(r, Pp) / self.d2lng_dr2(r, Pp)
            r = np.where((r_new > 0) & (r_new < 1), r_new, r)
        out[pos] = np.where(interior, r, 0.0)
        return out

    def x_star(self, P):
        return 1.0 - self.r_star(P)

    # ----- payoff curve -------------------------------------------------
    def lnv(self, P):
        P = np.atleast_1d(np.asarray(P, float))
        r = self.r_star(P)
        out = np.where(P > 0, self.lng(r, np.where(P > 0, P, 1.0)), 0.0)
        return out

    def v(self, P):
        return np.exp(self.lnv(P))

    def dlnv(self, P):
        """d ln v / dP by the envelope theorem: d ln g / dP at r*(P)."""
        p = self.p
        P = np.atleast_1d(np.asarray(P, float))
        R = p.b * self.r_star(P)
        h = _h(R, P)
        hP = R ** 2 / (R + P) ** 2
        c = R / P
        return (p.gamma * p.beta * hP / (1 + p.beta * h)
                - p.Theta + p.dl * p.Theta * (self.phi(c) - c * self.dphi(c)))

    def learning_term(self, P):
        """d ln Q / dP at r*(P): 'more to learn from'."""
        p = self.p
        P = np.atleast_1d(np.asarray(P, float))
        R = p.b * self.r_star(P)
        return p.gamma * p.beta * (R ** 2 / (R + P) ** 2) / (1 + p.beta * _h(R, P))

    def erosion_term(self, P):
        """-d ln C / dP at r*(P): 'more to miss'."""
        p = self.p
        P = np.atleast_1d(np.asarray(P, float))
        c = p.b * self.r_star(P) / P
        return p.Theta * (1 - p.dl * (self.phi(c) - c * self.dphi(c)))

    def eps_v(self, P):
        """Elasticity d ln v / d ln P."""
        P = np.atleast_1d(np.asarray(P, float))
        return P * self.dlnv(P)

    def eps_x(self, P, h: float = 1e-4):
        """Elasticity of the writing share with respect to density."""
        P = np.atleast_1d(np.asarray(P, float))
        up, dn = self.x_star(P * np.exp(h)), self.x_star(P * np.exp(-h))
        return (np.log(up) - np.log(dn)) / (2 * h)

    def coverage(self, P):
        P = np.atleast_1d(np.asarray(P, float))
        return self.p.b * self.r_star(P) / P

    def erosion_loss(self, P):
        """1 - C at the optimum: share of credit lost to overlap."""
        P = np.atleast_1d(np.asarray(P, float))
        c = self.coverage(P)
        return 1 - np.exp(-self.p.Theta * P * (1 - self.p.dl * self.phi(c)))

    def ln_credit(self, P):
        """ln C at the optimum (minus the erosion intensity)."""
        P = np.atleast_1d(np.asarray(P, float))
        c = self.coverage(P)
        return -self.p.Theta * P * (1 - self.p.dl * self.phi(c))

    def ln_social(self, P, tau: float = 0.0, extra: float = 0.0):
        """ln of social output per unit of writing speed at private choices.

        tau    share of credit lost to overlap that is a transfer to the earlier
               paper (priority credit), so that social erosion is (1-tau) times
               private erosion: ln C_s = (1 - tau) ln C;
        extra  additional social loss per covering paper not averted by
               detection, theta (lam_s - lam) with Delta_s = Delta, so that
               s(P) = v(P) exp(-extra P).
        """
        P = np.atleast_1d(np.asarray(P, float))
        return self.lnv(P) - tau * self.ln_credit(P) - extra * P

    # ----- landmarks ------------------------------------------------------
    def peak(self, Pmax: float = 1e4):
        """Location and height of the peak of v (unique by Theorem 1)."""
        grid = np.exp(np.linspace(np.log(1e-3), np.log(Pmax), 400))
        lv = self.lnv(grid)
        i = int(np.argmax(lv))
        lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
        if i == 0:
            return 0.0, 1.0
        res = minimize_scalar(lambda q: -self.lnv(np.exp(q))[0],
                              bounds=(np.log(lo), np.log(hi)), method="bounded",
                              options={"xatol": 1e-10})
        Ph = float(np.exp(res.x))
        return Ph, float(np.exp(-res.fun))

    def crossing(self, F: float, branch: str = "falling", kappa: float = 0.0,
                 Pmax: float = 1e5):
        """Solve v(P) exp(-kappa P) = F on the requested branch.

        Returns nan when the crossing does not exist on that branch.
        """
        f = lambda P: float(self.lnv(P)[0] - kappa * P - np.log(F))
        Ph = self.peak_w(kappa)[0]
        top = f(Ph) if Ph > 0 else -np.log(F)
        if branch == "falling":
            if top < 0:
                return np.nan
            hi = max(2 * Ph, 1.0)
            while f(hi) > 0:
                hi *= 2
                if hi > Pmax:
                    return np.nan
            return brentq(f, Ph, hi, xtol=1e-12)
        if top < 0 or f(1e-9) > 0:
            return np.nan
        return brentq(f, 1e-9, Ph, xtol=1e-12)

    def peak_w(self, kappa: float):
        """Peak of w(P) = v(P) exp(-kappa P)."""
        if kappa == 0.0:
            return self.peak()
        grid = np.exp(np.linspace(np.log(1e-3), np.log(1e4), 400))
        lw = self.lnv(grid) - kappa * grid
        i = int(np.argmax(lw))
        if i == 0:
            return 0.0, 1.0
        lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
        res = minimize_scalar(lambda q: -(self.lnv(np.exp(q))[0] - kappa * np.exp(q)),
                              bounds=(np.log(lo), np.log(hi)), method="bounded",
                              options={"xatol": 1e-10})
        Pw = float(np.exp(res.x))
        return Pw, float(np.exp(-res.fun))


class GridCurve:
    """Pre-tabulated payoff curve for fast interpolation in simulations."""

    def __init__(self, curve: PayoffCurve, Pmin: float = 1e-2, Pmax: float = 5e3,
                 n: int = 4000):
        self.curve = curve
        self.lp = np.linspace(np.log(Pmin), np.log(Pmax), n)
        P = np.exp(self.lp)
        self.lv = curve.lnv(P)
        self.lx = np.log(curve.x_star(P))
        self.Phat, self.vhat = curve.peak()

    def lnv(self, P):
        return np.interp(np.log(P), self.lp, self.lv)

    def x(self, P):
        return np.exp(np.interp(np.log(P), self.lp, self.lx))
