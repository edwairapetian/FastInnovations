"""Catching up with a topic's literature.

A newcomer must read the topic's relevant stock S at speed b_c before she can
contribute.  Papers that arrive while she reads join the stock and must be
read too, and the relevance of unread papers decays at rate delta.  Her
backlog B therefore follows

    dB/dt = P(t) - delta B - b_c,      B(0) = S,

and catch-up ends when B reaches zero.  With a constant flow P the catch-up
time is

    T(P, S) = (1/delta) ln(1 + delta S / (b_c - P))      for P < b_c,

and infinite for P >= b_c: nobody can catch up with a topic whose flow
exceeds the speed at which newcomers absorb it (the speed limit).  In a
stationary topic S = P / delta, so T = -(1/delta) ln(1 - P / b_c), which is
P / (delta b_c) to first order in P / b_c.

Careers end at rate rho and topics die at rate xi.  A newcomer to a
stationary topic keeps the survival factor

    Gamma(P) = exp(-(rho + xi) T) = (1 - P / b_c)^nu,     nu = (rho + xi) / delta,

and the share of a stationary topic's members who have finished catching up
(and write) is (1 - P / b_c)^(rho / delta).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class CatchUp:
    b_c: float = np.inf    # catch-up reading speed (papers per period); inf = frictionless
    delta: float = 0.25    # decay of the relevance of unread papers
    rho: float = 0.10      # rate at which careers in a topic end
    xi: float = 0.0        # rate at which topics become obsolete

    def with_(self, **kw) -> "CatchUp":
        return replace(self, **kw)

    @property
    def nu(self) -> float:
        """Elasticity of the newcomer's survival factor in (1 - P / b_c)."""
        return (self.rho + self.xi) / self.delta

    @property
    def nu_m(self) -> float:
        """Elasticity of the writing share of members in (1 - P / b_c)."""
        return self.rho / self.delta

    @property
    def frictionless(self) -> bool:
        return not np.isfinite(self.b_c)

    # ----- catch-up time ------------------------------------------------------
    def time(self, P, S=None):
        """Catch-up time with a constant flow P and stock S (stationary if S is None)."""
        P = np.asarray(P, float)
        if self.frictionless:
            return np.zeros_like(P)
        if S is None:
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(P < self.b_c, -np.log1p(-P / self.b_c) / self.delta, np.inf)
        S = np.asarray(S, float)
        gap = self.b_c - P
        with np.errstate(divide="ignore", invalid="ignore"):
            T = np.log1p(self.delta * S / np.where(gap > 0, gap, 1.0)) / self.delta
        return np.where(gap > 0, T, np.where(S <= 0, 0.0, np.inf))

    def ln_gamma(self, P, S=None):
        """ln of the newcomer's survival factor exp(-(rho + xi) T)."""
        P = np.asarray(P, float)
        if self.frictionless:
            return np.zeros_like(P)
        if S is None:
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(P < self.b_c, self.nu * np.log1p(-P / self.b_c), -np.inf)
        return -(self.rho + self.xi) * self.time(P, S)

    def ln_writing_share(self, P):
        """ln of the share of a stationary topic's members who write."""
        P = np.asarray(P, float)
        if self.frictionless:
            return np.zeros_like(P)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(P < self.b_c, self.nu_m * np.log1p(-P / self.b_c), -np.inf)

    def eps_gamma(self, P):
        """Catch-up elasticity -d ln Gamma / d ln P = nu (P/b_c) / (1 - P/b_c) (stationary)."""
        P = np.asarray(P, float)
        if self.frictionless:
            return np.zeros_like(P)
        x = P / self.b_c
        return np.where(x < 1, self.nu * x / (1 - x), np.inf)

    def eps_members(self, P):
        """d ln(1 / writing share) / d ln P = nu_m (P/b_c) / (1 - P/b_c)."""
        P = np.asarray(P, float)
        if self.frictionless:
            return np.zeros_like(P)
        x = P / self.b_c
        return np.where(x < 1, self.nu_m * x / (1 - x), np.inf)


def backlog_path_time(S0, P_path, t_grid, b_c, delta):
    """Catch-up time for a general flow path by integrating dB/dt = P - delta B - b_c.

    P_path is evaluated on t_grid; returns the first time B reaches zero (inf if never).
    Used to verify the closed forms.
    """
    B = float(S0)
    if B <= 0:
        return 0.0
    for k in range(1, len(t_grid)):
        dt = t_grid[k] - t_grid[k - 1]
        P = 0.5 * (P_path[k] + P_path[k - 1])
        # exact step for constant P over dt
        Bstar = (P - b_c) / delta
        Bn = Bstar + (B - Bstar) * np.exp(-delta * dt)
        if Bn <= 0:
            # time within the step at which B hits zero
            frac = np.log((B - Bstar) / (0 - Bstar)) / delta
            return t_grid[k - 1] + frac
        B = Bn
    return np.inf
