"""Credit block: from the arXiv citation results to (lam, ell, Delta/lam, q, sigma).

Lemma 1 of the paper.  When a covering paper is detected the researcher pivots
(keeping 1 - kappa_p) or proceeds as a follow-on keeping delta ~ G.  With G
exponential with mean mu,

    proceed share     q    = exp(-tau / mu),          tau = 1 - kappa_p
    follow-on credit  E[delta | proceed] = tau + mu   (memorylessness)
    kept on detection 1 - ell = E max(delta, tau) = tau + mu q.

Targets (Fact 4): unaware duplicates keep 1 - lam = 0.81 of comparable credit;
aware follow-ons earn 1.29.  The aware share A = q phi / (1 - (1 - q) phi)
(Fact 5, 0.38) then pins the detection probability phi at the typical
topic, and sigma follows from phi = 1 - exp(-sigma c*).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CreditTargets:
    unaware_penalty: float = 0.19   # lam
    aware_premium: float = 0.29     # E[delta | proceed] - 1
    aware_share: float = 0.38       # A, midpoint of 35-42%
    pivot_cost: float = 0.30        # kappa_p, share of a project lost by pivoting


def credit_block(t: CreditTargets = CreditTargets()):
    lam = t.unaware_penalty
    tau = 1.0 - t.pivot_cost
    mu = 1.0 + t.aware_premium - tau
    q = float(np.exp(-tau / mu))
    ell = 1.0 - (tau + mu * q)
    if not 0 <= ell < lam:
        raise ValueError("credit targets violate 0 <= ell < lam")
    A = t.aware_share
    phi = A / (q + A * (1 - q))
    return dict(lam=lam, ell=ell, dl=(lam - ell) / lam, q=q, mu=mu, tau=tau, phi=phi)


def sigma_for(phi: float, coverage: float) -> float:
    """Search precision that gives detection probability phi at coverage c*."""
    return float(-np.log(1 - phi) / coverage)


def aware_share(phi, q):
    phi = np.asarray(phi, float)
    return q * phi / (1 - (1 - q) * phi)


def flagged_rate_index(P, phi, q):
    """Flagged overlap per unit of theta * psi: P [1 - (1 - q) phi]."""
    return np.asarray(P, float) * (1 - (1 - q) * np.asarray(phi, float))
