"""Numerical checks of every analytical result in the paper.

Each test names the result it verifies.  Random draws cover the parameter
space allowed by Assumptions A1-A5; tolerances reflect grid and root-finding
error only.
"""
import numpy as np
import pytest

from ftmodel.equilibrium import Economy, Stationary
from ftmodel.lifecycle import (LifeCycleParams, composition_bias, consolidation_value,
                               g_star, g_star_re, g_star_re_bound, mature_density,
                               simulate, stopping_density_asymptotic, stopping_density_re)
from ftmodel.payoff import GridCurve, PayoffCurve, ReaderParams

rng = np.random.default_rng(20261021)


def random_params(n):
    out = []
    while len(out) < n:
        p = ReaderParams(b=100.0,
                         beta=float(np.exp(rng.uniform(np.log(0.03), np.log(1.5)))),
                         gamma=float(rng.uniform(0.2, 0.9)),
                         Theta=float(np.exp(rng.uniform(np.log(5e-4), np.log(0.02)))),
                         dl=float(rng.uniform(0.0, 0.95)),
                         sigma=float(np.exp(rng.uniform(np.log(0.5), np.log(8.0)))))
        if p.b * p.gamma * p.beta > 1.0:  # A5: reading the first paper is worth its time
            out.append(p)
    return out


PARAMS = random_params(25)
BASE = ReaderParams(beta=0.2, gamma=0.5, Theta=0.004, dl=0.42, sigma=3.0)


# ---------------------------------------------------------------- Theorem 1
@pytest.mark.parametrize("p", PARAMS)
def test_theorem1_log_concave_and_single_peaked(p):
    c = PayoffCurve(p)
    P = np.linspace(1e-3, 30.0 / p.Theta, 4000)
    lv = c.lnv(P)
    d2 = np.diff(lv, 2)
    assert d2.max() < 1e-7, "ln v must be concave in P"
    i = int(np.argmax(lv))
    assert np.all(np.diff(lv[: i + 1]) > -1e-12)
    assert np.all(np.diff(lv[i:]) < 1e-12)
    assert abs(c.lnv(1e-9)[0]) < 1e-6, "v(0) = 1"
    assert lv[-1] < 0, "v -> 0 as P -> infinity"


@pytest.mark.parametrize("p", PARAMS[:10])
def test_theorem1_envelope(p):
    c = PayoffCurve(p)
    for P in [0.5, 5.0, 20.0, 80.0, 300.0]:
        h = 1e-5 * P
        fd = (c.lnv(P + h)[0] - c.lnv(P - h)[0]) / (2 * h)
        assert abs(fd - c.dlnv(P)[0]) < 1e-6 * max(1.0, abs(fd))


@pytest.mark.parametrize("p", PARAMS)
def test_lemma_reading_rises_with_density(p):
    c = PayoffCurve(p)
    P = np.exp(np.linspace(np.log(0.01), np.log(5.0 / p.Theta), 600))
    r = c.r_star(P)
    assert np.all(np.diff(r) > -1e-9), "r*(P) nondecreasing (Topkis)"


def test_initial_slope_condition():
    """A4: v'(0+) equals the sup over rays of the first-order gain."""
    c = PayoffCurve(BASE)
    p = BASE
    rho = np.linspace(0.0, 50.0, 200001)
    gain = (p.gamma * p.beta * rho / (1 + rho) - rho / p.b
            - p.Theta * (1 - p.dl * (1 - np.exp(-p.sigma * rho))))
    P = 1e-6
    fd = (np.exp(c.lnv(P)[0]) - 1) / P
    assert abs(fd - gain.max()) < 1e-4


# ---------------------------------------------------------------- Proposition 1
def test_prop1_spillover_sign():
    c = PayoffCurve(BASE)
    Ph, _ = c.peak()
    for P in [0.3 * Ph, 0.9 * Ph, 1.1 * Ph, 3 * Ph]:
        ext = c.learning_term(P)[0] - c.erosion_term(P)[0]
        assert abs(ext - c.dlnv(P)[0]) < 1e-12
        assert np.sign(ext) == np.sign(Ph - P)


# ---------------------------------------------------------------- Propositions 2-4
@pytest.mark.parametrize("kappa", [0.0, 0.004, 0.01])
@pytest.mark.parametrize("eta", [0.0, 0.05, 0.3])
def test_prop2_equilibrium_on_falling_branch(kappa, eta):
    c = PayoffCurve(BASE)
    s = Stationary(c, Economy(kappa=kappa, eta=eta, Fbar=0.95, Mbar=50.0, N=1000.0))
    eq = s.equilibrium()
    Pw, ww = s.peak_w()
    assert eq["P"] > Pw
    assert s.eps_w(eq["P"])[0] < 0
    assert abs(eq["w"] - eq["F"]) < 1e-9
    # payoff per researcher equals the frontier: dissipation
    assert eq["w"] < ww


@pytest.mark.parametrize("kappa", [0.0, 0.006])
@pytest.mark.parametrize("eta", [0.02, 0.2, 1.0])
def test_prop3_volume_elasticity(kappa, eta):
    c = PayoffCurve(BASE)
    s = Stationary(c, Economy(kappa=kappa, eta=eta, Fbar=0.95, Mbar=50.0))
    e1 = s.equilibrium(N=1000.0, explorers=False)
    e2 = s.equilibrium(N=1000.0 * np.exp(0.002), explorers=False)
    fd = (np.log(e2["P"]) - np.log(e1["P"])) / 0.002
    E = s.volume_elasticity(np.sqrt(e1["P"] * e2["P"]))
    assert abs(fd - E) < 2e-3 * max(1, E) + 1e-4


def test_prop3_pinned_when_frontier_elastic():
    c = PayoffCurve(BASE)
    s = Stationary(c, Economy(kappa=0.004, eta=0.0, Fbar=0.95, Mbar=50.0))
    Ps = [s.equilibrium(N=N, explorers=False)["P"] for N in (100.0, 1000.0, 10000.0)]
    assert np.ptp(Ps) < 1e-8
    # writing speed is neutral for density
    Pa = [Stationary(c, s.e.with_(a=a)).equilibrium(explorers=False)["P"] for a in (2.0, 4.0, 8.0)]
    assert np.ptp(Pa) < 1e-8


@pytest.mark.parametrize("kappa", [0.0, 0.006])
def test_prop4_planner(kappa):
    c = PayoffCurve(BASE)
    s = Stationary(c, Economy(kappa=kappa, eta=0.1, Fbar=0.95, Mbar=50.0, N=1000.0))
    pl = s.planner()
    Pw, _ = s.peak_w()
    assert Pw < pl["P"] < pl["eq"]["P"]
    assert pl["M"] > pl["eq"]["M"]
    assert pl["Y"] > pl["Y_eq"]
    assert abs(s.planner_condition(pl["P"], pl["M"])) < 1e-5
    assert pl["subsidy"] > 0


def test_prop4_large_topic_limit():
    c = PayoffCurve(BASE)
    gaps = []
    for a in (4.0, 0.4, 0.04):
        s = Stationary(c, Economy(a=a, kappa=0.004, eta=0.0, Fbar=0.95, N=1e4))
        gaps.append(s.planner()["P"] / s.peak_w()[0] - 1)
    assert gaps[0] > gaps[1] > gaps[2] > 0
    assert gaps[2] < 0.01


def test_prop4_writing_wedge_vanishes_at_optimum():
    c = PayoffCurve(BASE)
    s = Stationary(c, Economy(a=0.2, kappa=0.004, eta=0.0, Fbar=0.95, N=1e4))
    fp, pl = s.full_planner(), s.planner()
    assert abs(fp["Y"] / pl["Y"] - 1) < 1e-3
    assert abs(fp["x"] - fp["x_private"]) < 0.01


# ---------------------------------------------------------------- Propositions 5-7
def test_prop5_returns_gradient_across_mature_topics():
    """Across mature topics that differ in fertility, d ln(zeta v)/d ln P = kappa P."""
    gc = GridCurve(PayoffCurve(BASE))
    lc = LifeCycleParams(F=0.95, kappa_s=0.001)
    lz = np.array([0.0, 0.01])
    Pm = np.array([mature_density(gc, lc, l) for l in lz])
    y = lz + gc.lnv(Pm)
    slope = np.diff(y)[0] / np.diff(np.log(Pm))[0]
    assert abs(slope - lc.kappa * np.sqrt(Pm[0] * Pm[1])) < 2e-3


def test_prop6_overcrowding_with_rising_returns():
    """Mature equilibria can sit on the rising side of v while w' < 0."""
    c = PayoffCurve(BASE)
    s = Stationary(c, Economy(kappa=0.012, eta=0.0, Fbar=0.95, Mbar=50.0))
    eq = s.equilibrium()
    assert c.dlnv(eq["P"])[0] > 0 > s.eps_w(eq["P"])[0]
    pl = s.planner()
    assert pl["P"] < eq["P"]


def test_prop7_overshoot_threshold():
    gc = GridCurve(PayoffCurve(BASE))
    lc = LifeCycleParams(F=0.95, kappa_s=0.003)
    Pm = mature_density(gc, lc)
    gs = [0.0, 0.1, 0.3, 1.0, 3.0, 30.0]
    Pc = [stopping_density_asymptotic(gc, lc, g) for g in gs]
    assert abs(Pc[0] - Pm) < 1e-6 * Pm
    assert np.all(np.diff(Pc) > 0)
    gst = g_star(gc, lc)
    assert gst > 0
    assert stopping_density_asymptotic(gc, lc, gst * 0.99) < gc.Phat < stopping_density_asymptotic(gc, lc, gst * 1.01)


def test_prop7_finite_birth_stops_later():
    """With a finite birth flow the stock lags more, so P_c >= P_c^inf(g)."""
    gc = GridCurve(PayoffCurve(BASE))
    lc = LifeCycleParams(F=0.95, kappa_s=0.003)
    for g in (0.1, 0.4, 1.2):
        t = np.linspace(0, 200, 400001)
        P = lc.P0 * np.exp(g * t)
        S = P * (1 - np.exp(-(g + lc.delta) * t)) / (g + lc.delta)
        E = gc.lnv(np.minimum(P, 4e3)) - lc.kappa_s * S - np.log(lc.F)
        Pc = P[np.argmax(E <= 0)]
        assert Pc >= stopping_density_asymptotic(gc, lc, g) * (1 - 1e-4)


@pytest.mark.parametrize("ks", [0.0025, 0.003, 0.0035])
def test_prop7_rational_expectations(ks):
    """Forward-looking entry: g* <= g*_RE <= delta (Phat / P_m - 1); topics that
    grow faster than the bound overshoot, slower than g* do not; with fast
    diffusion entry overshoots further than myopic entry, past the frictionless
    density."""
    gc = GridCurve(PayoffCurve(BASE))
    lc = LifeCycleParams(F=0.95, kappa_s=ks)
    if not gc.lnv(gc.Phat) - lc.kappa * gc.Phat < np.log(lc.F) < gc.lnv(gc.Phat):
        pytest.skip("mature topics must lie on the rising side of v")
    Pm = mature_density(gc, lc)
    if Pm >= gc.Phat:
        pytest.skip("mature topics must lie on the rising side of v")
    g0, g1, g2 = g_star(gc, lc), g_star_re(gc, lc), g_star_re_bound(gc, lc)
    assert g0 <= g1 + 1e-9 <= g2 + 2e-9
    assert stopping_density_re(gc, lc, g2 * 1.05) > gc.Phat
    assert stopping_density_re(gc, lc, g0 * 0.95) < gc.Phat
    assert stopping_density_re(gc, lc, g1 * 0.98) < gc.Phat < stopping_density_re(gc, lc, g1 * 1.02)
    Pf = gc.curve.crossing(lc.F, "falling")
    g_fast = 200.0
    assert stopping_density_re(gc, lc, g_fast) > stopping_density_asymptotic(gc, lc, g_fast)
    assert stopping_density_re(gc, lc, 1e6) > Pf
    # U(P) averages v along the consolidation path: equals v at P_m
    assert abs(consolidation_value(gc, lc, Pm) / np.exp(gc.lnv(Pm)) - 1) < 1e-9


def test_composition_bias_formula():
    """Small-acceleration limit equals Cov(ln P, (T - a)^+) / T."""
    gc = GridCurve(PayoffCurve(BASE))
    lc = LifeCycleParams(F=0.95, kappa_s=0.003, n_g=9, n_z=1, dt=0.1)
    cs = simulate(gc, lc)
    T = 7.0
    w = cs.w / cs.w.sum()
    x = np.clip(T - cs.age, 0, None)
    cov = ((w * (cs.lnP - (w * cs.lnP).sum())) * (x - (w * x).sum())).sum()
    assert abs(composition_bias(cs, T, dg=1e-6) - cov / T) < 1e-4
    assert composition_bias(cs, T) < 0


def test_prop6_social_value():
    """Social output: uniform wedge s = v exp(-extra P); proportional transfer
    ln s = ln v - tau ln C.  Overcrowding survives transfers in the calibrated
    region where |eps_w(P_m)| exceeds theta lam P_m."""
    c = PayoffCurve(BASE)
    P = np.array([3.0, 10.0, 30.0])
    assert np.allclose(c.ln_social(P, extra=0.01), c.lnv(P) - 0.01 * P)
    r = c.r_star(P)
    lnC = -BASE.Theta * P * (1 - BASE.dl * c.phi(BASE.b * r / P))
    assert np.allclose(c.ln_social(P, tau=0.4), c.lnv(P) - 0.4 * lnC)
    kappa = 0.012
    s = Stationary(c, Economy(kappa=kappa, eta=0.0, Fbar=0.95, Mbar=50.0))
    Pm = s.equilibrium()["P"]
    ew = abs(float(s.eps_w(Pm)[0]))
    grid = np.exp(np.linspace(np.log(0.5), np.log(3 * Pm), 4000))
    for tau in (0.0, 0.5, 1.0):
        Po = grid[np.argmax(c.ln_social(grid, tau=tau) - kappa * grid)]
        if ew > BASE.Theta * Pm:
            assert Po < Pm


# ---------------------------------------------------------------- Propositions 9-10
def test_prop10_sign_with_elastic_frontier():
    """eta > 0: d ln(Y/N) / d omega = -eta d ln M* / d omega >= 0 (corrected sign)."""
    eco = Economy(kappa=0.004, eta=0.2, Fbar=0.95, Mbar=50.0)
    res = []
    for beta in (0.2, 0.2 * np.exp(0.01)):
        s = Stationary(PayoffCurve(BASE.with_(beta=beta)), eco)
        eq = s.equilibrium(N=1e4, explorers=False)
        Y, M = s.output(eq["P"], 1e4, explorers=False)
        res.append((np.log(Y / 1e4), np.log(M)))
    dy, dm = res[1][0] - res[0][0], res[1][1] - res[0][1]
    assert dm < 0 and dy > 0
    # in the large-topic limit Y/N = a F(M*), so d ln(Y/N) = -eta d ln M*
    assert abs(dy - (-eco.eta * dm)) < 0.05 * abs(dy)


def test_prop10_reading_improvements_dissipated():
    """eta = 0: better reading raises density but not output per researcher."""
    out = []
    for beta in (0.2, 0.3):
        c = PayoffCurve(BASE.with_(beta=beta))
        s = Stationary(c, Economy(kappa=0.004, eta=0.0, Fbar=0.95))
        eq = s.equilibrium(N=1e4, explorers=False)
        Y, _ = s.output(eq["P"], 1e4, explorers=False)
        out.append((eq["P"], Y / 1e4))
    assert out[1][0] > out[0][0]
    assert abs(out[1][1] - out[0][1]) < 1e-9


def test_prop9_stepping_on_toes():
    c = PayoffCurve(BASE)
    s = Stationary(c, Economy(kappa=0.004, eta=0.15, Fbar=0.95, Mbar=50.0))
    e1 = s.equilibrium(N=1000.0, explorers=False)
    e2 = s.equilibrium(N=1000.0 * np.exp(0.002), explorers=False)
    Y1 = s.output(e1["P"], 1000.0, False)[0]
    Y2 = s.output(e2["P"], 1000.0 * np.exp(0.002), False)[0]
    fd = (np.log(Y2) - np.log(Y1)) / 0.002
    lam = s.stepping_on_toes(e1["P"], s.volume_elasticity(e1["P"]))
    assert abs(fd - lam) < 1e-3


# ---------------------------------------------------------------- Lemma 1 (credit)
def test_lemma1_credit_microfoundation():
    """Monte Carlo of Poisson coverage with multiplicative erosion."""
    r = np.random.default_rng(1)
    theta, P, lam, ell, phi = 0.02, 40.0, 0.19, 0.11, 0.6
    n = 400_000
    K = r.poisson(theta * P, size=n)
    Kd = r.binomial(K, phi)
    Ku = K - Kd
    credit = (1 - lam) ** Ku * (1 - ell) ** Kd
    analytic = np.exp(-theta * P * (lam - (lam - ell) * phi))
    assert abs(credit.mean() - analytic) < 3e-3
