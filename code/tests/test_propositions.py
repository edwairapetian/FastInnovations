"""Numerical checks of every analytical result in the paper.

Each test names the result it verifies.  Random draws cover the parameter
space allowed by the assumptions; tolerances reflect grid and root-finding
error only.
"""
import numpy as np
import pytest

from ftmodel.catchup import CatchUp, backlog_path_time
from ftmodel.equilibrium import Economy, Stationary
from ftmodel.lifecycle import (LifeCycleParams, composition_bias, g_star_myopic, g_star_peak,
                               g_star_re, g_star_re_bound, g_takeoff, life_cycle_accounts,
                               mature_density, re_value_ln, simulate, stall_recovery_gap,
                               stopping_density_myopic, stopping_density_re)
from ftmodel.payoff import GridCurve, PayoffCurve, ReaderParams

rng = np.random.default_rng(20261028)


def random_params(n):
    out = []
    while len(out) < n:
        p = ReaderParams(b=100.0,
                         beta=float(np.exp(rng.uniform(np.log(0.03), np.log(1.5)))),
                         gamma=float(rng.uniform(0.2, 0.9)),
                         Theta=float(np.exp(rng.uniform(np.log(5e-4), np.log(0.02)))),
                         dl=float(rng.uniform(0.0, 0.95)),
                         sigma=float(np.exp(rng.uniform(np.log(0.5), np.log(8.0)))),
                         Theta_c=float(rng.choice([0.0, np.exp(rng.uniform(np.log(1e-4), np.log(5e-3)))])))
        if p.b * p.gamma * p.beta > 1.0:  # reading the first paper is worth its time
            out.append(p)
    return out


PARAMS = random_params(25)
BASE = ReaderParams(beta=0.2, gamma=0.5, Theta=0.004, dl=0.42, sigma=3.0)
# a calibrated-like curve: mature topics on the rising side of v under catch-up
CAL = ReaderParams(beta=39.0, gamma=0.156, Theta=0.00407, dl=0.37, sigma=1.58)
CU = CatchUp(b_c=40.0, delta=0.25, rho=0.10, xi=0.04)


def economy_at(curve, cu, P_target, s, eta_F=0.0, a=4.0, N=1000.0):
    """Stationary economy whose equilibrium density is P_target with M = Mbar."""
    om = s / (cu.rho + cu.xi + s) if s > 0 else 0.0
    Ft = float(np.exp(curve.lnw(P_target, cu)[0]))
    vt = float(curve.v(P_target)[0])
    probe = Stationary(curve, Economy(a=a, cu=cu, founders=False))
    Mb = N / float(probe.members(P_target)[0])
    return Economy(a=a, cu=cu, e=(Ft - om * vt) / (1 - om), s=s, eta_F=eta_F, Mbar=Mb,
                   founders=s > 0, N=N)


# ---------------------------------------------------------------- Theorem 1
@pytest.mark.parametrize("p", PARAMS)
def test_theorem1_log_concave_and_single_peaked(p):
    c = PayoffCurve(p)
    P = np.linspace(1e-3, 30.0 / (p.Theta + p.Theta_c), 4000)
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
def test_reading_rises_with_density(p):
    c = PayoffCurve(p)
    P = np.exp(np.linspace(np.log(0.01), np.log(5.0 / p.Theta), 600))
    r = c.r_star(P)
    assert np.all(np.diff(r) > -1e-9), "r*(P) nondecreasing (Topkis)"


def test_initial_slope_condition():
    """v'(0+) equals the sup over rays of the first-order gain."""
    c = PayoffCurve(BASE)
    p = BASE
    rho = np.linspace(0.0, 50.0, 200001)
    gain = (p.gamma * p.beta * rho / (1 + rho) - rho / p.b
            - p.Theta * (1 - p.dl * (1 - np.exp(-p.sigma * rho))))
    P = 1e-6
    fd = (np.exp(c.lnv(P)[0]) - 1) / P
    assert abs(fd - gain.max()) < 1e-4


# ---------------------------------------------------------------- time shares (Lemma)
@pytest.mark.parametrize("p", PARAMS[:8] + [CAL])
def test_time_share_value_of_reading(p):
    """d ln v / d ln b = r*/(1 - r*) (envelope + first-order condition), so per unit of
    time saved a faster reader gains exactly what a faster producer gains."""
    c = PayoffCurve(p)
    for P in (2.0, 10.0, 40.0):
        h = 1e-5
        up = PayoffCurve(p.with_(b=p.b * np.exp(h))).lnv(P)[0]
        dn = PayoffCurve(p.with_(b=p.b * np.exp(-h))).lnv(P)[0]
        fd = (up - dn) / (2 * h)
        r = float(c.r_star(P)[0])
        assert abs(fd - r / (1 - r)) < 1e-6
        # per unit of total time saved: reading saves r d ln b, production saves x d ln a
        if r > 0:
            assert abs(fd / r - 1.0 / (1 - r)) < 1e-5


# ---------------------------------------------------------------- Proposition 1
def test_prop1_spillover_sign():
    c = PayoffCurve(BASE)
    Ph, _ = c.peak()
    for P in [0.3 * Ph, 0.9 * Ph, 1.1 * Ph, 3 * Ph]:
        ext = c.learning_term(P)[0] - c.erosion_term(P)[0]
        assert abs(ext - c.dlnv(P)[0]) < 1e-12
        assert np.sign(ext) == np.sign(Ph - P)


# ---------------------------------------------------------------- Lemma 2: catch-up
@pytest.mark.parametrize("P,S", [(5.0, 20.0), (15.0, 60.0), (30.0, 10.0), (38.0, 152.0)])
def test_lemma_catchup_time_closed_form(P, S):
    t = np.linspace(0, 400, 400001)
    T_num = backlog_path_time(S, np.full_like(t, P), t, CU.b_c, CU.delta)
    T = float(CU.time(P, S))
    assert abs(T_num - T) < 1e-6 * max(1.0, T)
    # stationary topic: S = P / delta gives T = -(1/delta) ln(1 - P / b_c)
    Ts = float(CU.time(P))
    assert abs(Ts - float(CU.time(P, P / CU.delta))) < 1e-10
    assert abs(float(CU.ln_gamma(P)) + (CU.rho + CU.xi) * Ts) < 1e-10


def test_speed_limit_no_catch_up():
    """Theorem (speed limit) (i): with P >= b_c a positive backlog never clears."""
    for P in (CU.b_c, 1.2 * CU.b_c, 3 * CU.b_c):
        assert not np.isfinite(CU.time(P, 1.0))
        assert not np.isfinite(CU.time(P))
        t = np.linspace(0, 200, 20001)
        assert not np.isfinite(backlog_path_time(1.0, np.full_like(t, P), t, CU.b_c, CU.delta))
    # the first-order approximation T ~ S / b_c understates catch-up time
    P = 20.0
    assert CU.time(P) > (P / CU.delta) / CU.b_c


def test_writing_share_of_members():
    """Stationary topic: the share of members who have caught up is (1 - P/b_c)^(rho/delta)."""
    cu = CU
    P = 25.0
    T = float(cu.time(P))
    # entrants arrive at a constant rate and leave at rate rho
    share = np.exp(-cu.rho * T)
    assert abs(np.log(share) - float(cu.ln_writing_share(P))) < 1e-12


# ---------------------------------------------------------------- Lemma 3: frontier value
def test_frontier_value_renewal_reward():
    """F = (1 - omega) e + omega U_F is the long-run output rate of the explore option:
    exploring until success (rate s) or the end of the career (rate rho), then founding
    a topic that pays U until the career ends or the topic dies (rate rho + xi)."""
    r = np.random.default_rng(7)
    rho, xi, s, e, U = 0.10, 0.04, 0.2, 0.8, 1.3
    n = 400_000
    t_end = r.exponential(1 / rho, n)        # career length from the start of exploring
    t_succ = r.exponential(1 / s, n)
    explore = np.minimum(t_end, t_succ)
    founded = t_succ < t_end
    t_die = r.exponential(1 / xi, n)
    found_len = np.where(founded, np.minimum(t_end - t_succ, t_die), 0.0)
    out = e * explore + U * found_len
    length = explore + found_len
    om = s / (rho + xi + s)
    assert abs(out.sum() / length.sum() - ((1 - om) * e + om * U)) < 5e-3


# ---------------------------------------------------------------- Propositions 2-4
@pytest.mark.parametrize("b_c", [np.inf, 60.0, 40.0])
@pytest.mark.parametrize("s", [0.0, 0.14])
def test_prop2_equilibrium_on_falling_branch(b_c, s):
    c = PayoffCurve(CAL)
    cu = CU.with_(b_c=b_c)
    Pw, ww = c.peak_w(cu)
    eco = economy_at(c, cu, 1.6 * Pw if np.isfinite(b_c) else 1.3 * Pw, s)
    st = Stationary(c, eco)
    eq = st.equilibrium()
    assert eq["P"] > Pw
    assert st.eps_w(eq["P"])[0] < 0
    assert abs(eq["w"] - eq["F"]) < 1e-9
    assert eq["w"] < ww  # dissipation of the rent at the peak of w
    if np.isfinite(b_c):
        assert eq["P"] < b_c  # speed limit


@pytest.mark.parametrize("s", [0.0, 0.14, 0.4])
@pytest.mark.parametrize("eta_F", [0.01, 0.05])
def test_prop3_volume_elasticity_with_founders(s, eta_F):
    c = PayoffCurve(CAL)
    eco = economy_at(c, CU, 8.0, s, eta_F=eta_F)
    st = Stationary(c, eco)
    e1 = st.equilibrium()
    e2 = st.equilibrium(N=eco.N * np.exp(0.002))
    fd = (np.log(e2["P"]) - np.log(e1["P"])) / 0.002
    E = st.volume_elasticity(e1["P"], e1["eta_F"], e1["theta_F"])
    assert abs(fd - E) < 2e-3 * max(1, E) + 1e-4
    # stepping on toes in equilibrium: d ln Y / d ln N = 1 - E |eps_w|
    lam = st.lambda_eq(e1["P"], E)
    dY = (np.log(e2["Y_per_N"] * eco.N * np.exp(0.002)) - np.log(e1["Y_per_N"] * eco.N)) / 0.002
    assert abs(dY - lam) < 1e-3
    # holding open topics fixed the elasticity is 1 + eps_w / D < lam
    assert st.lambda_fixed_topics(e1["P"]) < lam


def test_prop3_pinned_when_frontier_elastic():
    c = PayoffCurve(CAL)
    eco = economy_at(c, CU, 8.0, 0.14, eta_F=0.0)
    st = Stationary(c, eco)
    Ps = [st.equilibrium(N=N)["P"] for N in (100.0, 1000.0, 10000.0)]
    assert np.ptp(Ps) < 1e-8
    Pa = [Stationary(c, eco.with_(a=a)).equilibrium()["P"] for a in (2.0, 4.0, 8.0)]
    assert np.ptp(Pa) < 1e-8


def test_prop4_planner_peak_and_finite():
    c = PayoffCurve(CAL)
    eco = economy_at(c, CU, 8.0, 0.14)
    st = Stationary(c, eco)
    pl = st.planner()
    Pw, ww = st.peak_w()
    assert pl["P"] == Pw and pl["gain"] > 0
    pf = st.planner_finite()
    eq = st.equilibrium()
    assert Pw < pf["P"] < eq["P"]


# ---------------------------------------------------------------- Proposition 5, speed limit
def test_prop5_gradient_equals_catchup_elasticity():
    """Across mature topics that differ in fertility, d ln(zeta v)/d ln P_m = eps_Gamma(P_m),
    which exceeds the incumbents' log premium -ln Gamma(P_m)."""
    gc = GridCurve(PayoffCurve(CAL), n=3000)
    lc = LifeCycleParams(F=1.75, b_c=40.0)
    lz = np.array([0.0, 0.002])
    Pm = np.array([mature_density(gc, lc, l) for l in lz])
    y = lz + gc.lnv(Pm)
    slope = np.diff(y)[0] / np.diff(np.log(Pm))[0]
    Pmid = np.sqrt(Pm[0] * Pm[1])
    assert abs(slope - float(lc.cu.eps_gamma(Pmid))) < 5e-3
    assert float(lc.cu.eps_gamma(Pmid)) > -float(lc.cu.ln_gamma(Pmid))


def test_speed_limit_theorem():
    """(ii) mature density below b_c, premium unbounded as fertility rises;
    (iii) myopic stopping below b_c; (iv) a production shock closes dense topics."""
    gc = GridCurve(PayoffCurve(CAL), n=3000)
    lc = LifeCycleParams(F=1.75, b_c=40.0, P0=2.0)
    lzs = [0.0, 0.05, 0.2, 0.5, 1.0, 3.0]
    Pm = np.array([mature_density(gc, lc, l) for l in lzs])
    prem = -lc.cu.ln_gamma(Pm)
    assert np.all(Pm < lc.b_c) and np.all(np.diff(Pm) > 0) and np.all(np.diff(prem) > 0)
    assert Pm[-1] / lc.b_c > 0.95 and prem[-1] > 10 * prem[0]
    for g in (0.2, 1.0, 5.0, 50.0):
        assert stopping_density_myopic(gc, lc, g) < lc.b_c
    for alpha in (0.3, 1.0):
        P = Pm[0]
        assert np.isfinite(lc.cu.time(P))
        closes = (1 + alpha) * P >= lc.b_c
        assert closes == (not np.isfinite(lc.cu.time((1 + alpha) * P)))


def test_overcrowding_with_rising_returns():
    c = PayoffCurve(CAL)
    eco = economy_at(c, CU, 8.0, 0.0)
    st = Stationary(c, eco)
    eq = st.equilibrium()
    assert c.dlnv(eq["P"])[0] > 0 > st.eps_w(eq["P"])[0]
    assert st.planner()["P"] < eq["P"]


# ---------------------------------------------------------------- Proposition 9: too fast
@pytest.fixture(scope="module")
def lifecycle_setup():
    gc = GridCurve(PayoffCurve(CAL), n=3000)
    lc = LifeCycleParams(F=1.75, b_c=40.0, P0=2.0)
    assert gc.lnv(lc.P0) > np.log(lc.F), "the first follower of a newborn topic beats the frontier"
    return gc, lc


def test_prop9_myopic_threshold(lifecycle_setup):
    """Take-off and overshooting under myopic entry with the exact (finite-birth) stock."""
    gc, lc = lifecycle_setup
    Pm = mature_density(gc, lc)
    gs = [0.02, 0.05, 0.1, 0.2, 0.3, 1.0, 3.0, 30.0]
    Pc = np.array([stopping_density_myopic(gc, lc, g) for g in gs])
    assert np.all(np.diff(Pc) >= -1e-9), "the stopping density rises with the diffusion rate"
    assert np.all(Pc < lc.b_c)
    # take-off: slower topics stall below the mature density (in the valley where w < F)
    gt = g_takeoff(gc, lc)
    assert 0 < gt < np.inf
    assert stopping_density_myopic(gc, lc, 0.95 * gt) < Pm < stopping_density_myopic(gc, lc, 1.05 * gt) * (1 + 1e-6)
    stall = stopping_density_myopic(gc, lc, 0.95 * gt)
    assert float(gc.lnv(stall) + lc.cu.ln_gamma(stall)) < np.log(lc.F), "stalls happen where w < F"
    for g in (0.3 * gt, 0.7 * gt, 0.95 * gt):
        assert stall_recovery_gap(gc, lc, g) < 0, "a stalled topic is never worth joining again"
    # overshooting threshold = max(take-off, value at the peak beats F)
    gs_ = g_star_myopic(gc, lc)
    assert stopping_density_myopic(gc, lc, 0.98 * gs_) <= gc.Phat < stopping_density_myopic(gc, lc, 1.02 * gs_)
    gp = g_star_peak(gc, lc)
    assert abs(gs_ - max(gt, gp)) < 1e-3 * gs_
    # closed form at the peak with the asymptotic stock, an upper bound on the exact rate
    gi = g_star_peak(gc, lc, stock="asymptotic")
    k = (gc.vhat / lc.F) ** (1 / lc.nu) - 1
    assert abs(gi - (lc.delta * gc.Phat / ((lc.b_c - gc.Phat) * k) - lc.delta)) < 1e-9
    assert gp <= gi + 1e-9
    for g in (0.2, 0.6, 2.0):
        assert stopping_density_myopic(gc, lc, g) >= stopping_density_myopic(gc, lc, g, stock="asymptotic") - 1e-6


def test_prop9_forward_looking(lifecycle_setup):
    gc, lc = lifecycle_setup
    Pm = mature_density(gc, lc)
    assert Pm < gc.Phat < lc.b_c
    g1, g2 = g_star_re(gc, lc), g_star_re_bound(gc, lc)
    assert g1 <= g2 + 1e-9
    assert stopping_density_re(gc, lc, g2 * 1.05) > gc.Phat
    assert stopping_density_re(gc, lc, g1 * 0.98) < gc.Phat < stopping_density_re(gc, lc, g1 * 1.02)
    # the forward-looking value counts payoffs only after catch-up: at the mature
    # density it equals w(P_m) = F
    lnF = np.log(lc.F)
    assert abs(re_value_ln(gc, lc, Pm, Pm / lc.delta, 0.0, Pm) - lnF) < 1e-6
    # fast topics: foresight carries entry past the myopic stopping density
    assert stopping_density_re(gc, lc, 5.0) > stopping_density_myopic(gc, lc, 5.0)


def test_composition_bias_formula(lifecycle_setup):
    gc, lc = lifecycle_setup
    cs = simulate(gc, lc.with_(n_g=9, n_z=1, dt=0.1, g_med=0.2, g_disp=0.5))
    T = 7.0
    w = cs.w / cs.w.sum()
    x = np.clip(T - cs.age, 0, None)
    cov = ((w * (cs.lnP - (w * cs.lnP).sum())) * (x - (w * x).sum())).sum()
    assert abs(composition_bias(cs, T, dg=1e-6) - cov / T) < 1e-4
    assert composition_bias(cs, T) < 0


# ---------------------------------------------------------------- Theorem: frontier incidence
@pytest.mark.parametrize("s", [0.0, 0.14, 0.4])
@pytest.mark.parametrize("eta_F", [0.0, 0.02])
def test_frontier_incidence(s, eta_F):
    """d ln(Y/N) = lam_F f + (1 - lam_F) h against exact re-solved equilibria, for a
    learning tool (f = theta_F h + eta_F m), a catch-up tool (founders never catch up)
    and a production tool (acts like a larger community)."""
    c = PayoffCurve(CAL)
    eco = economy_at(c, CU, 8.0, s, eta_F=eta_F)
    st = Stationary(c, eco)
    eq = st.equilibrium()
    P, th, et = eq["P"], eq["theta_F"], eq["eta_F"]
    d = 1e-3
    # learning tool
    c1 = PayoffCurve(CAL.with_(beta=CAL.beta * np.exp(d)))
    eq1 = Stationary(c1, eco).equilibrium()
    h = float(c1.lnv(P)[0] - c.lnv(P)[0])
    m_ = -float(np.log(c1.x_star(P)[0] / c.x_star(P)[0]))  # members per topic at fixed P
    inc = st.incidence(P, th * h + et * m_, h, th, et)
    assert abs(np.log(eq1["Y_per_N"] / eq["Y_per_N"]) - inc["dlnY"]) < 2e-3 * h
    # catch-up tool
    cu1 = CU.with_(b_c=CU.b_c * np.exp(d))
    eq2 = Stationary(c, eco.with_(cu=cu1)).equilibrium()
    hc = float(cu1.ln_gamma(P) - CU.ln_gamma(P))
    mc = float(CU.ln_writing_share(P) - cu1.ln_writing_share(P))
    inc2 = st.incidence(P, et * mc, hc, th, et)
    assert abs(np.log(eq2["Y_per_N"] / eq["Y_per_N"]) - inc2["dlnY"]) < 2e-3 * hc
    # production tool
    eq3 = Stationary(c, eco.with_(a=eco.a * np.exp(d))).equilibrium()
    E = st.volume_elasticity(P, et, th)
    assert abs(np.log(eq3["Y_per_N"] / eq["Y_per_N"]) / d - st.lambda_eq(P, E)) < 2e-3
    # ordering of pass-through: production >= learning >= catch-up
    pl, pc = inc["dlnY"] / h, inc2["dlnY"] / hc
    assert st.lambda_eq(P, E) >= pl - 1e-6 and pl >= pc - 1e-6
    if s == 0.0 and eta_F == 0.0:
        assert abs(pl) < 1e-3 and abs(pc) < 1e-3  # without founders or depletion: full dissipation


def test_incidence_lower_bound_from_volume_elasticity():
    """With eps_U = eps_v, a pure newcomer-side tool survives entry at rate
    1 - (1 - E D) |eps_w| / (|eps_w| + theta_F eps_v) >= E D."""
    c = PayoffCurve(CAL)
    for s in (0.0, 0.14):
        eco = economy_at(c, CU, 8.0, s, eta_F=0.02)
        st = Stationary(c, eco)
        eq = st.equilibrium()
        P, th, et = eq["P"], eq["theta_F"], eq["eta_F"]
        E = st.volume_elasticity(P, et, th)
        ew, ev, D = abs(float(st.eps_w(P)[0])), float(c.eps_v(P)[0]), float(st.D(P)[0])
        lam = st.incidence(P, 0.0, 1.0, th, et)["lam_F"]
        assert abs((1 - lam) - (1 - (1 - E * D) * ew / (ew + th * ev))) < 1e-10
        assert 1 - lam >= E * D - 1e-12


# ---------------------------------------------------------------- audience
def test_reads_per_paper():
    """Reads per paper (b/a) r*/(1 - r*): faster production lowers it one for one at fixed P."""
    c = PayoffCurve(CAL)
    P = 8.0
    r = float(c.r_star(P)[0])
    A = lambda a, b: (b / a) * r / (1 - r)
    assert abs(np.log(A(4.0 * 1.3, 100.0) / A(4.0, 100.0)) + np.log(1.3)) < 1e-12
    # total reads in a topic equal members' reading time: n_w r b
    nw = P / (4.0 * (1 - r))
    assert abs(nw * r * 100.0 - P * A(4.0, 100.0)) < 1e-9


# ---------------------------------------------------------------- policy on the life cycle
def test_policy_flat_subsidy_kills_newborn_topics(lifecycle_setup):
    gc, lc = lifecycle_setup
    margin = float(np.exp(gc.lnv(lc.P0))) - lc.F
    cs = simulate(gc, lc.with_(n_g=5, n_z=1, subsidy=1.01 * margin))
    assert np.all(cs.phase[:, 1:] == 1), "no topic grows: newborn topics attract no followers"
    cs2 = simulate(gc, lc.with_(n_g=5, n_z=1, subsidy=0.5 * margin))
    assert np.any(cs2.phase == 0)


def test_policy_stock_tax_targets_peak_of_w(lifecycle_setup):
    gc, lc = lifecycle_setup
    Pw, ww = gc.curve.peak_w(lc.cu)
    tau = lc.delta * (ww - lc.F) / Pw
    Pm_tax = mature_density(gc, lc.with_(stock_tax=tau))
    assert abs(Pm_tax / Pw - 1) < 2e-3
    # the tax is zero for a newborn topic: entry into it is unaffected
    cs0 = simulate(gc, lc.with_(n_g=3, n_z=1))
    cs1 = simulate(gc, lc.with_(n_g=3, n_z=1, stock_tax=tau))
    assert np.all(cs1.phase[:, 1] == cs0.phase[:, 1])


def test_life_cycle_accounts_catch_up_imputation(lifecycle_setup):
    """Catch-up members imputed from the flow of new writers: in a mature topic the share of
    members who write is (1 - P/b_c)^(rho/delta), so output per member-year is
    v(P_m) e^{-rho T} = F e^{xi T} (the newcomer's value also prices the risk that the
    topic dies while she catches up)."""
    gc, lc = lifecycle_setup
    one = lc.with_(n_g=1, g_disp=0.0, g_med=0.4, n_z=1, zeta_sd=0.0, dt=0.05)
    cs = simulate(gc, one)
    acc = life_cycle_accounts(gc, one, 4.0, e=lc.F, s_rate=0.14, cs=cs, paths=True)
    mat = np.where(cs.phase[0] == 2)[0][5:-5]
    assert len(mat) > 100
    Pm = float(np.exp(cs.lnP[0, mat[0]]))
    T = float(lc.cu.time(Pm))
    nw, nc = acc["nw"][0, mat], acc["nc"][0, mat]
    assert np.allclose(nw / (nw + nc), np.exp(float(lc.cu.ln_writing_share(Pm))), rtol=1e-6)
    per_member = np.exp(cs.y[0, mat]) * nw / (nw + nc)
    assert np.allclose(per_member, lc.F * np.exp(lc.xi * T), rtol=2e-3)
    assert 0 < acc["catchup"] < acc["writers"] and 0 < acc["explorers"] < 1


# ---------------------------------------------------------------- Lemma 1 (credit)
def test_lemma1_credit_microfoundation():
    """Monte Carlo of Poisson coverage with multiplicative erosion, earlier and concurrent."""
    r = np.random.default_rng(1)
    theta, P, lam, ell, phi = 0.02, 40.0, 0.19, 0.11, 0.6
    theta_c, lam_c = 0.004, 0.21
    n = 400_000
    K = r.poisson(theta * P, size=n)
    Kd = r.binomial(K, phi)
    Ku = K - Kd
    Kc = r.poisson(theta_c * P, size=n)
    credit = (1 - lam) ** Ku * (1 - ell) ** Kd * (1 - lam_c) ** Kc
    analytic = np.exp(-theta * P * (lam - (lam - ell) * phi) - theta_c * lam_c * P)
    assert abs(credit.mean() - analytic) < 3e-3


def test_social_value_identities():
    c = PayoffCurve(BASE)
    P = np.array([3.0, 10.0, 30.0])
    assert np.allclose(c.ln_social(P, extra=0.01), c.lnv(P) - 0.01 * P)
    r = c.r_star(P)
    lnC = -BASE.Theta * P * (1 - BASE.dl * c.phi(BASE.b * r / P))
    assert np.allclose(c.ln_social(P, tau=0.4), c.lnv(P) - 0.4 * lnC)
