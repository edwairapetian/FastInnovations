"""Quantitative results for the paper: calibration, incidence of research tools,
speed limit, life cycles, welfare and policy.

Everything is computed from a calibration dictionary (results/calibration.json)
with the keys produced by scripts/run_calibration.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .credit import CreditTargets, aware_share, credit_block, flagged_rate_index
from .equilibrium import Economy, Stationary
from .lifecycle import (LifeCycleParams, composition_bias, founder_fixed_point, founder_value,
                        g_star_myopic, g_star_re, g_star_re_bound, g_takeoff, life_cycle_accounts,
                        mature_density, phase_shares, simulate, stopping_density_myopic,
                        stopping_density_re, test4_moments)
from .payoff import GridCurve, PayoffCurve, ReaderParams

OMEGAS = (0.0, 0.25, 0.5, 0.75)


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
        self.cu = self.lc.cu
        self.credit = c["credit"]
        self.fr = c["frontier"]
        self.a = c["a"]

    def mature(self, curve=None, cu=None, F=None):
        curve = curve or self.curve
        cu = self.cu if cu is None else cu
        F = self.lc.F if F is None else F
        return curve.crossing(F, "falling", cu=cu)

    def economy(self, omega: float | None = None, **kw) -> Economy:
        """Stationary economy that reproduces the calibrated mature density with M = 1.

        omega is the founders' probability (baseline from the calibration); e is set so
        that F(1, P_m) equals the calibrated frontier value, and eta_F so that the
        volume elasticity equals its target.
        """
        om = self.fr["omega"] if omega is None else omega
        Pm = self.mature()
        F, vPm = self.lc.F, float(self.curve.v(Pm)[0])
        s = om * (self.cu.rho + self.cu.xi) / (1 - om) if om > 0 else 0.0
        e = (F - om * vPm) / (1 - om)
        probe = Stationary(self.curve, Economy(a=self.a, cu=self.cu, e=e, s=s, founders=om > 0))
        th = probe.theta_F(1.0, Pm)
        eta = probe.eta_from_E(Pm, self.cal["E_target"], th)
        base = dict(a=self.a, cu=self.cu, e=e, s=s, eta_F=eta, Mbar=1.0, founders=om > 0,
                    N=float(probe.members(Pm)[0]))
        base.update(kw)
        return Economy(**base)


# --------------------------------------------------------------------------- headline numbers
def summary(m: Model) -> dict:
    cv, lc, cu = m.curve, m.lc, m.cu
    Ph, vh = cv.peak()
    Pw, ww = cv.peak_w(cu)
    Pm = m.mature()
    F = lc.F
    P_fric = cv.crossing(F, "falling")
    eco = m.economy()
    st = Stationary(cv, eco)
    eps_w = float(st.eps_w(Pm)[0])
    eps_v = float(cv.eps_v(Pm)[0])
    eps_g = float(cu.eps_gamma(Pm))
    D = float(st.D(Pm)[0])
    E = m.cal["E_target"]
    th = st.theta_F(1.0, Pm)
    eta = eco.eta_F
    cs = simulate(m.grid, lc)
    ph = phase_shares(cs)
    mom_p = test4_moments(cs, "pooled")
    mom_w = test4_moments(cs, "within")
    phi_m = float(cv.phi(cv.coverage(Pm))[0])
    q = m.credit["q"]
    r_m = float(cv.r_star(Pm)[0])
    x_cs = m.grid.x(np.exp(cs.lnP))
    vP0 = float(cv.v(lc.P0)[0])
    tau = lc.delta * (ww - F) / Pw
    Pm_top = float(np.nanmax(cs.Pm))
    return dict(
        Phat=Ph, vhat=vh, Pw=Pw, what=ww, Pm=Pm, v_Pm=float(cv.v(Pm)[0]), F=F, b_c=lc.b_c,
        P_frictionless=P_fric, premium=float(-cu.ln_gamma(Pm)), T=float(cu.time(Pm)),
        T_peak=float(cu.time(Ph)), speed_ratio=Pm / lc.b_c, speed_ratio_top=Pm_top / lc.b_c,
        rent=np.log(vh / F), dissipated_frictionless=1 - F / vh,
        planner_gain=ww / F - 1, subsidy=(ww - F) / F, stock_tax=tau,
        newborn_margin=np.log(vP0 / F), v_P0=vP0,
        eps_w=eps_w, eps_v=eps_v, eps_gamma=eps_g, D=D, E=E, eta=eta, theta_F=th,
        theta_F_lc=m.fr["theta_F_lc"], omega=m.fr["omega"],
        lam_eq=1 - E * abs(eps_w), lam_fixed=1 + eps_w / D,
        x_m=1 - r_m, r_m=r_m, read_cs=float((cs.w * (1 - x_cs)).sum() / cs.w.sum()),
        time_value_read=r_m / (1 - r_m), reads_per_paper=(m.reader.b / m.a) * r_m / (1 - r_m),
        coverage_m=float(cv.coverage(Pm)[0]), phi_m=phi_m,
        aware_m=float(aware_share(phi_m, q)), erosion_m=float(cv.erosion_loss(Pm)[0]),
        phases=ph, moments_pooled=mom_p, moments_within=mom_w, beyond_true=mom_p["beyond"],
        g_star=g_star_myopic(m.grid, lc), g_star_inf=g_star_myopic(m.grid, lc, stock="asymptotic"),
        g_star_re=g_star_re(m.grid, lc), g_star_re_bound=g_star_re_bound(m.grid, lc),
        comp_bias_T7=composition_bias(cs, 7.0), U_F=m.fr["U_F"],
    )


# --------------------------------------------------------------------------- incidence of tools
def _tools(m: Model, size: float):
    """(name, channel label, kwargs for the stationary economy and curve) for each tool."""
    k = 1 + size
    p = m.reader
    cb = credit_block(CreditTargets(**{**m.cal["credit_targets"],
                                       "pivot_cost": m.cal["credit_targets"]["pivot_cost"] / 2}))
    return [
        ("Faster production", r"$a\times%.1f$" % k, dict(a=m.a * k), None, "production"),
        ("Faster reading", r"$b\times%.1f$" % k, {}, p.with_(b=p.b * k), "learning"),
        ("Better learning", r"$\beta\times%.1f$" % k, {}, p.with_(beta=p.beta * k), "learning"),
        ("Better search", r"$\sigma\times%.1f$" % k, {}, p.with_(sigma=p.sigma * k), "learning"),
        ("Cheaper pivots", r"$\kappa_p\div 2$", {}, p.with_(dl=cb["dl"]), "learning"),
        ("Faster catch-up", r"$b_c\times%.1f$" % k, dict(cu=m.cu.with_(b_c=m.cu.b_c * k)), None, "catch-up"),
    ]


def incidence_table(m: Model, size: float = 0.3, omegas=OMEGAS) -> list[dict]:
    """Long-run effects of research tools in the stationary economy, solved exactly.

    For each tool: its value at fixed topics (the change in ln w at the mature density),
    the long-run change in output per researcher for each founders' probability omega
    (e reset so that the mature density is the calibrated one, eta_F set from the volume
    elasticity), and the footprints at the baseline omega.
    """
    Pm = m.mature()
    q0 = m.credit["q"]
    rows = []
    base = {}
    for om in omegas:
        eco = m.economy(omega=om)
        eq = Stationary(m.curve, eco).equilibrium()
        base[om] = (eco, eq)
    om0 = m.fr["omega"]
    eco0 = m.economy(omega=om0)
    eq0 = Stationary(m.curve, eco0).equilibrium()
    phi0 = float(m.curve.phi(m.curve.coverage(eq0["P"]))[0])
    g0 = g_star_myopic(m.grid, m.lc)
    for name, chan, eco_kw, rp_new, kind in _tools(m, size):
        curve = PayoffCurve(rp_new) if rp_new is not None else m.curve
        cu = eco_kw.get("cu", m.cu)
        a_new = eco_kw.get("a", m.a)
        h = float(curve.lnw(Pm, cu)[0] - m.curve.lnw(Pm, m.cu)[0]) + np.log(a_new / m.a)
        row = dict(name=name, channel=chan, kind=kind, fixed=h)
        for om in omegas:
            eco, eq = base[om]
            eq1 = Stationary(curve, eco.with_(**eco_kw)).equilibrium()
            row[f"long_{om}"] = float(np.log(eq1["Y_per_N"] / eq["Y_per_N"]))
        # footprints at the baseline founders' probability
        eq1 = Stationary(curve, eco0.with_(**eco_kw)).equilibrium()
        P1 = eq1["P"]
        phi1 = float(curve.phi(curve.coverage(P1))[0])
        q1 = q0
        if name == "Cheaper pivots":
            q1 = credit_block(CreditTargets(**{**m.cal["credit_targets"],
                                               "pivot_cost": m.cal["credit_targets"]["pivot_cost"] / 2}))["q"]
        lc1 = m.lc.with_(b_c=cu.b_c)
        g1 = g_star_myopic(GridCurve(curve, n=2000), lc1.with_(F=eq1["F"] / eq1["F"] * m.lc.F)) \
            if rp_new is not None or "cu" in eco_kw else g0
        row.update(dP=float(np.log(P1 / eq0["P"])), dM=float(np.log(eq1["M"] / eq0["M"])),
                   dpapers=float(np.log(a_new * eq1["x"] / (m.a * eq0["x"]))),
                   dflag=float(np.log(flagged_rate_index(P1, phi1, q1) / flagged_rate_index(eq0["P"], phi0, q0))),
                   daware=float(aware_share(phi1, q1) - aware_share(phi0, q0)),
                   dypr=row[f"long_{om0}"], dgstar=float(g1 - g0))
        rows.append(row)
    return rows


def incidence_formulas(m: Model, omegas=OMEGAS) -> list[dict]:
    """Pass-through rates from the frontier-incidence formulas at the mature density."""
    Pm = m.mature()
    out = []
    for om in omegas:
        eco = m.economy(omega=om)
        st = Stationary(m.curve, eco)
        th = st.theta_F(1.0, Pm)
        ew = abs(float(st.eps_w(Pm)[0]))
        ev = float(m.curve.eps_v(Pm)[0])
        D = float(st.D(Pm)[0])
        den = ew + th * ev + eco.eta_F * D
        lam = ew / den
        out.append(dict(omega=om, theta_F=th, eta_F=eco.eta_F, lam_F=lam,
                        pass_learning=1 - lam * (1 - th), pass_catchup=1 - lam,
                        production=1 - m.cal["E_target"] * ew))
    return out


def life_cycle_incidence(m: Model, size: float = 0.3, omegas=OMEGAS, grid=None) -> list[dict]:
    """Pass-through on the life cycle with founders who stay (frontier depletion off).

    For each omega, explorers' output e is set so that the calibrated F solves
    F = (1 - omega) e + omega U_F(F); each tool then moves F to the new fixed point.
    The long-run change in output per researcher is ln(a' F' / a F) (Little's law) and,
    with diffusion rents and catch-up counted, the change in life-cycle output per
    researcher-year.
    """
    grid = grid or dict(n_g=21, n_z=21, dt=0.1, horizon=100.0)
    lc0 = m.lc.with_(**grid)
    gc0 = GridCurve(m.curve, n=2000)
    cs0 = simulate(gc0, lc0)
    UF0 = founder_value(cs0, lc0)
    Pm = m.mature()
    rows = []
    tools = [t for t in _tools(m, size) if t[0] in ("Faster production", "Faster reading", "Better learning",
                                                      "Faster catch-up")]
    for om in omegas:
        e = (lc0.F - om * UF0) / (1 - om)
        s_rate = om * (lc0.rho + lc0.xi) / (1 - om) if om > 0 else m.fr["s"]
        acc0 = life_cycle_accounts(gc0, lc0, m.a, e, s_rate, cs=cs0)
        row = dict(omega=om, e=e, U_F=UF0, Y0=acc0["Y_per_aN"])
        for name, chan, eco_kw, rp_new, kind in tools:
            curve = PayoffCurve(rp_new) if rp_new is not None else m.curve
            gc1 = GridCurve(curve, n=2000) if rp_new is not None else gc0
            lc1 = lc0.with_(b_c=eco_kw["cu"].b_c) if "cu" in eco_kw else lc0
            a1 = eco_kw.get("a", m.a)
            F1, _ = founder_fixed_point(gc1, lc1, e, om, F_lo=0.8 * lc0.F, F_hi=1.25 * lc0.F)
            acc1 = life_cycle_accounts(gc1, lc1.with_(F=F1), a1, e, s_rate)
            h = float(curve.lnw(Pm, lc1.cu)[0] - m.curve.lnw(Pm, m.cu)[0]) + np.log(a1 / m.a)
            row[name] = dict(fixed=h, dlnF=float(np.log(a1 * F1 / (m.a * lc0.F))),
                             dY=float(np.log(a1 * acc1["Y_per_aN"] / (m.a * acc0["Y_per_aN"]))))
        rows.append(row)
    return rows


def audience(m: Model, rows: list[dict], alphas=(0.0, 0.5, 1.0), size: float = 0.3) -> dict:
    """Output valued by readership: Y_alpha = A^alpha Y, with A = (b / a) r / (1 - r) reads per paper.

    Uses the baseline-omega long-run effects; A is evaluated at the new equilibrium density.
    """
    eco0 = m.economy()
    eq0 = Stationary(m.curve, eco0).equilibrium()
    r0 = float(m.curve.r_star(eq0["P"])[0])
    A0 = (m.reader.b / m.a) * r0 / (1 - r0)
    out = {}
    for name, chan, eco_kw, rp_new, kind in _tools(m, size):
        curve = PayoffCurve(rp_new) if rp_new is not None else m.curve
        eq1 = Stationary(curve, eco0.with_(**eco_kw)).equilibrium()
        r1 = float(curve.r_star(eq1["P"])[0])
        b1 = rp_new.b if rp_new is not None else m.reader.b
        A1 = (b1 / eco_kw.get("a", m.a)) * r1 / (1 - r1)
        dY = float(np.log(eq1["Y_per_N"] / eq0["Y_per_N"]))
        out[name] = {f"alpha_{al}": dY + al * float(np.log(A1 / A0)) for al in alphas}
        out[name]["dlnA"] = float(np.log(A1 / A0))
    return dict(A0=A0, rows=out)


# --------------------------------------------------------------------------- social value
def social_block(m: Model, taus=(0.0, 0.25, 0.5, 0.75, 1.0)) -> dict:
    """Social value versus private credit (Proposition on social value).

    Uniform wedge: s = v exp(-theta (lam_s - lam) P); the equilibrium is overcrowded
    relative to the social optimum iff tau = (lam - lam_s) / lam < |eps_w(P_m)| / (theta lam P_m).
    Proportional transfer: ln C_s = (1 - tau) ln C; the social optimum is the peak of
    ln s + ln Gamma, found numerically.
    """
    cv, cu = m.curve, m.cu
    Pm = m.mature()
    eps_w = float(cv.eps_v(Pm)[0] - cu.eps_gamma(Pm))
    ThP = cv.p.Theta * Pm
    out = dict(Pm=Pm, eps_w=eps_w, theta_lam_Pm=ThP, erosion_m=float(cv.erosion_loss(Pm)[0]),
               tau_crit_uniform=abs(eps_w) / ThP, tau_max_uniform=1 - cv.p.dl)
    top = np.log(cu.b_c * (1 - 1e-6))

    def obj(t):
        return lambda q: -(cv.ln_social(np.exp(q), tau=t)[0] + float(cu.ln_gamma(np.exp(q))))

    def P_opt(t):
        r = minimize_scalar(obj(t), bounds=(np.log(0.5), top), method="bounded", options={"xatol": 1e-9})
        return float(np.exp(r.x)), -float(r.fun)

    rows = []
    for t in taus:
        Po, val = P_opt(t)
        base = cv.ln_social(Pm, tau=t)[0] + float(cu.ln_gamma(Pm))
        rows.append(dict(tau=t, P_social=Po, overcrowded=bool(Po < Pm), gain=float(np.exp(val - base) - 1)))
    out["proportional"] = rows
    try:
        out["tau_crit_proportional"] = brentq(lambda t: P_opt(t)[0] - Pm, 0.0, 5.0, xtol=1e-6)
    except ValueError:
        out["tau_crit_proportional"] = float("nan")
    return out


# --------------------------------------------------------------------------- life cycles
def re_overshooting(m: Model, gs=(0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0)) -> dict:
    """Myopic and forward-looking stopping densities (exact catch-up, finite birth flow)."""
    gc, lc = m.grid, m.lc
    rows = [dict(g=g, myopic=stopping_density_myopic(gc, lc, g),
                 rational=stopping_density_re(gc, lc, g)) for g in gs]
    return dict(g_star=g_star_myopic(gc, lc), g_star_inf=g_star_myopic(gc, lc, stock="asymptotic"),
                g_star_re=g_star_re(gc, lc), g_star_re_bound=g_star_re_bound(gc, lc),
                P_frictionless=m.curve.crossing(lc.F, "falling"), b_c=lc.b_c, rows=rows)


def closure(m: Model, alphas=(1.0, 2.0, 3.0)) -> list[dict]:
    """A production shock raises every topic's flow on impact; topics pushed past b_c close.

    Reports, for each shock, the share of mature researcher-years in topics that close and
    how long they stay closed (flow back below b_c at rate rho); the first entry gives the
    smallest shock that closes the densest mature topic.
    """
    cs = simulate(m.grid, m.lc)
    w = cs.w * (cs.phase == 2)
    w = w / w.sum()
    P = np.exp(cs.lnP)
    a_min = float(m.lc.b_c / P[cs.phase == 2].max() - 1)
    out = [dict(alpha_min=a_min)]
    for al in alphas:
        closed = (1 + al) * P > m.lc.b_c
        dur = np.where(closed, np.log((1 + al) * P / m.lc.b_c) / m.lc.rho, 0.0)
        share = float((w * closed).sum())
        out.append(dict(alpha=al, share_closed=share,
                        mean_years_closed=float((w * dur).sum() / max(share, 1e-12)) if share > 0 else 0.0,
                        threshold_ratio=1 / (1 + al)))
    return out


def policy_block(m: Model, grid=None) -> dict:
    """Welfare on the life-cycle economy: laissez-faire, a flat frontier subsidy equal to the
    stationary optimum, and an entry tax proportional to the stock (zero in newborn topics).

    Founders stay (baseline omega); F solves the founders' fixed point under each policy;
    output per researcher-year counts exploration, catch-up and writing.
    """
    grid = grid or dict(n_g=21, n_z=21, dt=0.1, horizon=100.0)
    lc0 = m.lc.with_(**grid)
    gc = GridCurve(m.curve, n=2000)
    om, s_rate = m.fr["omega"], m.fr["s"]
    cs0 = simulate(gc, lc0)
    UF0 = founder_value(cs0, lc0)
    e = (lc0.F - om * UF0) / (1 - om)
    acc0 = life_cycle_accounts(gc, lc0, m.a, e, s_rate, cs=cs0)
    Pw, ww = m.curve.peak_w(m.cu)
    sub = ww - lc0.F
    tau = lc0.delta * sub / Pw
    out = dict(e=e, U_F=UF0, Y_lf=acc0["Y_per_aN"], shares_lf=acc0, subsidy=sub, stock_tax=tau,
               newborn_margin=float(np.log(m.curve.v(lc0.P0)[0] / lc0.F)),
               g_takeoff=float(g_takeoff(gc, lc0)), g_star=float(g_star_myopic(gc, lc0)))
    for name, kw in (("subsidy", dict(subsidy=sub)), ("stock_tax", dict(stock_tax=tau))):
        lcp = lc0.with_(**kw)
        F1, UF1 = founder_fixed_point(gc, lcp, e, om, F_lo=0.6 * lc0.F, F_hi=1.3 * lc0.F)
        acc1 = life_cycle_accounts(gc, lcp.with_(F=F1), m.a, e, s_rate)
        newborn_ok = bool(m.curve.v(lc0.P0)[0] > F1 + kw.get("subsidy", 0.0))
        out[name] = dict(F=F1, U_F=UF1, Y=acc1["Y_per_aN"], gain=acc1["Y_per_aN"] / acc0["Y_per_aN"] - 1,
                         newborn_followers=newborn_ok, shares=acc1,
                         Pm=float(mature_density(gc, lcp.with_(F=F1))),
                         g_takeoff=float(g_takeoff(gc, lcp.with_(F=F1))),
                         g_star=float(g_star_myopic(gc, lcp.with_(F=F1))))
    # the regime a flat subsidy pushes the economy into when it exceeds the newborn margin
    return out


# --------------------------------------------------------------------------- utilities
def write_json(obj, path):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=conv)
