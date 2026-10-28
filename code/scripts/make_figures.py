"""Build every figure, table and number macro used in the paper.

Reads results/calibration.json (baseline) and results/calibration_within.json
(robustness) and writes to paper/figures, paper/tables and paper/numbers.tex.
"""
import argparse
import json
import os
import sys
import warnings

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

from ftmodel.calibration import TARGETS  # noqa: E402
from ftmodel.equilibrium import Stationary  # noqa: E402
from ftmodel.lifecycle import (chi_bar, g_takeoff, g_types, simulate, stall_recovery_gap,  # noqa: E402
                               stopping_density_myopic, stopping_density_re, thresholds_stock_learning)
from ftmodel.report import (OMEGAS, Model, audience, closure, incidence_formulas,  # noqa: E402
                            incidence_table, life_cycle_incidence, policy_block, re_overshooting,
                            social_block, summary, write_json)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG = os.path.join(ROOT, "paper", "figures")
TAB = os.path.join(ROOT, "paper", "tables")
NUM = os.path.join(ROOT, "paper", "numbers.tex")
RES = os.path.join(ROOT, "code", "results")

# Reference categorical palette (light mode), fixed order; text in neutral inks.
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "lines.linewidth": 1.5, "legend.frameon": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def label(ax, x, y, s, color=INK2, **kw):
    ax.annotate(s, (x, y), fontsize=8, color=color, **kw)


def logfmt(ax):
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def fig_payoff(m, S):
    cv, cu = m.curve, m.cu
    top = min(1.3 * S["P_frictionless"], 0.999 * cu.b_c * 3)
    P = np.exp(np.linspace(np.log(0.5), np.log(top), 700))
    v = cv.v(P)
    w = np.exp(cv.lnw(P, cu))
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.6))
    ax = axs[0]
    ax.plot(P, v, color=C1, label="incumbent $v(P)$")
    ax.plot(P[P < cu.b_c], w[P < cu.b_c], color=C2, label="newcomer $w(P)$")
    ax.axhline(S["F"], color=INK2, lw=0.8, ls="--")
    lo_y = 0.8 * S["F"]
    ax.set_ylim(lo_y, 1.12 * S["vhat"])
    ax.annotate("frontier $F$", (0.55, S["F"] * 0.995), fontsize=8, color=INK2, va="top")
    for k, (x, s_) in enumerate([(S["Pm"], "$P^m$"), (S["Phat"], "$\\hat P$"), (cu.b_c, "$b_c$")]):
        ax.axvline(x, color=GRID if s_ != "$b_c$" else C2, lw=0.8, ls="-" if s_ != "$b_c$" else ":")
        label(ax, x * 1.06, lo_y * (1.01 + 0.05 * k), s_)
    ax.set_xscale("log")
    ax.set_xlabel("density $P$ (papers per year)")
    ax.set_title("(a) payoff curves")
    ax.legend(loc="upper left", fontsize=7.5)
    ax = axs[1]
    lt, et = P * cv.learning_term(P), P * cv.erosion_term(P)
    ax.plot(P, lt, color=C1, label="more to learn from")
    ax.plot(P, et, color=C2, label="more to miss")
    ax.plot(P, lt - et, color=INK, lw=1.0, label="net: $\\varepsilon_v$")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.axvline(S["Phat"], color=GRID, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("density $P$")
    ax.set_title("(b) elasticity of $v$, decomposed")
    ax.legend(fontsize=7.5)
    ax = axs[2]
    ax.plot(P, cv.r_star(P), color=C1)
    ax.set_xscale("log")
    ax.set_xlabel("density $P$")
    ax.set_title("(c) optimal reading share $r^*(P)$")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_payoff_curve.pdf"))
    plt.close(fig)


def fig_planner(m, S):
    cv = m.curve
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ax = axs[0]
    Pf = S["P_frictionless"]
    P = np.exp(np.linspace(np.log(0.5), np.log(2 * Pf), 600))
    v = cv.v(P)
    ax.plot(P, v, color=C1, label="$v(P)$")
    ax.axhline(S["F"], color=INK2, lw=0.8, ls="--")
    ax.fill_between(P, S["F"], v, where=(v > S["F"]), color=C1, alpha=0.12, lw=0)
    ax.axvline(S["Phat"], color=GRID, lw=0.8)
    ax.axvline(Pf, color=GRID, lw=0.8)
    label(ax, S["Phat"] * 1.04, S["F"] + 0.01, "planner\n$\\hat P$")
    label(ax, Pf * 1.04, S["F"] + 0.01, "equilibrium\n$v=F$")
    ax.set_xscale("log")
    ax.set_xlabel("density $P$")
    ax.set_title("(a) frictionless: rents competed away")
    ax = axs[1]
    eta = np.linspace(0, 0.6, 400)
    ew, D, th, ev = abs(S["eps_w"]), S["D"], S["theta_F"], S["eps_v"]
    E = eta / (eta * D + ew + th * ev)
    ax.plot(eta, E, color=C1, label="volume elasticity $\\mathcal{E}$")
    ax.plot(eta, E * ew, color=C2, label="$1-\\lambda_J=\\mathcal{E}|\\varepsilon_w|$")
    ax.axhspan(0.0, 0.27, color=INK2, alpha=0.08, lw=0)
    label(ax, 0.42, 0.14, "arXiv range\nof $\\mathcal{E}$")
    ax.axvline(S["eta"], color=GRID, lw=0.8)
    ax.set_xlabel("frontier elasticity $\\eta_F$")
    ax.set_title("(b) volume elasticity and stepping on toes")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_planner_volume.pdf"))
    plt.close(fig)


def fig_lifecycle(m, S, RE):
    gc, lc = m.grid, m.lc
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
    ax = axs[0]
    gt = S["g_takeoff"]
    gs = [0.8 * gt] + list(g_types(lc.with_(n_g=6))[0]) if np.isfinite(gt) and gt > 0 else list(g_types(lc.with_(n_g=7))[0])
    t = np.linspace(0, 40, 2000)
    cols = plt.cm.Blues(np.linspace(0.45, 0.95, len(gs)))
    for g, col in zip(gs, cols):
        one = lc.with_(g_med=g, g_disp=0.0, n_g=1, zeta_sd=0.0, n_z=1, horizon=40.0, dt=0.02)
        cs = simulate(gc, one)
        stalled = g < gt
        ax.plot(cs.age[0], np.exp(cs.lnP[0]), color=C4 if stalled else col, lw=1.2, ls="--" if stalled else "-")
        if g in (gs[0], gs[-1]):
            k = int(np.argmax(cs.lnP[0])) if not stalled else int(0.45 * len(cs.age[0]))
            label(ax, cs.age[0][k] + 0.6, np.exp(cs.lnP[0][k]) * (1.08 if not stalled else 1.25),
                  f"$g={g:.2f}$" + (" (stalls)" if stalled else ""), color=INK2)
    ax.axhline(S["Phat"], color=C2, lw=0.9, ls="--")
    label(ax, 31, S["Phat"] * 1.05, "peak $\\hat P$", color=INK2)
    ax.axhline(S["Pm"], color=INK2, lw=0.8, ls=":")
    label(ax, 29, S["Pm"] * 0.80, "mature $P^m$", color=INK2)
    ax.set_yscale("log")
    logfmt(ax)
    ax.set_xlim(0, 40)
    ax.set_xlabel("topic age (years)")
    ax.set_ylabel("density $P$")
    ax.set_title("(a) life cycles by diffusion rate $g$")
    ax = axs[1]
    g = np.exp(np.linspace(np.log(0.01), np.log(5), 160))
    Pc = [stopping_density_myopic(gc, lc, x) for x in g]
    ax.plot(g, Pc, color=C1, label="myopic newcomers")
    gr = np.exp(np.linspace(np.log(max(0.05, 1.05 * gt if np.isfinite(gt) else 0.05)), np.log(5), 40))
    ax.plot(gr, [stopping_density_re(gc, lc, x) for x in gr], color=C3, ls="--",
            label="forward-looking newcomers")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 0.88))
    ax.axhline(S["Phat"], color=C2, lw=0.9, ls="--")
    ax.axhline(S["Pm"], color=INK2, lw=0.8, ls=":")
    ax.axhline(lc.b_c, color=C2, lw=0.9, ls=":")
    ax.axvline(S["g_star"], color=GRID, lw=0.8)
    if np.isfinite(gt) and gt > 0.01:
        ax.axvline(gt, color=GRID, lw=0.8, ls="--")
        label(ax, gt * 1.08, lc.P0 * 1.2, "take-off")
    label(ax, S["g_star"] * 1.1, S["Pm"] * 1.05, "$g^*$")
    label(ax, 0.012, S["Phat"] * 1.04, "$\\hat P$")
    label(ax, 0.012, S["Pm"] * 1.04, "$P^m$")
    label(ax, 0.012, lc.b_c * 1.05, "speed limit $b_c$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    logfmt(ax)
    ax.set_xlabel("diffusion rate $g$ (per year)")
    ax.set_title("(b) stopping density")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_lifecycle.pdf"))
    plt.close(fig)


def fig_test4(m, S):
    cs = simulate(m.grid, m.lc.with_(n_g=41, n_z=51, dt=0.05))
    w0 = cs.w / cs.w.sum()
    e, we = np.polynomial.hermite_e.hermegauss(7)
    we = we / we.sum()
    lp = (cs.lnP[:, :, None] + 0.3 * e[None, None, :]).ravel()
    wf = (w0[:, :, None] * we[None, None, :]).ravel()
    y = np.repeat(cs.y[:, :, None], e.size, axis=2).ravel()
    mu = (wf * lp).sum()
    sd = np.sqrt((wf * (lp - mu) ** 2).sum())
    z = (lp - mu) / sd
    bins = np.linspace(-2.5, 2.5, 21)
    idx = np.digitize(z, bins)
    xs, ys = [], []
    for k in range(1, len(bins)):
        sel = idx == k
        if wf[sel].sum() > 2e-3:
            xs.append(0.5 * (bins[k - 1] + bins[k]))
            ys.append((wf[sel] * y[sel]).sum() / wf[sel].sum())
    o = np.argsort(z)
    cw = np.cumsum(wf[o])
    mid = o[(cw > 0.4) & (cw <= 0.6)]
    y0 = (wf[mid] * y[mid]).sum() / wf[mid].sum()
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
    ax = axs[0]
    ax.plot(xs, np.array(ys) - y0, color=C1, lw=1.0, alpha=0.6, label="model, binned")
    zz = np.linspace(-2, 2.5, 100)
    ax.plot(zz, 0.094 * zz - 0.047 * zz ** 2, color=INK2, lw=1.0, ls="--", label="arXiv quadratic (not targeted)")
    zq = np.array([-1.4, -0.53, 0.0, 0.53, 1.4])
    qm = S["moments_pooled"]["quint"]
    ax.plot(zq[[0, 2, 3, 4]], [TARGETS["Q1"], 0.0, TARGETS["Q4"], TARGETS["Q5"]], ls="none", marker="s",
            ms=5, color=C2, label="arXiv fifths")
    ax.plot(zq, qm, ls="none", marker="D", ms=5, mfc="white", mew=1.4, color=C1, label="model fifths")
    ax.set_ylim(-0.3, 0.12)
    ax.set_xlabel("observed topic density (SD from mean)")
    ax.set_ylabel("log returns vs middle fifth")
    ax.set_title("(a) returns and density")
    ax.legend(fontsize=7, loc="lower right")
    lpt, wt, ph = cs.lnP.ravel(), w0.ravel(), cs.phase.ravel()
    mut = (wt * lpt).sum()
    sdt = np.sqrt((wt * (lpt - mut) ** 2).sum())
    zt = (lpt - mut) / sdt
    zhat = (np.log(cs.Phat) - mut) / sdt
    ax = axs[1]
    hb = np.linspace(-3, 3.5, 40)
    bottom = np.zeros(len(hb) - 1)
    for k, (name, col) in enumerate([("growth", C3), ("consolidating or declining", C2), ("mature", C1)]):
        hist, _ = np.histogram(zt[ph == k], bins=hb, weights=wt[ph == k])
        ax.bar(hb[:-1], hist, width=np.diff(hb) * 0.92, bottom=bottom, align="edge", color=col,
               edgecolor="white", linewidth=0.6, label=name)
        bottom += hist
    ax.axvline(zhat, color=INK2, lw=0.9, ls="--")
    label(ax, zhat + 0.1, bottom.max() * 0.62,
          f"{100 * S['beyond_true']:.0f}% of\nresearcher-years\nbeyond the\npeak of $v$")
    ax.set_xlabel("true topic density (SD from mean)")
    ax.set_ylabel("share of researcher-years")
    ax.set_title("(b) where researchers work, by phase")
    ax.set_xlim(-3.2, 3.6)
    ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(0.0, 1.0), handlelength=1.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_test4.pdf"))
    plt.close(fig)


def fig_incidence(m, S, rows, FM, LC):
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.5))
    ax = axs[0]
    show = [r for r in rows if r["name"] in ("Faster production", "Faster reading", "Better learning",
                                             "Faster catch-up")]
    names = [r["name"] for r in show]
    yy = np.arange(len(show))[::-1]
    om0 = m.fr["omega"]
    # bars: long-run value as a share of the value at fixed topics
    hgt = 0.25
    for k, (key, col, lab) in enumerate([("long_0.0", C2, "no founders ($\\omega=0$)"),
                                         (f"long_{om0}", C1, f"founders stay ($\\omega={om0:.2g}$)")]):
        vals = np.array([100 * r[key] / r["fixed"] for r in show])
        ax.barh(yy + (0.5 - k) * hgt, vals, height=hgt, color=col, label=lab)
    lc_row = [r for r in LC if abs(r["omega"] - om0) < 1e-9][0]
    vals = np.array([100 * lc_row[n]["dY"] / lc_row[n]["fixed"] for n in names])
    ax.barh(yy - 1.5 * hgt, vals, height=hgt, color=C3, label="life cycle, founders stay")
    ax.set_yticks(yy)
    ax.set_yticklabels(names)
    ax.grid(axis="y", visible=False)
    ax.axvline(100, color=INK2, lw=0.6, ls=":")
    ax.set_xlim(0, 110)
    ax.set_xlabel("long-run gain, % of the gain at fixed topics")
    ax.set_title("(a) how much of each tool survives entry")
    ax.legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=2)
    ax = axs[1]
    om = np.array([r["omega"] for r in FM])
    ax.plot(om, [100 * r["pass_learning"] for r in FM], color=C1, marker="o", ms=3.5,
            label="learning tool (formula)")
    ax.plot(om, [100 * r["pass_catchup"] for r in FM], color=C2, marker="o", ms=3.5,
            label="catch-up tool (formula)")
    ax.plot(om, [100 * r["production"] for r in FM], color=INK2, lw=1.0, ls="--", label="production tool")
    lco = np.array([r["omega"] for r in LC])
    ax.plot(lco, [100 * r["Faster reading"]["dlnF"] / r["Faster reading"]["fixed"] for r in LC], color=C1,
            ls=":", marker="s", ms=3.5, mfc="white", label="reading, life cycle ($F$ only)")
    ax.plot(lco, [100 * r["Faster catch-up"]["dlnF"] / r["Faster catch-up"]["fixed"] for r in LC], color=C2,
            ls=":", marker="s", ms=3.5, mfc="white", label="catch-up, life cycle ($F$ only)")
    ax.axvline(om0, color=GRID, lw=0.8)
    ax.set_xlabel("founders' probability $\\omega$")
    ax.set_ylabel("pass-through (%)")
    ax.set_ylim(-2, 105)
    ax.set_title("(b) pass-through and founders")
    ax.legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_incidence.pdf"))
    plt.close(fig)


def fmt(x, d=2):
    return f"{x:.{d}f}"


def tables(m, S, rows, Sw, FM, LC, AU, SB, RE, PB):
    cal = m.cal
    th = cal["theta"]
    cb = cal["credit"]
    fx = cal["fixed"]
    fr = cal["frontier"]
    lines = [r"\begin{tabular}{@{}llll@{}}", r"\toprule",
             r"Parameter & Meaning & Value & Source \\", r"\midrule",
             r"\multicolumn{4}{@{}l}{\emph{Set externally}} \\",
             rf"$b$ & papers read per year at full-time reading & {fx['b']:.0f} & normalisation (unit of $P$) \\",
             rf"$a$ & projects per year at full-time production & {cal['a']:.0f} & normalisation \\",
             rf"$\lambda$ & credit lost by an unaware duplicate & {cb['lam']:.2f} & Fact 3 ($-19\%$) \\",
             rf"$\kappa_p$ & credit lost by pivoting & {cal['credit_targets']['pivot_cost']:.2f} & assumption \\",
             rf"$\Delta/\lambda$, $q$ & detection value; share proceeding & {cb['dl']:.2f}, {cb['q']:.2f} & Fact 3 ($+29\%$), \cref{{lem:credit}} \\",
             rf"$\delta$ & decay of the stock's relevance & {fx['delta']:.2f} & half-life $\approx$ 2.8 years \\",
             rf"$\rho$ & rate of leaving a topic & {fx['rho']:.2f} & ten-year topic tenure \\",
             rf"$\xi$ & topic obsolescence & {fx['xi']:.2f} & 25-year topic life \\",
             rf"$P_0$ & flow of a newborn topic & {fx['P0']:.0f} & a few papers per year \\",
             rf"$s$ & explorers' success rate & {fr['s']:.2f} & $\omega=s/(\rho+\xi+s)={fr['omega']:.2f}$ \\",
             r"\multicolumn{4}{@{}l}{\emph{Calibrated (\cref{tab:fit})}} \\",
             rf"$\beta$, $\gamma$ & learning efficiency, curvature & {th['beta']:.2f}, {th['gamma']:.3f} & hump in returns \\",
             rf"$\theta\lambda$ & erosion per earlier paper & {th['Theta']:.4f} & fall at the top \\",
             rf"$F$ & frontier value & {th['F']:.3f} & sparsest fifth \\",
             rf"$b_c$ & catch-up reading speed (speed limit) & {th['b_c']:.1f} & premium, share beyond peak \\",
             rf"$\sigma$ & search precision & {m.reader.sigma:.2f} & recall-corrected aware share \\",
             rf"$g_{{\rm med}}$, $s_g$ & diffusion rate: median, log-sd & {th['g_med']:.3f}, {th['g_disp']:.2f} & share beyond peak, fifths \\",
             rf"$s_\zeta$ & log-sd of topic fertility & {th['zeta_sd']:.3f} & gradient across topics \\",
             rf"$e$ & explorers' output (life cycle) & {fr['e_lc']:.3f} & $F=(1-\omega)e+\omega\bar U_F$ \\",
             rf"$\eta_F$ & frontier depletion & {fr['eta_F']:.4f} & $\mathcal{{E}}={cal['E_target']:.2f}$ (Fact 1), \eqref{{eq:E}} \\",
             r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "calibration.tex"), "w").write("\n".join(lines) + "\n")
    # ---- fit table
    mp, mwm = S["moments_pooled"], Sw["moments_within"]

    def row(name, tgt, a, b, d=3, targeted=True):
        f_ = lambda x: f"${fmt(x, d)}$"
        t = "" if tgt is None else f_(tgt)
        mark = "" if targeted else r"$^\dagger$"
        return rf"{name}{mark} & {t} & {f_(a)} & {f_(b)} \\"
    q = mp["quint"]
    qw = mwm["quint"]
    lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
             r"Moment & Data & Baseline & Within-topic \\", r"\midrule",
             row("Gradient per SD of density", 0.067, mp["lin"], mwm["lin"]),
             row("Sparsest fifth vs.\\ middle", TARGETS["Q1"], q[0], qw[0]),
             row("Fourth fifth vs.\\ middle", TARGETS["Q4"], q[3], qw[3]),
             row("Densest fifth vs.\\ middle", TARGETS["Q5"], q[4], qw[4]),
             row("Researcher-years beyond the peak", 0.12, S["beyond_true"], Sw["beyond_true"]),
             row("Incumbents' log premium $(\\rho+\\xi)T$", 0.15, S["premium"], Sw["premium"]),
             row("Aware share of flagged overlap", TARGETS["aware"], S["aware_m"], Sw["aware_m"], 2),
             row("Reading share of incumbents' time", TARGETS["read"], S["read_cs"], Sw["read_cs"], 3),
             row("Peak of $v$ (SD above mean)", 1.0, mp["zhat"], mwm["zhat"], 2, targeted=False),
             row("Quadratic: linear term", 0.094, mp["b1"], mwm["b1"], 3, targeted=False),
             row("Quadratic: squared term", -0.047, mp["b2"], mwm["b2"], 3, targeted=False),
             r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "fit.tex"), "w").write("\n".join(lines) + "\n")
    # ---- incidence table (stationary, exact re-solve)
    om0 = m.fr["omega"]
    lines = [r"\small", r"\setlength{\tabcolsep}{3.6pt}", r"\begin{tabular}{@{}llrrrrrrrr@{}}", r"\toprule",
             r" & & \multicolumn{3}{c}{Output per researcher} & \multicolumn{5}{c}{Footprints ($\omega=%.2g$)} \\" % om0,
             r"\cmidrule(lr){3-5}\cmidrule(l){6-10}",
             r"Tool & Shock & Fixed & Long run & Long run & Density & Topics & Papers/ & Flagged & Aware \\",
             r" & & topics & $\omega=0$ & $\omega=%.2g$ & & & researcher & overlap & (pp) \\" % om0, r"\midrule"]
    for r in rows:
        vals = [r["fixed"], r["long_0.0"], r[f"long_{om0}"], r["dP"], r["dM"], r["dpapers"], r["dflag"], r["daware"]]
        lines.append(rf"{r['name']} & {r['channel']} & " + " & ".join(f"${100 * v:+.1f}$" for v in vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\par\smallskip\footnotesize Stationary economy, solved exactly. Log points ($\times100$) of a 30\% "
              r"shock (pivot cost halved). \emph{Fixed topics}: change in the newcomer's payoff $w$ at the mature "
              r"density, which for a speed-up equals the time it saves (\cref{lem:timeshares}). \emph{Long run}: "
              r"change in output per researcher $aF$, with the frontier's depletion $\eta_F$ set so that the volume "
              r"elasticity is 0.15, for explorers who never stay ($\omega=0$) and for founders who stay in the topics "
              r"they open with probability $\omega$. The flagged overlap rate is relative to its baseline level, which "
              r"is a normalisation."]
    open(os.path.join(TAB, "ai_channels.tex"), "w").write("\n".join(lines) + "\n")
    # ---- life-cycle incidence and readership
    lines = [r"\small", r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{Faster reading} & \multicolumn{2}{c}{Faster catch-up} & \multicolumn{2}{c}{Faster production} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}",
             r" & formula & life cycle & formula & life cycle & formula & life cycle \\", r"\midrule",
             r"\multicolumn{7}{@{}l}{\emph{A. Share of the gain at fixed topics that survives entry (\%)}} \\"]
    for f_, l_ in zip(FM, LC):
        lines.append(rf"$\omega={f_['omega']:.2f}$ & {100 * f_['pass_learning']:.0f} & "
                     rf"{100 * l_['Faster reading']['dY'] / l_['Faster reading']['fixed']:.0f} & "
                     rf"{100 * f_['pass_catchup']:.0f} & {100 * l_['Faster catch-up']['dY'] / l_['Faster catch-up']['fixed']:.0f} & "
                     rf"{100 * f_['production']:.0f} & {100 * l_['Faster production']['dY'] / l_['Faster production']['fixed']:.0f} \\")
    lines += [r"\midrule",
              r"\multicolumn{7}{@{}l}{\emph{B. Long-run gain valued by readership, $\omega=%.2g$ (log points $\times100$)}} \\" % om0]
    for al in (0.0, 0.5, 1.0):
        rr, rc, rp = AU["rows"]["Faster reading"], AU["rows"]["Faster catch-up"], AU["rows"]["Faster production"]
        lines.append(rf"$\alpha={al:.1f}$ & \multicolumn{{2}}{{c}}{{${100 * rr[f'alpha_{al}']:+.1f}$}} & "
                     rf"\multicolumn{{2}}{{c}}{{${100 * rc[f'alpha_{al}']:+.1f}$}} & \multicolumn{{2}}{{c}}{{${100 * rp[f'alpha_{al}']:+.1f}$}} \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\par\smallskip\footnotesize Panel A: \emph{formula} is the frontier-incidence pass-through at the mature "
              r"density (\cref{thm:incidence}) with founders' topics valued at maturity and $\eta_F$ set from the volume "
              r"elasticity; \emph{life cycle} re-solves the founders' fixed point $F=(1-\omega)e+\omega\bar U_F(F)$ on "
              r"the simulated life cycle (no depletion) and measures output per researcher-year with exploration, "
              r"catch-up and diffusion rents counted. Panel B: output valued as $A^\alpha Y$, where $A=(b/a)r^*/(1-r^*)$ "
              r"is reads per paper; $\alpha=0$ is credited output, $\alpha=1$ readership."]
    open(os.path.join(TAB, "incidence_lc.tex"), "w").write("\n".join(lines) + "\n")
    # ---- welfare table
    lines = [r"\begin{tabular}{@{}lr@{}}", r"\toprule", r"Object & Value \\", r"\midrule",
             r"\multicolumn{2}{@{}l}{\emph{Payoff curve and catch-up}} \\",
             rf"Peak of the payoff curve $\hat P$ (papers per year) & {S['Phat']:.1f} \\",
             rf"Mature density $P^m$; speed limit $b_c$ & {S['Pm']:.1f}; {S['b_c']:.1f} \\",
             rf"Catch-up time at $P^m$ / at $\hat P$ (years) & {S['T']:.2f} / {S['T_peak']:.2f} \\",
             rf"Incumbents' premium over the frontier, $\Gamma(P^m)^{{-1}}-1$ & {100 * (np.exp(S['premium']) - 1):.1f}\% \\",
             rf"Cross-topic returns gradient $\varepsilon_\Gamma(P^m)$ & {S['eps_gamma']:.3f} \\",
             rf"Slope of the entrant payoff curve $|\varepsilon_w(P^m)|$ & {abs(S['eps_w']):.3f} \\",
             rf"Newborn margin $\ln[v(P_0)/F]$ & {100 * S['newborn_margin']:.2f}\% \\",
             r"\multicolumn{2}{@{}l}{\emph{Welfare and policy}} \\",
             rf"Frictionless benchmark: rent dissipated, $1-F/\hat v$ & {100 * S['dissipated_frictionless']:.1f}\% \\",
             rf"Stationary planner's gain, $\hat w/F-1$ & {100 * S['planner_gain']:.1f}\% \\",
             rf"Planner's gain if all overlap losses are transfers ($\tau=1$) & {100 * SB['proportional'][-1]['gain']:.1f}\% \\",
             rf"Life cycle: gain from a flat frontier subsidy $s^o$ & {100 * PB['subsidy']['gain']:.2f}\% \\",
             rf"Life cycle: gain from an entry tax proportional to the stock & {100 * PB['stock_tax']['gain']:.2f}\% \\",
             r"\multicolumn{2}{@{}l}{\emph{Topic life cycles}} \\",
             rf"Take-off threshold $g_0$ (per year) & {S['g_takeoff']:.3f} \\",
             rf"Overshooting threshold $g^*$, myopic newcomers & {S['g_star']:.3f} \\",
             rf"Closed-form bound at the peak, $g^*_\infty$ & {S['g_star_inf']:.3f} \\",
             rf"Overshooting threshold, forward-looking newcomers & {S['g_star_re']:.3f} \\",
             rf"Sufficient bound, forward-looking & {S['g_star_re_bound']:.3f} \\",
             r"\multicolumn{2}{@{}l}{\emph{Growth}} \\",
             rf"Short-run bias in $\mathcal{{E}}$ from topic-age composition (7 years) & ${S['comp_bias_T7']:.3f}$ \\",
             rf"Frontier depletion $\eta_F$ (from $\mathcal{{E}}={S['E']:.2f}$) & {S['eta']:.4f} \\",
             rf"Stepping on toes, topics adjust: $1-\mathcal{{E}}|\varepsilon_w|$ & {S['lam_eq']:.3f} \\",
             rf"Stepping on toes, open topics fixed: $1+\varepsilon_w/D$ & {S['lam_fixed']:.3f} \\",
             r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "welfare.tex"), "w").write("\n".join(lines) + "\n")


def numbers(m, S, rows, Sw, FM, LC, AU, SB, RE, PB, CL):
    by = {r["name"]: r for r in rows}
    om0 = m.fr["omega"]
    fm = {round(r["omega"], 4): r for r in FM}
    lc = {round(r["omega"], 4): r for r in LC}
    f0, fb = fm[0.0], fm[round(om0, 4)]
    l0, lb = lc[0.0], lc[round(om0, 4)]
    au = AU["rows"]
    pct = lambda x, d=0: f"{100 * x:.{d}f}\\%"
    share = lambda r, k: r[k] / r["fixed"]
    lcshare = lambda r, n, k="dY": r[n][k] / r[n]["fixed"]
    per_hour = lambda r, k: (by["Faster production"][k] / by["Faster production"]["fixed"]) / (r[k] / r["fixed"])
    macros = {
        "calPhat": f"{S['Phat']:.0f}", "calPm": f"{S['Pm']:.1f}", "calvhat": f"{S['vhat']:.2f}",
        "calF": f"{S['F']:.3f}", "calT": f"{S['T']:.1f}", "calTpeak": f"{S['T_peak']:.1f}",
        "calPremium": pct(np.exp(S["premium"]) - 1), "calBc": f"{S['b_c']:.0f}",
        "calSpeedRatio": pct(S["speed_ratio"]), "calSpeedRatioTop": pct(S["speed_ratio_top"]),
        "calRent": pct(S["vhat"] / S["F"] - 1),
        "calDissip": pct(S["dissipated_frictionless"]),
        "calPlannerGain": pct(S["planner_gain"], 1), "calSubsidy": pct(S["subsidy"], 1),
        "calNewborn": pct(S["newborn_margin"], 2),
        "calEta": f"{S['eta']:.3f}", "calEtaNoF": f"{m.fr['eta_F_no_founders']:.3f}",
        "calLambda": f"{S['lam_eq']:.3f}", "calLambdaFixed": f"{S['lam_fixed']:.3f}",
        "calEpsW": f"{abs(S['eps_w']):.3f}", "calEpsV": f"{S['eps_v']:.3f}", "calEpsG": f"{S['eps_gamma']:.3f}",
        "calInvEpsW": f"{1 / abs(S['eps_w']):.0f}",
        "calLambdaLo": f"{1 - 0.27 * abs(S['eps_w']):.3f}",
        "calBeyond": pct(S["beyond_true"]), "calZhat": f"{S['moments_pooled']['zhat']:.2f}",
        "calLin": pct(S["moments_pooled"]["lin"], 1),
        "calQone": f"${100 * S['moments_pooled']['quint'][0]:.1f}$\\%",
        "calQfour": f"${100 * S['moments_pooled']['quint'][3]:+.1f}$\\%",
        "calQfive": f"${100 * S['moments_pooled']['quint'][4]:+.1f}$\\%",
        "calWQfour": f"${100 * Sw['moments_within']['quint'][3]:+.1f}$\\%",
        "calWQfive": f"${100 * Sw['moments_within']['quint'][4]:+.1f}$\\%",
        "calXm": f"{S['x_m']:.2f}", "calRm": pct(S["r_m"], 1), "calReadCS": pct(S["read_cs"], 1),
        "calTimeValueRead": f"{S['time_value_read']:.3f}",
        "calReadsPerPaper": f"{S['reads_per_paper']:.1f}",
        "calCoverage": f"{S['coverage_m']:.2f}", "calPhi": f"{S['phi_m']:.2f}",
        "calAware": pct(S["aware_m"]), "calErosion": pct(S["erosion_m"], 1),
        "calShareGrowth": pct(S["phases"]["growth"]),
        "calShareCons": pct(S["phases"]["consolidation"]),
        "calShareMature": pct(S["phases"]["mature"]),
        "calOmega": f"{om0:.2g}", "calThetaF": f"{S['theta_F']:.2f}", "calThetaFlc": f"{S['theta_F_lc']:.2f}",
        "calUFratio": f"{S['U_F'] / S['F']:.3f}",
        # incidence (stationary, exact)
        "calWriteFixed": f"{100 * by['Faster production']['fixed']:.1f}",
        "calWriteOut": f"{100 * by['Faster production'][f'long_{om0}']:.1f}",
        "calReadFixed": f"{100 * by['Faster reading']['fixed']:.2f}",
        "calReadOutNoF": f"{100 * by['Faster reading']['long_0.0']:.2f}",
        "calReadOut": f"{100 * by['Faster reading'][f'long_{om0}']:.2f}",
        "calReadPassNoF": pct(share(by["Faster reading"], "long_0.0")),
        "calReadPass": pct(share(by["Faster reading"], f"long_{om0}")),
        "calLearnPass": pct(share(by["Better learning"], f"long_{om0}")),
        "calCatchFixed": f"{100 * by['Faster catch-up']['fixed']:.2f}",
        "calCatchPassNoF": pct(share(by["Faster catch-up"], "long_0.0")),
        "calCatchPass": pct(share(by["Faster catch-up"], f"long_{om0}")),
        "calPerHourNoF": f"{per_hour(by['Faster reading'], 'long_0.0'):.1f}",
        "calPerHour": f"{per_hour(by['Faster reading'], f'long_{om0}'):.1f}",
        "calReadDens": f"{100 * by['Faster reading']['dP']:.1f}",
        "calCatchDens": f"{100 * by['Faster catch-up']['dP']:.1f}",
        "calCatchGstar": f"{abs(100 * by['Faster catch-up']['dgstar']):.1f}",
        "calSearchAware": f"{100 * by['Better search']['daware']:.1f}",
        # formulas
        "calLamF": f"{fb['lam_F']:.2f}", "calLamFnoF": f"{f0['lam_F']:.2f}",
        "calPassLearnFormula": pct(fb["pass_learning"]), "calPassCatchFormula": pct(fb["pass_catchup"]),
        "calPassNoF": pct(f0["pass_learning"]),
        "calPassLearnHi": pct(fm[0.75]["pass_learning"]), "calPassCatchHi": pct(fm[0.75]["pass_catchup"]),
        # life cycle
        "calLCReadF": pct(lcshare(lb, "Faster reading", "dlnF")),
        "calLCCatchF": pct(lcshare(lb, "Faster catch-up", "dlnF")),
        "calLCReadY": pct(lcshare(lb, "Faster reading")), "calLCReadYnoF": pct(lcshare(l0, "Faster reading")),
        "calLCCatchY": pct(lcshare(lb, "Faster catch-up")), "calLCCatchYnoF": pct(lcshare(l0, "Faster catch-up")),
        "calLCLearnY": pct(lcshare(lb, "Better learning")),
        "calLCProdY": pct(lcshare(lb, "Faster production")),
        "calLCRent": pct(lb["Y0"] / m.lc.F - 1, 1),
        # readership
        "calAudWrite": f"{100 * au['Faster production']['alpha_1.0']:.1f}",
        "calAudRead": f"{100 * au['Faster reading']['alpha_1.0']:.1f}",
        "calAudWriteHalf": f"{100 * au['Faster production']['alpha_0.5']:.1f}",
        "calAudReadHalf": f"{100 * au['Faster reading']['alpha_0.5']:.1f}",
        "calReadsDrop": pct(-au["Faster production"]["dlnA"]),
        # life cycles
        "calGstar": pct(S["g_star"]), "calGstarExact": pct(S["g_star"], 1),
        "calGstarInf": pct(S["g_star_inf"], 1),
        "calGtakeoff": pct(S["g_takeoff"], 1),
        "calStallGap": f"{S['stall_gap']:.3f}",
        "calChiBar": f"{S['chi_bar']:.2f}",
        "calChiHalf": f"{S['stock_learning'][1]['chi']:.2f}",
        "calGtakeoffChiHalf": pct(S["stock_learning"][1]["g_takeoff"], 1),
        "calGstarChiHalf": pct(S["stock_learning"][1]["g_star"], 1),
        "calGtakeoffChiHi": pct(S["stock_learning"][2]["g_takeoff"], 1),
        "calGstarChiHi": pct(S["stock_learning"][2]["g_star"], 1),
        "calGstarRE": pct(S["g_star_re"], 1), "calGstarREbound": pct(S["g_star_re_bound"], 1),
        "calCompBias": f"{abs(S['comp_bias_T7']):.3f}",
        "calREmyopicThree": f"{[r for r in RE['rows'] if r['g'] == 2.0][0]['myopic']:.0f}",
        "calRErationalThree": f"{[r for r in RE['rows'] if r['g'] == 2.0][0]['rational']:.0f}",
        "calPfric": f"{S['P_frictionless']:.0f}", "calPw": f"{S['Pw']:.1f}",
        "calWPlannerGain": pct(Sw["planner_gain"], 1),
        "calWGmed": pct(Sw["g_med"], 0),
        # social value
        "calTauCrit": f"{SB['tau_crit_uniform']:.1f}", "calThetaLamPm": f"{SB['theta_lam_Pm']:.3f}",
        "calPsocialFull": f"{SB['proportional'][-1]['P_social']:.1f}",
        "calPsocialZero": f"{SB['proportional'][0]['P_social']:.1f}",
        "calGainTransfer": pct(SB["proportional"][-1]["gain"], 1),
        # policy on the life cycle
        "calPolSubsidyGain": pct(PB["subsidy"]["gain"], 2), "calPolTaxGain": pct(PB["stock_tax"]["gain"], 2),
        "calPolSubsidyF": pct(PB["subsidy"]["F"] / m.lc.F - 1, 2),
        "calPolTaxPm": f"{PB['stock_tax']['Pm']:.1f}",
        "calPolSubsidyGtakeoff": pct(PB["subsidy"]["g_takeoff"], 1),
        "calPolTaxGtakeoff": pct(PB["stock_tax"]["g_takeoff"], 1),
        "calPolLFGtakeoff": pct(PB["g_takeoff"], 1),
        # closure after a production shock
        "calCloseAlpha": f"{1 + CL[0]['alpha_min']:.1f}",
        "calCloseFour": pct([c for c in CL[1:] if c["alpha"] == 3.0][0]["share_closed"], 0),
    }
    with open(NUM, "w") as f:
        f.write("% Generated by code/scripts/make_figures.py -- do not edit by hand.\n")
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


def main():
    global FIG, TAB, NUM, RES
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=RES)
    ap.add_argument("--paper", default=os.path.join(ROOT, "paper"))
    args = ap.parse_args()
    RES = args.results
    FIG, TAB = os.path.join(args.paper, "figures"), os.path.join(args.paper, "tables")
    NUM = os.path.join(args.paper, "numbers.tex")
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    cal = json.load(open(os.path.join(RES, "calibration.json")))
    calw = json.load(open(os.path.join(RES, "calibration_within.json")))
    m, mw = Model(cal), Model(calw)
    S, Sw = summary(m), summary(mw)
    S["g_takeoff"] = g_takeoff(m.grid, m.lc)
    Sw["g_takeoff"] = g_takeoff(mw.grid, mw.lc)
    Sw["g_med"] = mw.cal["theta"]["g_med"]
    gt = S["g_takeoff"]
    S["stall_gap"] = max(stall_recovery_gap(m.grid, m.lc, f * gt) for f in (0.2, 0.5, 0.9)) \
        if np.isfinite(gt) and gt > 0 else float("nan")
    S["chi_bar"] = chi_bar(m.curve, m.lc)
    S["stock_learning"] = [thresholds_stock_learning(m.curve, m.grid, m.lc, c)
                           for c in (0.0, 0.5 * S["chi_bar"], 0.9 * S["chi_bar"])]
    rows = incidence_table(m)
    FM = incidence_formulas(m)
    LC = life_cycle_incidence(m)
    AU = audience(m, rows)
    SB, RE = social_block(m), re_overshooting(m)
    PB = policy_block(m)
    CL = closure(m)
    write_json(dict(summary=S, summary_within=Sw, incidence=rows, formulas=FM, life_cycle=LC, audience=AU,
                    social=SB, rational_expectations=RE, policy=PB, closure=CL),
               os.path.join(RES, "results.json"))
    fig_payoff(m, S)
    fig_planner(m, S)
    fig_lifecycle(m, S, RE)
    fig_test4(m, S)
    fig_incidence(m, S, rows, FM, LC)
    tables(m, S, rows, Sw, FM, LC, AU, SB, RE, PB)
    numbers(m, S, rows, Sw, FM, LC, AU, SB, RE, PB, CL)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in S.items()
                      if not isinstance(v, (dict, list))}, indent=1, default=float))


if __name__ == "__main__":
    main()
