"""Build every figure, table and number macro used in the paper.

Reads results/calibration.json (baseline) and results/calibration_within.json
(robustness) and writes to paper/figures, paper/tables and paper/numbers.tex.
"""
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

from ftmodel.calibration import TARGETS, evaluate, Fixed  # noqa: E402
from ftmodel.equilibrium import Stationary  # noqa: E402
from ftmodel.lifecycle import (g_types, simulate, stopping_density_asymptotic,  # noqa: E402
                               test4_moments)
from ftmodel.report import Model, ai_counterfactuals, summary, write_json  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG = os.path.join(ROOT, "paper", "figures")
TAB = os.path.join(ROOT, "paper", "tables")
RES = os.path.join(ROOT, "code", "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# Reference categorical palette (light mode), fixed order; text in neutral inks.
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
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


def fig_payoff(m, S):
    cv = m.curve
    P = np.exp(np.linspace(np.log(0.5), np.log(1.3 * S["P_frictionless"]), 600))
    v, w = cv.v(P), cv.v(P) * np.exp(-m.kappa * P)
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.6))
    ax = axs[0]
    ax.plot(P, v, color=C1, label="incumbent $v(P)$")
    ax.plot(P, w, color=C2, label="newcomer $w(P)$")
    ax.axhline(S["F"], color=INK2, lw=0.8, ls="--")
    lo_y = 0.7
    ax.set_ylim(lo_y, 1.32)
    label(ax, 0.55, S["F"] + 0.012, "frontier $F$")
    for x, s_ in [(S["Pm"], "$P^m$"), (S["Phat"], "$\\hat P$"), (S["P_frictionless"], "$P^*_{\\kappa=0}$")]:
        ax.axvline(x, color=GRID, lw=0.8)
        label(ax, x * 1.06, lo_y + 0.015, s_)
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
    s = Stationary(cv, m.economy())
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
    ew, D = abs(S["eps_w"]), S["D"]
    E = eta / (eta * D + ew)
    ax.plot(eta, E, color=C1, label="volume elasticity $\\mathcal{E}$")
    ax.plot(eta, eta * ew / (eta * D + ew), color=C2, label="$1-\\lambda=\\mathcal{E}|\\varepsilon_w|$")
    ax.axhspan(0.04, 0.27, color=INK2, alpha=0.08, lw=0)
    label(ax, 0.42, 0.14, "arXiv range\nof $\\mathcal{E}$")
    ax.axvline(S["eta"], color=GRID, lw=0.8)
    ax.set_xlabel("frontier elasticity $\\eta$")
    ax.set_title("(b) volume elasticity and stepping on toes")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_planner_volume.pdf"))
    plt.close(fig)


def fig_lifecycle(m, S):
    gc, lc = m.grid, m.lc
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ax = axs[0]
    gs, _ = g_types(lc.with_(n_g=7))
    t = np.linspace(0, 40, 2000)
    cols = plt.cm.Blues(np.linspace(0.45, 0.95, len(gs)))
    for g, col in zip(gs, cols):
        one = lc.with_(g_med=g, g_disp=0.0, n_g=1, zeta_sd=0.0, n_z=1, horizon=40.0, dt=0.02)
        cs = simulate(gc, one)
        ax.plot(cs.age[0], np.exp(cs.lnP[0]), color=col, lw=1.2)
        if g in (gs[0], gs[-1]):
            k = int(np.argmax(cs.lnP[0]))
            label(ax, cs.age[0][k] + 0.6, np.exp(cs.lnP[0][k]) * 1.02, f"$g={g:.2f}$", color=INK2)
    ax.axhline(S["Phat"], color=C2, lw=0.9, ls="--")
    label(ax, 1, S["Phat"] * 1.05, "peak $\\hat P$", color=INK2)
    ax.axhline(S["Pm"], color=INK2, lw=0.8, ls=":")
    label(ax, 25, S["Pm"] * 0.86, "mature $P^m$", color=INK2)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(0, 40)
    ax.set_xlabel("topic age (years)")
    ax.set_ylabel("density $P$")
    ax.set_title("(a) life cycles by diffusion rate $g$")
    ax = axs[1]
    g = np.exp(np.linspace(np.log(0.01), np.log(5), 200))
    Pc = [stopping_density_asymptotic(gc, lc, x) for x in g]
    ax.plot(g, Pc, color=C1, label="$P_c^\\infty(g)$")
    ax.axhline(S["Phat"], color=C2, lw=0.9, ls="--")
    ax.axhline(S["Pm"], color=INK2, lw=0.8, ls=":")
    ax.axhline(S["P_frictionless"], color=INK2, lw=0.8, ls="-.")
    ax.axvline(S["g_star"], color=GRID, lw=0.8)
    label(ax, S["g_star"] * 1.1, S["Pm"] * 1.05, "$g^*$")
    label(ax, 0.012, S["Phat"] * 1.04, "$\\hat P$")
    label(ax, 0.012, S["Pm"] * 1.04, "$P^m$")
    label(ax, 0.012, S["P_frictionless"] * 0.8, "$\\bar P_F$ (frictionless)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("diffusion rate $g$ (per year)")
    ax.set_title("(b) stopping density")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_lifecycle.pdf"))
    plt.close(fig)


def fig_test4(m, S):
    cs = simulate(m.grid, m.lc.with_(n_g=41, n_z=41, dt=0.02))
    w = cs.w / cs.w.sum()
    mu = (w * cs.lnP).sum()
    sd = np.sqrt((w * (cs.lnP - mu) ** 2).sum())
    z = ((cs.lnP - mu) / sd).ravel()
    y = cs.y.ravel()
    wf = w.ravel()
    ph = cs.phase.ravel()
    bins = np.linspace(-2.5, 2.5, 21)
    idx = np.digitize(z, bins)
    xs, ys = [], []
    for k in range(1, len(bins)):
        sel = idx == k
        if wf[sel].sum() > 2e-3:
            xs.append(0.5 * (bins[k - 1] + bins[k]))
            ys.append((wf[sel] * y[sel]).sum() / wf[sel].sum())
    # normalise to the middle fifth
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
    ax.plot(zq[[0, 2, 3, 4]], [-0.08, 0.0, 0.05, 0.01], ls="none", marker="s", ms=5, color=C2, label="arXiv fifths")
    ax.plot(zq, qm, ls="none", marker="D", ms=5, mfc="white", mew=1.4, color=C1, label="model fifths")
    ax.axvline(S["moments_pooled"]["zhat"], color=GRID, lw=0.8)
    label(ax, S["moments_pooled"]["zhat"] + 0.05, -0.27, "peak of $v$")
    ax.set_ylim(-0.3, 0.12)
    ax.set_xlabel("topic density (SD from mean)")
    ax.set_ylabel("log returns vs middle fifth")
    ax.set_title("(a) returns and density")
    ax.legend(fontsize=7, loc="lower right")
    ax = axs[1]
    hb = np.linspace(-3, 3.5, 40)
    bottom = np.zeros(len(hb) - 1)
    for k, (name, col) in enumerate([("growth", C3), ("consolidation", C2), ("mature", C1)]):
        hist, _ = np.histogram(z[ph == k], bins=hb, weights=wf[ph == k])
        ax.bar(hb[:-1], hist, width=np.diff(hb) * 0.92, bottom=bottom, align="edge", color=col,
               edgecolor="white", linewidth=0.6, label=name)
        bottom += hist
    ax.axvline(S["moments_pooled"]["zhat"], color=INK2, lw=0.9, ls="--")
    label(ax, S["moments_pooled"]["zhat"] + 0.08, bottom.max() * 0.85,
          f"{100 * S['beyond_true']:.0f}% of researcher-years\nbeyond the peak")
    ax.set_xlabel("topic density (SD from mean)")
    ax.set_ylabel("share of researcher-years")
    ax.set_title("(b) where researchers work, by life-cycle phase")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_test4.pdf"))
    plt.close(fig)


def fig_ai(rows):
    metrics = [("dP", "density $P^*$"), ("dM", "number of topics"), ("dpapers", "papers per researcher"),
               ("dypr", "output per researcher"), ("dflag", "flagged overlap"), ("daware", "aware share (pp)")]
    names = [r["name"] for r in rows]
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.2), sharey=True)
    for ax, (key, title) in zip(axs.ravel(), metrics):
        vals = np.array([r[key] for r in rows]) * 100
        yy = np.arange(len(rows))[::-1]
        ax.barh(yy, vals, color=[C1 if x >= 0 else C2 for x in vals], height=0.62)
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_title(title)
        ax.set_yticks(yy)
        ax.set_yticklabels(names)
        ax.grid(axis="y", visible=False)
        lim = max(1.0, np.abs(vals).max() * 1.35)
        ax.set_xlim(-lim, lim)
        for yv, x in zip(yy, vals):
            if abs(x) >= 0.05:
                ax.annotate(f"{x:+.1f}", (x, yv), xytext=(3 if x >= 0 else -3, 0), textcoords="offset points",
                            ha="left" if x >= 0 else "right", va="center", fontsize=6.5, color=INK2)
    for ax in axs[1]:
        ax.set_xlabel("percent (log points)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_ai_channels.pdf"))
    plt.close(fig)


def fmt(x, d=2):
    return f"{x:.{d}f}"


def tables(m, S, rows, Sw, mw):
    cal = m.cal
    th = cal["theta"]
    # ---- calibration table
    cb = cal["credit"]
    fx = cal["fixed"]
    lines = [r"\begin{tabular}{@{}llll@{}}", r"\toprule",
             r"Parameter & Meaning & Value & Source \\", r"\midrule",
             r"\multicolumn{4}{@{}l}{\emph{Set externally}} \\",
             rf"$b$ & papers read per year at full-time reading & {fx['b']:.0f} & normalisation (unit of $P$) \\",
             rf"$\lambda$ & credit lost by an unaware duplicate & {cb['lam']:.2f} & Fact 4 ($-19\%$) \\",
             rf"$\kappa_p$ & credit lost by pivoting & {cal['credit_targets']['pivot_cost']:.2f} & assumption \\",
             rf"$\Delta/\lambda$, $q$ & detection value; share proceeding & {cb['dl']:.2f}, {cb['q']:.2f} & Fact 4 ($+29\%$), \cref{{lem:credit}} \\",
             rf"$\sigma$ & search precision & {fx['sigma']:.1f} & aware share (Fact 5), see text \\",
             rf"$\delta$ & decay of the stock's relevance & {fx['delta']:.2f} & half-life $\approx$ 2.8 years \\",
             rf"$\rho$ & rate of leaving a topic & {fx['rho']:.2f} & ten-year topic tenure \\",
             rf"$\xi$ & topic obsolescence & {fx['xi']:.2f} & 25-year topic life \\",
             rf"$P_0$ & flow of a newborn topic & {fx['P0']:.0f} & a few papers per year \\",
             r"\multicolumn{4}{@{}l}{\emph{Calibrated (\cref{tab:fit})}} \\",
             rf"$\beta$, $\gamma$ & learning efficiency, curvature & {th['beta']:.2f}, {th['gamma']:.3f} & hump in returns \\",
             rf"$\theta\lambda$ & erosion per earlier paper & {th['Theta']:.4f} & fall at the top \\",
             rf"$F$ & frontier payoff & {th['F']:.3f} & sparsest fifth \\",
             rf"$b_c$ & catch-up reading speed & {th['b_c']:.0f} & premium, share beyond peak \\",
             rf"$g_{{\rm med}}$, $s_g$ & diffusion rate: median, log-sd & {th['g_med']:.3f}, {th['g_disp']:.2f} & share beyond peak, fifths \\",
             rf"$s_\zeta$ & log-sd of topic fertility & {th['zeta_sd']:.3f} & gradient across topics \\",
             rf"$\eta$ & frontier elasticity & {cal['eta']:.4f} & $\mathcal{{E}}=0.2$ (Fact 1), \eqref{{eq:eta}} \\",
             r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "calibration.tex"), "w").write("\n".join(lines) + "\n")
    # ---- fit table
    mp, mwm = S["moments_pooled"], Sw["moments_within"]
    def row(name, tgt, a, b, d=3, targeted=True):
        t = "" if tgt is None else fmt(tgt, d)
        mark = "" if targeted else r"$^\dagger$"
        return rf"{name}{mark} & {t} & {fmt(a, d)} & {fmt(b, d)} \\"
    q = mp["quint"]
    qw = mwm["quint"]
    lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
             r"Moment & arXiv & Baseline & Within-topic \\", r"\midrule",
             row("Gradient per SD of density", 0.067, mp["lin"], mwm["lin"]),
             row("Sparsest fifth vs.\\ middle", -0.08, q[0], qw[0]),
             row("Fourth fifth vs.\\ middle", 0.05, q[3], qw[3]),
             row("Densest fifth vs.\\ middle", 0.01, q[4], qw[4]),
             row("Researcher-years beyond the peak", 0.12, S["beyond_true"], Sw["beyond_true"]),
             row("Incumbents' log premium $\\kappa P^m$", 0.15, S["premium"], Sw["premium"]),
             row("Peak of $v$ (SD above mean)", 1.0, mp["zhat"], mwm["zhat"], 2, targeted=False),
             row("Quadratic: linear term", 0.094, mp["b1"], mwm["b1"], 3, targeted=False),
             row("Quadratic: squared term", -0.047, mp["b2"], mwm["b2"], 3, targeted=False),
             row("Aware share of flagged overlap", 0.38, S["aware_m"], Sw["aware_m"], 2, targeted=False),
             r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "fit.tex"), "w").write("\n".join(lines) + "\n")
    # ---- AI table
    lines = [r"\small", r"\begin{tabular}{@{}llrrrrrrr@{}}", r"\toprule",
             r"Channel & Shock & Density & Topics & Papers/ & Output/ & Flagged & Aware & $g^*$ \\",
             r" & & & & researcher & researcher & overlap & share (pp) & (pp) \\", r"\midrule"]
    for r in rows:
        lines.append(rf"{r['name']} & {r['channel']} & {100*r['dP']:+.1f} & {100*r['dM']:+.1f} & {100*r['dpapers']:+.1f} & "
                     rf"{100*r['dypr']:+.1f} & {100*r['dflag']:+.1f} & {100*r['daware']:+.1f} & {100*r['dgstar']:+.1f} \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\par\smallskip\footnotesize Long-run stationary effects in log points ($\times100$) of a 30\% shock "
              r"(catch-up speed doubled, pivot cost halved). Density is $P^*$; output per researcher is $aF(M^*)$. "
              r"The flagged overlap rate is relative to its baseline level, which is a normalisation (\cref{sec:facts}). "
              r"$g^*$ is the overshooting threshold of \cref{prop:toofast}."]
    open(os.path.join(TAB, "ai_channels.tex"), "w").write("\n".join(lines) + "\n")
    # ---- welfare table
    lines = [r"\begin{tabular}{@{}lr@{}}", r"\toprule", r"Object & Value \\", r"\midrule",
             rf"Peak of the payoff curve $\hat P$ (papers per year) & {S['Phat']:.1f} \\",
             rf"Gain from cumulative knowledge at the peak, $\hat v/v(P_0)-1$ & {100*(S['vhat']/m.curve.v(m.lc.P0)[0]-1):.1f}\% \\",
             rf"Mature density $P^m$ & {S['Pm']:.1f} \\",
             rf"Catch-up time $T=S/b_c$ (years) & {S['T']:.2f} \\",
             rf"Incumbents' premium over the frontier, $e^{{\kappa P^m}}-1$ & {100*(np.exp(S['premium'])-1):.1f}\% \\",
             rf"Rent at the peak, $\hat v/F-1$ & {100*(S['vhat']/S['F']-1):.1f}\% \\",
             rf"Frictionless benchmark: rent dissipated, $1-F/\hat v$ & {100*S['dissipated_frictionless']:.1f}\% \\",
             rf"Catch-up economy: planner's gain, $\hat w/F-1$ & {100*S['planner_gain']:.1f}\% \\",
             rf"Optimal frontier subsidy, $s^o/F$ & {100*S['subsidy']:.1f}\% \\",
             rf"Overshooting threshold $g^*$ (per year) & {S['g_star']:.3f} \\",
             rf"Slope of the entrant payoff curve $|\varepsilon_w(P^m)|$ & {abs(S['eps_w']):.3f} \\",
             rf"Frontier elasticity $\eta$ (from $\mathcal{{E}}=0.2$) & {S['eta']:.4f} \\",
             rf"Stepping-on-toes parameter $\lambda=1-\mathcal{{E}}|\varepsilon_w|$ & {S['lam']:.3f} \\",
             r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "welfare.tex"), "w").write("\n".join(lines) + "\n")


def numbers(m, S, rows):
    vP0 = float(m.curve.v(m.lc.P0)[0])
    by = {r["name"]: r for r in rows}
    macros = {
        "calPhat": f"{S['Phat']:.0f}", "calPm": f"{S['Pm']:.1f}", "calvhat": f"{S['vhat']:.2f}",
        "calF": f"{S['F']:.3f}", "calT": f"{S['T']:.1f}", "calPremium": f"{100*(np.exp(S['premium'])-1):.0f}\\%",
        "calRent": f"{100*(S['vhat']/S['F']-1):.0f}\\%",
        "calGainPeak": f"{100*(S['vhat']/vP0-1):.0f}\\%",
        "calDissip": f"{100*S['dissipated_frictionless']:.0f}\\%",
        "calPlannerGain": f"{100*S['planner_gain']:.1f}\\%", "calSubsidy": f"{100*S['subsidy']:.1f}\\%",
        "calGstar": f"{100*S['g_star']:.0f}\\%", "calEta": f"{S['eta']:.3f}", "calLambda": f"{S['lam']:.3f}",
        "calEpsW": f"{abs(S['eps_w']):.3f}",
        "calLambdaLo": f"{1 - 0.27 * abs(S['eps_w']):.3f}", "calLambdaHi": f"{1 - 0.04 * abs(S['eps_w']):.3f}", "calEpsV": f"{S['eps_v']:.3f}",
        "calBeyond": f"{100*S['beyond_true']:.0f}\\%", "calZhat": f"{S['moments_pooled']['zhat']:.2f}",
        "calLin": f"{100*S['moments_pooled']['lin']:.1f}\\%",
        "calQone": f"{100*S['moments_pooled']['quint'][0]:.1f}\\%",
        "calQfour": f"{100*S['moments_pooled']['quint'][3]:.1f}\\%",
        "calQfive": f"{100*S['moments_pooled']['quint'][4]:.1f}\\%",
        "calXm": f"{S['x_m']:.2f}", "calCoverage": f"{S['coverage_m']:.2f}", "calPhi": f"{S['phi_m']:.2f}",
        "calAware": f"{100*S['aware_m']:.0f}\\%", "calErosion": f"{100*S['erosion_m']:.1f}\\%",
        "calShareGrowth": f"{100*S['phases']['growth']:.0f}\\%",
        "calShareCons": f"{100*S['phases']['consolidation']:.0f}\\%",
        "calShareMature": f"{100*S['phases']['mature']:.0f}\\%",
        "calWriteOut": f"{100*by['Faster drafting']['dypr']:.1f}",
        "calReadDens": f"{100*by['Faster reading']['dP']:.1f}",
        "calReadOut": f"{100*by['Faster reading']['dypr']:.2f}",
        "calCatchDens": f"{100*by['Faster catch-up']['dP']:.1f}",
        "calCatchGstar": f"{100*by['Faster catch-up']['dgstar']:.1f}",
    }
    with open(os.path.join(ROOT, "paper", "numbers.tex"), "w") as f:
        f.write("% Generated by code/scripts/make_figures.py -- do not edit by hand.\n")
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


def main():
    cal = json.load(open(os.path.join(RES, "calibration.json")))
    calw = json.load(open(os.path.join(RES, "calibration_within.json")))
    m, mw = Model(cal), Model(calw)
    S, Sw = summary(m), summary(mw)
    rows = ai_counterfactuals(m)
    write_json(dict(summary=S, summary_within=Sw, ai=rows), os.path.join(RES, "results.json"))
    fig_payoff(m, S)
    fig_planner(m, S)
    fig_lifecycle(m, S)
    fig_test4(m, S)
    fig_ai(rows)
    tables(m, S, rows, Sw, mw)
    numbers(m, S, rows)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in S.items()
                      if not isinstance(v, dict)}, indent=1))
    for r in rows:
        print(r["name"], {k: round(100 * v, 2) for k, v in r.items() if k not in ("name", "channel")})


if __name__ == "__main__":
    main()
