"""Numerical checks of the baseline model in baseline/baseline.tex, with its tables and chart.

Checks, each against a simulation or a finite difference:
  1. where she works: the shares pi_jk and the expected credit rate W_j   (Proposition 1)
  2. the credit rate of the chosen line has the same distribution in every field (Proposition 2)
  3. deeper or wider: the home share rises with pace a/b, is unchanged by balanced progress,
     and falls with AI scouting alpha                                       (Proposition 3)
  4. time shares: d ln W / d ln a = average writing share, d ln W / d ln b = 1 - that (Proposition 4)
  5. the equilibrium exists and is unique: the fixed-point iteration converges to the same point
     from random starts, and the flows add up; the two neutralities          (Proposition 5)
  6. counterfactuals from observed shares alone (exact changes)             (Section 4)

It also writes the two tables of the worked example (tables/*.tex) and the figure
(figures/mechanism.pdf).

Run:  python3 baseline/check_baseline.py
"""
import os
from math import gamma

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)


def q(u, sig=1.0):
    """Gap efficiency: what understanding u lets her find and trust (log-concave, q(1) = 1)."""
    return u ** sig


def tau(a, b, m, u):
    """Time per paper: writing plus reading what she does not know of the paper's literature."""
    return 1.0 / a + m * (1.0 - u) / b


def shares(G, u, m, a, b, alpha, theta, sig=1.0):
    ut = u + alpha * (1.0 - u)                        # effective understanding when scouting
    c = q(ut, sig) / tau(a, b, m[None, :], u)         # credit rate per unit of raw gap value
    T = G[None, :] * c ** theta
    Phi = T.sum(axis=1)
    return T / Phi[:, None], Phi, c


def equilibrium(A, N, u, m, a, b, alpha, theta, eta, P0=None, tol=1e-13):
    """Stationary equilibrium by iterating the map of Proposition 5 (a contraction for eta > -1/2)."""
    K = len(A)
    P = np.ones(K) if P0 is None else P0.copy()
    ut = u + alpha * (1.0 - u)
    t = tau(a, b, m[None, :], u)
    c = q(ut) / t
    for it in range(100000):
        Phi = (A[None, :] * P[None, :] ** (-eta) * c ** theta).sum(axis=1)
        Pn = (A * (N[:, None] * c ** theta / t / Phi[:, None]).sum(axis=0)) ** (1.0 / (1.0 + eta))
        if np.max(np.abs(np.log(Pn) - np.log(P))) < tol:
            return Pn, it
        P = Pn
    raise RuntimeError("no convergence")


# ---------------------------------------------------------------- a small world: 5 fields on a line
K = 5
dist = np.abs(np.arange(K)[:, None] - np.arange(K)[None, :])
u = 0.9 * np.exp(-0.45 * dist)                         # she knows 90% at home, less farther away
m = np.array([40.0, 35.0, 45.0, 40.0, 30.0])
A = np.array([1.0, 1.4, 0.8, 1.2, 1.0])
N = np.array([1.0, 2.0, 1.5, 1.0, 0.5])
a, b, alpha, theta, eta = 3.0, 100.0, 0.0, 4.0, 0.5
ok = True


def report(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS  " if cond else "FAIL  ") + name + ("   " + detail if detail else ""))


# 1 and 2: simulate the choice of one researcher (home = field 2) with given gaps
G = A * np.array([0.8, 1.1, 0.9, 1.3, 1.0])
pi, Phi, c = shares(G, u, m, a, b, alpha, theta)
j, n = 2, 2_000_000
Z = (G[None, :] / rng.exponential(size=(n, K))) ** (1.0 / theta)      # Frechet(G_k, theta)
V = c[j][None, :] * Z
k_star = V.argmax(axis=1)
Vmax = V.max(axis=1)
emp = np.bincount(k_star, minlength=K) / n
report("1a shares pi_jk", np.max(np.abs(emp - pi[j])) < 2e-3, f"max abs error {np.max(np.abs(emp - pi[j])):.4f}")
W = gamma(1 - 1 / theta) * Phi[j] ** (1 / theta)
report("1b expected credit rate W_j", abs(Vmax.mean() / W - 1) < 3e-3, f"simulated {Vmax.mean():.4f}, formula {W:.4f}")
cond_means = np.array([Vmax[k_star == k].mean() for k in range(K)])
report("2a credit rate of the chosen line is the same in every field",
       np.max(np.abs(cond_means / W - 1)) < 1e-2, "means by field / W: " + np.array2string(cond_means / W, precision=3))
t = tau(a, b, m, u[j])
per_paper = np.array([(Vmax[k_star == k] * t[k]).mean() for k in range(K)])
report("2b credit per paper equals W_j * tau_jk (more away than at home)",
       np.max(np.abs(per_paper / (W * t) - 1)) < 1e-2 and per_paper[j] == per_paper.min(),
       "relative to home: " + np.array2string(per_paper / per_paper[j], precision=2))

# 2c: with thinner tails (log-normal values) selection offsets only part of the cost of distance
Zl = np.exp(rng.normal(0.0, 0.5, size=(n, K))) * G[None, :] ** (1.0 / theta)
Vl = c[j][None, :] * Zl
kl = Vl.argmax(axis=1)
ml = np.array([Vl.max(axis=1)[kl == k].mean() for k in range(K)])
report("2c log-normal values: chosen lines away from home pay less than chosen lines at home",
       np.all(np.delete(ml, j) < ml[j]), "means relative to home: " + np.array2string(ml / ml[j], precision=3))

# 3: deeper or wider (given gaps)
home = lambda a_, b_, al_: shares(G, u, m, a_, b_, al_, theta)[0][j, j]
report("3a home share rises with writing speed (reading fixed)", home(6.0, b, 0.0) > home(3.0, b, 0.0),
       f"{home(3.0, b, 0.0):.3f} -> {home(6.0, b, 0.0):.3f}")
report("3b home share falls with reading speed (writing fixed)", home(a, 200.0, 0.0) < home(a, 100.0, 0.0),
       f"{home(a, 100.0, 0.0):.3f} -> {home(a, 200.0, 0.0):.3f}")
report("3c balanced progress leaves the home share unchanged", abs(home(6.0, 200.0, 0.0) - home(3.0, 100.0, 0.0)) < 1e-12)
report("3d home share falls with AI scouting", home(a, b, 0.5) < home(a, b, 0.0), f"{home(a, b, 0.0):.3f} -> {home(a, b, 0.5):.3f}")

# 4: time shares
h = 1e-6
lnW = lambda a_, b_: np.log(shares(G, u, m, a_, b_, alpha, theta)[1][j]) / theta
ea = (lnW(a * (1 + h), b) - lnW(a * (1 - h), b)) / (2 * h)
eb = (lnW(a, b * (1 + h)) - lnW(a, b * (1 - h))) / (2 * h)
omega = (1 / a) / tau(a, b, m, u[j])
wbar = (pi[j] * omega).sum()
report("4a d ln W / d ln a = average writing share of time", abs(ea - wbar) < 1e-6, f"{ea:.5f} vs {wbar:.5f}")
report("4b d ln W / d ln b = one minus that", abs(eb - (1 - wbar)) < 1e-6, f"{eb:.5f} vs {1 - wbar:.5f}")
big = 1.0 / tau(1e9, b, m, u[j])
report("4c with instant writing the paper rate is b / (m (1 - u))", np.allclose(big, b / (m * (1 - u[j])), rtol=1e-6))



def papers(a_, b_, al_):
    """Papers per unit of time of a researcher at home in j, given the gaps."""
    p = shares(G, u, m, a_, b_, al_, theta)[0][j]
    return (p / tau(a_, b_, m, u[j])).sum()


var = (pi[j] * (omega - wbar) ** 2).sum()
ra = (np.log(papers(a * (1 + h), b, alpha)) - np.log(papers(a * (1 - h), b, alpha))) / (2 * h)
rb = (np.log(papers(a, b * (1 + h), alpha)) - np.log(papers(a, b * (1 - h), alpha))) / (2 * h)
report("4d d ln R / d ln a = wbar + (1 + theta) Var / wbar  (papers rise by more than credit)",
       abs(ra - (wbar + (1 + theta) * var / wbar)) < 1e-6 and ra > ea, f"papers {ra:.4f}, credit {ea:.4f}")
report("4e d ln R / d ln b = (1 - wbar) - (1 + theta) Var / wbar  (papers rise by less than credit)",
       abs(rb - ((1 - wbar) - (1 + theta) * var / wbar)) < 1e-6 and rb < eb, f"papers {rb:.4f}, credit {eb:.4f}")
al0 = 0.2                                              # scouting: derivatives at an interior alpha
pi_s, Phi_s, _ = shares(G, u, m, a, b, al0, theta)
ut_s = u[j] + al0 * (1 - u[j])
g = (1 - u[j]) / ut_s                                  # (q'/q)(u~)(1 - u) for q(u) = u
om = (1 / a) / tau(a, b, m, u[j])
wb_s = (pi_s[j] * om).sum()
lnW_al = lambda al_: np.log(shares(G, u, m, a, b, al_, theta)[1][j]) / theta
wal = (lnW_al(al0 + h) - lnW_al(al0 - h)) / (2 * h)
ral = (np.log(papers(a, b, al0 + h)) - np.log(papers(a, b, al0 - h))) / (2 * h)
cov = (pi_s[j] * (om - wb_s) * (g - (pi_s[j] * g).sum())).sum()
report("4f d ln W / d alpha = sum_k pi_jk (q'/q)(1 - u)", abs(wal - (pi_s[j] * g).sum()) < 1e-6, f"{wal:.5f}")
report("4g d ln R / d alpha = theta Cov(omega, g) / wbar  (scouting lowers papers, raises credit)",
       abs(ral - theta * cov / wb_s) < 1e-6 and ral < 0 < wal, f"papers {ral:.4f}, credit {wal:.4f}")

# 5: equilibrium
P_star, its = equilibrium(A, N, u, m, a, b, alpha, theta, eta)
starts = [equilibrium(A, N, u, m, a, b, alpha, theta, eta, P0=np.exp(rng.normal(0, 3, K)))[0] for _ in range(20)]
report("5a iteration converges to one point from 20 random starts",
       max(np.max(np.abs(np.log(s) - np.log(P_star))) for s in starts) < 1e-9, f"{its} iterations from P = 1")
pi_e, _, _ = shares(A * P_star ** (-eta), u, m, a, b, alpha, theta)
flows = (N[:, None] * pi_e / tau(a, b, m[None, :], u)).sum(axis=0)
report("5b flows add up: P_k = sum_j N_j pi_jk / tau_jk", np.allclose(flows, P_star, rtol=1e-9))
P_fast, _ = equilibrium(A, N, u, m, 2 * a, b, alpha, theta, eta)
pi_f, _, _ = shares(A * P_fast ** (-eta), u, m, 2 * a, b, alpha, theta)
report("5c in equilibrium, faster writing raises every home share",
       np.all(np.diag(pi_f) > np.diag(pi_e)), "home shares " + np.array2string(np.diag(pi_e), precision=3) + " -> " + np.array2string(np.diag(pi_f), precision=3))

# 5d: mild "papers open gaps" (eta < 0) is covered by the same argument as long as eta > -1/2
P_neg, its_neg = equilibrium(A, N, u, m, a, b, alpha, theta, -0.3)
starts = [equilibrium(A, N, u, m, a, b, alpha, theta, -0.3, P0=np.exp(rng.normal(0, 3, K)))[0] for _ in range(20)]
report("5d unique also when papers open gaps (eta = -0.3)",
       max(np.max(np.abs(np.log(s_) - np.log(P_neg))) for s_ in starts) < 1e-9, f"{its_neg} iterations")

# 5e: the two neutralities (exact in equilibrium, any world)
lam = 1.7
P_bal, _ = equilibrium(A, N, u, m, lam * a, lam * b, alpha, theta, eta)
pi_b, Phi_b, _ = shares(A * P_bal ** (-eta), u, m, lam * a, lam * b, alpha, theta)
_, Phi_e, _ = shares(A * P_star ** (-eta), u, m, a, b, alpha, theta)
report("5e balanced progress: papers x lam, shares unchanged, credit x lam^(1 - eta/theta)",
       np.allclose(P_bal, lam * P_star, rtol=1e-9) and np.allclose(pi_b, pi_e, atol=1e-10)
       and np.allclose((Phi_b / Phi_e) ** (1 / theta), lam ** (1 - eta / theta), rtol=1e-9))
P_sc, _ = equilibrium(A, lam * N, u, m, a, b, alpha, theta, eta)
pi_sc, Phi_sc, _ = shares(A * P_sc ** (-eta), u, m, a, b, alpha, theta)
report("5f more researchers everywhere: papers x lam, shares unchanged, credit x lam^(-eta/theta)",
       np.allclose(P_sc, lam * P_star, rtol=1e-9) and np.allclose(pi_sc, pi_e, atol=1e-10)
       and np.allclose((Phi_sc / Phi_e) ** (1 / theta), lam ** (-eta / theta), rtol=1e-9))

# 5g: a symmetric world (fields on a circle): equal flows, and shares that do not depend on crowding
Ks = 6
cd = np.minimum(np.abs(np.arange(Ks)[:, None] - np.arange(Ks)[None, :]), Ks - np.abs(np.arange(Ks)[:, None] - np.arange(Ks)[None, :]))
us, ms, As, Ns = 0.9 * np.exp(-0.45 * cd), np.full(Ks, 40.0), np.ones(Ks), np.ones(Ks)
sym = []
for eta_ in (0.0, 0.5, 2.0):
    Ps, _ = equilibrium(As, Ns, us, ms, a, b, alpha, theta, eta_)
    sym.append((Ps, shares(As * Ps ** (-eta_), us, ms, a, b, alpha, theta)[0]))
report("5g symmetric world: equal flows; shares independent of crowding",
       all(np.allclose(P_, P_[0], rtol=1e-10) for P_, _ in sym) and all(np.allclose(sym[0][1], p_, atol=1e-10) for _, p_ in sym))

# 5i: symmetric world: the equilibrium elasticity of credit is the one given the gaps,
#     less eta/theta times the elasticity of papers
eta_s = 0.5


def sym_credit(a_):
    Ps, _ = equilibrium(As, Ns, us, ms, a_, b, alpha, theta, eta_s)
    return np.log(shares(As * Ps ** (-eta_s), us, ms, a_, b, alpha, theta)[1][0]) / theta


pis = shares(As, us, ms, a, b, alpha, theta)[0][0]
oms = (1 / a) / tau(a, b, ms, us[0])
wbs = (pis * oms).sum()
vs = (pis * (oms - wbs) ** 2).sum()
hh = 1e-5
eq_el = (sym_credit(a * (1 + hh)) - sym_credit(a * (1 - hh))) / (2 * hh)
report("5i symmetric world: credit elasticity = wbar - (eta/theta) x papers elasticity",
       abs(eq_el - (wbs - eta_s / theta * (wbs + (1 + theta) * vs / wbs))) < 1e-6, f"{eq_el:.5f}")

# 5h: scale that is uneven pushes work outwards (numerical illustration, not a theorem)
N2 = N.copy()
N2[j] *= 2
P_n, _ = equilibrium(A, N2, u, m, a, b, alpha, theta, eta)
pi_n, _, _ = shares(A * P_n ** (-eta), u, m, a, b, alpha, theta)
report("5h doubling the researchers of one field lowers its home share", pi_n[j, j] < pi_e[j, j],
       f"{pi_e[j, j]:.4f} -> {pi_n[j, j]:.4f}")

# 6: exact changes: the new equilibrium from the old shares, the writing shares, theta and eta alone
a1, b1, al1 = 2.0 * a, 1.5 * b, 0.3
t0 = tau(a, b, m[None, :], u)
om0 = (1 / a) / t0                                      # writing shares before
X0 = N[:, None] * pi_e / t0                             # papers by home and field before
that = om0 / (a1 / a) + (1 - om0) / (b1 / b)            # change in time per paper
qhat = q(u + al1 * (1 - u)) / q(u + alpha * (1 - u))    # change in gap efficiency
Phat = np.ones(K)
for _ in range(100000):
    num = pi_e * Phat[None, :] ** (-eta) * (qhat / that) ** theta
    pi_new = num / num.sum(axis=1, keepdims=True)
    Pn = ((X0 / P_star[None, :]) * (pi_new / pi_e) / that).sum(axis=0)
    if np.max(np.abs(np.log(Pn) - np.log(Phat))) < 1e-13:
        break
    Phat = Pn
What = num.sum(axis=1) ** (1 / theta)
P_dir, _ = equilibrium(A, N, u, m, a1, b1, al1, theta, eta)
pi_dir, Phi_dir, _ = shares(A * P_dir ** (-eta), u, m, a1, b1, al1, theta)
report("6  exact changes reproduce the new equilibrium (shares, flows, credit)",
       np.allclose(pi_new, pi_dir, atol=1e-9) and np.allclose(Phat, P_dir / P_star, rtol=1e-9)
       and np.allclose(What, (Phi_dir / Phi_e) ** (1 / theta), rtol=1e-9))


# ---------------------------------------------------------------- the worked example (tables in the note)
# One field at home, two next door, twenty far away; equal abundance of gaps; q(u) = u.
# Numbers chosen so that the split of her papers is close to the one on arXiv in 2022
# (60% at home, 20% near, 20% far).  An illustration, not an estimate.
EX = dict(a=3.0, b=100.0, m=40.0, theta=4.0, u=np.array([0.9, 0.7, 0.5]), n=np.array([1.0, 2.0, 20.0]))


def example(a_, b_, al_):
    """Shares of time and of papers (home, next door, far away), papers a year and Phi."""
    uu, nn, th = EX["u"], EX["n"], EX["theta"]
    t = tau(a_, b_, EX["m"], uu)
    c = q(uu + al_ * (1 - uu)) / t
    w = nn * c ** th
    pi_ = w / w.sum()
    R = (pi_ / t).sum()
    return dict(t=t, pi=pi_, R=R, Phi=w.sum(), s=(pi_ / t) / R)


def pct(x, digits=0):
    return f"{100 * x:.{digits}f}" + r"\%"


def tables():
    os.makedirs(os.path.join(HERE, "tables"), exist_ok=True)
    uu = EX["u"]
    e0 = example(EX["a"], EX["b"], 0.0)
    d = (e0["t"] / e0["t"][0]) * (q(uu[0]) / q(uu))
    rows = [
        (r"Number of such fields", [f"{x:.0f}" for x in EX["n"]]),
        (r"Share of a paper's literature she knows, $u_{jk}$", [pct(x) for x in uu]),
        (r"Papers she must read for one paper, $\ell_{jk}$", [f"{EX['m'] * (1 - x):.0f}" for x in uu]),
        (r"Years per paper, $\tau_{jk}$", [f"{x:.2f}" for x in e0["t"]]),
        (r"Papers a year on a line, $1/\tau_{jk}$", [f"{1 / x:.1f}" for x in e0["t"]]),
        (r"Share of her time on the line spent writing, $\omega_{jk}$", [pct((1 / EX["a"]) / x) for x in e0["t"]]),
        (r"Gap efficiency, $q(\tilde u_{jk})$", [f"{q(x):.1f}" for x in uu]),
        (r"Distance from home, $d_{jk}$", [f"{x:.2f}" for x in d]),
        (r"Share of her time, all such fields together", [pct(x) for x in e0["pi"]]),
        (r"Share of her papers, all such fields together", [pct(x) for x in e0["s"]]),
        (r"Credit per paper on a chosen line, home $=1$", [f"{x:.2f}" for x in e0["t"] / e0["t"][0]]),
    ]
    with open(os.path.join(HERE, "tables", "example.tex"), "w") as f:
        for name, vals in rows:
            f.write(name + " & " + " & ".join(vals) + r" \\" + "\n")
    scen = [("Today", (3.0, 100.0, 0.0)), ("Writing twice as fast", (6.0, 100.0, 0.0)),
            ("Reading twice as fast", (3.0, 200.0, 0.0)), (r"AI scouts, $\alpha=\tfrac12$", (3.0, 100.0, 0.5)),
            ("Both twice as fast", (6.0, 200.0, 0.0))]

    def chg(x):
        return "--" if abs(x) < 1e-9 else (f"$+{100 * x:.0f}" if x > 0 else f"$-{100 * abs(x):.0f}") + r"\%$"

    print("\nworked example")
    with open(os.path.join(HERE, "tables", "scenarios.tex"), "w") as f:
        for name, par in scen:
            e = example(*par)
            dR, dW = e["R"] / e0["R"] - 1, (e["Phi"] / e0["Phi"]) ** (1 / EX["theta"]) - 1
            f.write(" & ".join([name, pct(e["s"][0]), pct(e["s"][1]), pct(e["s"][2]), f"{e['R']:.1f}", chg(dR), chg(dW)]) + r" \\" + "\n")
            print(f"  {name:36s} time {np.round(e['pi'], 3)}  papers {np.round(e['s'], 3)}  R {e['R']:.3f} ({100 * dR:+.1f}%)  credit {100 * dW:+.1f}%")
    print("  distance", np.round(d, 3), " time distance", np.round(e0["t"] / e0["t"][0], 3),
          " limit as writing becomes free", np.round((1 - uu) / (1 - uu[0]), 2))
    om = (1 / EX["a"]) / e0["t"]
    print(f"  share of all her time spent writing: {(e0['pi'] * om).sum():.4f}")
    # Section 4.4: relative to home, any change in capabilities that raises the near share raises the far share by more
    worst, n_up = np.inf, 0
    for ah in np.geomspace(0.25, 8, 25):
        for bh in np.geomspace(0.25, 8, 25):
            for al in np.linspace(0.0, 0.9, 19):
                e = example(EX["a"] * ah, EX["b"] * bh, al)
                dn = np.log(e["s"][1] / e["s"][0]) - np.log(e0["s"][1] / e0["s"][0])
                df = np.log(e["s"][2] / e["s"][0]) - np.log(e0["s"][2] / e0["s"][0])
                if dn > 1e-12:
                    n_up += 1
                    worst = min(worst, df - dn)
    report("7  worked example: whenever capabilities raise the near share (relative to home), the far share rises by more",
           worst > 0, f"{n_up} cases, smallest gap {worst:.3f} log points")
    print("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED")


# ---------------------------------------------------------------- the figure
def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
    INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
    plt.rcParams.update({
        "font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 8.5, "axes.titlesize": 9,
        "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "xtick.major.size": 0, "ytick.major.size": 0, "lines.linewidth": 1.8, "legend.frameon": False,
        "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03})
    aa = np.geomspace(1, 300, 200)
    mm, bb = EX["m"], EX["b"]
    uh, un, uf = EX["u"]                               # understanding at home, next door, far away
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.1, 2.5), gridspec_kw={"wspace": 0.36})
    for uu, col, lab in ((uh, BLUE, "At home (knows 90%)"), (un, ORANGE, "Next door (knows 70%)"), (uf, AQUA, "Far away (knows 50%)")):
        y = 1.0 / tau(aa, bb, mm, uu)
        ax1.plot(aa, y, color=col, label=lab)
        below = uu == uf                                  # the lowest line is labelled from below
        ax1.annotate(f"limit {bb / (mm * (1 - uu)):.0f}", (aa[-1], y[-1]), xytext=(-4, -5 if below else 5),
                     textcoords="offset points", ha="right", va="top" if below else "bottom", color=INK2, fontsize=8)
    ax1.set_xscale("log")
    ax1.set_xticks([1, 3, 10, 30, 100, 300])
    ax1.set_xticklabels(["1", "3", "10", "30", "100", "300"])
    ax1.set_xlim(1, 300)
    ax1.set_ylim(0, 27)
    ax1.set_xlabel("writing speed a (papers a year if she only wrote)")
    ax1.set_ylabel("papers a year on the line")
    ax1.set_title("Reading becomes the bottleneck", loc="left", fontweight="bold", color=INK, pad=6)
    ax1.legend(loc="upper left", handlelength=1.4, borderaxespad=0.2)

    home = lambda a_, b_, al_: np.array([example(x, y, al_)["s"][0] for x, y in zip(a_, np.broadcast_to(b_, a_.shape))])
    ax2.plot(aa, 100 * home(aa, bb, 0.0), color=BLUE, label="Reading fixed")
    ax2.plot(aa, 100 * home(aa, bb * aa / EX["a"], 0.0), color=ORANGE, label="Reading keeps pace")
    ax2.plot(aa, 100 * home(aa, bb, 0.5), color=AQUA, label="Reading fixed, AI scouts")
    ax2.set_xscale("log")
    ax2.set_xticks([1, 3, 10, 30, 100, 300])
    ax2.set_xticklabels(["1", "3", "10", "30", "100", "300"])
    ax2.set_xlim(1, 300)
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("writing speed a")
    ax2.set_ylabel("share of her papers written at home (%)")
    ax2.set_title("Deeper or wider", loc="left", fontweight="bold", color=INK, pad=6)
    ax2.legend(loc="lower right", handlelength=1.4, borderaxespad=0.2)
    fig.savefig(os.path.join(HERE, "figures", "mechanism.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    tables()
    figure()
