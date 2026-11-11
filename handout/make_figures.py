"""Charts for the handout (handout/figures/*.pdf).

Every number is computed from files in this repository:
  data/arxiv_meta.parquet            the 2026 rise in papers per author
  code/results/calibration.json      the calibrated model (via code/ftmodel)
  code/results/results.json          the model's fit to the returns profile
The pass-through chart uses the values established by the review and listed in
docs/paper_diagnosis.md (Section 4.2).

Run:  python3 handout/make_figures.py            (reuses figures/surge_data.json if present)
      python3 handout/make_figures.py --recompute
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FIG = os.path.join(HERE, "figures")
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, os.path.join(ROOT, "empirics"))

# Reference categorical palette (fixed order) and neutral inks; text never wears a series colour.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

# A TrueType Helvetica-compatible face, so that the PDFs embed clean TrueType fonts.
plt.rcParams.update({
    "font.family": ["Liberation Sans", "DejaVu Sans"],
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "lines.linewidth": 1.8, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


def title(ax, text):
    ax.set_title(text, loc="left", fontweight="bold", color=INK, pad=6)


def dot(ax, x, y, color, size=5.5):
    ax.plot([x], [y], "o", ms=size, color=color, mec="white", mew=1.2, zorder=5)


# ------------------------------------------------------------------ 1. the 2026 surge
def surge_data(recompute=False):
    cache = os.path.join(FIG, "surge_data.json")
    if os.path.exists(cache) and not recompute:
        return json.load(open(cache))
    import pandas as pd
    from names import author_keys
    df = pd.read_parquet(os.path.join(ROOT, "data", "arxiv_meta.parquet"),
                         columns=["authors", "n_authors", "v1"])
    df["y"] = df.v1.dt.year
    df["m"] = df.v1.dt.month
    df = df[df.y >= 2019]
    monthly = df[(df.y < 2026) | (df.m <= 9)].groupby(["y", "m"]).size()
    d = df[(df.m <= 9) & (df.n_authors <= 30)].copy()
    d["keys"] = [author_keys(a, full=True) for a in d.authors]
    d["w"] = 1.0 / d["keys"].str.len().clip(lower=1)
    e = d[["y", "keys", "w"]].explode("keys").dropna()
    e = e[e["keys"].str.contains(" ") & (e["keys"].str.split().str[0].str.len() > 1)]  # full first names
    g = e.groupby(["y", "keys"]).agg(n=("w", "size"), frac=("w", "sum")).reset_index()
    out = {"years": [], "frac": [], "whole": [], "cap3": [], "papers": [], "monthly": []}
    for y, s in g.groupby("y"):
        out["years"].append(int(y))
        out["frac"].append(float(s.frac.mean()))
        out["whole"].append(float(s.n.mean()))
        out["cap3"].append(float(s.n.clip(upper=3).mean()))
        out["papers"].append(int(((df.y == y) & (df.m <= 9)).sum()))
    out["monthly"] = [[int(y), int(m), int(n)] for (y, m), n in monthly.items()]
    json.dump(out, open(cache, "w"))
    return out


def fig_surge(recompute=False):
    s = surge_data(recompute)
    yrs = np.array(s["years"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.1, 2.45), gridspec_kw={"wspace": 0.32})
    # (a) submissions per month
    mm = np.array(s["monthly"])
    t = mm[:, 0] + (mm[:, 1] - 0.5) / 12
    ax1.plot(t, mm[:, 2] / 1000, color=BLUE)
    ax1.set_ylim(0, None)
    ax1.set_xlim(2019, 2027.2)
    ax1.set_xticks(range(2019, 2027))
    ax1.set_xticklabels([str(y) if y % 2 else "" for y in range(2019, 2027)])
    ax1.set_ylabel("thousand papers per month")
    title(ax1, "New arXiv submissions per month")
    dot(ax1, t[-1], mm[-1, 2] / 1000, BLUE)
    ax1.annotate(f"Sept. 2026\n{mm[-1, 2] / 1000:.0f} thousand", (t[-1], mm[-1, 2] / 1000),
                 xytext=(-6, -4), textcoords="offset points", ha="right", va="top", color=INK2, fontsize=8)
    # (b) papers per author, Jan-Sep, index 2019 = 100
    fr = 100 * np.array(s["frac"]) / s["frac"][0]
    c3 = 100 * np.array(s["cap3"]) / s["cap3"][0]
    ax2.plot(yrs, fr, color=BLUE, label="Fractional papers per author")
    ax2.plot(yrs, c3, color=ORANGE, label="Papers per author, at most three counted")
    for series, col in ((fr, BLUE), (c3, ORANGE)):
        dot(ax2, yrs[-1], series[-1], col)
    ax2.annotate(f"{fr[-1]:.0f}", (yrs[-1], fr[-1]), xytext=(7, 0), textcoords="offset points",
                 va="center", color=INK, fontsize=8)
    ax2.annotate(f"{c3[-1]:.0f}", (yrs[-1], c3[-1]), xytext=(7, 0), textcoords="offset points",
                 va="center", color=INK, fontsize=8)
    ax2.set_xlim(2018.7, 2027)
    ax2.set_xticks(yrs)
    ax2.set_xticklabels([str(y) if y % 2 else "" for y in yrs])
    ax2.set_ylim(84, 113)
    ax2.set_yticks([90, 95, 100, 105, 110])
    ax2.set_ylabel("index, 2019 = 100")
    title(ax2, "Papers per author, January to September")
    ax2.legend(loc="lower left", handlelength=1.4, borderaxespad=0.2)
    fig.savefig(os.path.join(FIG, "surge.pdf"))
    plt.close(fig)
    return s


# ------------------------------------------------------------------ 2. pass-through under three accountings
def fig_passthrough():
    rows = ["Theorem 3 as stated\n(the abstract)", "The paper's own\ncount of members", "The paper's own\nlife-cycle simulation"]
    nof = [14, 53, 40]            # no founders
    fou = [74, 86.5, 63]          # founders, omega = 1/2 (85-88 for the member count)
    lab = ["74", "85–88", "63"]
    fig, ax = plt.subplots(figsize=(6.1, 2.25))
    y = np.arange(3)[::-1]
    h = 0.28
    ax.barh(y + h / 2 + 0.02, nof, height=h, color=BLUE, label="No founders")
    ax.barh(y - h / 2 - 0.02, fou, height=h, color=ORANGE, label="Founders stay (ω = ½)")
    for yi, v in zip(y, nof):
        ax.text(v + 1.2, yi + h / 2 + 0.02, f"{v}", va="center", color=INK, fontsize=8)
    for yi, v, l in zip(y, fou, lab):
        ax.text(v + 1.2, yi - h / 2 - 0.02, l, va="center", color=INK, fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(rows, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of the gain from a reading tool that survives entry (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", handlelength=1.0, handleheight=0.8, borderaxespad=0.1)
    fig.savefig(os.path.join(FIG, "passthrough.pdf"))
    plt.close(fig)


# ------------------------------------------------------------------ model helpers
def model():
    from ftmodel.report import Model
    return Model(json.load(open(os.path.join(ROOT, "code", "results", "calibration.json"))))


# ------------------------------------------------------------------ 3. a 12% production shock
def fig_shock():
    m = model()
    c, cu, Pm = m.curve, m.cu, m.mature()
    delta, rho, xi, bc = cu.delta, cu.rho, cu.xi, cu.b_c
    nu = (rho + xi) / delta

    def lnval(P, S):
        return float(c.lnv(P)[0]) - nu * np.log(1 + delta * S / (bc - P))

    a = 0.12
    S0, P1 = Pm / delta, Pm * (1 + a)
    base = lnval(Pm, S0)
    t = np.linspace(0, 6, 601)
    stock = np.array([100 * (lnval(P1, P1 / delta + (S0 - P1 / delta) * np.exp(-delta * x)) - base) for x in t])
    flow = 100 * (lnval(P1, P1 / delta) - base)
    cross = float(t[np.argmax(stock < 0)])
    tabs = np.log(1 + a) / rho

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.1, 2.45), gridspec_kw={"wspace": 0.34})
    # (a) excess flow of existing topics
    tt = np.linspace(0, 6, 601)
    stop = np.maximum(100 * ((1 + a) * np.exp(-rho * tt) - 1), 0)
    half = 100 * a * np.exp(-rho * tt)
    ax1.plot(tt, stop, color=BLUE, label="Entry stops, topics thin")
    ax1.plot(tt, half, color=ORANGE, label="Half-life of 7 years (App. B.8)")
    dot(ax1, tabs, 0, BLUE)
    ax1.annotate(f"gone after {tabs:.1f} years", (tabs, 0), xytext=(6, 9), textcoords="offset points",
                 color=INK2, fontsize=8)
    ax1.set_ylim(-0.6, 13.5)
    ax1.set_xlim(0, 6)
    ax1.set_xlabel("years after the shock")
    ax1.set_ylabel("excess flow of existing topics (%)")
    title(ax1, "How long the excess lasts")
    ax1.legend(loc="center right", handlelength=1.4, borderaxespad=0.1, bbox_to_anchor=(1.0, 0.36))
    # (b) the newcomer's payoff
    ax2.axhline(0, color=AXIS, lw=0.8)
    ax2.plot(t, stock, color=BLUE, label="Entry priced on the stock (Lemma 2)")
    ax2.plot(t, np.full_like(t, flow), color=ORANGE, label="Entry priced on the flow (App. B.8)")
    dot(ax2, 0, stock[0], BLUE)
    ax2.annotate(f"+{stock[0]:.2f} on impact", (0, stock[0]), xytext=(7, 7), textcoords="offset points",
                 color=INK2, fontsize=8)
    dot(ax2, cross, 0, BLUE)
    ax2.annotate(f"negative after {cross:.1f} years", (cross, 0), xytext=(9, 4),
                 textcoords="offset points", color=INK2, fontsize=8)
    ax2.annotate(f"{flow:.2f}".replace("-", "−"), (6, flow), xytext=(-2, 5), textcoords="offset points",
                 ha="right", color=INK2, fontsize=8)
    ax2.set_xlim(0, 6)
    ax2.set_ylim(-1.75, 0.75)
    ax2.set_yticks([-1.5, -1.0, -0.5, 0, 0.5])
    ax2.set_xlabel("years after the shock (flow held 12% higher)")
    ax2.set_ylabel("newcomer's payoff, log points")
    title(ax2, "What it does to a newcomer")
    ax2.legend(loc="lower left", handlelength=1.4, borderaxespad=0.1)
    fig.savefig(os.path.join(FIG, "shock.pdf"))
    plt.close(fig)
    return dict(impact=float(stock[0]), longrun=float(flow), cross=cross, absorb=float(tabs))


# ------------------------------------------------------------------ 4. the returns profile: data and model
def fig_fit():
    r = json.load(open(os.path.join(ROOT, "code", "results", "results.json")))
    mod = [100 * q for q in r["summary"]["moments_pooled"]["quint"]]
    data_x, data_y = [1, 3, 4, 5], [-11, 0, 2, -2]          # Fact 2; the second fifth is not reported
    fig, ax = plt.subplots(figsize=(4.3, 2.5))
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.plot(range(1, 6), mod, color=BLUE, marker="o", ms=5.5, mec="white", mew=1.2, label="Calibrated model")
    ax.plot(data_x, data_y, linestyle="none", marker="o", ms=5.5, color=ORANGE, mec="white", mew=1.2,
            label="arXiv (Fact 2)")
    ax.annotate(f"+{mod[-1]:.1f}", (5, mod[-1]), xytext=(7, 0), textcoords="offset points", va="center",
                color=INK, fontsize=8)
    ax.annotate("−2", (5, -2), xytext=(7, 0), textcoords="offset points", va="center", color=INK, fontsize=8)
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(["sparsest", "2nd", "middle", "4th", "densest"])
    ax.set_xlim(0.6, 5.7)
    ax.set_ylim(-13, 5)
    ax.set_yticks([-10, -5, 0, 5])
    ax.set_xlabel("fifth of topic density")
    ax.set_ylabel("returns relative to the middle fifth (%)")
    ax.legend(loc="lower right", handlelength=1.2, borderaxespad=0.2)
    fig.savefig(os.path.join(FIG, "fit.pdf"))
    plt.close(fig)
    return mod


# ------------------------------------------------------------------ 5. what makes a topic crowded for a newcomer
def fig_slope():
    m = model()
    c, Pm = m.curve, m.mature()
    learn = float(c.learning_term(Pm)[0]) * Pm
    over = -float(c.erosion_term(Pm)[0]) * Pm
    catch = -float(m.cu.eps_gamma(Pm))
    tot = learn + over + catch
    names = ["More to learn from", "More earlier work to miss", "More to read before joining", "Total: slope of the newcomer's payoff"]
    vals = [learn, over, catch, tot]
    cols = [BLUE, ORANGE, ORANGE, INK2]
    fig, ax = plt.subplots(figsize=(6.1, 1.75))
    y = np.arange(4)[::-1]
    ax.barh(y, vals, height=0.42, color=cols)
    ax.axvline(0, color=AXIS, lw=0.8)
    for yi, v in zip(y, vals):
        ax.text(v + (0.004 if v > 0 else -0.004), yi, f"{v:+.3f}".replace("-", "−"), va="center",
                ha="left" if v > 0 else "right", color=INK, fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, color=INK)
    ax.set_xlim(-0.15, 0.12)
    ax.set_xlabel("elasticity with respect to density, at the mature density")
    ax.grid(axis="y", visible=False)
    fig.savefig(os.path.join(FIG, "slope.pdf"))
    plt.close(fig)
    return dict(learn=learn, overlap=over, catchup=catch, total=tot)


# ------------------------------------------------------------------ 6. adopters of language models: the event study
def fig_llm():
    d = json.load(open(os.path.join(ROOT, "empirics", "results", "llm_authors.json")))["all_fields"]
    es = d["event_study"]
    keys = sorted(es.keys())
    x = {k: int(k[:4]) + (0.0 if k.endswith("H1") else 0.5) for k in keys}
    pre = [k for k in keys if k <= "2022H2"]
    post = [k for k in keys if k >= "2024H1"]
    pre_mean = float(np.mean([es[k][0] for k in pre]))
    post_mean = float(np.mean([es[k][0] for k in post]))
    fig, ax = plt.subplots(figsize=(6.1, 2.5))
    ax.axhline(0, color=AXIS, lw=0.8)
    for grp in (pre, post):
        xs = [x[k] for k in grp]
        ys = [es[k][0] for k in grp]
        se = [1.96 * es[k][1] for k in grp]
        ax.errorbar(xs, ys, yerr=se, fmt="none", ecolor=BLUE, elinewidth=1.0, alpha=0.55, zorder=3)
        ax.plot(xs, ys, linestyle="none", marker="o", ms=5.5, color=BLUE, mec="white", mew=1.2, zorder=5,
                label="Adopters relative to non-adopters" if grp is pre else None)
    ax.plot([x[pre[0]] - 0.15, x[pre[-1]] + 0.15], [pre_mean, pre_mean], color=ORANGE, label="Mean of the period")
    ax.plot([x[post[0]] - 0.15, x[post[-1]] + 0.15], [post_mean, post_mean], color=ORANGE)
    ax.annotate(f"mean before: {pre_mean:.3f}".replace("-", "−"), (2019.75, -0.008), ha="center",
                color=INK, fontsize=8)
    ax.annotate(f"mean after: {post_mean:.3f}\ndifference: +{post_mean - pre_mean:.3f}\n= the reported effect"
                .replace("-0", "−0"), (2025.2, -0.066), ha="center", va="bottom", color=INK, fontsize=8)
    ax.annotate("reference period\n(last half-year\nbefore ChatGPT)", (2022.5, 0), xytext=(0, 20),
                textcoords="offset points", ha="center", color=INK2, fontsize=7.5)
    ax.annotate("2023: adopters are\nclassified on these\npapers (not shown)", (2023.25, -0.056), ha="center",
                color=MUTED, fontsize=7.5)
    ax.axvspan(2022.95, 2023.8, color=GRID, alpha=0.6, lw=0)
    ax.set_xlim(2018.7, 2026.6)
    ax.set_xticks(range(2019, 2027))
    ax.set_ylim(-0.068, 0.05)
    ax.set_yticks([-0.06, -0.04, -0.02, 0, 0.02, 0.04])
    ax.set_ylabel("log papers per half-year, relative to 2022H2")
    ax.legend(loc="upper left", handlelength=1.4, borderaxespad=0.1)
    fig.savefig(os.path.join(FIG, "llm.pdf"))
    plt.close(fig)
    return dict(pre_mean=pre_mean, post_mean=post_mean, did=post_mean - pre_mean, reported=d["did_2024_26"][0])


if __name__ == "__main__":
    rc = "--recompute" in sys.argv
    s = fig_surge(rc)
    print("surge: years", s["years"])
    print("  fractional papers per author:", [round(v, 3) for v in s["frac"]])
    print("  capped at three:", [round(v, 3) for v in s["cap3"]])
    print("  papers Jan-Sep:", s["papers"])
    fig_passthrough()
    print("shock:", fig_shock())
    print("fit (model fifths):", [round(v, 1) for v in fig_fit()])
    print("slope:", fig_slope())
    print("llm:", fig_llm())
