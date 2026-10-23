"""Write the empirical tables (paper/tables/emp_*.tex), number macros
(paper/numbers_emp.tex) and copy empirical figures into paper/figures."""
import json
import shutil

R = json.load(open("empirics/results/rd.json"))
K = json.load(open("empirics/results/rank_design.json"))
E = json.load(open("empirics/results/entry_policies.json"))
C = json.load(open("empirics/results/cap.json"))
L = json.load(open("empirics/results/llm_authors.json"))


def bs(r, scale=1.0, d=3):
    return f"${scale * r['b']:+.{d}f}$ & $({scale * r['se']:.{d}f})$"


def pair(v, scale=1.0, d=3):
    return f"${scale * v[0]:+.{d}f}$ & $({scale * v[1]:.{d}f})$"


# ---------------------------------------------------------------- listing table
rd, rk = R["h60_d5"], K["batch_fe"]
rows = [
    r"\begin{tabular}{@{}lrrrl@{}}", r"\toprule",
    r"Outcome & Estimate & (SE) & Mean & $N$ \\", r"\midrule",
    r"\multicolumn{5}{@{}l}{\emph{A. Discontinuity at the daily cutoff: submitted just after (top of next listing) vs.\ just before (end of current listing)}} \\",
]
for key, lab in (("pos_frac", "Position in listing (0 = top, 1 = bottom)"),
                 ("top5", "Among the first five papers"),
                 ("n_authors", "Number of authors (balance)"),
                 ("abs_len", "Abstract length, words (balance)"),
                 ("log_prior", "log(1 + submitter's prior submissions) (balance)"),
                 ("lcites_ins", "log(1 + citations), HEP (INSPIRE)"),
                 ("lcites", "log(1 + citations), all fields (Semantic Scholar)")):
    r = rd[key]
    rows.append(rf"{lab} & {bs(r)} & {r['mean_before']:.3f} & {r['n']:,} \\")
rows.append(r"\midrule")
rows.append(r"\multicolumn{5}{@{}l}{\emph{B. Listing rank in high-energy physics: effect of log position, holding submission time fixed}} \\")
for key, lab in (("lcites", "log(1 + citations)"),
                 ("cite_pct", "Citation percentile in category-year"),
                 ("N50", r"Later overlapping papers by others, $\tau_{50}$"),
                 ("A50", r"\quad of which cite the focal paper (aware)"),
                 ("U50", r"\quad of which do not (unaware)"),
                 ("N75", r"Later overlapping papers by others, $\tau_{75}$"),
                 ("A75", r"\quad aware"),
                 ("U75", r"\quad unaware"),
                 ("S50", r"Later overlapping papers by the same authors, $\tau_{50}$")):
    r = rk[key]
    rows.append(rf"{lab} & {bs(r)} & {r['mean']:.3f} & {r['n']:,} \\")
rows += [r"\bottomrule", r"\end{tabular}"]
open("paper/tables/emp_listing.tex", "w").write("\n".join(rows) + "\n")

# ---------------------------------------------------------------- policy table
en, cs, ll = E["endorsement"], E["cs_reviews"], L
dr = cs["dose_response"]
rows = [
    r"\begin{tabular}{@{}lrrl@{}}", r"\toprule",
    r"Design and outcome & Estimate & (SE) & Pre-period mean \\", r"\midrule",
    r"\multicolumn{4}{@{}l}{\emph{A. Endorsement tightening (entry cost): Mathematics 10 Dec 2025, all sections 21 Jan 2026}} \\",
    rf"First-time submitter share, Mathematics, vs.\ year before & {pair(en['new_sub']['yoy_did_math'])} & {en['new_sub']['pre_mean_math']:.3f} \\",
    rf"First-time submitter share, other sections, vs.\ year before & {pair(en['new_sub']['yoy_did_other'])} & {en['new_sub']['pre_mean_other']:.3f} \\",
    rf"First-time submitter share, Mathematics vs.\ others, 10 Dec--20 Jan & {pair(en['new_sub']['staggered_math_vs_other'])} & \\",
    rf"Submitters with no prior arXiv authorship, Mathematics & {pair(en['no_track']['yoy_did_math'])} & {en['no_track']['pre_mean_math']:.3f} \\",
    rf"Submitters with no prior arXiv authorship, other sections & {pair(en['no_track']['yoy_did_other'])} & {en['no_track']['pre_mean_other']:.3f} \\",
    rf"log number of first-time submitters, other sections & {pair(en['lnew']['yoy_did_other'])} & \\",
    r"\midrule",
    r"\multicolumn{4}{@{}l}{\emph{B. Peer review required for CS review and position papers (catch-up cost), 31 Oct 2025}} \\",
    rf"Review-type share of papers, CS vs.\ other archives & {pair(cs['did'])} & {cs['cs_pre_mean_2025_jan_oct']:.3f} \\",
    rf"\quad with a CS-specific linear trend & {pair(cs['did_with_cs_trend'])} & \\",
    rf"\quad leakage: non-CS primary cross-listed to CS & {pair(cs['leakage_noncs_primary_crosslisted_cs'])} & \\",
    rf"Within CS: change in review share on pre-period review share & {pair(dr['first_stage_d_rev_on_expo'], d=2)} & {dr['expo_mean']:.3f} \\",
    rf"\quad placebo, one year earlier & {pair(dr['first_stage_placebo_prev_year'], d=2)} & \\",
    rf"Within CS: change in first-time-author share on review share & {pair(dr['reduced_form_first_author_share'], d=2)} & \\",
    rf"\quad placebo, one year earlier & {pair(dr['reduced_form_placebo_prev_year'], d=2)} & \\",
    r"\midrule",
    r"\multicolumn{4}{@{}l}{\emph{C. Two submissions per submitter per month (writing cap), 1 Oct 2026}} \\",
    rf"Third-or-later submissions in days 1--15, single-field names: Jan--Sep 2026 & ${100*C['excess_clean_2026_jan_sep']:.2f}\%$ & & \\",
    rf"\quad October 2026 (residual: shared names and proceedings accounts) & ${100*C['excess_clean_2026_oct']:.2f}\%$ & & \\",
    rf"log papers per weekday, Oct 1--15 vs.\ Sep 1--15: 2026 & ${C['dlog_oct_sep_2026']:+.3f}$ & & \\",
    rf"\quad same change in 2024 and 2025 & ${C['dlog_oct_sep_2024']:+.3f}$, ${C['dlog_oct_sep_2025']:+.3f}$ & & \\",
    rf"Category dose-response: change on exposure (vs.\ 2025) & {pair([C['dose_response']['b_expo_DiD'], C['dose_response']['se_DiD']], d=2)} & {C['dose_response']['mean_expo']:.3f} \\",
    r"\midrule",
    r"\multicolumn{4}{@{}l}{\emph{D. Author-level adoption of LLM-style writing (drafting speed)}} \\",
    rf"log(1 + papers per half-year), adopters vs.\ others, 2024--26, all fields & {pair(ll['all_fields']['did_2024_26'])} & {ll['all_fields']['mean_papers_pre_adopters']:.2f} \\",
    rf"\quad fields where AI is not the subject (no cs, stat, eess) & {pair(ll['non_ai_fields']['did_2024_26'])} & {ll['non_ai_fields']['mean_papers_pre_adopters']:.2f} \\",
    r"\bottomrule", r"\end{tabular}"]
open("paper/tables/emp_policies.tex", "w").write("\n".join(rows) + "\n")

# ---------------------------------------------------------------- macros
def pct(x, d=1):
    return f"{100*x:.{d}f}\\%"

m = {
    "empRDtopfive": f"{rd['top5']['b']:.2f}",
    "empRDtopfiveBefore": f"{rd['top5']['mean_before']:.2f}",
    "empRDcitesIns": f"{rd['lcites_ins']['b']:+.3f}", "empRDcitesInsSE": f"{rd['lcites_ins']['se']:.3f}",
    "empRDprior": f"{rd['log_prior']['b']:+.3f}", "empRDpriorSE": f"{rd['log_prior']['se']:.3f}",
    "empRankN": f"{K['n']:,}".replace(",", "{,}"),
    "empRankSD": f"{K['resid_sd_lpos']:.2f}",
    "empRankCites": f"{abs(rk['lcites']['b']):.3f}", "empRankCitesSE": f"{rk['lcites']['se']:.3f}",
    "empRankCitesMain": f"{abs(K['main']['lcites']['b']):.3f}", "empRankCitesMainSE": f"{K['main']['lcites']['se']:.3f}",
    "empRankUsevfive": f"{rk['U75']['b']:+.4f}", "empRankUsevfiveSE": f"{rk['U75']['se']:.4f}",
    "empRankUsevfiveMean": f"{rk['U75']['mean']:.3f}",
    "empRankUsevfiveUB": pct((rk['U75']['b'] + 1.96 * rk['U75']['se']) / rk['U75']['mean']),
    "empRankUsevfiveMDE": pct(2.8 * rk['U75']['se'] / rk['U75']['mean'], 0),
    "empRankAsevfive": f"{rk['A75']['b']:+.4f}", "empRankAsevfiveSE": f"{rk['A75']['se']:.4f}",
    "empRankSelf": f"{rk['S50']['b']:+.3f}", "empRankSelfSE": f"{rk['S50']['se']:.3f}",
    "empAwareInsFifty": pct(K["aware_share_nonself"]["50"], 0),
    "empAwareInsSevfive": pct(K["aware_share_nonself"]["75"], 0),
    "empAwareInsNinety": pct(K["aware_share_nonself"]["90"], 0),
    "empSelfShare": pct(K["self_share_50"], 0),
    "empEndMath": f"{abs(100*en['new_sub']['yoy_did_math'][0]):.1f}", "empEndMathSE": f"{100*en['new_sub']['yoy_did_math'][1]:.1f}",
    "empEndOther": f"{abs(100*en['new_sub']['yoy_did_other'][0]):.1f}", "empEndOtherSE": f"{100*en['new_sub']['yoy_did_other'][1]:.1f}",
    "empEndStag": f"{abs(100*en['new_sub']['staggered_math_vs_other'][0]):.1f}", "empEndStagSE": f"{100*en['new_sub']['staggered_math_vs_other'][1]:.1f}",
    "empEndOtherRel": pct(-en['new_sub']['yoy_did_other'][0] / en['new_sub']['pre_mean_other'], 0),
    "empEndMathRel": pct(-en['new_sub']['yoy_did_math'][0] / en['new_sub']['pre_mean_math'], 0),
    "empEndNoTrackOtherRel": pct(-en['no_track']['yoy_did_other'][0] / en['no_track']['pre_mean_other'], 0),
    "empCSdid": f"{abs(100*cs['did'][0]):.2f}", "empCSdidSE": f"{100*cs['did'][1]:.2f}",
    "empCSpre": f"{100*cs['cs_pre_mean_2025_jan_oct']:.1f}", "empCSpost": f"{100*cs['cs_post_mean_2025_11_2026_10']:.1f}",
    "empCSperMonthPre": f"{cs['cs_reviews_per_month_pre']:.0f}", "empCSperMonthPost": f"{cs['cs_reviews_per_month_post']:.0f}",
    "empCSdoseFS": f"{abs(dr['first_stage_d_rev_on_expo'][0]):.2f}", "empCSdoseFSSE": f"{dr['first_stage_d_rev_on_expo'][1]:.2f}",
    "empCSdoseRF": f"{dr['reduced_form_first_author_share'][0]:.2f}", "empCSdoseRFSE": f"{dr['reduced_form_first_author_share'][1]:.2f}",
    "empCSdoseRFpl": f"{dr['reduced_form_placebo_prev_year'][0]:.2f}", "empCSdoseRFplSE": f"{dr['reduced_form_placebo_prev_year'][1]:.2f}",
    "empCapPre": pct(C['excess_clean_2026_jan_sep'], 2), "empCapOct": pct(C['excess_clean_2026_oct'], 2),
    "empCapRemoved": pct(C['excess_clean_2026_jan_sep'] - C['excess_clean_2026_oct'], 1),
    "empCapVol": f"{C['dlog_oct_sep_2026']:.3f}",
    "empCapVolPrev": f"{(C['dlog_oct_sep_2024'] + C['dlog_oct_sep_2025']) / 2:.3f}",
    "empLLMall": f"{ll['all_fields']['did_2024_26'][0]:.3f}", "empLLMallSE": f"{ll['all_fields']['did_2024_26'][1]:.3f}",
    "empLLMnonai": f"{ll['non_ai_fields']['did_2024_26'][0]:.3f}", "empLLMnonaiSE": f"{ll['non_ai_fields']['did_2024_26'][1]:.3f}",
    "empLLMallPct": pct(ll['all_fields']['did_2024_26'][0] * (1 + ll['all_fields']['mean_papers_pre_adopters'])
                        / ll['all_fields']['mean_papers_pre_adopters'], 0),
    "empLLMn": f"{ll['all_fields']['n_submitters']:,}".replace(",", "{,}"),
    "empLLMadopters": f"{ll['n_adopters']:,}".replace(",", "{,}"),
    "empLLMpre": f"{ll['all_fields']['max_abs_pre']:.3f}",
}
with open("paper/numbers_emp.tex", "w") as f:
    f.write("% Generated by empirics/12_tables.py -- do not edit by hand.\n")
    for k, v in m.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
for fn in ("rd_listing", "rank_design", "endorsement", "cs_reviews", "llm_authors"):
    shutil.copy(f"empirics/figures/{fn}.pdf", f"paper/figures/emp_{fn}.pdf")
print(open("paper/numbers_emp.tex").read())
