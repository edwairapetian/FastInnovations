"""Calibrate the model to the arXiv facts and write results/calibration*.json.

By default the script starts from the stored calibrated parameters and only
re-polishes them (a few minutes).  Pass --search to rerun the full Sobol scan
plus Nelder-Mead procedure of ftmodel.calibration.calibrate (about an hour).
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

from ftmodel.calibration import (NAMES, Fixed, bounds_log, calibrate, evaluate, loss,  # noqa: E402
                                 objective, save, to_calibration)

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "..", "results")

# Seeds for the search: the previous calibration (approximate catch-up) with the catch-up
# speed raised so that the exact entrant payoff curve crosses the frontier.
SEEDS = {
    "pooled": dict(beta=39.00924472058154, gamma=0.15606463422321795, Theta=0.004071334010563669, F=1.76, g_med=0.18228300634123112, g_disp=0.2896105525788173, zeta_sd=0.052097496927168405, b_c=40.0, sigma=1.578655390208228),
    "within": dict(beta=38.72673879522025, gamma=0.1864477011960337, Theta=0.0022838867701565927, F=1.95, g_med=0.062067461825457165, g_disp=0.7304193599842183, zeta_sd=0.21594324990383268, b_c=35.0, sigma=1.9255371260274847),
}

# Outcome of the search with exact catch-up and the reading-share target (28 October 2026):
# a scan of 2048 Sobol points (400 and 456 admissible), Nelder-Mead from the best three and
# from the seeds above, winner selected by the loss on the fine grid.  Both winners came from
# the seeds (fine losses 20.59 pooled, 12.55 within); the best Sobol starts reached 27.78 and
# 26.49.  Further Nelder-Mead rounds lowered the coarse-grid loss but not the fine-grid loss,
# and relaxing the lower bound on g_med to 0.01 left the within-topic solution unchanged.
STORED = {
    "pooled": dict(beta=39.46967098701789, gamma=0.15739536245589017, Theta=0.006620149684571212, F=1.7599292981135015, g_med=0.17967740850560096, g_disp=0.29438545476416333, zeta_sd=0.04270172066993396, b_c=38.63831786659503, sigma=1.60553443168833),
    "within": dict(beta=38.85867667913382, gamma=0.17979464945073645, Theta=0.003321954519848275, F=1.9338456818980552, g_med=0.05054183873531595, g_disp=0.7368535455190637, zeta_sd=0.1905327117606038, b_c=38.53418830613412, sigma=1.9160624798103323),
}


def polish(theta, fx, maxiter=800):
    lo, hi = bounds_log()
    x0 = np.log([theta[k] for k in NAMES])
    res = minimize(objective, x0, args=(fx,), method="Nelder-Mead",
                   options=dict(maxiter=maxiter, xatol=1e-4, fatol=1e-4, adaptive=True))
    x = np.clip(res.x, lo, hi) if res.fun < objective(x0, fx) else x0
    return {k: float(v) for k, v in zip(NAMES, np.exp(x))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", action="store_true", help="rerun the full search")
    ap.add_argument("--no-polish", action="store_true")
    ap.add_argument("--only", choices=["pooled", "within"], default=None)
    ap.add_argument("--n-scan", type=int, default=2048)
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    for est in ("pooled", "within"):
        if args.only and est != args.only:
            continue
        fx = Fixed(estimator=est)
        if args.search:
            theta, _ = calibrate(fx, n_scan=args.n_scan, seeds=[SEEDS[est]])
        else:
            theta = STORED[est] if args.no_polish else polish(STORED[est], fx)
        out, info = evaluate(theta, fx, fine=True)
        cal = to_calibration(theta, fx)
        cal["fit"] = dict(loss=loss(out), moments=out,
                          quadratic=dict(b1=info["raw"]["b1"], b2=info["raw"]["b2"], zpeak=info["raw"]["zpeak"]),
                          sd_lnP=info["raw"]["sd_lnP"], zhat=info["raw"]["zhat"])
        name = "calibration.json" if est == "pooled" else "calibration_within.json"
        save(cal, os.path.join(RESULTS, name))
        print(est, json.dumps({k: round(v, 4) for k, v in out.items()}), "loss", round(loss(out), 3), flush=True)
        print("   theta", {k: round(v, 5) for k, v in theta.items()}, flush=True)
        print("   theta_full", json.dumps(theta), flush=True)


if __name__ == "__main__":
    main()
