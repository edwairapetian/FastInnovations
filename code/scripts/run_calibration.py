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

# Outcome of ftmodel.calibration.calibrate: a scan of 2048 Sobol points, Nelder-Mead from the
# best three and from the best points of earlier calibration rounds, selected on the fine grid.
STORED = {
    "pooled": dict(beta=41.670204640794815, gamma=0.15892230181229888, Theta=0.0022051927403809826, F=1.7719723396629103, g_med=0.1754282236992788, g_disp=0.30500168478496453, zeta_sd=0.0664067433552573, b_c=28.62981381245267, sigma=1.5430366809195095),
    "within": dict(beta=33.535472849974575, gamma=0.17703507338435204, Theta=0.0009851964668296444, F=1.8164037956957853, g_med=0.07243572400229235, g_disp=0.7260824284867119, zeta_sd=0.20731119134274156, b_c=28.136787915381646, sigma=1.821495646633994),
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
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    for est in ("pooled", "within"):
        if args.only and est != args.only:
            continue
        fx = Fixed(estimator=est)
        if args.search:
            theta, _ = calibrate(fx)
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


if __name__ == "__main__":
    main()
