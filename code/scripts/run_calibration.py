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

# Outcome of the full search (scan of 8192 Sobol points + Nelder-Mead from the best five).
STORED = {
    "pooled": dict(beta=1.89278, gamma=0.12617, Theta=0.0016, F=1.00529, g_med=0.19011,
                   g_disp=0.74744, zeta_sd=0.07278, b_c=20.0, sigma=3.0),
    "within": dict(beta=1.34959, gamma=0.25236, Theta=0.00061, F=1.24494, g_med=0.47575,
                   g_disp=0.6746, zeta_sd=0.07933, b_c=20.0, sigma=3.0),
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
