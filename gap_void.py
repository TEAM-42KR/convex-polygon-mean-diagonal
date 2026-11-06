#!/usr/bin/env python3
"""Gap-void functions H_m, m = 2..8.

H_m(x) = P(a fixed interval of length x in RP^1 contains none of the m row
lines of a Haar m x 2 frame).  Computed in double precision from the
recurrence (see hm_core.py)
    H_2(x) = (1-2x/pi)_+,  H_m(0) = 1,
    H_m'(x) = -m(m-2)/pi int_0^1 t^(m-3) H_{m-1}(atan(t tan x)) dt.
Checks:
  * H_3 against 1 - 3x/pi + 3x^2/pi^2 - 3 Li2(sin^2 x)/(2 pi^2)        (tol 1e-12)
  * H_4 against 1 - 4x/pi + (12/pi^2)(1 - x cot x)
        + (1/pi^3)[12 cot x (x^2 - 2l + l^2) - 24x - 6Q(x)],
        l = log cos x, Q(x) = int_0^x Li2(-tan^2 u) du             (tol 1e-12)
  * H_m(pi/2) = 0 (the recurrence is integrated forward from H_m(0) = 1) (tol 1e-12)
  * int_0^{pi/2} x H_m''(x) dx = 1                                     (tol 1e-10)
  * Monte Carlo: Haar frames from Gaussian rows, row line angle
    theta_k = atan2(Q_k2, Q_k1) mod pi; fraction of frames with no theta_k in
    [0, x) at x = 0.3, 0.7, 1.1, 1.4                                      (|z| < 4)
Output: out/Hm.dat with columns x, H2, ..., H8 (121 points).
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import sys
import numpy as np
import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Checker, haar_frames, write_table
from hm_core import GapVoid, graded_rule

SEED = 20261109
NMC = 400_000
ZMAX = 4.0
mp.mp.dps = 30


def H3_closed(x):
    x = mp.mpf(x)
    return 1 - 3 * x / mp.pi + 3 * x ** 2 / mp.pi ** 2 - 3 * mp.polylog(2, mp.sin(x) ** 2) / (2 * mp.pi ** 2)


def H4_closed(x):
    x = mp.mpf(x)
    l = mp.log(mp.cos(x))
    Q = mp.quad(lambda u: mp.polylog(2, -mp.tan(u) ** 2), [0, x])
    return (1 - 4 * x / mp.pi + 12 / mp.pi ** 2 * (1 - x * mp.cot(x))
            + (12 * mp.cot(x) * (x ** 2 - 2 * l + l ** 2) - 24 * x - 6 * Q) / mp.pi ** 3)


def main():
    ck = Checker("gap_void")
    G = GapVoid(8)
    xs = [0.01, 0.1, 0.3, 0.5, 0.7, np.pi / 4, 1.0, 1.2, 1.4, 1.5, 1.55, 1.57]
    e3 = max(abs(G.H(3, x) - float(H3_closed(x))) for x in xs)
    ck.check("H_3 recurrence vs closed form (12 points)", e3 < 1e-12, max_abs_err=e3, tol=1e-12)
    e4 = max(abs(G.H(4, x) - float(H4_closed(x))) for x in xs)
    ck.check("H_4 recurrence vs closed form (12 points)", e4 < 1e-12, max_abs_err=e4, tol=1e-12)
    xg = np.array([0.05, 0.4, 1.0, 1.5, 1.5707])
    e3d = max(abs(G.d2H(3, x) - float(-6 * mp.log(mp.cos(x)) / (mp.pi ** 2 * mp.sin(x) ** 2))) /
              max(1.0, abs(float(-6 * mp.log(mp.cos(x)) / (mp.pi ** 2 * mp.sin(x) ** 2)))) for x in xg)
    ck.check("H_3'' vs -6 log(cos x)/(pi^2 sin^2 x)", e3d < 1e-10, max_rel_err=e3d, tol=1e-10)
    g, w = graded_rule(0, np.pi / 2, 16, 0.3, 30, "b")
    for m in range(3, 9):
        ck.check("H_%d(pi/2) = 0" % m, abs(G.end_value[m]) < 1e-12, value=G.end_value[m], tol=1e-12)
    for m in range(3, 9):
        nm = float(np.sum(w * g * G.d2H(m, g)))
        ck.check("int x H_%d''(x) dx = 1" % m, abs(nm - 1) < 1e-10, value=nm, tol=1e-10)
    # monotonicity / range sanity on a fine grid
    xf = np.linspace(0, np.pi / 2, 2001)
    ok = all(np.all(np.diff(G.H(m, xf)) <= 1e-14) and G.H(m, xf).min() > -1e-14 for m in range(2, 9))
    ck.check("H_m decreasing and >= 0 on [0,pi/2] (m=2..8)", ok)

    rng = np.random.default_rng(SEED)
    xmc = np.array([0.3, 0.7, 1.1, 1.4])
    for m in range(2, 9):
        z = haar_frames(rng, NMC, m)
        th = np.mod(np.angle(z), np.pi)                      # row line in RP^1 = [0, pi)
        thmin = th.min(axis=1)
        for x in xmc:
            p = np.mean(thmin >= x)
            se = np.sqrt(max(p * (1 - p), 1.0 / NMC) / NMC)
            zz = (p - G.H(m, x)) / se
            ck.check("MC H_%d(%.1f)" % (m, x), abs(zz) < ZMAX, mc=p, se=se, recurrence=float(G.H(m, x)),
                     z=zz, tol="|z|<4")
    # data table
    X = np.linspace(0, np.pi / 2, 121)
    rows = [[x] + [float(G.H(m, x)) if x < np.pi / 2 else 0.0 for m in range(2, 9)] for x in X]
    write_table("Hm.dat", ["x"] + ["H%d" % m for m in range(2, 9)], rows)
    ck.info.update(seed=SEED, n_mc=NMC,
                   samples={"H%d(%g)" % (m, x): float(G.H(m, x)) for m in range(2, 9) for x in (0.5, 1.0)})
    return ck.finish()


if __name__ == "__main__":
    sys.exit(main())
