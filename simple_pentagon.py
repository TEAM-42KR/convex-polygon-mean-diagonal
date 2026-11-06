#!/usr/bin/env python3
"""Probability that the symmetric random pentagon is simple.

J = int_0^1 [ (1-a)log(1-a) + (1+a)log(1+a) - 2a log(1+1/a)
              + (1-a^2) log(1+1/a) log((1+a)/(1-a)) ] da
  = (pi^2 + 48L + 12L^2 - 48)/18,   L = log 2,
and P(simple) = 1/2 - 15 J/(4 pi^2) = 7/24 + (10 - 10L - (5/2)L^2)/pi^2.
Checks: J at 50 digits (tolerance 1e-50), the two forms of P(simple) agree,
and a Monte Carlo estimate (edges e_k = z_k^2/2 of a Haar 2-frame in label
order, no sorting; simple = no two non-adjacent edges intersect; N = 10^6).
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import sys
import numpy as np
import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Checker, haar_frames, seg_cross

mp.mp.dps = 60
TOL = mp.mpf(10) ** -50
SEED = 20261110
NMC = 1_000_000
ZMAX = 4.0


def jint(a):
    return ((1 - a) * mp.log(1 - a) + (1 + a) * mp.log(1 + a) - 2 * a * mp.log(1 + 1 / a)
            + (1 - a * a) * mp.log(1 + 1 / a) * mp.log((1 + a) / (1 - a)))


def main():
    ck = Checker("simple_pentagon")
    L = mp.log(2)
    J = mp.quad(jint, [0, mp.mpf(1) / 2, 1])
    Jc = (mp.pi ** 2 + 48 * L + 12 * L ** 2 - 48) / 18
    ck.check("J = (pi^2 + 48L + 12L^2 - 48)/18", abs(J - Jc) < TOL, integral=mp.nstr(J, 52),
             closed_form=mp.nstr(Jc, 52), abs_err=float(abs(J - Jc)), tol="1e-50")
    P1 = mp.mpf(1) / 2 - 15 * J / (4 * mp.pi ** 2)
    P2 = mp.mpf(7) / 24 + (10 - 10 * L - mp.mpf(5) / 2 * L ** 2) / mp.pi ** 2
    ck.check("1/2 - 15J/(4pi^2) = 7/24 + (10-10L-(5/2)L^2)/pi^2", abs(P1 - P2) < TOL,
             value=mp.nstr(P2, 40), abs_err=float(abs(P1 - P2)), tol="1e-50")

    rng = np.random.default_rng(SEED)
    simple = 0
    B = 250_000
    for _ in range(NMC // B):
        z = haar_frames(rng, B, 5)
        V = np.concatenate([np.zeros((B, 1), complex), np.cumsum(z * z / 2, axis=1)], axis=1)
        bad = np.zeros(B, bool)
        for i in range(5):
            j = (i + 2) % 5                     # the 5 non-adjacent edge pairs (i, i+2)
            bad |= seg_cross(V[:, i], V[:, i + 1], V[:, j], V[:, j + 1])
        simple += int((~bad).sum())
    p = simple / NMC
    se = np.sqrt(p * (1 - p) / NMC)
    z = (p - float(P2)) / se
    ck.check("MC P(simple pentagon)", abs(z) < ZMAX, mc=p, se=se, exact=float(P2), z=z, tol="|z|<4")
    ck.info.update(seed=SEED, n_mc=NMC, J=mp.nstr(J, 55), P_simple=mp.nstr(P2, 55))
    return ck.finish()


if __name__ == "__main__":
    sys.exit(main())
