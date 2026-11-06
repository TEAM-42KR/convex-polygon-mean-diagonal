#!/usr/bin/env python3
"""Unconditional chord mean (no convexity; edges in label order).

For the closed perimeter-1 polygon with edges e_k = z_k^2/2 (Haar 2-frame),
    E|e_1 + ... + e_k| = (m0 m2 - m1^2) / S,
    alpha = (k-1)/2, beta = (n-k-1)/2, m_j = B(alpha + j, beta),
    S = G(alpha) G(beta) G(alpha+1/2) G(beta+1/2)
        / ( G(alpha+beta+1/2) G(alpha+beta+1) G(3/2) )      (G = Gamma).
Checks: the exact values pi^2/32 (n=4), 4/15 (n=5), 3pi^2/128 (n=6,k=2),
1/4 (n=6,k=3) at 50 digits, the symmetry k <-> n-k, and Monte Carlo with
unsorted Haar frames for n = 5, 6, 7 and every k = 2..n-2.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import sys
import numpy as np
import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Checker, haar_frames

mp.mp.dps = 60
SEED = 20261108
NMC = 1_000_000
ZMAX = 4.0


def chord_mean(n, k):
    a = mp.mpf(k - 1) / 2
    b = mp.mpf(n - k - 1) / 2
    m0, m1, m2 = (mp.beta(a + j, b) for j in range(3))
    S = (mp.gamma(a) * mp.gamma(b) * mp.gamma(a + 0.5) * mp.gamma(b + 0.5)
         / (mp.gamma(a + b + 0.5) * mp.gamma(a + b + 1) * mp.gamma(mp.mpf(3) / 2)))
    return (m0 * m2 - m1 ** 2) / S


def main():
    ck = Checker("chord_selberg")
    exact = {(4, 2): mp.pi ** 2 / 32, (5, 2): mp.mpf(4) / 15,
             (6, 2): 3 * mp.pi ** 2 / 128, (6, 3): mp.mpf(1) / 4}
    for (n, k), v in exact.items():
        f = chord_mean(n, k)
        err = abs(f - v)
        ck.check("closed form n=%d k=%d" % (n, k), err < mp.mpf(10) ** -50,
                 formula=mp.nstr(f, 30), exact=mp.nstr(v, 30), abs_err=float(err), tol="1e-50")
    for n in range(4, 13):
        err = max(abs(chord_mean(n, k) - chord_mean(n, n - k)) for k in range(2, n - 1))
        ck.check("symmetry k<->n-k, n=%d" % n, err < mp.mpf(10) ** -50, abs_err=float(err), tol="1e-50")

    rng = np.random.default_rng(SEED)
    for n in (5, 6, 7):
        acc1 = np.zeros(n - 3)
        acc2 = np.zeros(n - 3)
        B = 200_000
        for _ in range(NMC // B):
            z = haar_frames(rng, B, n)
            V = np.cumsum(z * z / 2, axis=1)          # label order, no sorting
            for k in range(2, n - 1):
                x = np.abs(np.roll(V, -k, axis=1) - V).mean(axis=1)
                acc1[k - 2] += x.sum()
                acc2[k - 2] += (x * x).sum()
        for k in range(2, n - 1):
            mean = acc1[k - 2] / NMC
            se = np.sqrt((acc2[k - 2] / NMC - mean ** 2) / (NMC - 1))
            ex = float(chord_mean(n, k))
            z = (mean - ex) / se
            ck.check("MC n=%d k=%d" % (n, k), abs(z) < ZMAX, mc=mean, se=se, formula=ex, z=z, tol="|z|<4")
    ck.info.update(seed=SEED, n_mc=NMC,
                   values={"%d,%d" % (n, k): mp.nstr(chord_mean(n, k), 40)
                           for n in range(4, 9) for k in range(2, n - 1)})
    return ck.finish()


if __name__ == "__main__":
    sys.exit(main())
