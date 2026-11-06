#!/usr/bin/env python3
"""Exact Monte Carlo sampler of the random convex n-gon (perimeter 1).

Law: Haar 2-frame Q in R^n, z_k = Q_k1 + i Q_k2, edges e_k = z_k^2 / 2
(closed, perimeter sum|e_k| = 1), conditioned on convexity.  A closed edge
set gives a convex polygon exactly when the edges are traversed in angular
order, so the conditioned law is obtained exactly by sorting the edges by
arg(e_k).  Q is produced from an n x 2 standard Gaussian matrix G by
Q = G (G^T G)^(-1/2).

Estimates:
  M_n    = E[mean of the n(n-3)/2 diagonals]    n = 4..12, 16, 20, 30, 40
  d(n,k) = E[|e_i + ... + e_{i+k-1}|] (sorted)  n <= 8
The per-polygon estimator averages over all n cyclic starting points, and
M_n = (1/(n-3)) sum_{k=2}^{n-2} d(n,k) per polygon.
Checks: z-score against the exact reference values (n = 4..8), |z| < 4.
Output: out/mc_scaled.dat with columns n, n(M_n - 2/pi^2), SE.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Checker, haar_frames, write_table

SEED = 20261107
ZMAX = 4.0
# exact reference values (closed forms evaluated to 55 digits)
REF = {
    "M4": "0.3453051847481618355032977789205408860013184513634833978",
    "M5": "0.3154621014556979611662880552884315465420673744805453633",
    "M6": "0.2958556888751911490912019732607904837860657228457401777",
    "M7": "0.2820120392645604931934154596208657878305703644530818003",
    "M8": "0.2717258936742568282403978417258471104682420164008997004",
    "d62": "0.2821387234598579905052671919829763000772416858404397297",
    "d63": "0.3232896197058574662630715358164188512037137968563410737",
    "d72": "0.2524038522839323627263822968095553359392710750397520086",
    "d73": "0.3116202262451886236604486224321762397218696538664115919",
    "d82": "0.2271363414020088984428194938755787383047365332849458014",
    "d83": "0.2937483582774511231767249981447828013775279273363596966",
    "d84": "0.3168600690123640979629002245885124729766811607618875059",
}
REF["d42"] = REF["M4"]
REF["d52"] = REF["M5"]

# sample sizes (chosen for <= ~5 minutes single-thread in total)
NSAMP = {4: 2_000_000, 5: 2_000_000, 6: 2_000_000, 7: 2_000_000, 8: 2_000_000,
         9: 1_000_000, 10: 1_000_000, 11: 1_000_000, 12: 1_000_000,
         16: 600_000, 20: 500_000, 30: 300_000, 40: 200_000}
BATCH_ELEMS = 2_000_000          # B * n per batch


def sorted_chords(rng, B, n):
    """Per-polygon averages of the k-chord lengths, k = 2..n-2 (shape (B, n-3))."""
    z = haar_frames(rng, B, n)
    e = z * z / 2
    e = e / np.abs(e).sum(axis=1, keepdims=True)      # perimeter exactly 1
    order = np.argsort(np.angle(e), axis=1)
    e = np.take_along_axis(e, order, axis=1)
    V = np.cumsum(e, axis=1)                          # vertices (last = 0 up to rounding)
    out = np.empty((B, n - 3))
    for k in range(2, n - 1):
        out[:, k - 2] = np.abs(np.roll(V, -k, axis=1) - V).mean(axis=1)
    return out


def estimate(rng, n, N):
    B = max(1000, BATCH_ELEMS // n)
    s1 = np.zeros(n - 2)        # columns: d(n,2..n-2), M_n
    s2 = np.zeros(n - 2)
    done = 0
    while done < N:
        b = min(B, N - done)
        ch = sorted_chords(rng, b, n)
        x = np.column_stack([ch, ch.mean(axis=1)])
        s1 += x.sum(axis=0)
        s2 += (x * x).sum(axis=0)
        done += b
    mean = s1 / N
    se = np.sqrt((s2 / N - mean ** 2) / (N - 1))
    return mean, se


def main():
    ck = Checker("sorted_sampler")
    rng = np.random.default_rng(SEED)
    rows = []
    res = {}
    for n in sorted(NSAMP):
        N = NSAMP[n]
        mean, se = estimate(rng, n, N)
        M, sM = mean[-1], se[-1]
        res[n] = {"N": N, "M": M, "SE": sM,
                  "d": {k: [mean[k - 2], se[k - 2]] for k in range(2, n - 1)}}
        scaled = n * (M - 2 / np.pi ** 2)
        rows.append([n, scaled, n * sM])
        print("n=%2d  N=%8d  M_n=%.7f +- %.7f   n(M_n-2/pi^2)=%.5f +- %.5f"
              % (n, N, M, sM, scaled, n * sM), flush=True)
        if n <= 8:
            z = (M - float(REF["M%d" % n])) / sM
            ck.check("M_%d vs exact" % n, abs(z) < ZMAX, mc=M, se=sM,
                     exact=float(REF["M%d" % n]), z=z, tol="|z|<4")
            for k in range(2, n // 2 + 1):
                # per polygon the k-chord and (n-k)-chord averages coincide
                m_, s_ = mean[k - 2], se[k - 2]
                key = "d%d%d" % (n, k)
                z = (m_ - float(REF[key])) / s_
                ck.check("d(%d,%d) vs exact" % (n, k), abs(z) < ZMAX, mc=m_, se=s_,
                         exact=float(REF[key]), z=z, tol="|z|<4")
    ck.info["estimates"] = {str(n): {"N": r["N"], "M": float(r["M"]), "SE": float(r["SE"]),
                                     "d": {str(k): [float(a), float(b)] for k, (a, b) in r["d"].items()}}
                            for n, r in res.items()}
    ck.info["seed"] = SEED
    write_table("mc_scaled.dat", ["n", "scaled", "se"], rows)
    return ck.finish()


if __name__ == "__main__":
    sys.exit(main())
