#!/usr/bin/env python3
"""Double-precision evaluation of the finite-n reduction for the chord means.

    d(n,k) = [binom(n,k)/n] int_0^1 q_{n,k}(t) P_{k,m}(t) dt,      m = n - k,
    q_{n,k}(t) = [k(k-1)m(m-1)/(4n(n^2-1))] t^(k-2) (1-t^2)^2
                 2F1(n/2+1, k+1; n+2; 1-t^2),
    P_{2,m}(t) = (2/pi) int_0^{pi/2} H_m(atan(t cot u) + atan(t tan u)) du,
    P_{k,m}(t) = int_{u,v>0, u+v<pi/2} H_k''(u+v) H_m(atan(t cot u) + atan(t cot v)) du dv
                 (k >= 3),
with H_m extended by 0 beyond pi/2 (H_m from hm_core.py).

Numerics: int_0^1 q_{n,k} dt equals the unconditional chord mean
(chord_selberg.py), so we integrate q (P - 1) and add that exact mass back;
this removes the endpoint behaviour of q at t = 0.  All outer and inner rules
are composite Gauss-Legendre rules, geometrically graded toward the endpoint
singularities.  For k >= 3 the inner domain is reduced to the exact support
{ell < pi/2} = {2 atan t < g < pi/2, |u - v| < acos(cos g (1+t^2)/(1-t^2))},
g = u + v.  q is evaluated with mpmath's 2F1 (15 digits).

Checks: d(5,2), d(6,2), d(6,3), d(7,2), d(7,3) against the exact closed-form
values (tolerance 1e-7), the quadrature mass of q against the exact chord
mean (tolerance 1e-10), and M_5, M_6, M_7 assembled from the d(n,k).
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import sys
import time
from math import comb
import numpy as np
import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Checker
from hm_core import GapVoid, graded_rule

PI = np.pi
HP = np.pi / 2
TOL = 1e-7
REF = {
    (5, 2): "0.3154621014556979611662880552884315465420673744805453633",
    (6, 2): "0.2821387234598579905052671919829763000772416858404397297",
    (6, 3): "0.3232896197058574662630715358164188512037137968563410737",
    (7, 2): "0.2524038522839323627263822968095553359392710750397520086",
    (7, 3): "0.3116202262451886236604486224321762397218696538664115919",
}
REF_M = {5: "0.3154621014556979611662880552884315465420673744805453633",
         6: "0.2958556888751911490912019732607904837860657228457401777",
         7: "0.2820120392645604931934154596208657878305703644530818003"}


def q_weight(n, k, t):
    m = n - k
    out = []
    with mp.workdps(60):          # 1 - t^2 must not round to 1 for tiny t
        c = mp.mpf(k * (k - 1) * m * (m - 1)) / (4 * n * (n * n - 1))
        for tt in t:
            tt = mp.mpf(tt)
            out.append(float(c * tt ** (k - 2) * (1 - tt ** 2) ** 2
                             * mp.hyp2f1(mp.mpf(n) / 2 + 1, k + 1, n + 2, 1 - tt ** 2)))
    return np.array(out)


def chord_mass(n, k):
    """Exact int_0^1 q_{n,k}(t) dt = unconditional chord mean (Selberg form)."""
    a = mp.mpf(k - 1) / 2
    b = mp.mpf(n - k - 1) / 2
    m0, m1, m2 = (mp.beta(a + j, b) for j in range(3))
    S = (mp.gamma(a) * mp.gamma(b) * mp.gamma(a + 0.5) * mp.gamma(b + 0.5)
         / (mp.gamma(a + b + 0.5) * mp.gamma(a + b + 1) * mp.gamma(mp.mpf(3) / 2)))
    return (m0 * m2 - m1 ** 2) / S


def P2(G, m, t, rule):
    u, w = rule                                   # rule on [0, pi/4]; integrand symmetric in u <-> pi/2-u
    ell = np.arctan(t / np.tan(u)) + np.arctan(t * np.tan(u))
    return 4 / PI * np.sum(w * G.H(m, ell))


def Pk(G, k, m, t, npts=12, levels=16):
    g0 = 2 * np.arctan(t)
    g, wg = graded_rule(g0, HP, npts, 0.3, levels, "both")
    z, wz = graded_rule(0.0, 1.0, npts, 0.3, levels, "b")
    h = np.arccos(np.clip(np.cos(g) * (1 + t * t) / (1 - t * t), -1, 1))
    u = (g[:, None] + h[:, None] * z[None, :]) / 2       # u - v = h z, z in [0,1]: half the
    v = (g[:, None] - h[:, None] * z[None, :]) / 2       # domain; x2 symmetry and Jacobian 1/2 cancel
    ell = np.arctan(t / np.tan(u)) + np.arctan(t / np.tan(v))
    inner = h * np.sum(wz * G.H(m, ell), axis=1)
    return float(np.sum(wg * G.d2H(k, g) * inner))


def d_nk(G, n, k):
    m = n - k
    t, wt = graded_rule(0.0, 1.0, 10, 0.3, 20, "both")
    q = q_weight(n, k, t)
    mass = chord_mass(n, k)
    if k == 2:
        rule = graded_rule(0.0, PI / 4, 12, 0.3, 25, "a")
        P = np.array([P2(G, m, tt, rule) for tt in t])
    else:
        P = np.array([Pk(G, k, m, tt) for tt in t])
    pref = comb(n, k) / n
    d = pref * (float(np.sum(wt * q * (P - 1))) + float(mass))
    mass_err = abs(float(np.sum(wt * q)) - float(mass))
    return d, mass_err, float(mass)


def main():
    ck = Checker("finite_n_reduction")
    G = GapVoid(8)
    d = {}
    for (n, k) in [(5, 2), (6, 2), (6, 3), (7, 2), (7, 3)]:
        t0 = time.time()
        val, merr, mass = d_nk(G, n, k)
        d[(n, k)] = val
        ck.check("mass int q_{%d,%d} = chord mean" % (n, k), merr < 1e-10, quad=mass + merr, exact=mass,
                 abs_err=merr, tol=1e-10)
        ex = float(mp.mpf(REF[(n, k)]))
        ck.check("d(%d,%d) reduction vs closed form" % (n, k), abs(val - ex) < TOL, reduction=val,
                 exact=ex, abs_err=abs(val - ex), tol=TOL, seconds=round(time.time() - t0, 1))
    for n in (5, 6, 7):
        M = sum(d[(n, min(k, n - k))] for k in range(2, n - 1)) / (n - 3)
        ex = float(mp.mpf(REF_M[n]))
        ck.check("M_%d = mean_k d(%d,k)" % (n, n), abs(M - ex) < TOL, reduction=M, exact=ex,
                 abs_err=abs(M - ex), tol=TOL)
    ck.info["d"] = {"%d,%d" % key: repr(v) for key, v in d.items()}
    return ck.finish()


if __name__ == "__main__":
    sys.exit(main())
