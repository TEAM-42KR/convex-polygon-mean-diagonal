#!/usr/bin/env python3
"""Checks of the printed formulas of Section 5 (fixed-dimension representation).

1. Contour formula (5.1) for H_m, retyped from the paper, against the closed forms (3.2) for H_3
   (x = 0.3, 0.9, 1.4) and (3.3) for H_4 (x = 0.8).  25 working digits, tolerance 1e-20.
2. The change of variables of Lemma 5.2: with sin(u+v) = sech(zeta), sin(l_t(u,v)) = sech(xi), t = e^-beta,
     sinh(xi) = cos(u-v) sinh(beta) cosh(zeta) - sinh(zeta) cosh(beta),
     R = |sin(u-v)| sinh(beta) cosh(zeta),   |d l_t / d delta| = 2 R / cosh(xi)^2   (u, v = U/2 +- delta),
   at 12 random points (30 digits; the derivative is numerical).
3. The outer formula of Theorem 5.3, retyped from the paper, with the exact gap-void functions H_2, H_3 and
   H_3'': d(4,2) = M_4, d(5,2) = M_5, d(6,3) against the reference values.  Fixed double-exponential /
   trapezoid rules at 20 digits; tolerance 1e-12.
4. Kernel of Proposition 5.4 (generating function) at random points: |phi|, |Phi| <= 1; the symmetrized bracket
   is O(|eta|^2) as eta -> 0; the partial sums over n of (n-1) z^(n-2) [B0 C^(n-2) - (B+ A+^(n-2) + B- A-^(n-2))/2]
   converge to the Psi form.  (The five-dimensional integral itself is not evaluated.)
"""
import json
import os
import random
import sys

import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Checker

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------- 1. contour formula (5.1) ----------
def Omega(x, s, y):
    rho = (1 - y) / (1 + y)
    L = (mp.log(1 + rho * mp.expj(-2 * x * (1 - s))) - mp.log(1 + rho * mp.expj(2 * x * s))
         + mp.log(mp.sin(x * s)) - mp.log(mp.sin(x * (1 - s))))
    return mp.mpf(1) / 2 + x / (2 * mp.pi) - 1j / (2 * mp.pi) * L


def H_contour(m, x):
    def f(s, y):
        W = Omega(x, s, y)
        return y ** (m - 3) * mp.re((1 - W) ** m - (2 - W) ** m)
    return 1 + (m - 2) * x / mp.pi * mp.quad(f, [0, mp.mpf(1) / 2, 1], [0, 1])


def H3(x):
    return 1 - 3 * x / mp.pi + 3 * x ** 2 / mp.pi ** 2 - 3 / (2 * mp.pi ** 2) * mp.polylog(2, mp.sin(x) ** 2)


def H4(x):
    l = mp.log(mp.cos(x))
    J = mp.quad(lambda u: mp.polylog(2, -mp.tan(u) ** 2), [0, x])
    return (1 - 4 * x / mp.pi + 12 / mp.pi ** 2 * (1 - x * mp.cot(x))
            + (12 * mp.cot(x) * (x ** 2 - 2 * l + l ** 2) - 24 * x - 6 * J) / mp.pi ** 3)


# ---------- 3. outer formula of Theorem 5.3 ----------
def q(n, k, t):
    m = n - k
    c = mp.mpf(k * (k - 1) * m * (m - 1)) / (4 * n * (n * n - 1))
    a = mp.mpf(n) / 2 + 1
    if t >= mp.mpf("0.5"):
        F = mp.hyp2f1(a, k + 1, n + 2, 1 - t * t)
    else:  # Pfaff's transformation (the same function, numerically stable for small t)
        F = t ** (-2 * a) * mp.hyp2f1(a, n + 1 - k, n + 2, -(1 - t * t) / (t * t))
    return c * t ** (k - 2) * (1 - t * t) ** 2 * F


def shc(z):
    return mp.sinh(z) / z if z != 0 else mp.mpf(1)


def xof(z):
    return mp.atan(1 / mp.sinh(z))


def r_nodes(h, tlo, thi):
    out, j = [], int(mp.floor(tlo / h))
    while j * h <= thi:
        tau = j * h
        et = mp.exp(-tau)
        r = mp.exp(tau - et)
        out.append((r, r * (1 + et) * h))
        j += 1
    return out


def eta_nodes(r, nats, k):
    h = min(mp.mpf("4.93"), mp.mpf("7.85") / mp.sqrt(2 * r + mp.mpf("3.1"))) / nats
    lim = mp.mpf(nats) / max(k - 1, 1) / 2 + 40
    out, j = [], 0
    while True:
        e = j * h
        if 2 * r * mp.sinh(e) ** 2 > lim:
            break
        out.append((e, h if j else h / 2))
        j += 1
    return out


def sigma_nodes(h, tmax):
    out, K = [], int(tmax / h) + 1
    for j in range(-K, K + 1):
        tau = j * h
        z = mp.pi * mp.sinh(tau)
        s, sc = 1 / (1 + mp.exp(-z)), 1 / (1 + mp.exp(z))
        out.append((s, sc, h * mp.pi * mp.cosh(tau) * s * sc))
    return out


def eta_part(n, k, r, nats):
    """list of (cosh beta, weight * e^-beta q(e^-beta) / sqrt(shc shc))."""
    out = []
    for (e, w) in eta_nodes(r, nats, k):
        b = r * mp.cosh(2 * e)
        t = mp.exp(-b)
        out.append((mp.cosh(b), w * t * q(n, k, t) / mp.sqrt(shc(r * mp.sinh(e) ** 2) * shc(r * mp.cosh(e) ** 2))))
    return out


def d_k2(n, Hm, nats=45, h=mp.mpf(1) / 12):
    tot = 0
    for (r, wr) in r_nodes(h, mp.mpf("-3.6"), mp.mpf("4.6")):
        I = mp.fsum(w / mp.sqrt(cb + mp.cosh(r)) for (cb, w) in eta_part(n, 2, r, nats))
        tot += wr * mp.cosh(r) * Hm(xof(r)) * I
    return 4 * mp.sqrt(2) / (mp.pi * n) * mp.binomial(n, 2) * tot


def d_k(n, k, Hm, Hk2_sech, nats=45, h=mp.mpf(1) / 12, hs=mp.mpf(1) / 12):
    tot = 0
    sig = sigma_nodes(hs, 3.8)
    for (r, wr) in r_nodes(h, mp.mpf("-3.3"), mp.mpf("3.7")):
        ep = eta_part(n, k, r, nats)
        acc = 0
        for (s, sc, ws) in sig:
            xi, ze = r * s, r * sc
            I = mp.fsum(w / mp.sqrt(cb + mp.cosh(xi - ze)) for (cb, w) in ep)
            acc += ws * Hk2_sech(ze) * mp.cosh(xi) * Hm(xof(xi)) * I
        tot += wr * r * acc
    return 2 * mp.sqrt(2) / n * mp.binomial(n, k) * tot


# ---------- 4. kernel of Proposition 5.4 ----------
def phi(th, z):  # z = (z1, z2) real
    return 1 / (1 - 1j * (z[0] * mp.cos(th) + z[1] * mp.sin(th)))


def Phi(a, b, z):  # arc average over (a, b), closed form of int dtheta/(2pi) phi
    return mp.quad(lambda th: phi(th, z), [a, b]) / (2 * mp.pi)


def main():
    ck = Checker("fixed_dimension")
    # 1
    mp.mp.dps = 25
    for x in ("0.3", "0.9", "1.4"):
        x = mp.mpf(x)
        d = abs(H_contour(3, x) - H3(x))
        ck.check("(5.1) H_3 vs (3.2) at x=%s" % mp.nstr(x, 2), d < mp.mpf("1e-20"), abs_err=float(d), tol="1e-20")
    x = mp.mpf("0.8")
    d = abs(H_contour(4, x) - H4(x))
    ck.check("(5.1) H_4 vs (3.3) at x=0.8", d < mp.mpf("1e-20"), abs_err=float(d), tol="1e-20")
    # 2
    mp.mp.dps = 30
    random.seed(1)
    worst = mp.mpf(0)
    ell = lambda t, u, v: mp.atan(t / mp.tan(u)) + mp.atan(t / mp.tan(v))
    for _ in range(12):
        while True:
            U = mp.mpf(random.uniform(0.05, 1.5)); dl = mp.mpf(random.uniform(-0.49, 0.49)) * U
            t = mp.mpf(random.uniform(0.01, 0.99)); u, v = U / 2 + dl, U / 2 - dl
            L = ell(t, u, v)
            if L < mp.pi / 2:
                break
        ze, xi, be = mp.asech(mp.sin(U)), mp.asech(mp.sin(L)), -mp.log(t)
        R = mp.sqrt((mp.cosh(be) - mp.cosh(xi + ze)) * (mp.cosh(be) + mp.cosh(xi - ze)))
        dld = mp.diff(lambda dd: ell(t, U / 2 + dd, U / 2 - dd), dl)
        worst = max(worst, abs(abs(dld) - 2 * R / mp.cosh(xi) ** 2) / abs(dld),
                    abs(R - abs(mp.sin(u - v)) * mp.sinh(be) * mp.cosh(ze)) / R,
                    abs(mp.sinh(xi) - (mp.cos(u - v) * mp.sinh(be) * mp.cosh(ze) - mp.sinh(ze) * mp.cosh(be))))
    ck.check("Lemma 5.2 change of variables (12 random points)", worst < mp.mpf("1e-24"),
             worst_rel_err=float(worst), tol="1e-24")
    # 3
    mp.mp.dps = 20
    ref = json.load(open(os.path.join(HERE, "data", "reference_values.json")))
    H2 = lambda x: 1 - 2 * x / mp.pi
    H3pp_sech = lambda z: -6 / mp.pi ** 2 * mp.cosh(z) * mp.log(mp.tanh(z))  # H_3''(x(z)) / cosh z
    for (n, k, val) in ((4, 2, d_k2(4, H2)), (5, 2, d_k2(5, H3)), (6, 3, d_k(6, 3, H3, H3pp_sech))):
        key = {(4, 2): "M4", (5, 2): "M5", (6, 3): "d63"}[(n, k)]
        d = abs(val - mp.mpf(ref[key]))
        ck.check("Theorem 5.3 d(%d,%d) with exact H vs %s" % (n, k, key), d < mp.mpf("1e-12"),
                 value=mp.nstr(val, 18), abs_err=float(d), tol="1e-12")
    # 4
    mp.mp.dps = 20
    random.seed(7)
    psi = lambda w: (1 - w) ** -2 - 1 - 2 * w
    bad_bound, worst_sum, ratios = 0, mp.mpf(0), []
    for _ in range(4):
        xi = (mp.mpf(random.uniform(-2, 2)), mp.mpf(random.uniform(-2, 2)))
        et = (mp.mpf(random.uniform(-1, 1)), mp.mpf(random.uniform(-1, 1)))
        th = mp.mpf(random.uniform(0.2, 6.0))
        z = mp.mpc(random.uniform(-0.35, 0.35), random.uniform(-0.35, 0.35))  # |z| < 1/2: tail of N = 160 negligible

        def bracket(et, z=z, xi=xi, th=th, N=None):
            xp = (xi[0] + et[0], xi[1] + et[1]); xm = (xi[0] - et[0], xi[1] - et[1])
            B0 = phi(0, xi) * phi(th, xi); C = 1 / mp.sqrt(1 + xi[0] ** 2 + xi[1] ** 2)
            Bp = phi(0, xp) * phi(th, xp); Bm = phi(0, xm) * phi(th, xm)
            Ap = Phi(0, th, xp) + Phi(th, 2 * mp.pi, xi); Am = Phi(0, th, xm) + Phi(th, 2 * mp.pi, xi)
            if N is None:
                return B0 * psi(z * C) - (Bp * psi(z * Ap) + Bm * psi(z * Am)) / 2, (B0, Bp, Bm, Ap, Am, C)
            return mp.fsum((n - 1) * z ** (n - 2) * (B0 * C ** (n - 2) - (Bp * Ap ** (n - 2) + Bm * Am ** (n - 2)) / 2)
                           for n in range(4, N + 1))
        val, parts = bracket(et)
        bad_bound += sum(1 for p in parts if abs(p) > 1 + mp.mpf("1e-15"))
        worst_sum = max(worst_sum, abs(bracket(et, N=160) - val))
        e1 = (et[0] * mp.mpf("1e-3"), et[1] * mp.mpf("1e-3")); e2 = (e1[0] / 2, e1[1] / 2)
        ratios.append(abs(bracket(e1)[0] / bracket(e2)[0]))
    ck.check("Prop. 5.4 kernel: |B|, |A|, |C| <= 1", bad_bound == 0, violations=bad_bound)
    ck.check("Prop. 5.4 kernel: n-series sums to the Psi form", worst_sum < mp.mpf("1e-15"),
             abs_err=float(worst_sum), tol="1e-15")
    ck.check("Prop. 5.4 kernel: bracket = O(|eta|^2) (ratio at eta, eta/2 close to 4)",
             all(abs(r - 4) < mp.mpf("0.01") for r in ratios), ratios=" ".join(mp.nstr(r, 6) for r in ratios))
    return ck.finish()


if __name__ == "__main__":
    sys.exit(main())
