#!/usr/bin/env python3
"""Second and third coefficients of the formal large-n expansion
M_n = 2/pi^2 + c1/n + c2/n^2 + c3/n^3 + O(n^-4)   (Formal expansion 6.2).

Representation (Lemma 6.1): independent marks X_j = W_j u(theta_j), W_j ~ Exp(1),
theta_j ~ U(0, 2 pi), no conditioning on closure.  For an arc B of consecutive edges,
with (tau, x) the sums of (W_j, X_j) over B and (sigma, y) over the complement,
    |chord_B| = F(tau, x, sigma, y)
              = sqrt(|sigma x - tau y|^2 - (x cross y)^2) / ((tau + sigma)^2 - |x + y|^2).
For the tagged pair of Section 6 the half-width alpha is exactly uniform on (0, pi):
mark 1 = W_1 u(-alpha) lies in B, mark 2 = W_2 u(alpha) does not, and each of the
m = n - 2 untagged marks lies in B iff |theta_j| < alpha.  Hence
    D_n = (1/pi) int_0^pi E[F | alpha] d alpha,
and M_n follows from D_n by (6.2): M_n = ((n-1) D_n - 2/n) / (n-3).

Bulk expansion at fixed alpha (delta method).  F is homogeneous of degree 0, so
F(sum) = F(mu + eps) with mu the mean of one untagged mark (in R^6: weight and
vector inside B, weight and vector outside B) and eps = t/m + (Ybar - mu), t the
tagged marks.  The cumulant generating function of eps is explicit,
    log E e^{lambda.eps} = sum_k u^k [(lambda.a1)^k + (lambda.a2)^k]/k
                           + sum_{k>=2} u^{k-1} K_Y^{(k)}(lambda),       u = 1/m,
with a1 = (1, cos a, -sin a, 0, 0, 0), a2 = (0, 0, 0, 1, cos a, sin a) and K_Y the
cumulant function of one untagged mark (arc integrals of cos^j sin^l are finite
combinations of alpha and sin(k alpha)).  Since E[eps^beta] = O(u^ceil(|beta|/2)),
the Taylor polynomial of F at mu of degree 2U gives the exact coefficients k_j(alpha)
of E[F | alpha] = sum_{j<=U} k_j(alpha) m^-j + O(m^-U-1); all of this is finite
polynomial algebra in six variables, evaluated in mpmath.

The script computes delta_j = (1/pi) int_0^pi k_j(alpha) d alpha, j <= 3, by
Gauss-Legendre quadrature on (0, pi/2) (k_j(alpha) = k_j(pi - alpha)), converts
D_n = sum_j delta_j (n-2)^-j into the coefficients c_j of M_n, and compares with
c1 of Theorem 1.3 and the closed forms of c2, c3 printed in Formal expansion 6.2.
In the formal derivation the short-arc layers near alpha = 0 and alpha = pi first
contribute at order n^-4 (their inner expansions are polynomial up to that order),
so c2 and c3 are given by the bulk integrals alone; the script checks these
integrals and the printed closed forms, not the formal steps.  A built-in self-test runs
the same machinery on F = tau/(tau + sigma), whose conditional mean (p + u)/(1 + 2u),
p = alpha/pi, is known exactly.

Precision: 80 working digits; two quadrature rules (96 and 192 nodes) must agree to
1e-60, and the closed forms must hold to 1e-60.  Typical runtime: under a minute.
"""
import itertools
import math
import os
import sys
import time
from fractions import Fraction
from math import comb, factorial

import mpmath as mp
from mpmath.calculus.quadrature import GaussLegendre

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    from common import Checker  # the package's shared helper, if present
except Exception:  # standalone use: minimal equivalent
    import json
    import platform

    class Checker:
        def __init__(self, name):
            self.name, self.t0, self.checks, self.info = name, time.time(), [], {}

        def check(self, label, ok, **fields):
            ok = bool(ok)
            rec = {"check": label, "pass": ok}
            for k, v in fields.items():
                rec[k] = v if isinstance(v, (str, int, bool)) or v is None else float(v)
            self.checks.append(rec)
            extra = "  ".join("%s=%s" % (k, ("%.6g" % v) if isinstance(v, float) else v)
                              for k, v in rec.items() if k not in ("check", "pass"))
            print("%s  %s  %s" % ("PASS" if ok else "FAIL", label, extra), flush=True)
            return ok

        def finish(self):
            outdir = os.path.join(HERE, "out", "checks")
            os.makedirs(outdir, exist_ok=True)
            npass = sum(c["pass"] for c in self.checks)
            rec = {"script": self.name, "all_pass": npass == len(self.checks),
                   "n_checks": len(self.checks), "n_pass": npass,
                   "runtime_seconds": round(time.time() - self.t0, 2),
                   "python": sys.version.split()[0], "mpmath": mp.__version__,
                   "platform": platform.platform(), "checks": self.checks, "info": self.info}
            path = os.path.join(outdir, self.name + ".json")
            with open(path, "w") as f:
                json.dump(rec, f, indent=1)
            print("%s: %d/%d checks PASS, %.1f s, record %s" % (
                self.name, npass, len(self.checks), rec["runtime_seconds"],
                os.path.relpath(path, HERE)), flush=True)
            return 0 if rec["all_pass"] else 1

U = 3                 # expansion order in u = 1/m
DPS = 80              # working digits
RULES = (6, 7)        # Gauss-Legendre degrees: 3*2^(d-1) = 96 and 192 nodes
TOL = mp.mpf(10) ** -60
BASE = 16             # exponent encoding: a monomial in 6 variables -> one integer key


class Engine:
    """Polynomials in 6 variables as {key: coefficient}, truncated at total degree 2U."""

    def __init__(self, U):
        self.U = U
        self.MAXD = 2 * U
        keys = [e for e in itertools.product(range(self.MAXD + 1), repeat=6)
                if sum(e) <= self.MAXD]
        self.enc = lambda e: sum(x * BASE ** i for i, x in enumerate(e))
        self.DEG = {self.enc(e): sum(e) for e in keys}
        self.BF = {self.enc(e): math.prod(factorial(x) for x in e) for e in keys}  # beta!
        self.FC = {}

    def mul(self, A, B, maxd=None):
        maxd = self.MAXD if maxd is None else maxd
        Bg = {}
        for k, v in B.items():
            Bg.setdefault(self.DEG[k], []).append((k, v))
        C = {}
        for ka, va in A.items():
            da = self.DEG[ka]
            for db, lst in Bg.items():
                if da + db > maxd:
                    continue
                for kb, vb in lst:
                    C[ka + kb] = C.get(ka + kb, 0) + va * vb
        return C

    @staticmethod
    def add(A, B, cb=1):
        C = dict(A)
        for k, v in B.items():
            C[k] = C.get(k, 0) + cb * v
        return C

    def fourier(self, j, l):
        """cos^j sin^l = sum_k C_k e^{ik theta}; None when l is odd (integrates to 0
        over arcs symmetric about theta = 0)."""
        if (j, l) not in self.FC:
            if l % 2:
                self.FC[(j, l)] = None
            else:
                coef = {}
                for a in range(j + 1):
                    for b in range(l + 1):
                        k = (2 * a - j) + (2 * b - l)
                        coef[k] = coef.get(k, 0) + Fraction(comb(j, a) * comb(l, b) * (-1) ** (l - b))
                fac = Fraction((-1) ** (l // 2), 2 ** (j + l))
                self.FC[(j, l)] = {k: fac * v for k, v in coef.items() if v}
        return self.FC[(j, l)]

    def moments(self, alpha):
        """mu = E Y and M[j] = coefficient of u^j in E exp(lambda.eps) (polynomial in lambda)."""
        U, enc, KM = self.U, self.enc, self.U + 1
        sa, ca = mp.sin(alpha), mp.cos(alpha)
        sk = {k: mp.sin(k * alpha) for k in range(1, KM + 2)}

        def arc(j, l):        # (1/2pi) int_{-alpha}^{alpha} cos^j sin^l
            f = self.fourier(j, l)
            if f is None:
                return mp.mpf(0)
            tot = mp.mpf(0)
            for k, c in f.items():
                cc = mp.mpf(c.numerator) / c.denominator
                tot += cc * 2 * alpha if k == 0 else cc * 2 * sk[abs(k)] / abs(k)
            return tot / (2 * mp.pi)

        def full(j, l):       # (1/2pi) int_0^{2pi} cos^j sin^l
            f = self.fourier(j, l)
            if f is None:
                return mp.mpf(0)
            c = f.get(0, 0)
            return mp.mpf(c.numerator) / c.denominator

        # E exp(lambda.Y) = sum_k E[g(theta)^k] since E W^k = k!
        MY = {}
        for k in range(KM + 1):
            for i in range(k + 1):
                for j in range(k - i + 1):
                    l = k - i - j
                    ji = arc(j, l)
                    jo = full(j, l) - ji
                    mc = factorial(k) // (factorial(i) * factorial(j) * factorial(l))
                    if ji != 0:
                        key = enc((i, j, l, 0, 0, 0))
                        MY[key] = MY.get(key, 0) + mc * ji
                    if jo != 0:
                        key = enc((0, 0, 0, i, j, l))
                        MY[key] = MY.get(key, 0) + mc * jo
        X = {k: v for k, v in MY.items() if k != 0}
        KY, Xp = {}, {0: mp.mpf(1)}
        for j in range(1, KM + 1):                     # log(1 + X)
            Xp = self.mul(Xp, X, KM)
            for k, v in Xp.items():
                KY[k] = KY.get(k, 0) + mp.mpf((-1) ** (j + 1)) / j * v
        KYd = {d: {k: v for k, v in KY.items() if self.DEG[k] == d} for d in range(1, KM + 1)}
        mu = [KYd[1].get(BASE ** i, mp.mpf(0)) for i in range(6)]
        a1 = [1, ca, -sa, 0, 0, 0]
        a2 = [0, 0, 0, 1, ca, sa]
        lin1 = {BASE ** i: mp.mpf(a1[i]) for i in range(6) if a1[i] != 0}
        lin2 = {BASE ** i: mp.mpf(a2[i]) for i in range(6) if a2[i] != 0}
        Kt, P1, P2 = {}, {0: mp.mpf(1)}, {0: mp.mpf(1)}
        for k in range(1, U + 1):                      # tagged Exp(1) marks
            P1, P2 = self.mul(P1, lin1), self.mul(P2, lin2)
            Kt[k] = self.add({kk: v / k for kk, v in P1.items()}, {kk: v / k for kk, v in P2.items()})
        A = {j: self.add(Kt[j], KYd[j + 1]) for j in range(1, U + 1)}   # K_eps = sum_j u^j A_j
        M = {0: {0: mp.mpf(1)}}
        for j in range(1, U + 1):                      # exp: j M_j = sum_i i A_i M_{j-i}
            acc = {}
            for i in range(1, j + 1):
                acc = self.add(acc, self.mul(A[i], M[j - i]), i)
            M[j] = {kk: v / j for kk, v in acc.items()}
        return mu, M

    def taylor(self, mu, which="chord"):
        """Taylor polynomial at mu of F (chord) or of tau/(tau + sigma) (self-test)."""
        add, mul = self.add, self.mul

        def var(i):
            P = {BASE ** i: mp.mpf(1)}
            if mu[i] != 0:
                P[0] = mu[i]
            return P

        tau, x1, x2, sig, y1, y2 = [var(i) for i in range(6)]
        T = add(tau, sig)

        def inverse_series(D):                         # 1/D = (1/D0) sum (-d)^j
            D0 = D[0]
            d = {k: v / D0 for k, v in D.items() if k != 0}
            R, dp = {0: mp.mpf(1)}, {0: mp.mpf(1)}
            for j in range(1, self.MAXD + 1):
                dp = mul(dp, d)
                R = add(R, dp, (-1) ** j)
            return R, D0

        if which == "chord":
            vx = add(mul(sig, x1), mul(tau, y1), -1)
            vy = add(mul(sig, x2), mul(tau, y2), -1)
            cr = add(mul(x1, y2), mul(x2, y1), -1)
            Q = add(add(mul(vx, vx), mul(vy, vy)), mul(cr, cr), -1)
            C1, C2 = add(x1, y1), add(x2, y2)
            D = add(add(mul(T, T), mul(C1, C1), -1), mul(C2, C2), -1)
            Q0 = Q[0]
            q = {k: v / Q0 for k, v in Q.items() if k != 0}
            S, qp, bc = {0: mp.mpf(1)}, {0: mp.mpf(1)}, mp.mpf(1)
            for j in range(1, self.MAXD + 1):          # (1 + q)^(1/2)
                qp = mul(qp, q)
                bc = bc * (mp.mpf(1) / 2 - (j - 1)) / j
                S = add(S, qp, bc)
            R, D0 = inverse_series(D)
            sc = mp.sqrt(Q0) / D0
            return {k: v * sc for k, v in mul(S, R).items()}
        if which == "selftest":
            R, T0 = inverse_series(T)
            return {k: v / T0 for k, v in mul(tau, R).items()}
        raise ValueError(which)

    def coeffs(self, alpha, which="chord"):
        """k_0(alpha), ..., k_U(alpha)."""
        mu, M = self.moments(alpha)
        Fp = self.taylor(mu, which)
        out = []
        for j in range(self.U + 1):
            tot = mp.mpf(0)
            for k, v in M[j].items():
                if k in Fp:
                    tot += Fp[k] * self.BF[k] * v
            out.append(tot)
        return out


def integrate(eng, degree):
    """delta_j = (2/pi) int_0^{pi/2} k_j(alpha) d alpha by Gauss-Legendre."""
    nodes = GaussLegendre(mp.mp).calc_nodes(degree, mp.mp.prec)
    half = mid = mp.pi / 4
    tot = [mp.mpf(0)] * (eng.U + 1)
    for x, w in nodes:
        for i, v in enumerate(eng.coeffs(mid + half * x)):
            tot[i] += w * half * v
    return [2 * v / mp.pi for v in tot], len(nodes)


def m_to_n(dl):
    """D_n = sum_j delta_j (n-2)^-j  ->  M_n = sum_j c_j n^-j, M_n = ((n-1) D_n - 2/n)/(n-3)."""
    J = len(dl) - 1
    dn = [mp.mpf(0)] * (J + 1)
    for j in range(J + 1):
        for i in range(J + 1 - j):
            if j == 0 and i > 0:
                continue
            dn[j + i] += dl[j] * (comb(j + i - 1, i) if j > 0 else 1) * 2 ** i
    c = [mp.mpf(0)] * (J + 1)
    for k in range(J + 1):
        c[k] += dn[k]
        for i in range(1, k + 1):          # (n-1)/(n-3) = 1 + sum_i 2*3^(i-1) n^-i
            c[k] += 2 * 3 ** (i - 1) * dn[k - i]
        if k >= 2:                         # 2/(n(n-3)) = sum_{k>=2} 2*3^(k-2) n^-k
            c[k] -= 2 * 3 ** (k - 2)
    return c


def printed_forms():
    pi, z3, z5, z7 = mp.pi, mp.zeta(3), mp.zeta(5), mp.zeta(7)
    return {
        "c1": (1 + mp.mpf(7) / 2 * z3) / pi ** 2,
        "c2": -mp.mpf(3) / 8 - mp.mpf(7) / 16 * z3 + (84 + 93 * z5) / (16 * pi ** 2),
        "c3": (-mp.mpf(71) / 32 + mp.mpf(63) / 4 * z3 - mp.mpf(2511) / 128 * z5
               + (mp.mpf(123) / 8 + mp.mpf(21) / 2 * z3 - mp.mpf(3069) / 16 * z5
                  + mp.mpf(51435) / 256 * z7) / pi ** 2),
    }


PRINTED_DECIMALS = {"c2": "0.24171353884950172649636", "c3": "-0.41644512648653075829"}


def main():
    ck = Checker("asymptotic_c2c3")
    mp.mp.dps = DPS
    eng = Engine(U)

    al = mp.mpf("0.7")
    p = al / mp.pi
    st = eng.coeffs(al, "selftest")
    exact = [p] + [(-2) ** (j - 1) * (1 - 2 * p) for j in range(1, U + 1)]
    err = max(abs(a - b) for a, b in zip(st, exact))
    ck.check("self-test: tau/(tau+sigma) expansion is exact through u^%d" % U,
             err < mp.mpf(10) ** -(DPS - 10), max_abs_err=mp.nstr(err, 5))

    t = time.time()
    r1, n1 = integrate(eng, RULES[0])
    r2, n2 = integrate(eng, RULES[1])
    ck.info.update(quadrature_nodes=[n1, n2], seconds_quadrature=round(time.time() - t, 1),
                   working_digits=DPS)
    rule_diff = max(abs(a - b) for a, b in zip(r1, r2))
    ck.check("quadrature: %d- and %d-node rules agree for delta_0..delta_%d" % (n1, n2, U),
             rule_diff < TOL, max_abs_diff=mp.nstr(rule_diff, 5))
    ck.check("delta_0 = (1/pi) int r d alpha = 2/pi^2", abs(r2[0] - 2 / mp.pi ** 2) < TOL,
             abs_diff=mp.nstr(abs(r2[0] - 2 / mp.pi ** 2), 5))
    for j in range(1, U + 1):
        ck.info["delta%d" % j] = mp.nstr(r2[j], DPS - 15)

    c = m_to_n(r2)
    pf = printed_forms()
    where = {"c1": "Theorem 1.3", "c2": "Formal expansion 6.2", "c3": "Formal expansion 6.2"}
    for j in range(1, U + 1):
        name = "c%d" % j
        diff = c[j] - pf[name]
        ck.check("%s from the alpha-integrals = closed form of %s" % (name, where[name]),
                 abs(diff) < TOL, integral=mp.nstr(c[j], 40), abs_diff=mp.nstr(abs(diff), 5))
        ck.info[name] = mp.nstr(c[j], DPS - 15)
    for name, s in PRINTED_DECIMALS.items():
        ndig = len(s.split(".")[1])
        ok = abs(pf[name] - mp.mpf(s)) < mp.mpf(10) ** -ndig
        ck.check("%s: printed decimal expansion matches the closed form" % name, ok,
                 printed=s, closed_form=mp.nstr(pf[name], ndig + 3))
    return ck.finish()


if __name__ == "__main__":
    sys.exit(main())
