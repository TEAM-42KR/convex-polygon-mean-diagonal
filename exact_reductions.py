#!/usr/bin/env python3
"""Direct evaluation of the unreduced finite sums behind the exact reductions of
M_6 and M_7 (Sections 7.4 and 7.5 of the paper).

Input (data/, gzip-compressed text; format described in each file header):
    d62_sum.txt.gz   4599 terms, weight <= 6     d63_sum.txt.gz   6517 terms, weight <= 5
    d72_sum.txt.gz   1232 terms, weight <= 5     d73_sum.txt.gz   3371 terms, weight <= 5
Each line is a term c * L(w): c in Q(i), w = (a_1, ..., a_r) a word in the letters
0, -1, i, -i, -1/2, (-1+i)/2, (-1-i)/2, and L(w) the regularized value at +infinity
of the hyperlogarithm G(a_1, ..., a_r; z) = int_0^z dt/(t - a_1) G(a_2, ..., a_r; t)
(G(0^r; z) = log^r(z)/r!; L keeps the constant term in powers of log z).

Method (no stored relations, no fitted constants):
  * G(w; 1) for every word needed is computed by Taylor stepping along the segment
    [0, 1] in exact fixed-point integer arithmetic: if w = (a, w') then
    (t - a) d/dt G(w; t) = G(w'; t), which gives a two-term recurrence for the
    Taylor coefficients at each base point.  No letter lies on (0, 1], and the
    words are processed depth-first on their suffix tree, together with the
    complex-conjugate words (G of the conjugate word is the conjugate value).
  * Trailing zeros are removed with the shuffle relation and log(1) = 0.
  * L(w) follows from path composition through t = 1 and the inversion t -> 1/t:
        G(w; 1) = sum_{k=0}^{r} Phi(a_1..a_k) L(a_{k+1}..a_r),
    where Phi(a_1..a_k) is the integral over (+infinity -> 1), rewritten on (0, 1]
    by dt/(t - a) = ds/(s - 1/a) - ds/s  (a != 0),  dt/t = -ds/s.
The sums are assembled with exact rational coefficients, converted to the chord
means d_{6,2}, d_{6,3}, d_{7,2}, d_{7,3} with the normalizations in chord_from_sum()
(also stated in the data headers; for n = 6 the terms outside the sum are the
elementary part of the reduction, integrated separately in closed form), and
compared with the reduced closed forms printed in Sections 7.4, 7.5 and with
M_6 = (2 d_{6,2} + d_{6,3})/3, M_7 = (d_{7,2} + d_{7,3})/2 of Theorem 1.5.

Working precision 480 bits (about 144 digits); required agreement 1e-100.
Typical runtime: about 20 seconds, single thread.
"""
import gzip
import math
import os
import sys
import time
from fractions import Fraction
from functools import lru_cache

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(HERE, "data")
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

PREC = 480                      # fixed-point bits
ONE = 1 << PREC
DIGITS_REQUIRED = 100
mp.mp.dps = 160

# ---------------------------------------------------------------- letters
# codes 0..6 occur in the data; 7..9 are the inverses needed for t -> 1/t
F = Fraction
LETTER = {0: (F(0), F(0)), 1: (F(-1), F(0)), 2: (F(0), F(1)), 3: (F(0), F(-1)),
          4: (F(-1, 2), F(0)), 5: (F(-1, 2), F(1, 2)), 6: (F(-1, 2), F(-1, 2)),
          7: (F(-2), F(0)), 8: (F(-1), F(-1)), 9: (F(-1), F(1))}
INV = {1: 1, 2: 3, 3: 2, 4: 7, 5: 8, 6: 9, 7: 4, 8: 5, 9: 6}
CONJ = {0: 0, 1: 1, 2: 3, 3: 2, 4: 4, 5: 6, 6: 5, 7: 7, 8: 9, 9: 8}
for _a, (_r, _i) in LETTER.items():          # consistency of the tables
    assert LETTER[CONJ[_a]] == (_r, -_i)
    if _a:
        _d = _r * _r + _i * _i
        assert LETTER[INV[_a]] == (_r / _d, -_i / _d)


def conj_word(w):
    return tuple(CONJ[a] for a in w)


# ---------------------------------------------------------------- path
# base points on [0, 1]; step h <= 0.75 t_j keeps the 1/t recurrence stable and
# h / (distance to the nearest nonzero letter) <= 0.4 gives geometric convergence
TPTS = [F(0), F(1, 5), F(7, 20), F(49, 80), F(1)]


def _steps():
    steps = []
    for t0, t1 in zip(TPTS[:-1], TPTS[1:]):
        h = t1 - t0
        R = min(math.hypot(float(t0 - r), float(i)) for a, (r, i) in LETTER.items() if a)
        rho = float(h) / R
        N = int(math.ceil((PREC + 50) / math.log2(1 / rho))) + 8
        E = {}
        for a, (r, i) in LETTER.items():
            if t0 == 0 and a == 0:
                continue
            dr, di = t0 - r, -i                      # E = h / (t0 - a)
            den = dr * dr + di * di
            er, ei = h * dr / den, -h * di / den
            E[a] = ((er.numerator << PREC) // er.denominator,
                    (ei.numerator << PREC) // ei.denominator)
        steps.append((t0, N, E, rho))
    return steps


STEPS = _steps()


def advance(parent, a):
    """Taylor coefficients (scaled by h^k) of G((a,)+w'; t) at every base point from
    those of G(w'; t); returns (arrays, value at t = 1) as fixed-point integers."""
    out = []
    vr = vi = 0                                       # G vanishes at t = 0
    for j, (t0, N, E, _) in enumerate(STEPS):
        sre, sim = parent[j]
        if t0 == 0 and a == 0:                        # t f' = g at t = 0
            cre = [0] + [sre[k] // k for k in range(1, N + 1)]
            cim = [0] + [sim[k] // k for k in range(1, N + 1)]
        else:                                         # (t0 - a + x) f' = g
            er, ei = E[a]
            cre, cim = [vr], [vi]
            xr, xi = vr, vi
            if ei == 0:
                for k in range(N):
                    ur = sre[k] - k * xr
                    ui = sim[k] - k * xi
                    xr = ((ur * er) >> PREC) // (k + 1)
                    xi = ((ui * er) >> PREC) // (k + 1)
                    cre.append(xr)
                    cim.append(xi)
            else:
                for k in range(N):
                    ur = sre[k] - k * xr
                    ui = sim[k] - k * xi
                    xr, xi = (((ur * er - ui * ei) >> PREC) // (k + 1),
                              ((ur * ei + ui * er) >> PREC) // (k + 1))
                    cre.append(xr)
                    cim.append(xi)
        vr, vi = sum(cre), sum(cim)
        out.append((cre, cim))
    return out, (vr, vi)


def evaluate_words(words):
    """G(w; 1) for a suffix-closed, conjugation-closed set of convergent words."""
    children = {}
    for w in words:
        children.setdefault(w[1:], []).append(w[0])
    root = [([ONE] + [0] * N, [0] * (N + 1)) for (_, N, _, _) in STEPS]
    G = {}

    def visit(word, arrays):
        selfconj = conj_word(word) == word
        for a in sorted(children.get(word, ())):
            if selfconj and CONJ[a] < a:
                continue                              # mirror image of a visited child
            w = (a,) + word
            arr, (vr, vi) = advance(arrays, a)
            G[w] = (vr, vi)
            G[conj_word(w)] = (vr, -vi)
            if w in children:
                visit(w, arr)

    visit((), root)
    missing = [w for w in words if w not in G]
    assert not missing, "words not reached: %d" % len(missing)
    return G


# ---------------------------------------------------------------- algebra
@lru_cache(maxsize=None)
def trailing(word):
    """G(word; 1) as {convergent word: Fraction} (shuffle with G(0; 1) = 0)."""
    if not word:
        return {(): F(1)}
    m = 0
    for a in reversed(word):
        if a:
            break
        m += 1
    if m == 0:
        return {word: F(1)}
    if m == len(word):
        return {}
    v, k, out = word[:-1], len(word) - m, {}
    for pos in range(k):
        for u, c in trailing(v[:pos] + (0,) + v[pos:]).items():
            out[u] = out.get(u, 0) - c / m
    return {u: c for u, c in out.items() if c}


@lru_cache(maxsize=None)
def phi_words(word):
    """Integral over (+infinity -> 1) as {word on (0,1]: integer coefficient}."""
    out = {(): 1}
    for a in word:
        new = {}
        for v, c in out.items():
            if a:
                new[v + (INV[a],)] = new.get(v + (INV[a],), 0) + c
            new[v + (0,)] = new.get(v + (0,), 0) - c
        out = {v: c for v, c in new.items() if c}
    return out


def needed_words(term_words):
    need = set()
    for w in term_words:
        for i in range(len(w) + 1):
            suf = w[i:]
            need.update(trailing(suf))
            for k in range(1, len(suf) + 1):
                for v in phi_words(suf[:k]):
                    need.update(trailing(v))
    need.discard(())
    closed = set()
    for w in need:
        for i in range(len(w)):
            closed.add(w[i:])
            closed.add(conj_word(w[i:]))
    return closed


def fmul(x, y):
    return ((x[0] * y[0] - x[1] * y[1]) >> PREC, (x[0] * y[1] + x[1] * y[0]) >> PREC)


def frac_times(c, x):
    return ((c.numerator * x[0]) // c.denominator, (c.numerator * x[1]) // c.denominator)


class Values:
    def __init__(self, G):
        self.G = G
        self._g1, self._phi, self._L = {}, {}, {(): (ONE, 0)}

    def g1(self, w):                                  # G(w; 1), any word
        if w not in self._g1:
            tr, ti = 0, 0
            for u, c in trailing(w).items():
                x = (ONE, 0) if not u else self.G[u]
                xr, xi = frac_times(c, x)
                tr, ti = tr + xr, ti + xi
            self._g1[w] = (tr, ti)
        return self._g1[w]

    def phi(self, w):
        if w not in self._phi:
            tr, ti = 0, 0
            for u, c in phi_words(w).items():
                x = self.g1(u)
                tr, ti = tr + c * x[0], ti + c * x[1]
            self._phi[w] = (tr, ti)
        return self._phi[w]

    def L(self, w):                                   # regularized G(w; +infinity)
        if w not in self._L:
            vr, vi = self.g1(w)
            for k in range(1, len(w) + 1):
                pr, pi_ = fmul(self.phi(w[:k]), self.L(w[k:]))
                vr, vi = vr - pr, vi - pi_
            self._L[w] = (vr, vi)
        return self._L[w]


def to_mp(x):
    return mp.mpc(mp.ldexp(mp.mpf(x[0]), -PREC), mp.ldexp(mp.mpf(x[1]), -PREC))


# ---------------------------------------------------------------- data
def read_sum(name):
    terms = []
    with gzip.open(os.path.join(DATADIR, name + "_sum.txt.gz"), "rt") as f:
        declared = None
        for line in f:
            if line.startswith("#"):
                if line.startswith("# terms:"):
                    declared = int(line.split(":")[1])
                continue
            re_, im_, w = line.split()
            terms.append((F(re_), F(im_), tuple(int(ch) for ch in w) if w != "-" else ()))
    assert declared == len(terms), name
    return terms


def assemble(values, terms):
    sr = si = 0
    for cr, ci, w in terms:
        lr, li = values.L(w)
        # (cr + i ci)(lr + i li)
        sr += (cr.numerator * lr) // cr.denominator - (ci.numerator * li) // ci.denominator
        si += (cr.numerator * li) // cr.denominator + (ci.numerator * lr) // ci.denominator
    return to_mp((sr, si))


def digits(x):
    return float("inf") if x == 0 else float(-mp.log10(abs(x)))


# ---------------------------------------------------------------- closed forms
def closed_forms():
    pi, lg2 = mp.pi, mp.log(2)
    z3, z5, B = mp.zeta(3), mp.zeta(5), mp.polylog(4, mp.mpf(1) / 2)
    c = {}
    c["d62"] = (15 * pi**2 / 896 + 5 / (2 * pi**2) - mp.mpf(225) / (16 * pi**4)
                + 105 * z3 / (16 * pi**4) - 315 * z3**2 / (64 * pi**4))
    c["d63"] = (mp.mpf(419) / 1536 + 45 * lg2 / 128 - 225 / (64 * pi**2) - 315 * z3 / (128 * pi**2)
                + 225 / (8 * pi**4) + 4185 * z5 / (256 * pi**4))
    c["d72"] = (mp.mpf(64) / 105 + (2296 + 1568 * lg2) / (315 * pi**2)
                - (9216 - 128 * lg2 + 4704 * z3) / (105 * pi**4))
    c["d73"] = (-mp.mpf(36) / 35 - (240 + 256 * lg2 + 480 * lg2**2) / (35 * pi**2)
                + (11520 * B + 480 * lg2**4 + 384 * lg2**2 + 10080 * lg2 * z3 - 5760 * lg2
                   - 2304 * z3 + 3072) / (35 * pi**4))
    c["M6"] = (5 * pi**2 / 448 + mp.mpf(419) / 4608 + 15 * lg2 / 128 + 95 / (192 * pi**2)
               - 105 * z3 / (128 * pi**2) - 105 * z3**2 / (32 * pi**4) + 35 * z3 / (8 * pi**4)
               + 1395 * z5 / (256 * pi**4))
    c["M7"] = (-mp.mpf(22) / 105 + (68 - 368 * lg2 - 2160 * lg2**2) / (315 * pi**2)
               + (51840 * B + 2160 * lg2**4 + 1728 * lg2**2) / (315 * pi**4)
               - (25728 * lg2 + 17424 * z3 - 45360 * lg2 * z3) / (315 * pi**4))
    return c


def chord_from_sum(name, S):
    pi = mp.pi
    if name == "d62":
        return 15 * pi**2 / 256 - mp.mpf(45) / 64 + 35 / (12 * pi**2) + (
            S + 10 + 45 * pi**4 / 128 - 15 * pi**6 / 256) / pi**4
    if name == "d63":
        return mp.mpf(145) / 768 + 55 / (144 * pi**2) + S / pi**4
    if name == "d72":
        return 12 * S / pi**5
    if name == "d73":
        return 60 * S / pi**5
    raise ValueError(name)


TABLE = {"M6": "0.29585568887519114909120200", "M7": "0.28201203926456049319341550"}


def main():
    ck = Checker("exact_reductions")
    tol = mp.mpf(10) ** -DIGITS_REQUIRED
    names = ["d62", "d63", "d72", "d73"]
    sums = {n: read_sum(n) for n in names}
    for n in names:
        ck.info["terms_" + n] = len(sums[n])
        ck.info["max_weight_" + n] = max(len(w) for _, _, w in sums[n])

    # evaluator self-test on classical values (letters -1, i, -1/2 and the inversion)
    tests = {(1,): mp.log(2), (0, 1): mp.pi**2 / 12, (0, 0, 0, 0, 1): 15 * mp.zeta(5) / 16,
             (2,): mp.log(1 + 1j), (4,): mp.log(3),
             (0, 0, 2): mp.mpc(3 * mp.zeta(3) / 32, mp.pi**3 / 32)}   # -Li_3(-i)
    linf_tests = {(1,): mp.mpf(0), (0, 1): mp.pi**2 / 6, (0, 0, 1): mp.mpf(0)}
    t0 = time.time()
    words = needed_words([w for n in names for _, _, w in sums[n]] + list(tests) + list(linf_tests))
    G = evaluate_words(words)
    t_eval = time.time() - t0
    ck.info.update(words_evaluated=len(words), steps=[(str(s[0]), s[1], round(s[3], 3)) for s in STEPS],
                   seconds_hyperlog=round(t_eval, 1), precision_bits=PREC)
    print("evaluated G(w;1) for %d words in %.1f s (base points %s, terms per step %s)" % (
        len(words), t_eval, [str(s[0]) for s in STEPS] + ["1"], [s[1] for s in STEPS]), flush=True)
    vals = Values(G)
    err = max(abs(to_mp(vals.g1(w)) - v) for w, v in tests.items())
    err_l = max(abs(to_mp(vals.L(w)) - v) for w, v in linf_tests.items())
    ck.check("evaluator self-test: classical G(w;1) values", err < mp.mpf(10) ** -130,
             max_abs_err=mp.nstr(err, 5))
    ck.check("evaluator self-test: regularized values at infinity", err_l < mp.mpf(10) ** -130,
             max_abs_err=mp.nstr(err_l, 5))

    cf = closed_forms()
    d = {}
    for n in names:
        S = assemble(vals, sums[n])
        val = chord_from_sum(n, S.real)
        d[n] = val
        diff = val - cf[n]
        ck.check("d(%s,%s): %d-term sum = printed closed form" % (n[1], n[2], len(sums[n])),
                 abs(diff) < tol and abs(S.imag) < tol,
                 value=mp.nstr(val, 40), abs_diff=mp.nstr(abs(diff), 5),
                 digits=round(digits(diff), 1), imag_part=mp.nstr(abs(S.imag), 5))
        ck.info["value_" + n] = mp.nstr(val, 130)
    M6 = (2 * d["d62"] + d["d63"]) / 3
    M7 = (d["d72"] + d["d73"]) / 2
    for name, val in (("M6", M6), ("M7", M7)):
        diff = val - cf[name]
        ck.check("%s from the sums = closed form of Theorem 1.5" % name, abs(diff) < tol,
                 value=mp.nstr(val, 40), abs_diff=mp.nstr(abs(diff), 5),
                 digits=round(digits(diff), 1))
        tab = mp.mpf(TABLE[name])
        ck.check("%s agrees with the 25 printed digits of Table 1.1" % name,
                 abs(cf[name] - tab) < mp.mpf(10) ** -25,
                 printed=TABLE[name][:27], closed_form=mp.nstr(cf[name], 30))
        ck.info["value_" + name] = mp.nstr(val, 130)
    return ck.finish()


if __name__ == "__main__":
    sys.exit(main())
