#!/usr/bin/env python3
"""Alphabets of the threefold integral (4.2) of the paper (Section 4.2), by computer algebra.

The integrand of (4.2) is
    2 x s y q(xs) / [(1 + x^2 y^2)(1 + s^2 y^2)] * H_k'(atan z_x(y)) H_m'(atan z_s(y)),
    z_R(y) = R (y + 1/y) / (1 - R^2).
Checks:
  1. the letters in y (roots of z_R - a, a in {0, i, -i}, the pole y = 0 of z_R, and the poles
     of the rational kernel) are exactly {0, +-i, +-ix, +-i/x, +-is, +-i/s};
  2. the singularity factors of the y-integration (pairwise differences of the letters and of the
     base point 1, leading coefficients), factored over Q(i), are among
     {x, x +- 1, x +- i, x +- s, sx +- 1}, with none of these missing;
  3. after the x-integration over (0, 1) the factors in s are among {s, s +- 1, s +- i}, all linear.
Requires SymPy.
"""
import itertools
import os
import sys

import sympy as S

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Checker

x, s, y = S.symbols('x s y')
I = S.I


def y_roots(zexpr, a):
    num = S.numer(S.together(zexpr - a))
    return S.roots(S.Poly(num, y), y)


def main():
    ck = Checker("alphabet_check")
    letters = set()
    for R in (x, s):
        z = R * (y + 1 / y) / (1 - R ** 2)
        for a in (0, I, -I):
            for r in y_roots(z, a):
                letters.add(S.simplify(r))
        letters.add(S.Integer(0))                      # pole of z_R
    for R in (x, s):                                   # kernel poles 1 + R^2 y^2 = 0
        for r in S.roots(S.Poly(1 + R ** 2 * y ** 2, y), y):
            letters.add(S.simplify(r))
    A_y = sorted(letters, key=str)
    expected_y = {S.Integer(0), I, -I, I * x, -I * x, I / x, -I / x, I * s, -I * s, I / s, -I / s}
    ck.check("y-letters = {0, +-i, +-ix, +-i/x, +-is, +-i/s}",
             set(map(S.simplify, A_y)) == expected_y, letters=" ".join(map(str, A_y)))

    pts = list(A_y) + [S.Integer(1)]                   # letters and the base point y = 1
    facs_x, facs_s = set(), set()

    def add(expr):
        for f, _e in S.factor_list(S.numer(S.together(expr)), extension=I)[1]:
            P = S.Poly(f, x, s)
            if P.total_degree() == 0:
                continue
            fx = P.monic().as_expr()
            (facs_x if fx.has(x) else facs_s).add(fx)

    for a, b in itertools.combinations(pts, 2):
        d = S.simplify(a - b)
        if d != 0:
            add(d)
    for R in (x, s):                                   # leading coefficients R, 1 - R^2
        for f in (R, 1 - R, 1 + R):
            add(f)
    norm = lambda e: S.Poly(e, x, s).monic().as_expr()
    got = set(norm(f) for f in facs_x)
    exp_x = set(norm(f) for f in (x, x - 1, x + 1, x - I, x + I, x - s, x + s, s * x - 1, s * x + 1))
    ck.check("x-factors after y-integration = {x, x+-1, x+-i, x+-s, sx+-1}", got == exp_x,
             found=" ".join(sorted(map(str, got))),
             missing=" ".join(sorted(map(str, exp_x - got))) or "none",
             extra=" ".join(sorted(map(str, got - exp_x))) or "none")

    xr = {S.Integer(0), S.Integer(1)}                  # x-letters and the endpoints 0, 1
    for f in got:
        for r in S.roots(S.Poly(f, x), x):
            xr.add(S.simplify(r))
    facs_s2 = set()
    for a, b in itertools.combinations(list(xr), 2):
        d = S.simplify(a - b)
        if d == 0:
            continue
        for f, _e in S.factor_list(S.numer(S.together(d)), extension=I)[1]:
            P = S.Poly(f, s)
            if P.degree() >= 1:
                facs_s2.add(P.monic().as_expr())
    facs_s2 |= set(S.Poly(f, s).monic().as_expr() for f in facs_s if S.Poly(f, s).degree() >= 1)
    exp_s = {s, s - 1, s + 1, s - I, s + I}
    ck.check("s-factors after x-integration within {s, s+-1, s+-i}, all linear",
             facs_s2 <= exp_s and all(S.Poly(f, s).degree() == 1 for f in facs_s2),
             x_letters=" ".join(sorted(map(str, xr))), found=" ".join(sorted(map(str, facs_s2))))
    ck.info.update(sympy=S.__version__)
    return ck.finish()


if __name__ == "__main__":
    sys.exit(main())
