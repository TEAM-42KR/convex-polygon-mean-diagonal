#!/usr/bin/env python3
"""First-order correction of the large-n expansion M_n = 2/pi^2 + c1/n + ...

With r = sin(a)/pi, A = (a + sin a cos a)/pi, b = (a - sin a cos a)/pi,
    c1 = (1/pi) int_0^pi [ cos a (1 - 2A) + b(1-b)/(2r) - r cos(2a) ] da,
claimed closed form c1 = (1 + 7 zeta(3)/2)/pi^2.
Also checks the auxiliary integral int_0^pi a(pi - a)/sin a da = 7 zeta(3).
Precision: 60 working digits, tolerance 1e-50.
"""
import os
import sys
import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Checker

mp.mp.dps = 60
TOL = mp.mpf(10) ** -50


def integrand(a):
    s, c = mp.sin(a), mp.cos(a)
    r = s / mp.pi
    A = (a + s * c) / mp.pi
    b = (a - s * c) / mp.pi
    return c * (1 - 2 * A) + b * (1 - b) / (2 * r) - r * mp.cos(2 * a)


def main():
    ck = Checker("asymptotic_c1")
    c1 = mp.quad(integrand, [0, mp.pi / 2, mp.pi]) / mp.pi
    closed = (1 + 7 * mp.zeta(3) / 2) / mp.pi ** 2
    ck.check("c1 integral = (1 + 7 zeta(3)/2)/pi^2", abs(c1 - closed) < TOL,
             integral=mp.nstr(c1, 52), closed_form=mp.nstr(closed, 52),
             abs_err=float(abs(c1 - closed)), tol="1e-50")
    aux = mp.quad(lambda a: a * (mp.pi - a) / mp.sin(a), [0, mp.pi / 2, mp.pi])
    ck.check("int_0^pi a(pi-a)/sin a da = 7 zeta(3)", abs(aux - 7 * mp.zeta(3)) < TOL,
             integral=mp.nstr(aux, 52), closed_form=mp.nstr(7 * mp.zeta(3), 52),
             abs_err=float(abs(aux - 7 * mp.zeta(3))), tol="1e-50")
    ck.info.update(c1=mp.nstr(c1, 55), limit=mp.nstr(2 / mp.pi ** 2, 55))
    return ck.finish()


if __name__ == "__main__":
    sys.exit(main())
