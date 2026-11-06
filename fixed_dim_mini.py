#!/usr/bin/env python3
"""Light end-to-end evaluation of the fixed-dimension representation (Section 5 of the paper).

Runs the evaluator in fixed_dim/ with its 'mini' profile on one core (about five minutes):
  1. tables: H_m (m = 3..8) and the gap density H_k'' (k = 3, 4, 5) from the contour formula (5.1) at the
     Chebyshev nodes of the outer variables (ball arithmetic, python-flint);
  2. outer: d(n,k), 4 <= n <= 10, 2 <= k <= n/2, by the quadrature of Theorem 5.3, and M_n.
Checks (pass: abs(diff) < 1e-15; the mini profile reaches about 18-25 digits):
  M_4..M_8 against the closed forms (reference values), M_9, M_10 against the values of Table 1.1
  (Section 8, computed with the recurrence (3.1) instead of (5.1)).
Also checks that the stored production values (data/fixed_dim_values.json, 70 digits) agree with the
same references to 54 digits (n <= 8, reference length) and 60 digits (n = 9, 10).
Requires python-flint and mpmath.
"""
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "fixed_dim"))
import mpmath as mp
from common import Checker

REF_9_10 = {  # Table 1.1 (Section 8), 61 trusted digits
    9: "0.2637869975407623857441802931447992384940684480975227518034706",
    10: "0.2574768099076879134716507783866900751116429778778418519610442",
}


def main():
    ck = Checker("fixed_dim_mini")
    mp.mp.dps = 80
    ref = json.load(open(os.path.join(HERE, "data", "reference_values.json")))
    R = {n: mp.mpf(ref["M%d" % n]) for n in range(4, 9)}
    R.update({n: mp.mpf(v) for n, v in REF_9_10.items()})

    stored = json.load(open(os.path.join(HERE, "data", "fixed_dim_values.json")))
    for n in range(4, 11):
        v = mp.mpf(stored["M"][str(n)])
        need = 54 if n <= 8 else 60
        diff = abs(v - R[n])
        ck.check("stored M_%d (70 digits) vs reference" % n, diff < mp.mpf(10) ** (-need),
                 abs_diff=float(diff), tol="1e-%d" % need)

    import prod
    out = os.path.join(HERE, "out", "fixed_dim_mini")
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    t0 = time.time()
    prod.run_tables("mini", 1, out)
    t1 = time.time()
    prod.run_outer("mini", 1, out)
    t2 = time.time()
    mp.mp.dps = 40
    res = json.load(open(os.path.join(out, "outer.json")))
    for n in range(4, 11):
        v = mp.mpf(res["E"][str(n)])
        diff = abs(v - R[n])
        ck.check("mini-profile M_%d vs %s" % (n, "closed form" if n <= 8 else "Table 1.1"),
                 diff < mp.mpf("1e-15"), value=mp.nstr(v, 26), abs_diff=float(diff), tol="1e-15")
    ck.info.update(tables_seconds=round(t1 - t0, 1), outer_seconds=round(t2 - t1, 1))
    return ck.finish()


if __name__ == "__main__":
    sys.exit(main())
