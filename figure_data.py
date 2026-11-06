#!/usr/bin/env python3
"""Data tables for the figures.

out/polygon8.dat    : vertices of one sampled convex octagon (fixed seed,
                     perimeter 1, first vertex at the origin, edges in
                     angular order); columns i x y, the last row repeats
                     vertex 0 to close the path.
out/exact_scaled.dat: n, n (M_n - 2/pi^2) for n = 4..8 from the exact values.
Checks: the octagon closes, has perimeter 1 and is convex (all turns positive).
"""
import os
import sys
import numpy as np
import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Checker, haar_frames, write_table

SEED = 48   # first seed >= 1 whose sample has every exterior angle > 0.2 rad and every
            # edge longer than 0.05 (so all eight corners are visible in the figure)
mp.mp.dps = 60
REF_M = {
    4: "0.3453051847481618355032977789205408860013184513634833978",
    5: "0.3154621014556979611662880552884315465420673744805453633",
    6: "0.2958556888751911490912019732607904837860657228457401777",
    7: "0.2820120392645604931934154596208657878305703644530818003",
    8: "0.2717258936742568282403978417258471104682420164008997004",
}


def main():
    ck = Checker("figure_data")
    rng = np.random.default_rng(SEED)
    z = haar_frames(rng, 1, 8)[0]
    e = z * z / 2
    e = e[np.argsort(np.angle(e))]
    V = np.concatenate([[0j], np.cumsum(e)])
    ck.check("octagon closes", abs(V[-1]) < 1e-14, closure=abs(V[-1]), tol=1e-14)
    per = np.abs(e).sum()
    ck.check("octagon perimeter = 1", abs(per - 1) < 1e-14, perimeter=per, tol=1e-14)
    turns = (e.conjugate() * np.roll(e, -1)).imag
    ck.check("octagon convex (all turns positive)", np.all(turns > 0), min_turn=turns.min())
    ext = np.angle(np.roll(e, -1) / e)
    ck.check("illustration seed: exterior angles > 0.2, edges > 0.05",
             ext.min() > 0.2 and np.abs(e).min() > 0.05, min_angle=ext.min(), min_edge=np.abs(e).min())
    V[-1] = 0
    write_table("polygon8.dat", ["i", "x", "y"], [[i, V[i].real, V[i].imag] for i in range(9)])
    rows = []
    for n, s in REF_M.items():
        rows.append([n, "%.15f" % float(n * (mp.mpf(s) - 2 / mp.pi ** 2))])
    write_table("exact_scaled.dat", ["n", "scaled"], rows)
    ck.info.update(seed=SEED, scaled={str(n): r[1] for n, r in zip(REF_M, rows)})
    return ck.finish()


if __name__ == "__main__":
    sys.exit(main())
