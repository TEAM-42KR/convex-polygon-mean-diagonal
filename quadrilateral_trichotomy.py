#!/usr/bin/env python3
"""Quadrilateral trichotomy.

For a generic closed set of 4 edge vectors, the 3 cyclic orders up to rotation
and reversal, (1,2,3,4), (1,3,2,4), (1,2,4,3), give exactly one convex, one
simple non-convex and one self-intersecting quadrilateral.
Laws tested (10^5 samples each):
  (S) symmetric measure: e_k = z_k^2/2 from a Haar 4 x 2 frame;
  (A) asymmetric exchangeable law: v_j iid, v = (X, 0.3 Y + 0.5 X U) with
      X ~ Exp(1), Y ~ N(0,1), U ~ U(0,1) (not rotation invariant),
      edges = v_j - mean(v), then randomly permuted.
Classification by segment-intersection tests: self-intersecting iff an
opposite pair of sides crosses; otherwise convex iff all four turns have the
same orientation, else simple non-convex.

Criterion: convex P0P1P2P3 with diagonals meeting at O,
P0 = O + alpha u, P1 = O + beta v, P2 = O - gamma u, P3 = O - delta v.
  P0 P1 P2 P3'  (P3' = P0 + P2 - P3) is simple iff
        sign(alpha delta - beta gamma) =  sign(gamma delta - alpha beta);
  P0 P1 P2' P3  (P2' = P1 + P3 - P2) is simple iff
        sign(alpha delta - beta gamma) = -sign(gamma delta - alpha beta).
Checked on 10^5 random convex quadrilaterals (exact agreement required).
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Checker, haar_frames, seg_cross

SEED = 20261111
N = 100_000
CONVEX, SIMPLE, CROSSED = 0, 1, 2
ORDERS = [(0, 1, 2, 3), (0, 2, 1, 3), (0, 1, 3, 2)]


def classify(W):
    """W: (B,4) complex vertices.  Returns class codes (B,)."""
    crossed = seg_cross(W[:, 0], W[:, 1], W[:, 2], W[:, 3]) | seg_cross(W[:, 1], W[:, 2], W[:, 3], W[:, 0])
    e = np.roll(W, -1, axis=1) - W
    turn = np.sign((e.conjugate() * np.roll(e, -1, axis=1)).imag)
    convex = np.all(turn == turn[:, :1], axis=1)
    return np.where(crossed, CROSSED, np.where(convex, CONVEX, SIMPLE))


def from_edges(E):
    return np.concatenate([np.zeros((E.shape[0], 1), complex), np.cumsum(E[:, :3], axis=1)], axis=1)


def trichotomy(ck, label, E):
    cls = np.stack([classify(from_edges(E[:, list(o)])) for o in ORDERS], axis=1)
    counts = np.stack([(cls == c).sum(axis=1) for c in (CONVEX, SIMPLE, CROSSED)], axis=1)
    good = np.all(counts == 1, axis=1)
    frac = cls.T
    ck.check("trichotomy, law %s: one convex + one simple non-convex + one crossed" % label,
             good.all(), samples=len(E), violations=int((~good).sum()), tol="0 violations")
    return {"order%d" % i: [float(np.mean(frac[i] == c)) for c in (CONVEX, SIMPLE, CROSSED)]
            for i in range(3)}


def main():
    ck = Checker("quadrilateral_trichotomy")
    rng = np.random.default_rng(SEED)
    z = haar_frames(rng, N, 4)
    info_s = trichotomy(ck, "S (symmetric)", z * z / 2)
    X = rng.exponential(size=(N, 4))
    v = X + 1j * (0.3 * rng.standard_normal((N, 4)) + 0.5 * X * rng.random((N, 4)))
    E = v - v.mean(axis=1, keepdims=True)
    E = np.take_along_axis(E, np.argsort(rng.random((N, 4)), axis=1), axis=1)
    info_a = trichotomy(ck, "A (asymmetric exchangeable)", E)
    ck.info["fractions_[convex,simple,crossed]_per_order"] = {"S": info_s, "A": info_a}

    # criterion
    al, be, ga, de = (rng.uniform(0.02, 2.0, N) for _ in range(4))
    th1 = rng.uniform(0, 2 * np.pi, N)
    th2 = th1 + rng.uniform(0.05, np.pi - 0.05, N)
    u, w = np.exp(1j * th1), np.exp(1j * th2)
    O = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    P0, P1, P2, P3 = O + al * u, O + be * w, O - ga * u, O - de * w
    base = classify(np.stack([P0, P1, P2, P3], axis=1))
    ck.check("criterion setup: P0P1P2P3 convex", np.all(base == CONVEX), samples=N,
             violations=int((base != CONVEX).sum()), tol="0 violations")
    s1 = np.sign(al * de - be * ga)
    s2 = np.sign(ga * de - al * be)
    c1 = classify(np.stack([P0, P1, P2, P0 + P2 - P3], axis=1))
    c2 = classify(np.stack([P0, P1, P1 + P3 - P2, P3], axis=1))
    ok1 = (c1 != CROSSED) == (s1 == s2)
    ok2 = (c2 != CROSSED) == (s1 == -s2)
    ck.check("P0P1P2P3' simple iff sign(ad-bg) = sign(gd-ab)", ok1.all(), samples=N,
             violations=int((~ok1).sum()), tol="0 violations")
    ck.check("P0P1P2'P3 simple iff sign(ad-bg) = -sign(gd-ab)", ok2.all(), samples=N,
             violations=int((~ok2).sum()), tol="0 violations")
    ck.check("criterion samples: neither swap is convex", np.all((c1 != CONVEX) & (c2 != CONVEX)),
             samples=N, tol="0 violations")
    ck.info["seed"] = SEED
    return ck.finish()


if __name__ == "__main__":
    sys.exit(main())
