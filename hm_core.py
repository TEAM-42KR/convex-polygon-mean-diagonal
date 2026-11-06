"""Shared numerical core: the gap-void functions H_m and small helpers.

H_m(x) = probability that a fixed interval of length x in RP^1 contains none
of the m row lines of a Haar m x 2 orthonormal frame.

    H_2(x) = (1 - 2x/pi)_+ ,   H_m(0) = 1,   H_m(x) = 0 for x >= pi/2,
    H_m'(x) = -m(m-2)/pi * int_0^1 t^(m-3) H_{m-1}(atan(t tan x)) dt   (m >= 3).

Substituting s = atan(t tan x) gives the equivalent form
    tan(x)^(m-2) H_m'(x) = -m(m-2)/pi * F_m(x),
    F_m(x) = int_0^x tan(s)^(m-3) (1 + tan(s)^2) H_{m-1}(s) ds,
and differentiating once more,
    H_m''(x) = (m-2) (1 + tan(x)^2)/tan(x) * [ -m/pi H_{m-1}(x) - H_m'(x) ].

Numerics (double precision): piecewise Chebyshev interpolation on a mesh that
is geometrically graded toward x = pi/2 (where H_m has d^(m-1) log d type
behaviour, d = pi/2 - x).  On the two panels covering [0, pi/4] the t-form of
the derivative is used (Gauss-Legendre in t); for x >= pi/4 the s-form with
spectral cumulative integration is used.  H_m itself is the cumulative
integral of H_m' from 0, so H_m(pi/2) = 0 is a genuine consistency check.
"""
import numpy as np
from numpy.polynomial import chebyshev as C
from numpy.polynomial.legendre import leggauss

PI = np.pi
HP = np.pi / 2

# ---------------------------------------------------------------- quadrature
def gauss(a, b, n):
    x, w = leggauss(n)
    return (a + b) / 2 + (b - a) / 2 * x, (b - a) / 2 * w


def graded_rule(a, b, npts=12, ratio=0.3, levels=20, toward="b", core=1):
    """Composite Gauss-Legendre rule on [a,b], geometrically graded toward one
    or both endpoints ("a", "b", "both").  Returns nodes, weights."""
    L = b - a
    if toward == "both":
        m = (a + b) / 2
        x1, w1 = graded_rule(a, m, npts, ratio, levels, "a", core)
        x2, w2 = graded_rule(m, b, npts, ratio, levels, "b", core)
        return np.concatenate([x1, x2]), np.concatenate([w1, w2])
    # distances from the graded endpoint: L, L*ratio, L*ratio^2, ...
    d = L * ratio ** np.arange(levels + 1)
    d = np.concatenate([d, [0.0]])
    xs, ws = [], []
    for i in range(len(d) - 1):
        lo, hi = d[i + 1], d[i]
        n = npts if i > 0 else npts * core
        x, w = gauss(lo, hi, n)
        xs.append(x)
        ws.append(w)
    x = np.concatenate(xs)
    w = np.concatenate(ws)
    if toward == "b":
        return b - x, w
    return a + x, w


# ---------------------------------------------------------------- mesh
P_DEG = 26                      # Chebyshev degree per panel
RATIO = 0.3                     # geometric grading ratio toward pi/2
D_MIN = 1e-15                   # last mesh point is pi/2 - D_MIN (approx.)

_ref = -np.cos(np.pi * np.arange(P_DEG + 1) / P_DEG)        # [-1,1], ascending
_V = C.chebvander(_ref, P_DEG)
_Vinv = np.linalg.inv(_V)
# cumulative integral from -1 of the interpolant, evaluated at the nodes
_Icum = np.stack([C.chebval(_ref, C.chebint(np.eye(P_DEG + 1)[j], lbnd=-1))
                  for j in range(P_DEG + 1)], axis=1) @ _Vinv
_bw = np.ones(P_DEG + 1)
_bw[1::2] = -1
_bw[0] *= 0.5
_bw[-1] *= 0.5                   # barycentric weights (2nd-kind points)


def _breaks():
    b = [0.0, PI / 8, PI / 4]
    d = PI / 4
    while d * RATIO > D_MIN:
        d *= RATIO
        b.append(HP - d)
    return np.array(b)


BREAKS = _breaks()
NPAN = len(BREAKS) - 1
XN = np.array([(BREAKS[i] + BREAKS[i + 1]) / 2 + (BREAKS[i + 1] - BREAKS[i]) / 2 * _ref
               for i in range(NPAN)])                       # (NPAN, P_DEG+1)
HALF = (BREAKS[1:] - BREAKS[:-1]) / 2
XEND = BREAKS[-1]


def _interp(vals, x):
    """Piecewise barycentric interpolation of node values vals (NPAN, P+1)."""
    x = np.asarray(x, dtype=float)
    shp = x.shape
    xf = x.ravel()
    out = np.zeros_like(xf)
    inside = (xf >= 0) & (xf < XEND)
    xi = xf[inside]
    p = np.clip(np.searchsorted(BREAKS, xi, side="right") - 1, 0, NPAN - 1)
    s = (xi - (BREAKS[p] + BREAKS[p + 1]) / 2) / HALF[p]          # in [-1,1]
    diff = s[:, None] - _ref[None, :]
    exact = diff == 0
    diff[exact] = 1.0
    w = _bw[None, :] / diff
    num = np.sum(w * vals[p], axis=1)
    den = np.sum(w, axis=1)
    r = num / den
    hit = exact.any(axis=1)
    if hit.any():
        r[hit] = vals[p[hit]][exact[hit]]
    out[inside] = r
    return out.reshape(shp)


class GapVoid:
    """Tabulates H_m, H_m', H_m'' for m = 2..mmax and evaluates them anywhere."""

    def __init__(self, mmax=8, nt=48):
        self.mmax = mmax
        self.val = {}
        self.der = {}
        tq, wq = gauss(0.0, 1.0, nt)
        X = XN
        tanX = np.tan(X)
        sq_sec = 1 + tanX ** 2
        for m in range(3, mmax + 1):
            c = m * (m - 2) / PI
            dH = np.empty_like(X)
            # t-form on the first two panels (x <= pi/4)
            for p in (0, 1):
                arg = np.arctan(tq[None, :] * tanX[p][:, None])
                dH[p] = -c * np.sum(wq * tq ** (m - 3) * self.H(m - 1, arg), axis=1)
            # s-form for x >= pi/4: F_m(pi/4) from the t-form value
            F0 = -dH[1, -1] * np.tan(PI / 4) ** (m - 2) / c
            integrand = tanX ** (m - 3) * sq_sec * self.H(m - 1, X)
            F = F0
            for p in range(2, NPAN):
                Fp = F + HALF[p] * (_Icum @ integrand[p])
                dH[p] = -c * Fp / tanX[p] ** (m - 2)
                F = Fp[-1]
            # H_m = 1 + int_0^x H_m'
            Hv = np.empty_like(X)
            acc = 1.0
            for p in range(NPAN):
                Hv[p] = acc + HALF[p] * (_Icum @ dH[p])
                acc = Hv[p, -1]
            self.val[m] = Hv
            self.der[m] = dH
            self.end_value = getattr(self, "end_value", {})
            self.end_value[m] = acc        # H_m at pi/2 - D_MIN (should be ~0)

    # ----- evaluation (H_m extended by 0 for x >= pi/2)
    def H(self, m, x):
        x = np.asarray(x, dtype=float)
        if m == 2:
            return np.clip(1 - 2 * x / PI, 0.0, None) * (x >= 0)
        out = _interp(self.val[m], x)
        tail = (x >= XEND) & (x < HP)
        if np.any(tail):
            out = np.where(tail, self.end_value[m] * (HP - x) / (HP - XEND), out)
        return np.where(x <= 0, 1.0, out)

    def dH(self, m, x):
        x = np.asarray(x, dtype=float)
        if m == 2:
            return np.where(x < HP, -2 / PI, 0.0)
        return _interp(self.der[m], x)

    def d2H(self, m, x):
        """H_m''(x) for m >= 3, 0 < x < pi/2."""
        x = np.asarray(x, dtype=float)
        tx = np.tan(x)
        br = -m / PI * self.H(m - 1, x) - self.dH(m, x)
        return (m - 2) * (1 + tx * tx) / tx * br
