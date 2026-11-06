"""Gap-void functions H_m(x) (m=3..mmax) with exact x-derivatives (2nd-order duals), robust up to x -> pi/2.

Formula (contour formula (5.1) of the paper, Proposition 5.1, with t = s; symmetric under t <-> 1-t):
  H_m(x) = 1 + ((m-2)/pi) * x * 2 * int_0^{1/2} dt int_0^1 dy  y^(m-3) Re[(1-u)^m - (2-u)^m],
  u = 1/2 + x/(2pi) - (i/(2pi)) [ log(1+v e1) - log(1+v e2) + log sin(xt) - log sin(x(1-t)) ],
  e1 = exp(-2ix(1-t)), e2 = exp(2ixt), v = (1-y)/(1+y).
  Stable form near the corner: 1 + v e1 = 2y/(1+y) - 2iv sin(th) e^{i th}, th = x t + eps, eps = pi/2 - x.
The integrand has a log singularity at (t,y) = (-delta, 0), delta = eps/x, i.e. at distance delta OUTSIDE the
corner (0,0) of the domain; this produces the eps^2 log eps singularity of H_m at x = pi/2.
Quadrature: corner square [0,1/2]^2 in polar coordinates around (0,0) with a radial map uniform in log r down to
r ~ delta (DE-compressed below, DE-clustered at the outer edge) -> exponential convergence uniformly in delta;
strip [0,1/2]x[1/2,1] tensor tanh-sinh. m = 2: H_2(x) = 1 - 2x/pi exactly (the y-weight (m-2)y^(m-3) -> delta(y)).
All arithmetic in python-flint (arb/acb balls).
"""
from flint import arb, acb, ctx

I = acb(0, 1)

class D2:
    """a + b e + c e^2/2 with a,b,c acb (b = d/dx, c = d^2/dx^2)."""
    __slots__ = ('a', 'b', 'c')
    def __init__(s, a, b=0, c=0): s.a = a; s.b = b; s.c = c
    def __add__(s, o):
        if isinstance(o, D2): return D2(s.a + o.a, s.b + o.b, s.c + o.c)
        return D2(s.a + o, s.b, s.c)
    __radd__ = __add__
    def __sub__(s, o):
        if isinstance(o, D2): return D2(s.a - o.a, s.b - o.b, s.c - o.c)
        return D2(s.a - o, s.b, s.c)
    def __rsub__(s, o): return D2(o - s.a, -s.b, -s.c)
    def __neg__(s): return D2(-s.a, -s.b, -s.c)
    def __mul__(s, o):
        if isinstance(o, D2): return D2(s.a * o.a, s.a * o.b + s.b * o.a, s.a * o.c + 2 * s.b * o.b + s.c * o.a)
        return D2(s.a * o, s.b * o, s.c * o)
    __rmul__ = __mul__
    def f(s, f0, f1, f2): return D2(f0, f1 * s.b, f2 * s.b * s.b + f1 * s.c)

def dlog(z):
    a = z.a; ia = 1 / a
    return z.f(a.log(), ia, -ia * ia)

def dsin(z):
    a = z.a; sa = a.sin()
    return z.f(sa, a.cos(), -sa)

def dexpi(z):  # exp(i z)
    e = (I * z.a).exp()
    return z.f(e, I * e, -e)

def integrand(t, y, X, E, ms, mmax, need_deriv=True):
    """dict m -> (re value, re d/dx, re d2/dx2) of ((m-2)/pi) x * 2 * y^(m-3) Re[(1-u)^m-(2-u)^m] at (t,y).
    t, y: arb (accurate small values); X = D2(x,1,0); E = D2(eps,-1,0)."""
    pi = arb.pi()
    omt = 1 - t
    v = (1 - y) / (1 + y)
    th = X * t + E                     # = pi/2 - x(1-t), accurate when t, eps small
    sth = dsin(th)
    ei = dexpi(th)
    L1 = dlog(sth * ei * (-2 * I * v) + 2 * y / (1 + y))
    L2 = dlog(dexpi(X * (2 * t)) * v + 1)
    G = dlog(dsin(X * t)) - dlog(dsin(X * omt))
    S = L1 - L2 + G
    w = (S * I) * (1 / (2 * pi)) + (X * (-1 / (2 * pi))) + arb(1) / 2   # 1 - u
    w0, w1, w2 = w.a, w.b, w.c
    z0 = w0 + 1                                                       # 2 - u (same derivatives)
    out = {}
    p0 = acb(1); q0 = acb(1)          # w0^(m-2), z0^(m-2) running
    pm1 = acb(1); qm1 = acb(1)
    pw = [acb(1)]; qw = [acb(1)]
    for j in range(1, mmax + 1):
        pw.append(pw[-1] * w0); qw.append(qw[-1] * z0)
    x0 = X.a
    ypow = arb(1)
    for m in range(3, mmax + 1):
        if m > 3: ypow = ypow * y
        if m not in ms: continue
        P0 = pw[m] - qw[m]
        c = arb(2 * (m - 2)) / pi * ypow
        if need_deriv:
            P1 = m * (pw[m - 1] - qw[m - 1]) * w1
            P2 = m * (m - 1) * (pw[m - 2] - qw[m - 2]) * w1 * w1 + m * (pw[m - 1] - qw[m - 1]) * w2
            # times x (dual (x,1,0))
            out[m] = (c * (x0 * P0).real, c * (x0 * P1 + P0).real, c * (x0 * P2 + 2 * P1).real)
        else:
            out[m] = (c * (x0 * P0).real, arb(0), arb(0))
    return out

def ts_nodes(h, tmax):
    """tanh-sinh on (0,1): list of (s, 1-s, weight) with accurate complements."""
    pi = arb.pi(); out = []
    k = int(tmax / float(h)) + 1
    for j in range(-k, k + 1):
        tau = arb(j) * h
        z = pi * tau.sinh()
        s = 1 / (1 + (-z).exp()); sc = 1 / (1 + z.exp())
        wt = h * s * sc * pi * tau.cosh()
        out.append((s, sc, wt))
    return out

_GL = {}
def gl_nodes(n):
    """Gauss-Legendre on (-1,1) at the current precision: list of (x, w)."""
    key = (n, ctx.prec)
    if key not in _GL:
        _GL[key] = [arb.legendre_p_root(n, k, weight=True) for k in range(n)]
    return _GL[key]

def radial_nodes(h, tau_c, tau_lo, tau_hi):
    """rho in (0,1): rho = 1 - exp(-e^g), g = tau - exp(tau_c - tau). returns (rho, 1-rho, drho/dtau*h)."""
    import math
    out = []
    hf = float(h); j = int(math.floor(float(tau_lo) / hf)); thi = float(tau_hi)
    while True:
        tau = arb(j) * h
        if j * hf > thi: break
        ex = (tau_c - tau).exp()
        g = tau - ex
        eg = g.exp()
        omr = (-eg).exp()                 # 1 - rho
        rho = -((-eg).expm1())            # rho, accurate when small
        jac = omr * eg * (1 + ex) * h
        out.append((rho, omr, jac))
        j += 1
    return out

def H_all(x, eps, ms, prm, need_deriv=True):
    """x, eps = pi/2 - x given accurately (arb). prm: dict(hr, hphi, hs, nats).
    returns dict m -> (H, H', H'') ; m=2 analytic."""
    pi = arb.pi()
    mmax = max(ms)
    msx = [m for m in ms if m >= 3]
    X = D2(acb(x), acb(1), acb(0)); E = D2(acb(eps), acb(-1), acb(0))
    acc = {m: [arb(0), arb(0), arb(0)] for m in msx}
    def add(vals, wt):
        for m, (a, b, c) in vals.items():
            A = acc[m]; A[0] += a * wt; A[1] += b * wt; A[2] += c * wt
    nats = arb(prm.get('nats', 165))
    half = arb(1) / 2
    # --- strip [0,1/2] x [1/2,1]: t tanh-sinh (log edge t=0), y Gauss-Legendre (smooth)
    tn = ts_nodes(arb(prm['hs']), prm.get('tsmax', 4.7))
    gy = gl_nodes(prm.get('ny', 48))
    for (st, sct, wt) in tn:
        t = half * st
        for (gx, gw) in gy:
            y = arb(3) / 4 + gx / 4
            add(integrand(t, y, X, E, msx, mmax, need_deriv), wt * gw / 8)
    # --- corner square [0,1/2]^2, polar around (0,0)
    delta = eps / x
    dl = delta if delta < 1 else arb(1)
    pin = pi / 4
    an = ts_nodes(arb(prm['hphi']), prm.get('tsmax', 4.7))
    ga = [((1 + gx) / 2, (1 - gx) / 2, gw / 2) for (gx, gw) in gl_nodes(prm.get('na1', 64))]
    tau_hi = (nats.log() + arb('0.7'))
    for (lo_is_zero, ) in [(True,), (False,)]:
        for (sp, scp, wp) in (ga if lo_is_zero else an):
            if lo_is_zero:      # phi in (0, pi/4) (smooth, Gauss-Legendre): r_max = 1/(2 cos phi)
                phi = pin * sp; cphi = phi.cos(); sphi = phi.sin(); rmax = half / cphi
            else:               # phi in (pi/4, pi/2): use psi = pi/2 - phi = (pi/4) * scp accurately
                psi = pin * scp; cphi = psi.sin(); sphi = psi.cos(); rmax = half / sphi
            wph = wp * pin
            tau_c = (dl / rmax).log() - 2
            tau_lo = tau_c - (nats / 2 + 5).log()
            for (rho, omr, jac) in radial_nodes(arb(prm['hr']), tau_c, tau_lo, tau_hi):
                r = rmax * rho
                t = r * cphi; y = r * sphi
                add(integrand(t, y, X, E, msx, mmax, need_deriv), wph * rmax * jac * r)
    out = {m: (1 + acc[m][0], acc[m][1], acc[m][2]) for m in msx}
    if 2 in ms:
        out[2] = (1 - 2 * x / pi, -2 / pi, arb(0))
    return out

def x_eps_of_s(s):
    """x(s) = atan(csch s), eps = pi/2 - x = atan(sinh s) (both accurate)."""
    sh = s.sinh()
    return (1 / sh).atan(), sh.atan()
