"""Outer integrator of the fixed-dimension representation (Theorem 5.3 of the paper; here s = xi, D = zeta,
b = beta, and w = sqrt(2r) sinh v, so that b = r + w^2 = r cosh 2v).

d(n,k) = pref * int_0^inf dD f_k(D) int_0^inf ds cosh(s) G_m(s) I(s,D),  pref = binom(n,k)/(2n) * k/pi,  m = n-k,
I(s,D) = int_0^inf dw 2 gW(n,k; s+D+w^2) / sqrt(cosh(D+w^2/2) sinhc(w^2/2) (sinh(2s+D+w^2) + sinh D)),
G_m(s) = H_m(x(s)), f_k(D) = (pi/k) H_k''(x(D)) sech D (k>=3), f_2 = delta(D), x(s) = atan(csch s),
gW(n,k;b) = 2 e^{-b} q(n,k;e^{-b}), q(t) = k(k-1)m(m-1)/(4n(n^2-1)) t^(k-2) (1-t^2)^2 2F1(n/2+1,k+1;n+2;1-t^2).
Coordinates: r = s+D, sigma = s/r (ds dD = r dr dsigma); w = sqrt(2r) sinh v, Q = 2r + w^2 = 2r cosh^2 v:
  dw / sqrt(2 sinh(Q/2)) -> dv / sqrt(sinhc(Q/2)) removes the corner singularity (s,D,w)->0 exactly, and
  cosh(D+w^2/2) cosh(s+w^2/2) = (cosh(r+w^2) + cosh((1-2 sigma) r))/2.
  => d(n,k) = pref int_0^inf r dr int_0^1 dsigma f_k(D) cosh(s) G_m(s)
              int_0^inf dv 2 gW(n,k; r+w^2) / sqrt( (cosh(r+w^2)+cosh((1-2sigma)r))/2 * sinhc(w^2/2) sinhc(Q/2) ).
  k = 2: sigma = 1 (D = 0, s = r), no r Jacobian, f = 1.
Rules: r exp-type DE r = exp(tau - e^-tau); sigma tanh-sinh (r <= rsplit) or, for r > rsplit, split at 1/2 with
log-uniform maps in the absolute variable D (resp. s) so the singularities of f_k, G_m at D, s = +-i pi/2 stay at
fixed distance; v trapezoid (even integrand, strip |Im v| < pi/4).
"""
import mpmath as mp

def q_nk(n, k, t):
    m = n - k
    c = mp.mpf(k * (k - 1) * m * (m - 1)) / (4 * n * (n * n - 1))
    a = mp.mpf(n) / 2 + 1
    if t >= mp.mpf('0.5'):
        F = mp.hyp2f1(a, k + 1, n + 2, 1 - t * t)
    else:   # Pfaff
        F = t ** (-2 * a) * mp.hyp2f1(a, n + 1 - k, n + 2, -(1 - t * t) / (t * t))
    return c * t ** (k - 2) * (1 - t * t) ** 2 * F

def gW(n, k, b):
    t = mp.exp(-b)
    return 2 * t * q_nk(n, k, t)

def sinhc(z):
    return mp.sinh(z) / z if z != 0 else mp.mpf(1)

def de_exp_nodes(h, tlo, thi):
    """r = exp(tau - e^-tau): list of (r, dr/dtau * h)."""
    out = []
    j = int(mp.floor(tlo / h))
    while j * h <= thi:
        tau = j * h
        et = mp.exp(-tau)
        r = mp.exp(tau - et)
        out.append((r, r * (1 + et) * h))
        j += 1
    return out

def ts01(h, tmax):
    """tanh-sinh on (0,1): (s, 1-s, w)."""
    out = []
    k = int(tmax / h) + 1
    for j in range(-k, k + 1):
        tau = j * h
        z = mp.pi * mp.sinh(tau)
        s = 1 / (1 + mp.exp(-z)); sc = 1 / (1 + mp.exp(z))
        out.append((s, sc, h * s * sc * mp.pi * mp.cosh(tau)))
    return out

def logmap_nodes(L, c0, h, nats):
    """nodes on (0, L): u = L (1 - exp(-e^g)), g = tau - exp(tau_c - tau), tau_c = log(c0/L):
    log-uniform from ~c0 up, DE-compressed below, DE-clustered at L. returns (u, L-u, du/dtau*h)."""
    tau_c = mp.log(mp.mpf(c0) / L)
    tlo = tau_c - mp.log(nats + 10); thi = mp.log(nats) + mp.mpf('0.7')   # integrand ~ log u at u->0: cut at u ~ c0 e^-nats
    out = []
    j = int(mp.floor(tlo / h))
    while j * h <= thi:
        tau = j * h
        ex = mp.exp(tau_c - tau); g = tau - ex; eg = mp.exp(g)
        om = mp.exp(-eg); rho = -mp.expm1(-eg)
        out.append((L * rho, L * om, L * om * eg * (1 + ex) * h))
        j += 1
    return out

def v_nodes(r, hvfac, kmin, nats):
    """trapezoid on v in [0, vmax) (even integrand): (w2, Q, wt).
    Step: the integrand is analytic for |Im v| < pi/4 (small r) and, in w = sqrt(2r) sinh v, for |Im w| < 1.25
    (zeros of cosh(D + w^2/2), cosh(s + w^2/2) at w^2 = i pi - 2D, i pi - 2s); dw/dv ~ sqrt(2r + w^2) there."""
    hv = mp.mpf(hvfac) * min(mp.mpf('4.93'), mp.mpf('7.85') / mp.sqrt(2 * r + mp.mpf('3.1'))) / nats
    out = []
    lim = mp.mpf(nats) / max(kmin - 1, 1) / 2 + 40
    j = 0
    while True:
        v = j * hv
        sh = mp.sinh(v)
        w2 = 2 * r * sh * sh
        if w2 > lim: break
        out.append((w2, 2 * r + w2, hv if j > 0 else hv / 2))
        j += 1
    return out

def d_all(pairs, funcs, prm, part=(0, 1)):
    """pairs: list of (n,k). funcs: object with Gvec(s)->{m:G}, fvec(D)->{k:f}.
    prm: hr, hvfac, hsig, hlog, nats, rsplit, tsmax, tlo2, thi2, tlo3, thi3.
    part = (i, P): only r-nodes with index % P == i (partial sums add up).  Returns {(n,k): d}."""
    ip, P = part
    nats = prm['nats']
    res = {p: mp.mpf(0) for p in pairs}
    p2 = [p for p in pairs if p[1] == 2]
    p3 = [p for p in pairs if p[1] >= 3]
    pref = {(n, k): mp.binomial(n, k) / (2 * n) * k / mp.pi for (n, k) in pairs}
    # ---- k = 2 : s = r, D = 0
    if p2:
        for ir, (r, wr) in enumerate(de_exp_nodes(mp.mpf(prm['hr']), prm['tlo2'], prm['thi2'])):
            if ir % P != ip: continue
            vn = v_nodes(r, prm.get('hvfac', 1), 2, nats)
            G = funcs.Gvec(r)
            chr_ = mp.cosh(r)
            acc = {p: mp.mpf(0) for p in p2}
            for (w2, Q, wv) in vn:
                b = r + w2
                c = wv / mp.sqrt(mp.cosh(w2 / 2) * mp.cosh(r + w2 / 2) * sinhc(w2 / 2) * sinhc(Q / 2))
                for p in p2:
                    acc[p] += c * 2 * gW(p[0], p[1], b)
            for p in p2:
                res[p] += wr * chr_ * G[p[0] - 2] * acc[p]
    # ---- k >= 3
    if p3:
        kmin = min(p[1] for p in p3)
        sig_ts = ts01(mp.mpf(prm['hsig']), prm.get('tsmax', 4.7))
        for ir, (r, wr) in enumerate(de_exp_nodes(mp.mpf(prm['hr']), prm['tlo3'], prm['thi3'])):
            if ir % P != ip: continue
            vn = v_nodes(r, prm.get('hvfac', 1), kmin, nats)
            gv = []
            for (w2, Q, wv) in vn:
                b = r + w2
                pre = wv / mp.sqrt(sinhc(w2 / 2) * sinhc(Q / 2) / 2)
                gv.append((mp.cosh(b), pre, {p: 2 * gW(p[0], p[1], b) for p in p3}))
            # sigma nodes: (s, D, weight dsigma)
            if r <= prm['rsplit']:
                sn = [(r * s, r * sc, w) for (s, sc, w) in sig_ts]
            else:
                ln = logmap_nodes(r / 2, prm.get('c0', '0.3'), mp.mpf(prm['hlog']), nats)
                sn = [(r - u, u, wu / r) for (u, ru, wu) in ln] + [(u, r - u, wu / r) for (u, ru, wu) in ln]
            for (s, D, ws) in sn:
                G = funcs.Gvec(s); F = funcs.fvec(D)
                C = mp.cosh(s - D)
                acc = {p: mp.mpf(0) for p in p3}
                for (chb, pre, g) in gv:
                    c = pre / mp.sqrt(chb + C)
                    for p in p3:
                        acc[p] += c * g[p]
                base = wr * r * ws * mp.cosh(s)
                for p in p3:
                    res[p] += base * F[p[1]] * G[p[0] - p[1]] * acc[p]
    return {p: pref[p] * res[p] for p in pairs}
